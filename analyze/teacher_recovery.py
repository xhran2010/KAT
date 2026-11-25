"""Teacher recoverability (实验3) + Top-5 token overlap (实验4) 分析.

实验3（teacher recoverability）：
  学生 (ckpt, 训到 step 60) 在 MATH-500 上 rollout；teacher (Qwen3-8B) 逐位置
  reverse-KL + 训练阈值 τ 的滑窗规则 (L=64, t=32, N=4) 找 agreement 起点
  m = t + i_first；teacher 从三个切点 c ∈ {clip(m-δ,1,Lc), m, clip(m+δ,1,Lc)}
  接着生成，比较三种切点下的平均正确率（对照学生原始 rollout 的 baseline）。
  可选 --perm_control：保留 rollout 间对 m 做固定种子 derangement 得 m'，
  加 {m'-δ, m', m'+δ} 三个对照切点，检验 m 是否携带位置之外的信号；
  可选 --min_m：剔除左删失的触发 rollout（m 被滑窗下界 t 卡住）。

实验4（top-k overlap）：
  统计 pre/in/after 三阶段内 teacher×student Top-5 token 的重叠率。

Stages（幂等：产物存在且非 --force 则跳过）：
  rollout   -> {out_dir}/rollouts.pt                （复用 AS.stage1_generate_rollouts）
  prefill   -> {out_dir}/per_token.parquet          （schema 与 AS.stage2 完全一致）
  analyze   -> {out_dir}/partition.json + cutpoints.parquet + grading.json
  continue  -> {out_dir}/continuations.jsonl        （vLLM teacher 续写，支持断点续跑）
  report    -> {out_dir}/recovery_results.csv + figures/F_recovery_acc.png
               + tables/T_recovery_acc.{md,csv} + report.md
               （--perm_control 时另有 tables/T_interaction.csv 与配对交互分析）
  overlap   -> {out_dir}/tables/T_topk_overlap.{md,csv} + figures/F_topk_overlap.png
               + report_topk_overlap.md
  all       -> 依次全部（仅本地调试；sbatch 建议逐 stage 独立起进程）

Run:
  cd <repo_root>
  python analyze/teacher_recovery.py --stage all \
    --ckpt outputs/xxx/checkpoint-60 --out_dir analyze/recovery_work
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CUT_NAMES = ["m_minus_delta", "m", "m_plus_delta"]
PERM_CUT_NAMES = ["m_perm_minus_delta", "m_perm", "m_perm_plus_delta"]
STAGES = ["pre", "in", "after"]


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["rollout", "prefill", "analyze", "continue", "report", "overlap", "all"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--teacher", default="Qwen/Qwen3-8B")
    p.add_argument("--prompt_parquet", default="datasets/test_data/MATH-500/test.parquet")
    p.add_argument("--num_prompts", type=int, default=200)
    p.add_argument("--rollouts_per_prompt", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    # 滑窗 / loss 口径（与训练一致）
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--t", type=int, default=32)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--top_k_loss", type=int, default=64)
    p.add_argument("--loss_temperature", type=float, default=1.0)
    # 阈值
    p.add_argument("--threshold_step", type=int, default=60,
                   help="取 log_history 中 step ≤ 此值的最近一条有效 kl_trunc/threshold")
    p.add_argument("--threshold", type=float, default=None, help="直接覆盖 τ")
    # 切点 / teacher 续写
    p.add_argument("--delta", type=int, default=64)
    p.add_argument("--perm_control", action="store_true",
                   help="加入 permuted-m 对照切点：保留 rollout 间对 m 做固定种子 "
                        "index 级 derangement，得 m'，再取 {m'-δ, m', m'+δ} 三个对照切点")
    p.add_argument("--perm_seed", type=int, default=123)
    p.add_argument("--min_m", type=int, default=None,
                   help="剔除 m < min_m 的触发 rollout（左删失：滑窗从 t 起检测，"
                        "m=t 表示真实 agreement 起点被下界卡住，m 为有偏估计）；"
                        "仍计入 partition.json 的触发统计")
    p.add_argument("--num_continuations", type=int, default=4)
    p.add_argument("--continue_temperature", type=float, default=0.7)
    p.add_argument("--continue_top_p", type=float, default=0.95)
    p.add_argument("--continue_max_tokens", type=int, default=2048)
    # 模型加载
    p.add_argument("--attn_impl", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--vllm_max_model_len", type=int, default=4096)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--student_gpu", type=int, default=0)
    p.add_argument("--teacher_gpu", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  懒加载 / 通用小工具（torch / AS / transformers / vllm 全部延迟导入）
# ──────────────────────────────────────────────────────────────────────────────

def _import_AS():
    """导入 analyze.analyze_stages（处理仓库根 sys.path，与 AS 引 math500_eval 同法）。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import analyze.analyze_stages as AS
    return AS


def _skip(paths: list[Path], force: bool, tag: str) -> bool:
    """产物全部存在且非 --force 则跳过。"""
    if not force and all(p.exists() for p in paths):
        print(f"[{tag}] 产物已存在，跳过: {[str(p) for p in paths]}")
        return True
    return False


def _load_bundle(out_dir: str) -> dict:
    import torch
    path = Path(out_dir) / "rollouts.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，请先跑 --stage rollout")
    return torch.load(path, weights_only=False)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """rollout 级别均值的 bootstrap 95% CI。"""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _derange_indices(n: int, seed: int) -> np.ndarray:
    """固定种子的 index 级 derangement（perm[i] != i，要求 n ≥ 2）。

    先 rng.permutation，再从左到右把不动点与右邻（末位与首位）交换：换来的值
    不可能等于自身下标（同一值在 permutation 中只出现一次），故一遍扫描即可
    消除全部不动点。注意这是 index 级保证；若 m 有重复值，值级碰撞
    （m'[i] == m[i]）不可避免，需另行统计上报。"""
    if n < 2:
        raise ValueError(f"derangement 需要 n ≥ 2，当前 n={n}")
    perm = np.random.default_rng(seed).permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    assert not np.any(perm == np.arange(n)), "derangement 消除不动点失败"
    return perm


# ──────────────────────────────────────────────────────────────────────────────
#  阈值：容错加载（不用 AS.load_training_threshold）
# ──────────────────────────────────────────────────────────────────────────────

def load_threshold_tolerant(ckpt_dir: str, step_limit: int) -> tuple[float, int]:
    """返回 (τ, 来源 step)：log_history 中 step ≤ step_limit 且含有限正
    kl_trunc/threshold 的条目里取最大 step 那条。"""
    ts_path = Path(ckpt_dir) / "trainer_state.json"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)
    hist = _load_json(ts_path).get("log_history", [])
    best: tuple[int, float] | None = None
    for entry in hist:
        v = entry.get("kl_trunc/threshold")
        s = entry.get("step")
        if not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
            continue
        if not isinstance(s, (int, float)) or s > step_limit:
            continue
        if best is None or int(s) >= best[0]:
            best = (int(s), float(v))
    if best is None:
        raise RuntimeError(
            f"{ts_path} 的 log_history 中找不到 step ≤ {step_limit} 的有效 kl_trunc/threshold")
    return best[1], best[0]


def resolve_threshold(args) -> tuple[float, int | None]:
    if args.threshold is not None:
        print(f"[threshold] τ = {args.threshold:.10g} (CLI 覆盖)")
        return float(args.threshold), None
    tau, step = load_threshold_tolerant(args.ckpt, args.threshold_step)
    print(f"[threshold] τ = {tau:.10g} (来自 trainer_state.json step {step}, 上限 {args.threshold_step})")
    return tau, step


# ──────────────────────────────────────────────────────────────────────────────
#  stage: rollout —— 直接复用 AS.stage1_generate_rollouts
# ──────────────────────────────────────────────────────────────────────────────

def stage_rollout(args) -> None:
    AS = _import_AS()
    rollouts_path = Path(args.out_dir) / "rollouts.pt"
    ns = argparse.Namespace(
        ckpt=args.ckpt,
        prompt_parquet=args.prompt_parquet,
        num_prompts=args.num_prompts,
        rollouts_per_prompt=args.rollouts_per_prompt,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_len,  # AS 用的字段名带 length
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        enable_thinking=args.enable_thinking,
        out_dir=args.out_dir,
        skip_rollouts=(rollouts_path.exists() and not args.force),  # 幂等
    )
    AS.stage1_generate_rollouts(ns)
    print(f"[rollout] done -> {rollouts_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  stage: prefill —— 复制 AS.stage2_per_token 小改（attn_impl / 单卡双模型）
# ──────────────────────────────────────────────────────────────────────────────

def stage_prefill(args) -> None:
    per_token_path = Path(args.out_dir) / "per_token.parquet"
    if _skip([per_token_path], args.force, "prefill"):
        return
    import torch
    from transformers import AutoModelForCausalLM
    AS = _import_AS()

    bundle = _load_bundle(args.out_dir)
    student_dev = torch.device(f"cuda:{args.student_gpu}")
    teacher_dev = torch.device(f"cuda:{args.teacher_gpu}")
    print(f"[prefill] student on {student_dev}, teacher on {teacher_dev}, attn={args.attn_impl}")

    common_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        low_cpu_mem_usage=True,
    )
    print(f"[prefill] loading student from {args.ckpt}")
    student = AutoModelForCausalLM.from_pretrained(args.ckpt, **common_kwargs).to(student_dev).eval()
    print(f"[prefill] loading teacher from {args.teacher}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **common_kwargs).to(teacher_dev).eval()

    rows: list[dict] = []
    n = len(bundle["prompt_ids"])
    for ridx in range(n):
        prompt = bundle["prompt_ids"][ridx]
        completion = bundle["completion_ids"][ridx]
        if len(completion) < 2:
            continue
        full_ids = torch.tensor(list(prompt) + list(completion), dtype=torch.long).unsqueeze(0)
        Ltot = full_ids.size(1)
        Lprompt = len(prompt)
        Lcomp = len(completion)
        with torch.no_grad():
            s_out = student(full_ids.to(student_dev), use_cache=False)
            t_out = teacher(full_ids.to(teacher_dev), use_cache=False)
            s_logits = s_out.logits[:, Lprompt - 1: Ltot - 1, :].float()
            t_logits = t_out.logits[:, Lprompt - 1: Ltot - 1, :].to(s_logits.device).float()
            del s_out, t_out
            rk = AS.per_token_reverse_kl_topk_tail(
                s_logits, t_logits, top_k=args.top_k_loss, temperature=args.loss_temperature
            )
            Hs = AS.per_token_entropy(s_logits)
            Ht = AS.per_token_entropy(t_logits)
            s_top5 = s_logits.topk(5, dim=-1).indices.squeeze(0).cpu().numpy()
            t_top5 = t_logits.topk(5, dim=-1).indices.squeeze(0).cpu().numpy()
        rk_np = rk.squeeze(0).cpu().numpy().astype(np.float32)
        hs_np = Hs.squeeze(0).cpu().numpy().astype(np.float32)
        ht_np = Ht.squeeze(0).cpu().numpy().astype(np.float32)
        for pos in range(Lcomp):
            rows.append({
                "rollout_id": ridx,
                "position": pos,
                "completion_len": Lcomp,
                "kl": float(rk_np[pos]),
                "H_s": float(hs_np[pos]),
                "H_t": float(ht_np[pos]),
                "s_top1": int(s_top5[pos, 0]),
                "s_top2": int(s_top5[pos, 1]),
                "s_top3": int(s_top5[pos, 2]),
                "s_top4": int(s_top5[pos, 3]),
                "s_top5": int(s_top5[pos, 4]),
                "t_top1": int(t_top5[pos, 0]),
                "t_top2": int(t_top5[pos, 1]),
                "t_top3": int(t_top5[pos, 2]),
                "t_top4": int(t_top5[pos, 3]),
                "t_top5": int(t_top5[pos, 4]),
                "tok_id": int(completion[pos]),
            })
        if (ridx + 1) % 32 == 0:
            print(f"[prefill]   {ridx + 1}/{n} done; rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(per_token_path, index=False)
    print(f"[prefill] saved {per_token_path} ({len(df)} rows)")

    del student, teacher
    gc.collect()
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
#  stage: analyze —— 分段 (partition.json) + 切点 (cutpoints.parquet) + 判分 (grading.json)
# ──────────────────────────────────────────────────────────────────────────────

def stage_analyze(args) -> None:
    out = Path(args.out_dir)
    partition_path = out / "partition.json"
    cutpoints_path = out / "cutpoints.parquet"
    grading_path = out / "grading.json"
    if _skip([partition_path, cutpoints_path, grading_path], args.force, "analyze"):
        return
    AS = _import_AS()

    tau, tau_step = resolve_threshold(args)
    per_token_df = pd.read_parquet(out / "per_token.parquet")
    bundle = _load_bundle(args.out_dir)

    # ─── 逐 rollout 滑窗分段：m = t + i_first；(0, pre_end, in_end, S) ───
    info: list[dict] = []
    for rid, g in per_token_df.groupby("rollout_id", sort=True):
        g_sorted = g.sort_values("position")
        kl_series = g_sorted["kl"].to_numpy()
        Lcomp = int(g_sorted["completion_len"].iloc[0])
        is_trunc, i_first, i_last, (_, pre_end, in_end, S) = AS._find_stages_for_rollout(
            kl_series, tau, args.L, args.t, args.N)
        info.append({
            "rollout_id": int(rid),
            "completion_len": Lcomp,
            "is_trunc": bool(is_trunc),
            "i_first": int(i_first),
            "i_last": int(i_last),
            "m": int(args.t + i_first) if is_trunc else -1,
            "pre_end": int(pre_end),
            "in_end": int(in_end),
        })
    n_trunc = sum(1 for r in info if r["is_trunc"])

    # ─── 保留集：触发且（若给定 --min_m）m ≥ min_m；m < min_m 为左删失，剔除 ───
    triggered = [r for r in info if r["is_trunc"]]
    if args.min_m is not None:
        excluded = [r for r in triggered if r["m"] < args.min_m]
        retained = [r for r in triggered if r["m"] >= args.min_m]
    else:
        excluded, retained = [], triggered

    # ─── permuted-m 对照：保留集内对 m 做固定种子 index 级 derangement ───
    perm_m: list[int] | None = None
    n_value_collisions = 0
    if args.perm_control:
        m_vals = [r["m"] for r in retained]
        pidx = _derange_indices(len(retained), args.perm_seed)
        perm_m = [int(m_vals[j]) for j in pidx]
        n_value_collisions = sum(1 for a, b in zip(m_vals, perm_m) if a == b)
        print(f"[analyze] perm_control: seed={args.perm_seed}, n={len(retained)}, "
              f"值级碰撞（m'==m，m 重复值所致）= {n_value_collisions}")

    partition = {
        "threshold": float(tau),
        "threshold_source_step": tau_step,
        "threshold_step_limit": args.threshold_step,
        "L": args.L, "t": args.t, "N": args.N, "delta": args.delta,
        "n_rollouts": len(info),
        "n_trunc": int(n_trunc),
        "trigger_frac": float(n_trunc / max(1, len(info))),
        "min_m": args.min_m,
        "n_excluded_left_censored": len(excluded),
        "n_retained": len(retained),
        "excluded_left_censored": [{"rollout_id": r["rollout_id"], "m": r["m"]} for r in excluded],
        "rollouts": info,
    }
    if args.perm_control:
        partition["perm_seed"] = args.perm_seed
        partition["perm_n_value_collisions"] = int(n_value_collisions)
        partition["perm_mapping"] = [
            {"rollout_id": r["rollout_id"], "m": r["m"], "m_perm": mp}
            for r, mp in zip(retained, perm_m)]
    with open(partition_path, "w") as f:
        json.dump(partition, f, indent=2, ensure_ascii=False)
    print(f"[analyze] partition: {n_trunc}/{len(info)} rollouts 触发 "
          f"({partition['trigger_frac']:.1%}) -> {partition_path}")
    if args.min_m is not None:
        print(f"[analyze] min_m={args.min_m}: 左删失剔除 {len(excluded)} 条, 保留 {len(retained)} 条")

    # ─── cutpoints：仅保留集，真实切点 3 行/条（--perm_control 时另加对照 3 行/条）───
    cut_rows: list[dict] = []
    for i, r in enumerate(retained):
        m, Lc = r["m"], r["completion_len"]
        cuts = [("m_minus_delta", m - args.delta), ("m", m), ("m_plus_delta", m + args.delta)]
        if perm_m is not None:
            mp = perm_m[i]
            cuts += [("m_perm_minus_delta", mp - args.delta), ("m_perm", mp),
                     ("m_perm_plus_delta", mp + args.delta)]
        for name, raw in cuts:
            cut_pos = int(np.clip(raw, 1, Lc))
            cut_rows.append({
                "rollout_id": r["rollout_id"],
                "cut_name": name,
                "raw_pos": int(raw),
                "cut_pos": cut_pos,
                "clipped": bool(cut_pos != raw),
                "completion_len": Lc,
            })
    cut_df = pd.DataFrame(cut_rows)
    cut_df.to_parquet(cutpoints_path, index=False)
    n_clip = int(cut_df["clipped"].sum()) if len(cut_df) else 0
    print(f"[analyze] cutpoints: {len(cut_df)} 行 (clipped={n_clip}) -> {cutpoints_path}")

    # ─── grading：学生原始 rollout 判分（对照基线）───
    _is_correct = AS._import_is_correct()
    items = []
    for rid in range(len(bundle["completion_text"])):
        ct = bundle["completion_text"][rid]
        gt = bundle["ground_truth"][rid]
        items.append({"rollout_id": rid, "ground_truth": gt, "is_correct": bool(_is_correct(ct, gt))})
        if (rid + 1) % 50 == 0:
            print(f"[analyze]   graded {rid + 1}/{len(bundle['completion_text'])}", flush=True)
    n_correct = sum(1 for x in items if x["is_correct"])
    grading = {"n_rollouts": len(items), "n_correct": n_correct, "items": items}
    with open(grading_path, "w") as f:
        json.dump(grading, f, indent=2, ensure_ascii=False)
    print(f"[analyze] baseline: {n_correct}/{len(items)} correct "
          f"({n_correct / max(1, len(items)):.1%}) -> {grading_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  stage: continue —— vLLM teacher 从各切点续写
# ──────────────────────────────────────────────────────────────────────────────

def stage_continue(args) -> None:
    out = Path(args.out_dir)
    cont_path = out / "continuations.jsonl"
    import torch
    from vllm import LLM, SamplingParams
    try:
        from vllm import TokensPrompt
    except ImportError:
        from vllm.inputs import TokensPrompt

    bundle = _load_bundle(args.out_dir)
    cut_df = pd.read_parquet(out / "cutpoints.parquet")

    # 断点续跑：已有 ≥ num_continuations 条的 (rollout_id, cut_name) 跳过
    done: set[tuple[int, str]] = set()
    if cont_path.exists():
        if args.force:
            cont_path.unlink()
            print(f"[continue] --force：删除旧 {cont_path}")
        else:
            counts: dict[tuple[int, str], int] = {}
            with open(cont_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (int(rec["rollout_id"]), str(rec["cut_name"]))
                    counts[key] = counts.get(key, 0) + 1
            done = {k for k, v in counts.items() if v >= args.num_continuations}
            print(f"[continue] 续跑：已完成 {len(done)} 个 (rollout, cut)")

    tasks: list[dict] = []
    for rec in cut_df.to_dict("records"):
        rid, cname, cpos = int(rec["rollout_id"]), str(rec["cut_name"]), int(rec["cut_pos"])
        if (rid, cname) in done:
            continue
        prefix_ids = list(bundle["prompt_ids"][rid]) + list(bundle["completion_ids"][rid])[:cpos]
        tasks.append({"rollout_id": rid, "cut_name": cname, "cut_pos": cpos, "prefix_ids": prefix_ids})
    if not tasks:
        print(f"[continue] 无待跑任务 -> {cont_path}")
        return

    max_prefix = max(len(t["prefix_ids"]) for t in tasks)
    max_model_len = max(8192, max_prefix + args.continue_max_tokens)
    print(f"[continue] {len(tasks)} 任务, max_prefix={max_prefix}, max_model_len={max_model_len}")

    llm = LLM(
        model=args.teacher,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        seed=args.seed,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sp = SamplingParams(
        n=args.num_continuations,
        temperature=args.continue_temperature,
        top_p=args.continue_top_p,
        max_tokens=args.continue_max_tokens,
        seed=args.seed,
    )
    prompts = [TokensPrompt(prompt_token_ids=t["prefix_ids"]) for t in tasks]
    print(f"[continue] vLLM 批量生成 {len(prompts)} × n={args.num_continuations}")
    outs = llm.generate(prompts, sp)

    with open(cont_path, "a") as f:
        for task, o in zip(tasks, outs):
            for ci, sub in enumerate(o.outputs):
                f.write(json.dumps({
                    "rollout_id": task["rollout_id"],
                    "cut_name": task["cut_name"],
                    "cut_pos": task["cut_pos"],
                    "continuation_idx": ci,
                    "text": sub.text,
                    "finish_reason": sub.finish_reason,
                }, ensure_ascii=False) + "\n")
    print(f"[continue] appended {len(tasks) * args.num_continuations} 行 -> {cont_path}")

    # 释放 vLLM（与 AS 同法）
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  stage: report —— 判分聚合 + 图 + 表 + 中文报告
# ──────────────────────────────────────────────────────────────────────────────

def stage_report(args) -> None:
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    results_path = out / "recovery_results.csv"
    fig_path = fig_dir / "F_recovery_acc.png"
    report_path = out / "report.md"
    cut_df = pd.read_parquet(out / "cutpoints.parquet")
    has_perm = bool(set(PERM_CUT_NAMES) & set(cut_df["cut_name"].unique()))
    skip_paths = [results_path, fig_path, report_path,
                  tab_dir / "T_recovery_acc.csv", tab_dir / "T_recovery_acc.md"]
    if has_perm:
        skip_paths.append(tab_dir / "T_interaction.csv")
    if _skip(skip_paths, args.force, "report"):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    AS = _import_AS()
    _is_correct = AS._import_is_correct()

    bundle = _load_bundle(args.out_dir)
    partition = _load_json(out / "partition.json")
    grading = _load_json(out / "grading.json")
    tok = AutoTokenizer.from_pretrained(args.ckpt)

    conts: list[dict] = []
    with open(out / "continuations.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                conts.append(json.loads(line))
    if not conts:
        raise RuntimeError(f"{out / 'continuations.jsonl'} 为空，请先跑 --stage continue")
    print(f"[report] loaded {len(conts)} continuations")

    gt_map = {it["rollout_id"]: it["ground_truth"] for it in grading["items"]}
    base_map = {it["rollout_id"]: it["is_correct"] for it in grading["items"]}
    clip_map = {(int(r["rollout_id"]), str(r["cut_name"])): bool(r["clipped"])
                for r in cut_df.to_dict("records")}
    # 保留集 = cutpoints 中出现的 rollout（无 --min_m 时即触发集）
    retained_rids = sorted(int(r) for r in cut_df["rollout_id"].unique())

    # 前缀解码缓存：同一 (rollout, cut_pos) 只解码一次
    prefix_cache: dict[tuple[int, int], str] = {}

    def _prefix_text(rid: int, cpos: int) -> str:
        key = (rid, cpos)
        if key not in prefix_cache:
            prefix_cache[key] = tok.decode(list(bundle["completion_ids"][rid])[:cpos],
                                           skip_special_tokens=False)
        return prefix_cache[key]

    # ─── 逐条判分：完整答案 = 学生前缀 + teacher 续写 ───
    per_cont: list[dict] = []
    for i, c in enumerate(conts):
        rid = int(c["rollout_id"])
        full_text = _prefix_text(rid, int(c["cut_pos"])) + c["text"]
        per_cont.append({
            "rollout_id": rid,
            "cut_name": c["cut_name"],
            "cut_pos": int(c["cut_pos"]),
            "continuation_idx": int(c["continuation_idx"]),
            "finish_reason": c.get("finish_reason"),
            "is_correct": bool(_is_correct(full_text, gt_map[rid])),
        })
        if (i + 1) % 200 == 0:
            print(f"[report]   graded {i + 1}/{len(conts)}", flush=True)
    pc_df = pd.DataFrame(per_cont)

    # 每 (rollout, cut) 取 num_continuations 条平均
    rc = (pc_df.groupby(["rollout_id", "cut_name"])
          .agg(n_cont=("is_correct", "size"), acc=("is_correct", "mean"))
          .reset_index())
    rc["clipped"] = rc.apply(lambda r: clip_map.get((int(r["rollout_id"]), str(r["cut_name"])), False), axis=1)
    rc["baseline_correct"] = rc["rollout_id"].map(base_map)
    rc.to_csv(results_path, index=False)
    print(f"[report] saved {results_path} ({len(rc)} rows)")

    # ─── 每切点汇总：跨保留 rollout 平均 + bootstrap 95% CI（真实 3 + perm 3（若有））───
    present_cuts = [c for c in CUT_NAMES + PERM_CUT_NAMES
                    if c in set(cut_df["cut_name"].unique())]
    sum_rows: list[dict] = []
    for cname in present_cuts:
        sub = rc[rc["cut_name"] == cname]
        accs = sub["acc"].to_numpy()
        lo, hi = _bootstrap_ci(accs, n_boot=1000, seed=args.seed)
        sum_rows.append({
            "cut_name": cname,
            "n_rollouts": int(len(sub)),
            "n_clipped": int(sub["clipped"].sum()),
            "mean_acc": float(accs.mean()) if len(accs) else float("nan"),
            "ci_low": lo, "ci_high": hi,
        })
    # 学生 baseline（保留子集 + 全体；无 --min_m 时保留子集 = 触发子集）
    base_label = ("student_baseline(触发子集)" if partition.get("min_m") is None
                  else "student_baseline(保留子集)")
    base_trunc = np.array([float(base_map[r]) for r in retained_rids]) if retained_rids else np.array([])
    lo_b, hi_b = _bootstrap_ci(base_trunc, n_boot=1000, seed=args.seed)
    baseline_trunc_acc = float(base_trunc.mean()) if len(base_trunc) else float("nan")
    baseline_all_acc = grading["n_correct"] / max(1, grading["n_rollouts"])
    sum_rows.append({
        "cut_name": base_label,
        "n_rollouts": int(len(base_trunc)),
        "n_clipped": 0,
        "mean_acc": baseline_trunc_acc,
        "ci_low": lo_b, "ci_high": hi_b,
    })
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(tab_dir / "T_recovery_acc.csv", index=False)
    (tab_dir / "T_recovery_acc.md").write_text(
        summary.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")
    print(f"[report] summary:\n{summary.to_string(index=False)}")

    # ─── 图：柱 + CI 误差线 + baseline 虚线（期刊风格，灰阶可读，300 dpi）
    #      无 perm：三根单柱（旧版）；有 perm：{−Δ, m, +Δ} 三组 × (real, permuted) 双柱 ───
    cut_labels = {"m_minus_delta": f"m − {args.delta}", "m": "m", "m_plus_delta": f"m + {args.delta}"}

    def _plot_stats(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        rows_ = summary.set_index("cut_name").loc[names]
        mm = rows_["mean_acc"].to_numpy()
        ee = np.vstack([mm - rows_["ci_low"].to_numpy(), rows_["ci_high"].to_numpy() - mm])
        return mm, ee

    if has_perm:
        xs = np.arange(len(CUT_NAMES))
        w = 0.36
        means, errs = _plot_stats(CUT_NAMES)
        means_p, errs_p = _plot_stats(PERM_CUT_NAMES)
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.bar(xs - w / 2, means, width=w, color="#4878a8", edgecolor="black", linewidth=0.8,
               yerr=errs, capsize=3, error_kw={"lw": 1.0, "ecolor": "black"}, label="real (m)")
        ax.bar(xs + w / 2, means_p, width=w, color="#b0b7c0", edgecolor="black", linewidth=0.8,
               yerr=errs_p, capsize=3, error_kw={"lw": 1.0, "ecolor": "black"}, label="permuted (m′)")
        for x, v in list(zip(xs - w / 2, means)) + list(zip(xs + w / 2, means_p)):
            if np.isfinite(v):
                ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        y_top_vals = np.concatenate([means, means_p])
        xtick_labels = [cut_labels[c] for c in CUT_NAMES]
    else:
        plot_rows = summary[summary["cut_name"].isin(CUT_NAMES)]
        xs = np.arange(len(plot_rows))
        means, errs = _plot_stats(list(plot_rows["cut_name"]))
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.bar(xs, means, width=0.55, color="#4878a8", edgecolor="black", linewidth=0.8,
               yerr=errs, capsize=4, error_kw={"lw": 1.0, "ecolor": "black"})
        for x, v in zip(xs, means):
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        y_top_vals = means
        xtick_labels = [cut_labels[c] for c in plot_rows["cut_name"]]
    ax.axhline(baseline_trunc_acc, color="0.35", ls="--", lw=1.2,
               label=f"student baseline = {baseline_trunc_acc:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("teacher continuation cut point")
    ax.set_ylabel("mean accuracy (MATH-500)")
    y_top = float(np.nanmax(y_top_vals)) + 0.15 if np.isfinite(y_top_vals).any() else 1.0
    ax.set_ylim(0, min(1.05, max(1.0, y_top)))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Teacher recoverability by cut point", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[report] figure -> {fig_path}")

    # ─── 交互统计（仅 perm 存在时）：逐 rollout 配对，6 切点齐全者 ───
    interaction_md = ""
    if has_perm:
        all6 = CUT_NAMES + PERM_CUT_NAMES
        piv6 = (rc.pivot(index="rollout_id", columns="cut_name", values="acc")
                .reindex(columns=all6).dropna())
        n_paired = int(len(piv6))
        gap_real = (piv6["m_minus_delta"] - piv6["m_plus_delta"]).to_numpy()
        gap_perm = (piv6["m_perm_minus_delta"] - piv6["m_perm_plus_delta"]).to_numpy()
        interaction = gap_real - gap_perm
        center = (piv6["m"] - piv6["m_perm"]).to_numpy()
        # 敏感性：剔除 6 切点中任一被 clip（实际均为上界 clip）的 rollout
        clip_any = cut_df.groupby("rollout_id")["clipped"].any()
        keep_mask = np.array([not bool(clip_any.get(rid, False)) for rid in piv6.index])
        n_clip_dropped = n_paired - int(keep_mask.sum())

        def _stat_row(name: str, vals: np.ndarray, note: str = "") -> dict:
            lo, hi = _bootstrap_ci(vals, n_boot=1000, seed=args.seed)
            return {"stat": name, "n_rollouts": int(len(vals)),
                    "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                    "ci_low": lo, "ci_high": hi, "note": note}

        s_gr = _stat_row("gap_real = acc(m−Δ) − acc(m+Δ)", gap_real)
        s_gp = _stat_row("gap_perm = acc(m′−Δ) − acc(m′+Δ)", gap_perm)
        s_int = _stat_row("interaction = gap_real − gap_perm", interaction)
        s_ctr = _stat_row("center = acc(m) − acc(m′)", center)
        s_sens = _stat_row("interaction（剔除含 clip 切点的 rollout）",
                           interaction[keep_mask], note=f"dropped={n_clip_dropped}")
        inter_df = pd.DataFrame([s_gr, s_gp, s_int, s_ctr, s_sens])
        inter_df.to_csv(tab_dir / "T_interaction.csv", index=False)
        print(f"[report] interaction:\n{inter_df.to_string(index=False)}")

        def _sig(s: dict) -> bool:
            return (np.isfinite(s["ci_low"]) and np.isfinite(s["ci_high"])
                    and (s["ci_low"] > 0 or s["ci_high"] < 0))

        def _fmt(s: dict) -> str:
            return (f"{s['mean']:+.3f}（95% CI [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]，"
                    f"n={s['n_rollouts']}）")

        if _sig(s_int):
            concl_int = (f"交互项 = {_fmt(s_int)}，CI 不含 0，gap_real 显著"
                         f"{'大于' if s_int['mean'] > 0 else '小于'} gap_perm："
                         "说明 m 携带位置之外的信号、切点位置选取有效。")
        else:
            concl_int = (f"交互项 = {_fmt(s_int)}，CI 含 0，gap_real 与 gap_perm 无显著差异："
                         "观察到的切点效应不能与单纯的位置效应区分（不能排除仅是位置效应）。")
        if _sig(s_ctr):
            concl_ctr = (f"中心对比 acc(m) − acc(m′) = {_fmt(s_ctr)}，CI 不含 0，"
                         f"在 m 处切显著{'优于' if s_ctr['mean'] > 0 else '劣于'}在打乱位置 m′ 处切。")
        else:
            concl_ctr = (f"中心对比 acc(m) − acc(m′) = {_fmt(s_ctr)}，CI 含 0，"
                         "在 m 与 m′ 处切无显著差异。")
        concl_sens = (f"敏感性分析（剔除 {n_clip_dropped} 条 6 切点中任一被上界 clip 的 rollout）"
                      f"后交互项 = {_fmt(s_sens)}，与主分析结论"
                      f"{'一致' if _sig(s_sens) == _sig(s_int) else '不一致，需谨慎解读'}。")

        interaction_md = f"""
## Permuted-m 对照与交互分析（逐 rollout 配对，6 切点齐全 n={n_paired}）

判据：若 gap_real 显著大于 gap_perm（交互项 CI 不含 0），说明 m 携带位置之外的信号、切点位置选取有效；若无差异，则仅是位置效应。

{inter_df.to_markdown(index=False, floatfmt=".4g")}

- gap_real = {_fmt(s_gr)}；gap_perm = {_fmt(s_gp)}。
- {concl_int}
- {concl_ctr}
- {concl_sens}
"""

    # ─── 中文报告 ───
    tau = partition["threshold"]
    tau_step = partition.get("threshold_source_step")
    n_total = partition["n_rollouts"]
    n_trunc = partition["n_trunc"]

    extra_setup_rows = ""
    if partition.get("min_m") is not None:
        extra_setup_rows += (
            f"\n| min_m（左删失剔除）| 剔除 m < {partition['min_m']} 的触发 rollout "
            f"{partition.get('n_excluded_left_censored', 0)} 条"
            f"（左删失：滑窗从 t={args.t} 起检测，m={args.t} 表示真实 agreement 起点被下界卡住，"
            f"m 为有偏估计），保留 n={partition.get('n_retained')} |")
    if has_perm:
        extra_setup_rows += (
            f"\n| Permuted-m 对照 | perm_seed={partition.get('perm_seed')}，"
            f"index 级 derangement（perm 下标 ≠ 自身下标），"
            f"值级碰撞（m′==m）{partition.get('perm_n_value_collisions', 0)} 条；"
            f"对照切点 c′ ∈ {{clip(m′−{args.delta}, 1, Lc), m′, clip(m′+{args.delta}, 1, Lc)}} |")

    retained_note = ""
    if partition.get("min_m") is not None:
        retained_note = (
            f"\n- 左删失剔除（m < {partition['min_m']}）= "
            f"{partition.get('n_excluded_left_censored', 0)} 条，"
            f"保留参与切点比较 = {partition.get('n_retained')} 条。")

    def _row(cname):
        return summary[summary["cut_name"] == cname].iloc[0]

    r_minus, r_m, r_plus = _row("m_minus_delta"), _row("m"), _row("m_plus_delta")
    best = max([r_minus, r_m, r_plus], key=lambda r: r["mean_acc"])
    scope_word = "触发" if partition.get("min_m") is None else "保留"
    n_cuts_word = "六切点（真实 3 + permuted 3）" if has_perm else "三切点"
    inter_prod = f"\n- `{tab_dir / 'T_interaction.csv'}`" if has_perm else ""
    md = f"""# 实验3 · Teacher recoverability（从不同切点由 teacher 续写的正确率）

## 实验设置

| 项目 | 值 |
|---|---|
| Student ckpt | `{args.ckpt}` |
| Teacher | `{args.teacher}` |
| Prompt 源 | `{args.prompt_parquet}` 前 {args.num_prompts} 条 × {args.rollouts_per_prompt} rollout |
| Rollout 采样 | T={args.temperature}, top_p={args.top_p}, max_tokens={args.max_completion_length}, seed={args.seed} |
| 阈值 τ | {tau:.10g}（{"CLI 覆盖" if tau_step is None else f"trainer_state.json step {tau_step}（上限 --threshold_step {args.threshold_step}）"}） |
| 滑窗规则 | L={args.L}, t={args.t}, N={args.N}；m = t + i_first |
| 切点 | Δ={args.delta}；c ∈ {{clip(m−{args.delta}, 1, Lc), m, clip(m+{args.delta}, 1, Lc)}} |{extra_setup_rows}
| Teacher 续写 | n={args.num_continuations}/切点, T={args.continue_temperature}, top_p={args.continue_top_p}, max_tokens={args.continue_max_tokens}, seed={args.seed} |
| 判分 | `math500_eval._is_correct`（math_verify + \\boxed fallback），完整答案 = 学生前缀 + teacher 续写 |

## 触发情况

- 总 rollout 数 = {n_total}，触发滑窗规则 = {n_trunc}（**{n_trunc / max(1, n_total):.1%}**）。
- 仅{scope_word}的 rollout 参与切点续写比较。{retained_note}

## {n_cuts_word}正确率（{scope_word} rollout 上，每 rollout 先对 {args.num_continuations} 条续写取平均）

{summary.to_markdown(index=False, floatfmt=".4g")}

- 学生 baseline（{scope_word}子集，原始 rollout 直接判分）= **{baseline_trunc_acc:.3f}**（n={len(base_trunc)}，95% CI [{lo_b:.3f}, {hi_b:.3f}]）；全体 rollout baseline = {baseline_all_acc:.3f}。

![F_recovery_acc](figures/F_recovery_acc.png)

## 结果描述

- m−{args.delta} 切点平均正确率 = {r_minus['mean_acc']:.3f}（95% CI [{r_minus['ci_low']:.3f}, {r_minus['ci_high']:.3f}]，n={int(r_minus['n_rollouts'])}，clip {int(r_minus['n_clipped'])} 条）。
- m 切点平均正确率 = {r_m['mean_acc']:.3f}（95% CI [{r_m['ci_low']:.3f}, {r_m['ci_high']:.3f}]，n={int(r_m['n_rollouts'])}，clip {int(r_m['n_clipped'])} 条）。
- m+{args.delta} 切点平均正确率 = {r_plus['mean_acc']:.3f}（95% CI [{r_plus['ci_low']:.3f}, {r_plus['ci_high']:.3f}]，n={int(r_plus['n_rollouts'])}，clip {int(r_plus['n_clipped'])} 条）。
- 三切点中最高的是 **{best['cut_name']}**（{best['mean_acc']:.3f}）；与学生 baseline（{baseline_trunc_acc:.3f}）之差为 {best['mean_acc'] - baseline_trunc_acc:+.3f}。
- 若三切点 CI 相互重叠，则不能断言切点位置对可恢复性有显著影响，如实报告为无显著差异；反之，若越早切正确率越高，说明 agreement 起点 m 之后的学生轨迹已把 teacher 也"带偏"（teacher 顺着低 KL 前缀难以恢复）。
{interaction_md}
## 产物

- `{out / 'rollouts.pt'}`
- `{out / 'per_token.parquet'}`
- `{out / 'partition.json'}` / `{out / 'cutpoints.parquet'}` / `{out / 'grading.json'}`
- `{out / 'continuations.jsonl'}`
- `{results_path}`
- `{tab_dir / 'T_recovery_acc.csv'}` / `{tab_dir / 'T_recovery_acc.md'}`{inter_prod}
- `{fig_path}`
"""
    report_path.write_text(md, encoding="utf-8")
    print(f"[report] report -> {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  stage: overlap —— 实验4：三阶段 Top-5 token 重叠率
# ──────────────────────────────────────────────────────────────────────────────

def stage_overlap(args) -> None:
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "F_topk_overlap.png"
    report_path = out / "report_topk_overlap.md"
    if _skip([tab_dir / "T_topk_overlap.csv", tab_dir / "T_topk_overlap.md", fig_path, report_path],
             args.force, "overlap"):
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(out / "per_token.parquet")
    partition = _load_json(out / "partition.json")
    info = {r["rollout_id"]: r for r in partition["rollouts"]}

    # 每位置 overlap = |{s_top1..5} ∩ {t_top1..5}| / 5（向量化逐 k 比较）
    s_cols = [f"s_top{k}" for k in range(1, 6)]
    t_cols = [f"t_top{k}" for k in range(1, 6)]
    s_arr = df[s_cols].to_numpy()
    t_arr = df[t_cols].to_numpy()
    match = (s_arr[:, :, None] == t_arr[:, None, :]).any(axis=2)  # 每个 s 候选是否出现在 t top5
    df = df.copy()
    df["overlap"] = match.sum(axis=1) / 5.0

    # 阶段划分：仅触发 rollout；position < pre_end → pre；< in_end → in；否则 after
    def _stage_of(rid: int, pos: int) -> str | None:
        r = info.get(int(rid))
        if r is None:
            return None
        if not r["is_trunc"]:
            return "no_trunc"
        if pos < r["pre_end"]:
            return "pre"
        if pos < r["in_end"]:
            return "in"
        return "after"

    rid_np = df["rollout_id"].to_numpy()
    pos_np = df["position"].to_numpy()
    df["stage"] = [_stage_of(r, p) for r, p in zip(rid_np, pos_np)]
    print(f"[overlap] {len(df)} tokens; stage 计数 = {df['stage'].value_counts().to_dict()}")

    # (a) 每 rollout 每阶段先平均，再跨 rollout 平均（±std）；(b) 阶段内 token 池化
    rows: list[dict] = []
    for s in STAGES:
        sub = df[df["stage"] == s]
        per_roll = sub.groupby("rollout_id")["overlap"].mean()
        rows.append({
            "stage": s,
            "n_rollouts": int(per_roll.size),
            "n_tokens": int(len(sub)),
            "mean_per_rollout": float(per_roll.mean()) if per_roll.size else float("nan"),
            "std_per_rollout": float(per_roll.std(ddof=0)) if per_roll.size else float("nan"),
            "mean_pooled": float(sub["overlap"].mean()) if len(sub) else float("nan"),
        })
    # no_trunc rollout 整体平均作参考行
    nt = df[df["stage"] == "no_trunc"]
    nt_per_roll = nt.groupby("rollout_id")["overlap"].mean()
    rows.append({
        "stage": "no_trunc(整条参考)",
        "n_rollouts": int(nt_per_roll.size),
        "n_tokens": int(len(nt)),
        "mean_per_rollout": float(nt_per_roll.mean()) if nt_per_roll.size else float("nan"),
        "std_per_rollout": float(nt_per_roll.std(ddof=0)) if nt_per_roll.size else float("nan"),
        "mean_pooled": float(nt["overlap"].mean()) if len(nt) else float("nan"),
    })
    T = pd.DataFrame(rows)
    T.to_csv(tab_dir / "T_topk_overlap.csv", index=False)
    (tab_dir / "T_topk_overlap.md").write_text(T.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")
    print(f"[overlap] table:\n{T.to_string(index=False)}")

    # ─── 图：三阶段两种平均并排柱状（灰阶可读，300 dpi）───
    stage_rows = T[T["stage"].isin(STAGES)]
    xs = np.arange(len(stage_rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(xs - w / 2, stage_rows["mean_per_rollout"], width=w, color="#4878a8",
           edgecolor="black", linewidth=0.8,
           yerr=stage_rows["std_per_rollout"], capsize=3,
           error_kw={"lw": 1.0, "ecolor": "black"},
           label="per-rollout mean (±std)")
    ax.bar(xs + w / 2, stage_rows["mean_pooled"], width=w, color="#b0b7c0",
           edgecolor="black", linewidth=0.8, label="token-pooled mean")
    nt_ref = float(T[T["stage"] == "no_trunc(整条参考)"]["mean_per_rollout"].iloc[0])
    if np.isfinite(nt_ref):
        ax.axhline(nt_ref, color="0.35", ls="--", lw=1.2, label=f"no_trunc ref = {nt_ref:.3f}")
    for x, v in zip(xs - w / 2, stage_rows["mean_per_rollout"]):
        if np.isfinite(v):
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(stage_rows["stage"])
    ax.set_xlabel("stage")
    ax.set_ylabel("Top-5 overlap  |S∩T| / 5")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Student × Teacher Top-5 token overlap by stage", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"[overlap] figure -> {fig_path}")

    # ─── 中文报告 ───
    def _t(stage, col):
        sub = T[T["stage"] == stage]
        return float(sub.iloc[0][col]) if len(sub) else float("nan")

    tau = partition["threshold"]
    md = f"""# 实验4 · 三阶段 Student × Teacher Top-5 token 重叠率

## 实验设置

| 项目 | 值 |
|---|---|
| Student ckpt | `{args.ckpt}` |
| Teacher | `{args.teacher}` |
| 数据 | `{out / 'per_token.parquet'}`（prefill 逐位置 top-5，与训练 loss 同一前向）|
| 阈值 τ | {tau:.10g} |
| 滑窗规则 | L={partition['L']}, t={partition['t']}, N={partition['N']} |
| 重叠率定义 | 每位置 \\|{{s_top1..5}} ∩ {{t_top1..5}}\\| / 5 |
| 阶段划分 | 仅触发 rollout：position < pre_end → pre；< in_end → in；否则 after |

- 总 rollout = {partition['n_rollouts']}，触发 = {partition['n_trunc']}（{partition['trigger_frac']:.1%}）。

## 结果

{T.to_markdown(index=False, floatfmt=".4g")}

两种平均：
- (a) `mean_per_rollout`：每 rollout 每阶段先平均、再跨 rollout 平均（±std）；
- (b) `mean_pooled`：阶段内全部 token 池化平均。
- `no_trunc(整条参考)`：未触发 rollout 全序列的重叠率，作参考基线。

![F_topk_overlap](figures/F_topk_overlap.png)

## 结果描述

- pre / in / after 的 per-rollout 平均重叠率分别为 {_t('pre', 'mean_per_rollout'):.3f} / {_t('in', 'mean_per_rollout'):.3f} / {_t('after', 'mean_per_rollout'):.3f}（池化：{_t('pre', 'mean_pooled'):.3f} / {_t('in', 'mean_pooled'):.3f} / {_t('after', 'mean_pooled'):.3f}）；no_trunc 参考 = {_t('no_trunc(整条参考)', 'mean_per_rollout'):.3f}。
- in 段重叠率相对 pre 段的变化为 {_t('in', 'mean_per_rollout') - _t('pre', 'mean_per_rollout'):+.3f}：若显著升高，印证低 KL 触发段内 teacher/student 候选集合高度一致（agreement 而非教师误判）；若接近，则说明 top-5 重叠对该滑窗信号不敏感。
- after 段相对 in 段的变化为 {_t('after', 'mean_per_rollout') - _t('in', 'mean_per_rollout'):+.3f}，反映离开低 KL 段后两模型候选集合是否重新分化。

## 产物

- `{tab_dir / 'T_topk_overlap.csv'}` / `{tab_dir / 'T_topk_overlap.md'}`
- `{fig_path}`
- `{report_path}`
"""
    report_path.write_text(md, encoding="utf-8")
    print(f"[overlap] report -> {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────────────────

def _cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    stage_funcs = {
        "rollout": stage_rollout,
        "prefill": stage_prefill,
        "analyze": stage_analyze,
        "continue": stage_continue,
        "report": stage_report,
        "overlap": stage_overlap,
    }
    if args.stage == "all":
        # 仅本地调试用：GPU stage 之间尽力回收显存；sbatch 建议逐 stage 独立起进程
        for name in ["rollout", "prefill", "analyze", "continue", "report", "overlap"]:
            print(f"\n========== stage: {name} ==========")
            stage_funcs[name](args)
            _cleanup_gpu()
    else:
        stage_funcs[args.stage](args)
    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
