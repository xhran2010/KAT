"""Balanced 100-correct + 100-incorrect heatmap per checkpoint.

Pipeline:
  1. vLLM-rollout a pool of `--candidate_pool` prompts × 1 each (resumes
     existing pool if file present).
  2. Math-verify-grade each completion against DAPO ground truth.
  3. Stratified-sample `--n_correct` correct + `--n_incorrect` incorrect (deterministic, seed 42).
  4. HF prefill on the 200 selected rollouts only -> per-token reverse-KL.
  5. Compute z_t (partial-window warmup), normalize to [0,1] x grid, render
     a 1×2 heatmap (correct | incorrect), rows sorted triggered-then-non-triggered
     by normalized trigger position within each column.

Run:
  CUDA_VISIBLE_DEVICES=0,1 python \\
    analyze/heatmap_balanced.py \\
    --ckpt outputs/1.7b_p2048_eta0.3_s32/checkpoint-380 \\
    --threshold_step 379 \\
    --candidate_pool 1800 \\
    --out_dir analyze/work_heatmap_balanced_ckpt380
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--teacher", default="Qwen/Qwen3-8B")
    p.add_argument("--prompt_parquet",
                   default="data/dapo-math-17k/dapo_clean.parquet")
    p.add_argument("--candidate_pool", type=int, default=2000)
    p.add_argument("--n_correct", type=int, default=100)
    p.add_argument("--n_incorrect", type=int, default=100)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--L0", type=int, default=32)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--top_k_loss", type=int, default=64)
    p.add_argument("--loss_temperature", type=float, default=1.0)
    p.add_argument("--threshold_step", type=int, required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45)
    p.add_argument("--vllm_max_model_length", type=int, default=4096)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    p.add_argument("--student_gpu", type=int, default=0)
    p.add_argument("--teacher_gpu", type=int, default=1)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--n_bins", type=int, default=200)
    p.add_argument("--colormap", default="magma")
    p.add_argument("--skip_rollouts", action="store_true")
    p.add_argument("--skip_grade", action="store_true")
    p.add_argument("--skip_prefill", action="store_true")
    return p.parse_args()


# ---- shared math ----------------------------------------------------------

def _add_tail_bucket(log_probs):
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail = torch.log(-torch.expm1(log_sum))
    return torch.cat([log_probs, tail], dim=-1)


def per_token_reverse_kl_topk_tail(student_logits, teacher_logits, top_k=64, temperature=1.0):
    s = student_logits / temperature
    t = teacher_logits / temperature
    V = s.size(-1)
    if top_k <= 0 or top_k >= V:
        s_lp = F.log_softmax(s, dim=-1); t_lp = F.log_softmax(t, dim=-1)
        return (s_lp.exp() * (s_lp - t_lp)).sum(-1)
    _, support = s.topk(top_k, dim=-1)
    s_lp = F.log_softmax(s, dim=-1).gather(-1, support)
    t_lp = F.log_softmax(t, dim=-1).gather(-1, support)
    s_lp = _add_tail_bucket(s_lp); t_lp = _add_tail_bucket(t_lp)
    out = s_lp.exp() * (s_lp - t_lp)
    return torch.nan_to_num(out, nan=0.0).sum(-1)


def load_training_threshold(ckpt_dir: str, step: int) -> float:
    with open(Path(ckpt_dir) / "trainer_state.json") as f:
        s = json.load(f)
    for entry in s.get("log_history", []):
        if entry.get("step") == step:
            v = entry.get("kl_trunc/threshold")
            if isinstance(v, (int, float)) and v == v and v > 0:
                return float(v)
    raise RuntimeError(f"No valid kl_trunc/threshold at step={step}")


def _z_series(kl: np.ndarray, W: int) -> np.ndarray:
    n = kl.shape[0]
    z = np.full(n, np.nan, dtype=np.float32)
    if n < 1:
        return z
    csum = np.concatenate([[0.0], np.cumsum(kl.astype(np.float64))])
    t = np.arange(n)
    lo = np.maximum(0, t - W + 1)
    width = (t - lo + 1).astype(np.float64)
    z[:] = ((csum[t + 1] - csum[lo]) / width).astype(np.float32)
    return z


def _resample_to_grid(z: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full(n_bins, np.nan, dtype=np.float32)
    L = z.shape[0]
    if L < 1:
        return out
    src_x = (np.arange(L, dtype=np.float64) + 1.0) / float(L)
    tgt_x = (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
    idx = np.clip(np.searchsorted(src_x, tgt_x, side="right") - 1, 0, L - 1)
    out[:] = z[idx]
    return out


def _trigger_for(kl: np.ndarray, threshold: float, W: int, L0: int, N: int) -> int:
    S = kl.shape[0]
    if S - L0 < W or S - L0 < W + N - 1:
        return -1
    csum = np.concatenate([[0.0], np.cumsum(kl[L0:].astype(np.float64))])
    Wn = (S - L0) - W + 1
    win_mean = (csum[W: W + Wn] - csum[:Wn]) / float(W)
    low = win_mean < threshold
    if Wn < N:
        return -1
    conv = np.convolve(low.astype(np.int32), np.ones(N, dtype=np.int32), mode="valid")
    hits = conv >= N
    if not hits.any():
        return -1
    return L0 + int(np.argmax(hits)) + W


# ---- Stage 1: vLLM pool ----------------------------------------------------

def stage1_pool(args) -> dict[str, Any]:
    path = Path(args.out_dir) / "pool_rollouts.pt"
    if args.skip_rollouts and path.exists():
        print(f"[stage1] reuse {path}")
        return torch.load(path, weights_only=False)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    df = pd.read_parquet(args.prompt_parquet)
    records = df.to_dict("records")[: args.candidate_pool]
    prompt_strs, prompt_ids_list, ground_truths = [], [], []
    for r in records:
        p = r["prompt"]
        if hasattr(p, "__iter__") and not isinstance(p, str):
            messages = [dict(m) for m in list(p)]
        else:
            messages = [{"role": "user", "content": str(p)}]
        messages = [m for m in messages if m.get("role") != "assistant"]
        try:
            txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=args.enable_thinking)
        except TypeError:
            txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(txt, add_special_tokens=False).input_ids
        if len(ids) > args.max_prompt_length:
            ids = ids[-args.max_prompt_length:]
            txt = tok.decode(ids, skip_special_tokens=False)
        prompt_strs.append(txt); prompt_ids_list.append(ids)
        rm = r["reward_model"]
        ground_truths.append(str(rm["ground_truth"]) if isinstance(rm, dict) else "")

    print(f"[stage1] vLLM TP={args.vllm_tensor_parallel_size}, {len(prompt_strs)} prompts × n=1")
    llm = LLM(model=args.ckpt,
              gpu_memory_utilization=args.vllm_gpu_memory_utilization,
              max_model_len=args.vllm_max_model_length,
              tensor_parallel_size=args.vllm_tensor_parallel_size,
              seed=args.seed, dtype="bfloat16", enforce_eager=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_completion_length, seed=args.seed, n=1)
    outs = llm.generate(prompt_strs, sp)

    rollout_id, prompt_idx, p_ids, c_ids, c_text, fr = [], [], [], [], [], []
    for pi, o in enumerate(outs):
        sub = o.outputs[0]
        rollout_id.append(pi); prompt_idx.append(pi)
        p_ids.append(prompt_ids_list[pi])
        c_ids.append(list(sub.token_ids))
        c_text.append(sub.text); fr.append(sub.finish_reason)
    bundle = {"rollout_id": rollout_id, "prompt_idx": prompt_idx,
              "prompt_ids": p_ids, "completion_ids": c_ids,
              "completion_text": c_text, "finish_reasons": fr,
              "ground_truth": ground_truths}
    torch.save(bundle, path)
    print(f"[stage1] saved {path} ({len(c_ids)} rollouts)")
    del llm; gc.collect(); torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel(); destroy_distributed_environment()
    except Exception:
        pass
    return bundle


# ---- Stage 2: grade --------------------------------------------------------

def stage2_grade(args, bundle: dict) -> dict:
    path = Path(args.out_dir) / "pool_grading.json"
    if args.skip_grade and path.exists():
        with open(path) as f:
            return json.load(f)
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from math500_eval import _is_correct
    items = []
    for rid in range(len(bundle["completion_text"])):
        ct = bundle["completion_text"][rid]
        gt = bundle["ground_truth"][rid]
        items.append({"rollout_id": rid, "is_correct": bool(_is_correct(ct, gt)),
                      "ground_truth": gt})
    n_c = sum(1 for x in items if x["is_correct"])
    out = {"n_pool": len(items), "n_correct": n_c, "n_incorrect": len(items) - n_c, "items": items}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[stage2] graded pool: correct={n_c}/{len(items)} ({n_c/max(1,len(items)):.2%})")
    return out


# ---- Stage 3: balanced selection ------------------------------------------

def stage3_select(args, grading: dict) -> dict:
    items = grading["items"]
    rng = np.random.default_rng(args.seed)
    correct_ids = np.array([x["rollout_id"] for x in items if x["is_correct"]])
    incorrect_ids = np.array([x["rollout_id"] for x in items if not x["is_correct"]])
    n_c = min(args.n_correct, len(correct_ids))
    n_i = min(args.n_incorrect, len(incorrect_ids))
    pick_c = rng.choice(correct_ids, size=n_c, replace=False) if n_c > 0 else np.array([], dtype=int)
    pick_i = rng.choice(incorrect_ids, size=n_i, replace=False) if n_i > 0 else np.array([], dtype=int)
    sel = {"correct": [int(x) for x in pick_c], "incorrect": [int(x) for x in pick_i],
           "n_correct_available": int(len(correct_ids)),
           "n_incorrect_available": int(len(incorrect_ids)),
           "n_correct_picked": int(n_c), "n_incorrect_picked": int(n_i)}
    with open(Path(args.out_dir) / "selection.json", "w") as f:
        json.dump(sel, f, indent=2)
    print(f"[stage3] selected {n_c} correct (of {len(correct_ids)}) + {n_i} incorrect (of {len(incorrect_ids)})")
    return sel


# ---- Stage 4: HF prefill on selected only ---------------------------------

def stage4_prefill(args, bundle: dict, selection: dict) -> pd.DataFrame:
    path = Path(args.out_dir) / "per_token_kl_selected.parquet"
    if args.skip_prefill and path.exists():
        print(f"[stage4] reuse {path}")
        return pd.read_parquet(path)

    from transformers import AutoModelForCausalLM
    n_gpu = torch.cuda.device_count()
    sdev = torch.device(f"cuda:{args.student_gpu}") if n_gpu >= 1 else torch.device("cuda:0")
    tdev = torch.device(f"cuda:{args.teacher_gpu}") if n_gpu >= 2 else sdev
    print(f"[stage4] student→{sdev} teacher→{tdev}")
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True)
    print(f"[stage4] loading student {args.ckpt}")
    student = AutoModelForCausalLM.from_pretrained(args.ckpt, **kw).to(sdev).eval()
    print(f"[stage4] loading teacher {args.teacher}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **kw).to(tdev).eval()

    selected = sorted(set(selection["correct"]) | set(selection["incorrect"]))
    rows: list[dict] = []
    for idx, rid in enumerate(selected):
        prompt = bundle["prompt_ids"][rid]
        completion = bundle["completion_ids"][rid]
        if len(completion) < 2:
            continue
        full_ids = torch.tensor(prompt + completion, dtype=torch.long).unsqueeze(0)
        Ltot = full_ids.size(1); Lprompt = len(prompt); Lcomp = len(completion)
        with torch.no_grad():
            s_out = student(full_ids.to(sdev), use_cache=False)
            t_out = teacher(full_ids.to(tdev), use_cache=False)
            s_logits = s_out.logits[:, Lprompt - 1: Ltot - 1, :].float()
            t_logits = t_out.logits[:, Lprompt - 1: Ltot - 1, :].to(s_logits.device).float()
            del s_out, t_out
            rk = per_token_reverse_kl_topk_tail(
                s_logits, t_logits, top_k=args.top_k_loss, temperature=args.loss_temperature
            ).squeeze(0).cpu().numpy().astype(np.float32)
        for pos in range(Lcomp):
            rows.append({"rollout_id": int(rid), "position": pos,
                         "completion_len": Lcomp, "kl": float(rk[pos])})
        if (idx + 1) % 20 == 0:
            print(f"[stage4]   {idx+1}/{len(selected)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"[stage4] saved {path} ({len(df)} rows for {len(selected)} rollouts)")
    del student, teacher; gc.collect(); torch.cuda.empty_cache()
    return df


# ---- Stage 5: figure -------------------------------------------------------

def stage5_figure(args, df: pd.DataFrame, selection: dict, threshold: float, ckpt_tag: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    # compute z + trigger per rollout
    grouped = df.groupby("rollout_id", sort=False)
    per_roll: dict[int, dict] = {}
    for rid, g in grouped:
        g = g.sort_values("position")
        kl = g["kl"].to_numpy()
        z = _z_series(kl, args.W)
        trig = _trigger_for(kl, threshold, args.W, args.L0, args.N)
        per_roll[int(rid)] = {
            "kl": kl, "z": z,
            "completion_len": int(g["completion_len"].iloc[0]),
            "is_trunc": trig != -1,
            "trigger_pos": int(trig),
        }

    def _order_group(ids: list[int]) -> list[int]:
        meta = [(rid, per_roll[rid]["is_trunc"],
                 (per_roll[rid]["trigger_pos"] / per_roll[rid]["completion_len"])
                 if per_roll[rid]["is_trunc"] else 2.0,
                 per_roll[rid]["completion_len"])
                for rid in ids]
        # triggered first (by normalized trigger pos asc), then non-triggered (by completion_len asc)
        meta_trig = sorted([m for m in meta if m[1]], key=lambda m: m[2])
        meta_notr = sorted([m for m in meta if not m[1]], key=lambda m: m[3])
        return [m[0] for m in meta_trig] + [m[0] for m in meta_notr]

    cor_ids = _order_group(selection["correct"])
    inc_ids = _order_group(selection["incorrect"])

    def _build(ids: list[int]) -> np.ndarray:
        if not ids:
            return np.zeros((0, args.n_bins), dtype=np.float32)
        m = np.full((len(ids), args.n_bins), np.nan, dtype=np.float32)
        for i, rid in enumerate(ids):
            m[i, :] = _resample_to_grid(per_roll[rid]["z"], args.n_bins) / threshold
        return m

    mat_c = _build(cor_ids); mat_i = _build(inc_ids)
    finite = np.concatenate([mat_c[np.isfinite(mat_c)], mat_i[np.isfinite(mat_i)]])
    if finite.size > 0:
        vmin = max(float(np.percentile(finite, 0.5)), 1e-4)
        vmax = min(float(np.percentile(finite, 99.5)), 50.0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    cmap = plt.get_cmap(args.colormap).copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                             gridspec_kw={"width_ratios": [1, 1]})

    def _draw(ax, mat, ids, title):
        R = mat.shape[0]
        if R == 0:
            ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes, ha="center"); ax.set_title(title); return None
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", origin="upper",
                       cmap=cmap, norm=norm, interpolation="nearest",
                       extent=[0.0, 1.0, R - 0.5, -0.5])
        xs, ys = [], []
        n_trig = 0
        for i, rid in enumerate(ids):
            r = per_roll[rid]
            if r["is_trunc"]:
                xs.append(r["trigger_pos"] / r["completion_len"]); ys.append(i)
                n_trig += 1
        if xs:
            ax.scatter(xs, ys, s=10, c="black", marker="|", linewidths=0.7,
                       label=f"τ trigger (n={n_trig})")
            ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
        if 0 < n_trig < R:
            ax.axhline(n_trig - 0.5, color="red", lw=0.6, ls="--", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("normalized rollout position  (token_index / completion_len)")
        ax.set_ylabel("rollouts (triggered ↑ early → late, non-triggered ↓)")
        ax.set_title(title)
        return im

    n_c, n_i = len(cor_ids), len(inc_ids)
    n_c_trig = sum(1 for rid in cor_ids if per_roll[rid]["is_trunc"])
    n_i_trig = sum(1 for rid in inc_ids if per_roll[rid]["is_trunc"])
    im_c = _draw(axes[0], mat_c, cor_ids,
                 f"Correct  (n={n_c}, triggered={n_c_trig} → {n_c_trig/max(1,n_c):.1%})")
    im_i = _draw(axes[1], mat_i, inc_ids,
                 f"Incorrect  (n={n_i}, triggered={n_i_trig} → {n_i_trig/max(1,n_i):.1%})")
    im_ref = im_c if im_c is not None else im_i
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, pad=0.015, fraction=0.025)
        cbar.set_label("normalized sliding-window KL  z_t / s_k")
        try:
            cbar.ax.plot([0, 1], [1.0, 1.0], color="red", lw=1.2,
                         transform=cbar.ax.get_yaxis_transform())
        except Exception:
            pass
    fig.suptitle(
        f"Sustained low-KL regions split by final correctness   |   {ckpt_tag}   "
        f"|  balanced N=100+100  (τ={threshold:.4g})",
        fontsize=13
    )
    fig_path = fig_dir / "heatmap_balanced_by_correctness.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[stage5] saved {fig_path}")

    # stats
    def _g_stats(ids):
        if not ids:
            return {"n": 0, "n_triggered": 0, "trigger_rate": 0.0,
                    "mean_tau_norm": float("nan"), "median_tau_norm": float("nan"),
                    "mean_post_trigger_fraction": float("nan"),
                    "mean_completion_len": 0.0}
        clens = np.array([per_roll[r]["completion_len"] for r in ids])
        trig_mask = np.array([per_roll[r]["is_trunc"] for r in ids])
        n_tr = int(trig_mask.sum())
        if n_tr > 0:
            tn = np.array([per_roll[r]["trigger_pos"] / per_roll[r]["completion_len"]
                           for r in ids if per_roll[r]["is_trunc"]])
            pf = np.array([1.0 - per_roll[r]["trigger_pos"] / per_roll[r]["completion_len"]
                           for r in ids if per_roll[r]["is_trunc"]])
            return {"n": len(ids), "n_triggered": n_tr,
                    "trigger_rate": n_tr / len(ids),
                    "mean_tau_norm": float(tn.mean()),
                    "median_tau_norm": float(np.median(tn)),
                    "mean_post_trigger_fraction": float(pf.mean()),
                    "mean_completion_len": float(clens.mean())}
        return {"n": len(ids), "n_triggered": 0, "trigger_rate": 0.0,
                "mean_tau_norm": float("nan"), "median_tau_norm": float("nan"),
                "mean_post_trigger_fraction": float("nan"),
                "mean_completion_len": float(clens.mean())}

    cbc = {"correct": _g_stats(cor_ids), "incorrect": _g_stats(inc_ids),
           "threshold": float(threshold),
           "n_correct_available": selection["n_correct_available"],
           "n_incorrect_available": selection["n_incorrect_available"]}
    with open(out / "stats_balanced.json", "w") as f:
        json.dump(cbc, f, indent=2)
    print("[stage5] stats:")
    print(json.dumps(cbc, indent=2))
    return cbc


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    threshold = (load_training_threshold(args.ckpt, args.threshold_step)
                 if args.threshold is None else float(args.threshold))
    print(f"[main] τ = {threshold:.10g} (step {args.threshold_step})")

    bundle = stage1_pool(args)
    grading = stage2_grade(args, bundle)
    selection = stage3_select(args, grading)
    df = stage4_prefill(args, bundle, selection)
    stage5_figure(args, df, selection, threshold, Path(args.ckpt).name)
    print(f"[done] outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
