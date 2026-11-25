"""Rollout-level heatmap of sustained low-KL regions.

For each of N (default 1000) DAPO prompts, generate 1 rollout from the student
checkpoint, then compute per-token reverse-KL d_t (with the same support as
training: student top-64 + tail bucket, β=1). Build the rolling-window mean
z_t = mean(d_{t-W+1 .. t}) using W = L = 64, normalize by the training step's
threshold s_k = τ, and render a heatmap with rows sorted by trigger position.

Trigger detection mirrors distillation/kl_truncation.py: the first window
position where N=4 consecutive sliding-window means fall below τ.

Outputs (under --out_dir, default analyze/work_heatmap):
  rollouts.pt                — vLLM bundle (1000 rollouts)
  per_token_kl.parquet       — rollout_id, position, kl, completion_len
  z_matrix.npy               — (R, T_max) float32, NaN-padded
  trigger_info.parquet       — rollout_id, completion_len, is_trunc, trigger_pos
  figures/heatmap_sustained_lowkl.png
  stats.json                 — summary counts

Run:
  cd /path/to/kat-opd
  CUDA_VISIBLE_DEVICES=4,5,6,7 python \\
    analyze/heatmap_lowkl.py \\
    --ckpt outputs/1.7b_p2048_eta0.3_s32/checkpoint-380 \\
    --num_prompts 1000 \\
    --out_dir analyze/work_heatmap
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
    p.add_argument("--num_prompts", type=int, default=1000)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--W", type=int, default=64, help="sliding window size (matches training L)")
    p.add_argument("--L0", type=int, default=32, help="window start offset (matches training t)")
    p.add_argument("--N", type=int, default=4, help="consecutive low-window count for trigger")
    p.add_argument("--top_k_loss", type=int, default=64)
    p.add_argument("--loss_temperature", type=float, default=1.0)
    p.add_argument("--threshold_step", type=int, default=379)
    p.add_argument("--threshold", type=float, default=None,
                   help="Hard override; otherwise read from trainer_state.json at threshold_step.")
    p.add_argument("--out_dir", default="analyze/work_heatmap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--vllm_max_model_length", type=int, default=4096)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    p.add_argument("--student_gpu", type=int, default=0)
    p.add_argument("--teacher_gpu", type=int, default=1)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--skip_rollouts", action="store_true")
    p.add_argument("--skip_prefill", action="store_true")
    p.add_argument("--colormap", default="magma")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  per-token KL (identical to analyze_stages.py)
# ──────────────────────────────────────────────────────────────────────────────

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
        s_lp = F.log_softmax(s, dim=-1)
        t_lp = F.log_softmax(t, dim=-1)
        s_p = s_lp.exp()
        return (s_p * (s_lp - t_lp)).sum(-1)
    _, support = s.topk(top_k, dim=-1)
    s_lp_full = F.log_softmax(s, dim=-1)
    t_lp_full = F.log_softmax(t, dim=-1)
    s_topk = s_lp_full.gather(-1, support)
    t_topk = t_lp_full.gather(-1, support)
    s_lp_tail = _add_tail_bucket(s_topk)
    t_lp_tail = _add_tail_bucket(t_topk)
    s_p = s_lp_tail.exp()
    per_tok_kl = s_p * (s_lp_tail - t_lp_tail)
    per_tok_kl = torch.nan_to_num(per_tok_kl, nan=0.0)
    return per_tok_kl.sum(-1)


def load_training_threshold(ckpt_dir: str, step: int) -> float:
    with open(Path(ckpt_dir) / "trainer_state.json") as f:
        s = json.load(f)
    for entry in s.get("log_history", []):
        if entry.get("step") == step:
            v = entry.get("kl_trunc/threshold")
            if v is not None and isinstance(v, (int, float)) and v == v and v > 0:
                return float(v)
    raise RuntimeError(f"No valid kl_trunc/threshold at step={step}")


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 1: vLLM rollouts
# ──────────────────────────────────────────────────────────────────────────────

def stage1_rollouts(args) -> dict[str, Any]:
    path = Path(args.out_dir) / "rollouts.pt"
    if args.skip_rollouts and path.exists():
        print(f"[stage1] reuse {path}")
        return torch.load(path, weights_only=False)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    df = pd.read_parquet(args.prompt_parquet)
    records = df.to_dict("records")[: args.num_prompts]
    prompt_strs: list[str] = []
    prompt_ids_list: list[list[int]] = []
    for r in records:
        p = r["prompt"]
        if hasattr(p, "__iter__") and not isinstance(p, str):
            messages = [dict(m) for m in list(p)]
        else:
            messages = [{"role": "user", "content": str(p)}]
        messages = [m for m in messages if m.get("role") != "assistant"]
        try:
            txt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        except TypeError:
            txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(txt, add_special_tokens=False).input_ids
        if len(ids) > args.max_prompt_length:
            ids = ids[-args.max_prompt_length:]
            txt = tok.decode(ids, skip_special_tokens=False)
        prompt_strs.append(txt)
        prompt_ids_list.append(ids)

    print(f"[stage1] vLLM TP={args.vllm_tensor_parallel_size}, {len(prompt_strs)} prompts × n=1")
    llm = LLM(
        model=args.ckpt,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_length,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        seed=args.seed,
        dtype="bfloat16",
        enforce_eager=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_length,
        seed=args.seed,
        n=1,
    )
    outs = llm.generate(prompt_strs, sp)

    rid = 0
    rollout_id, prompt_idx, p_ids, c_ids, c_text, fr = [], [], [], [], [], []
    for pi, o in enumerate(outs):
        sub = o.outputs[0]
        rollout_id.append(rid); prompt_idx.append(pi)
        p_ids.append(prompt_ids_list[pi])
        c_ids.append(list(sub.token_ids))
        c_text.append(sub.text)
        fr.append(sub.finish_reason)
        rid += 1

    bundle = {
        "rollout_id": rollout_id,
        "prompt_idx": prompt_idx,
        "prompt_ids": p_ids,
        "completion_ids": c_ids,
        "completion_text": c_text,
        "finish_reasons": fr,
    }
    torch.save(bundle, path)
    mean_len = float(np.mean([len(c) for c in c_ids]))
    print(f"[stage1] saved {path} (n={len(c_ids)}, mean_completion_len={mean_len:.1f})")

    del llm; gc.collect(); torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel(); destroy_distributed_environment()
    except Exception:
        pass
    return bundle


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 2: per-token reverse-KL  (only metric we need)
# ──────────────────────────────────────────────────────────────────────────────

def stage2_per_token_kl(args, bundle: dict[str, Any]) -> pd.DataFrame:
    path = Path(args.out_dir) / "per_token_kl.parquet"
    if args.skip_prefill and path.exists():
        print(f"[stage2] reuse {path}")
        return pd.read_parquet(path)

    from transformers import AutoModelForCausalLM

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        sdev = torch.device(f"cuda:{args.student_gpu}")
        tdev = torch.device(f"cuda:{args.teacher_gpu}")
    else:
        sdev = tdev = torch.device("cuda:0")
    print(f"[stage2] student→{sdev} teacher→{tdev}")

    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True)
    print(f"[stage2] loading student {args.ckpt}")
    student = AutoModelForCausalLM.from_pretrained(args.ckpt, **kw).to(sdev).eval()
    print(f"[stage2] loading teacher {args.teacher}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **kw).to(tdev).eval()

    rows: list[dict[str, Any]] = []
    n = len(bundle["prompt_ids"])
    for ridx in range(n):
        prompt = bundle["prompt_ids"][ridx]
        completion = bundle["completion_ids"][ridx]
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
            rows.append({
                "rollout_id": ridx,
                "position": pos,
                "completion_len": Lcomp,
                "kl": float(rk[pos]),
            })
        if (ridx + 1) % 50 == 0:
            print(f"[stage2]   {ridx+1}/{n}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    print(f"[stage2] saved {path} ({len(df)} rows)")
    del student, teacher; gc.collect(); torch.cuda.empty_cache()
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 3: build z_t matrix + trigger detection + heatmap
# ──────────────────────────────────────────────────────────────────────────────

def _trigger_for_rollout(kl: np.ndarray, threshold: float, W: int, L0: int, N: int) -> int:
    """Return the trigger token position (window-end of the first low window in
    the first N-long consecutive run), or -1 if no trigger.

    Mirrors distillation/kl_truncation.py:
      win_mean[i] = mean(kl[L0+i : L0+i+W]); first i with N consecutive low → i_first;
      trigger token position (training mask start) = L0 + i_first + W.
    """
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
    i_first = int(np.argmax(hits))
    return L0 + i_first + W  # token-position where mask begins (== window-end of first low window)


def _z_series(kl: np.ndarray, W: int) -> np.ndarray:
    """Right-aligned rolling mean: z[t] = mean(kl[t-W+1 .. t]) for t ≥ W-1; NaN before.

    Length matches kl. NaN for t < W-1.
    """
    n = kl.shape[0]
    z = np.full(n, np.nan, dtype=np.float32)
    if n < W:
        return z
    csum = np.concatenate([[0.0], np.cumsum(kl.astype(np.float64))])
    # z[t] for t in [W-1, n-1]: sum(kl[t-W+1..t]) / W = (csum[t+1] - csum[t-W+1]) / W
    idx = np.arange(W - 1, n)
    z[idx] = ((csum[idx + 1] - csum[idx - W + 1]) / float(W)).astype(np.float32)
    return z


def stage3_heatmap(args, df: pd.DataFrame, threshold: float, bundle: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)

    # group by rollout, build z_t and trigger
    grouped = df.groupby("rollout_id", sort=True)
    rids = sorted(df["rollout_id"].unique().tolist())
    per_roll: list[dict[str, Any]] = []
    z_lists: dict[int, np.ndarray] = {}
    for rid in rids:
        g = grouped.get_group(rid).sort_values("position")
        kl = g["kl"].to_numpy()
        Lcomp = int(g["completion_len"].iloc[0])
        z = _z_series(kl, args.W)
        z_lists[rid] = z
        trig = _trigger_for_rollout(kl, threshold, args.W, args.L0, args.N)
        per_roll.append({
            "rollout_id": int(rid),
            "completion_len": Lcomp,
            "is_trunc": trig != -1,
            "trigger_pos": int(trig),
        })

    info = pd.DataFrame(per_roll)
    info.to_parquet(out / "trigger_info.parquet", index=False)

    # statistics
    n = len(info)
    n_trig = int(info["is_trunc"].sum())
    fr = bundle.get("finish_reasons", [])
    n_reach_max = int(sum(1 for f in fr if f == "length"))
    mean_len = float(info["completion_len"].mean())
    if n_trig > 0:
        trig = info[info["is_trunc"]]
        mean_tau = float(trig["trigger_pos"].mean())
        median_tau = float(trig["trigger_pos"].median())
        post_len = (trig["completion_len"] - trig["trigger_pos"]).to_numpy()
        mean_post = float(post_len.mean())
    else:
        mean_tau = median_tau = mean_post = float("nan")
    stats = {
        "num_rollouts": n,
        "triggered_count": n_trig,
        "trigger_rate": n_trig / max(1, n),
        "mean_tau": mean_tau,
        "median_tau": median_tau,
        "fraction_reaching_max_length": n_reach_max / max(1, n),
        "mean_rollout_length": mean_len,
        "mean_post_trigger_length": mean_post,
        "threshold": float(threshold),
        "W": args.W, "L0": args.L0, "N": args.N,
        "max_completion_length": args.max_completion_length,
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("[stage3] stats:")
    for k, v in stats.items():
        print(f"  {k} = {v}")

    # sort rows: triggered first (by trigger_pos ascending), then non-triggered
    trig_sorted = info[info["is_trunc"]].sort_values("trigger_pos")
    notrig = info[~info["is_trunc"]]
    row_order = trig_sorted["rollout_id"].tolist() + notrig["rollout_id"].tolist()

    # build (R, T_max) NaN-padded normalized z matrix
    T_max = int(info["completion_len"].max())
    R = len(row_order)
    mat = np.full((R, T_max), np.nan, dtype=np.float32)
    for i, rid in enumerate(row_order):
        z = z_lists[rid]
        Lz = z.shape[0]
        mat[i, :Lz] = z / float(threshold)
    np.save(out / "z_matrix.npy", mat)

    # heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    # vmin/vmax: clip extreme values for color scale stability
    finite = mat[np.isfinite(mat)]
    if finite.size > 0:
        vmin = max(float(np.percentile(finite, 0.5)), 1e-4)
        vmax = min(float(np.percentile(finite, 99.5)), 50.0)
        # Use LogNorm so y=1 (threshold) sits in the middle and low values stand out
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    # Use masked array so NaN is white (set by set_bad)
    cmap = plt.get_cmap(args.colormap).copy()
    cmap.set_bad(color="white")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest")

    # mark trigger positions on triggered rows with short black ticks
    if n_trig > 0:
        rid_to_row = {rid: i for i, rid in enumerate(row_order)}
        xs = []
        ys = []
        for _, r in trig_sorted.iterrows():
            ys.append(rid_to_row[int(r["rollout_id"])])
            xs.append(int(r["trigger_pos"]))
        # use scatter with small markers; black for high contrast
        ax.scatter(xs, ys, s=2.5, c="black", marker="|", linewidths=0.8, label=f"τ trigger (n={n_trig})")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    # divider line between triggered and non-triggered
    n_trig_rows = len(trig_sorted)
    if 0 < n_trig_rows < R:
        ax.axhline(n_trig_rows - 0.5, color="red", lw=0.6, ls="--", alpha=0.6)

    ax.set_xlabel("rollout token position")
    ax.set_ylabel("rollouts sorted by trigger position")
    ckpt_tag = Path(args.ckpt).name
    ax.set_title(
        "Rollout-specific sustained low-KL regions\n"
        f"({ckpt_tag}, N={n}, triggered={n_trig} ({n_trig/max(1,n):.1%}), "
        f"W={args.W}, N_consec={args.N}, τ={threshold:.4g})"
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("normalized sliding-window KL  z_t / s_k")
    cbar.ax.axhline(1.0, color="red", lw=1.2)  # threshold reference

    fig.tight_layout()
    out_png = fig_dir / "heatmap_sustained_lowkl.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[stage3] saved {out_png}")
    return stats


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.threshold is None:
        threshold = load_training_threshold(args.ckpt, args.threshold_step)
    else:
        threshold = float(args.threshold)
    print(f"[main] τ = {threshold:.10g} (step {args.threshold_step})")

    bundle = stage1_rollouts(args)
    df = stage2_per_token_kl(args, bundle)
    stage3_heatmap(args, df, threshold, bundle)
    print(f"[done] outputs at {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
