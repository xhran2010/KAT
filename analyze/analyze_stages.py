"""3-stage rollout analysis around KL truncation, conditioned on correctness.

Pipeline:
  1. vLLM rollouts:        200 prompts × 4 rollouts = 800 completions (seed 42, T=1, top_p=0.95, max=2048).
  2. HF prefill:           student (ckpt) and teacher (Qwen3-8B) — per-token reverse-KL (top-64 + tail),
                           per-token student/teacher entropy, top-1 token ids.
  3. Segment + grade:      replay distillation/kl_truncation.py with τ from trainer_state.json[step=379];
                           grade each rollout with math_verify against reward_model.ground_truth.
                           Buckets: correct / incorrect / no_trunc.
  4. Figures + tables:     F1 trigger-norm hist+CDF, F2 entropy (student/teacher) per (group×stage),
                           F3 KL per (group×stage), F4 wordclouds student & teacher per (group×stage).
                           T1 counts, T2 stage stats, T3 trigger pos, T4 grading summary.
  5. Chinese report.md.

Run:
  cd /path/to/kat-opd
  CUDA_VISIBLE_DEVICES=4,5,6,7 python analyze/analyze_stages.py \
    --ckpt outputs/1.7b_p2048_eta0.3_s32/checkpoint-380 \
    --teacher Qwen/Qwen3-8B \
    --prompt_parquet datasets/dapo-math-17k.parquet \
    --num_prompts 200 --rollouts_per_prompt 4 \
    --threshold_step 379 \
    --vllm_tensor_parallel_size 4 \
    --out_dir analyze/work
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--teacher", default="Qwen/Qwen3-8B")
    p.add_argument("--prompt_parquet",
                   default="data/dapo-math-17k/dapo_clean.parquet")
    p.add_argument("--num_prompts", type=int, default=200)
    p.add_argument("--rollouts_per_prompt", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--t", type=int, default=32)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--top_k_loss", type=int, default=64)
    p.add_argument("--loss_temperature", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=None,
                   help="Hard override. If None, read kl_trunc/threshold at --threshold_step.")
    p.add_argument("--threshold_step", type=int, default=379)
    p.add_argument("--threshold_expected", type=float, default=0.11906326469033957,
                   help="If set, assert loaded threshold matches within 1e-12.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--vllm_max_model_length", type=int, default=4096)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    p.add_argument("--student_gpu", type=int, default=0)
    p.add_argument("--teacher_gpu", type=int, default=1)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--skip_rollouts", action="store_true")
    p.add_argument("--skip_prefill", action="store_true")
    p.add_argument("--skip_grade", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers: per-token KL signal exactly matching training
# ──────────────────────────────────────────────────────────────────────────────

def _add_tail_bucket(log_probs):
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail = torch.log(-torch.expm1(log_sum))
    return torch.cat([log_probs, tail], dim=-1)


def per_token_reverse_kl_topk_tail(student_logits, teacher_logits, top_k=64, temperature=1.0):
    """Returns (B, S) per-token reverse-KL on student-selected top-k + tail bucket.

    Mirrors distillation/distillation_trainer.py::generalized_jsd_loss(beta=1, top_k=64,
    add_tail_bucket=True), then sums over the K+1 support axis. This is the exact signal
    that feeds _kl_trunc_apply via per_token_kl = jsd.sum(dim=-1).
    """
    s = student_logits / temperature
    t = teacher_logits / temperature
    V = s.size(-1)
    if top_k <= 0 or top_k >= V:
        s_lp = F.log_softmax(s, dim=-1)
        t_lp = F.log_softmax(t, dim=-1)
        s_p = s_lp.exp()
        kl = s_p * (s_lp - t_lp)
        return kl.sum(-1)
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


def per_token_entropy(logits):
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


# ──────────────────────────────────────────────────────────────────────────────
#  Threshold loader
# ──────────────────────────────────────────────────────────────────────────────

def load_training_threshold(ckpt_dir: str, step: int, expected: float | None) -> float:
    ts_path = Path(ckpt_dir) / "trainer_state.json"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)
    with open(ts_path) as f:
        s = json.load(f)
    hist = s.get("log_history", [])
    for entry in hist:
        if entry.get("step") == step:
            v = entry.get("kl_trunc/threshold")
            if v is None or not isinstance(v, (int, float)) or v != v or v <= 0:
                raise RuntimeError(f"kl_trunc/threshold missing or invalid at step {step}: {v}")
            v = float(v)
            if expected is not None and abs(v - expected) > 1e-12:
                raise RuntimeError(
                    f"threshold mismatch: loaded {v} vs expected {expected} (|diff|={abs(v-expected):.2e})"
                )
            return v
    raise RuntimeError(f"No log_history entry with step={step} in {ts_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 1: vLLM rollouts — 200 prompts × n=4 each
# ──────────────────────────────────────────────────────────────────────────────

def stage1_generate_rollouts(args) -> dict[str, Any]:
    rollouts_path = Path(args.out_dir) / "rollouts.pt"
    if args.skip_rollouts and rollouts_path.exists():
        print(f"[stage1] reuse {rollouts_path}")
        return torch.load(rollouts_path, weights_only=False)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[stage1] tokenizer from {args.ckpt}")
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"[stage1] prompts from {args.prompt_parquet} (first {args.num_prompts} rows)")
    df = pd.read_parquet(args.prompt_parquet)
    records = df.to_dict("records")[: args.num_prompts]

    prompts_raw: list[list[dict]] = []
    ground_truths: list[str] = []
    for r in records:
        p = r["prompt"]
        if hasattr(p, "__iter__") and not isinstance(p, str):
            messages = [dict(m) for m in list(p)]
        else:
            messages = [{"role": "user", "content": str(p)}]
        messages = [m for m in messages if m.get("role") != "assistant"]
        prompts_raw.append(messages)
        rm = r["reward_model"]
        gt = rm["ground_truth"] if isinstance(rm, dict) else None
        ground_truths.append(str(gt) if gt is not None else "")

    prompt_strs: list[str] = []
    prompt_ids_list: list[list[int]] = []
    for messages in prompts_raw:
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

    print(f"[stage1] vLLM TP={args.vllm_tensor_parallel_size} gpu_util={args.vllm_gpu_memory_utilization}")
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
        n=args.rollouts_per_prompt,
    )
    print(f"[stage1] generating {len(prompt_strs)} × n={args.rollouts_per_prompt} rollouts")
    outs = llm.generate(prompt_strs, sp)

    # outs is len(prompts); each has .outputs of length n=rollouts_per_prompt
    rollout_id, prompt_idx, p_ids, c_ids, c_text, fr, gts = [], [], [], [], [], [], []
    rid = 0
    for pi, o in enumerate(outs):
        for sub in o.outputs:
            rollout_id.append(rid)
            prompt_idx.append(pi)
            p_ids.append(prompt_ids_list[pi])
            c_ids.append(list(sub.token_ids))
            c_text.append(sub.text)
            fr.append(sub.finish_reason)
            gts.append(ground_truths[pi])
            rid += 1

    bundle = {
        "tokenizer_path": args.ckpt,
        "rollout_id": rollout_id,
        "prompt_idx": prompt_idx,
        "prompt_ids": p_ids,
        "completion_ids": c_ids,
        "completion_text": c_text,
        "finish_reasons": fr,
        "ground_truth": gts,
    }
    torch.save(bundle, rollouts_path)
    mean_len = float(np.mean([len(c) for c in c_ids]))
    print(f"[stage1] saved {rollouts_path} (n={len(c_ids)}, mean_completion_len={mean_len:.1f})")

    # release vLLM
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    return bundle


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 2: HF prefill + per-token KL / entropy / top-1
# ──────────────────────────────────────────────────────────────────────────────

def stage2_per_token(args, bundle: dict[str, Any]) -> pd.DataFrame:
    per_token_path = Path(args.out_dir) / "per_token.parquet"
    if args.skip_prefill and per_token_path.exists():
        print(f"[stage2] reuse {per_token_path}")
        return pd.read_parquet(per_token_path)

    from transformers import AutoModelForCausalLM

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 2:
        student_dev = torch.device(f"cuda:{args.student_gpu}")
        teacher_dev = torch.device(f"cuda:{args.teacher_gpu}")
    else:
        student_dev = teacher_dev = torch.device("cuda:0")
    print(f"[stage2] student on {student_dev}, teacher on {teacher_dev}")

    common_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    print(f"[stage2] loading student from {args.ckpt}")
    student = AutoModelForCausalLM.from_pretrained(args.ckpt, **common_kwargs).to(student_dev).eval()
    print(f"[stage2] loading teacher from {args.teacher}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, **common_kwargs).to(teacher_dev).eval()

    rows: list[dict[str, Any]] = []
    n = len(bundle["prompt_ids"])
    for ridx in range(n):
        prompt = bundle["prompt_ids"][ridx]
        completion = bundle["completion_ids"][ridx]
        if len(completion) < 2:
            continue
        full_ids = torch.tensor(prompt + completion, dtype=torch.long).unsqueeze(0)
        Ltot = full_ids.size(1)
        Lprompt = len(prompt)
        Lcomp = len(completion)
        with torch.no_grad():
            s_out = student(full_ids.to(student_dev), use_cache=False)
            t_out = teacher(full_ids.to(teacher_dev), use_cache=False)
            s_logits = s_out.logits[:, Lprompt - 1: Ltot - 1, :].float()
            t_logits = t_out.logits[:, Lprompt - 1: Ltot - 1, :].to(s_logits.device).float()
            del s_out, t_out
            rk = per_token_reverse_kl_topk_tail(
                s_logits, t_logits, top_k=args.top_k_loss, temperature=args.loss_temperature
            )
            Hs = per_token_entropy(s_logits)
            Ht = per_token_entropy(t_logits)
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
            print(f"[stage2]   {ridx+1}/{n} done; rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(per_token_path, index=False)
    print(f"[stage2] saved {per_token_path} ({len(df)} rows)")

    del student, teacher
    gc.collect()
    torch.cuda.empty_cache()
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 3a: segmentation
# ──────────────────────────────────────────────────────────────────────────────

def _find_stages_for_rollout(kl_series: np.ndarray, threshold: float, L: int, t: int, N: int):
    S = kl_series.shape[0]
    if S - t < L or S - t < L + N - 1:
        return False, -1, -1, (0, S, S, S)
    csum = np.concatenate([[0.0], np.cumsum(kl_series[t:].astype(np.float64))])
    W = (S - t) - L + 1
    win_mean = (csum[L: L + W] - csum[:W]) / float(L)
    low = win_mean < threshold
    if W < N:
        return False, -1, -1, (0, S, S, S)
    conv = np.convolve(low.astype(np.int32), np.ones(N, dtype=np.int32), mode="valid")
    hits = conv >= N
    if not hits.any():
        return False, -1, -1, (0, S, S, S)
    i_first = int(np.argmax(hits))
    i_last = i_first
    while i_last + 1 < W and low[i_last + 1]:
        i_last += 1
    pre_end = min(t + i_first, S)
    in_end = min(t + i_last + L, S)
    return True, i_first, i_last, (0, pre_end, in_end, S)


def stage3_segment(args, per_token_df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict]:
    L, t, N = args.L, args.t, args.N
    stage_col = np.empty(len(per_token_df), dtype=object)
    info: list[dict] = []
    grouped = per_token_df.groupby("rollout_id", sort=True)
    for rid, g in grouped:
        g_sorted = g.sort_values("position")
        idx = g_sorted.index.to_numpy()
        positions = g_sorted["position"].to_numpy()
        kl_series = g_sorted["kl"].to_numpy()
        is_trunc, i_first, i_last, (a, b, c, d) = _find_stages_for_rollout(kl_series, threshold, L, t, N)
        Lcomp = int(g_sorted["completion_len"].iloc[0])
        if not is_trunc:
            stage_per_pos = np.full(Lcomp, "no_trunc", dtype=object)
        else:
            stage_per_pos = np.empty(Lcomp, dtype=object)
            stage_per_pos[:b] = "pre"
            stage_per_pos[b:c] = "in"
            stage_per_pos[c:Lcomp] = "after"
        for row_idx, pos in zip(idx, positions):
            stage_col[row_idx] = stage_per_pos[pos]
        info.append({
            "rollout_id": int(rid),
            "completion_len": Lcomp,
            "is_trunc": bool(is_trunc),
            "i_first": int(i_first),
            "i_last": int(i_last),
            "trigger_token_pos": int(t + i_first) if is_trunc else -1,
            "trigger_norm": float((t + i_first) / Lcomp) if is_trunc and Lcomp > 0 else float("nan"),
            "pre_len": int(b - a),
            "in_len": int(c - b),
            "after_len": int(d - c),
        })
    per_token_df = per_token_df.copy()
    per_token_df["stage"] = stage_col
    summary = {
        "threshold": float(threshold),
        "L": L, "t": t, "N": N,
        "n_rollouts": len(info),
        "n_trunc": int(sum(1 for r in info if r["is_trunc"])),
        "rollouts": info,
    }
    return per_token_df, summary


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 3b: math grading
# ──────────────────────────────────────────────────────────────────────────────

def _import_is_correct():
    """Import math500_eval._is_correct from repo root (handle ad-hoc sys.path)."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from math500_eval import _is_correct
    return _is_correct


def stage3b_grade(args, bundle: dict) -> dict:
    grading_path = Path(args.out_dir) / "grading.json"
    if args.skip_grade and grading_path.exists():
        with open(grading_path) as f:
            return json.load(f)

    _is_correct = _import_is_correct()
    items = []
    for rid in range(len(bundle["completion_text"])):
        ct = bundle["completion_text"][rid]
        gt = bundle["ground_truth"][rid]
        correct = bool(_is_correct(ct, gt))
        items.append({"rollout_id": rid, "ground_truth": gt, "is_correct": correct})

    n_correct = sum(1 for x in items if x["is_correct"])
    out = {"n_rollouts": len(items), "n_correct": n_correct, "items": items}
    with open(grading_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[stage3b] graded {len(items)} rollouts; correct={n_correct} ({n_correct/max(1,len(items)):.1%})")

    # sanity print
    print("[stage3b] first 5 (ground_truth, is_correct, completion_tail):")
    for i in range(min(5, len(items))):
        tail = bundle["completion_text"][i][-180:].replace("\n", "↵")
        print(f"  rid={i} gt={items[i]['ground_truth']!r} ok={items[i]['is_correct']} tail={tail!r}")
    return out


def assemble_groups(per_token_df: pd.DataFrame, partition: dict, grading: dict) -> pd.DataFrame:
    """Add 'group' column ∈ {correct, incorrect, no_trunc} to per_token_df."""
    correct_map = {it["rollout_id"]: it["is_correct"] for it in grading["items"]}
    trunc_map = {r["rollout_id"]: r["is_trunc"] for r in partition["rollouts"]}

    def _grp(rid):
        is_t = trunc_map.get(int(rid), False)
        if not is_t:
            return "no_trunc"
        return "correct" if correct_map.get(int(rid), False) else "incorrect"

    df = per_token_df.copy()
    df["group"] = df["rollout_id"].map(_grp)
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 4: figures + tables
# ──────────────────────────────────────────────────────────────────────────────

GROUPS = ["correct", "incorrect"]
STAGES = ["pre", "in", "after"]
GROUP_COLORS = {"correct": "#1f77b4", "incorrect": "#d62728"}
STAGE_COLORS = {"pre": "#1f77b4", "in": "#d62728", "after": "#2ca02c"}

EN_STOPWORDS = {
    "the", "of", "to", "and", "is", "in", "it", "a", "an", "for", "on", "at", "as",
    "by", "be", "with", "or", "if", "we", "this", "that", "are", "was", "from", "so",
    "but", "not", "no", "do", "can", "you", "he", "she", "they", "them", "his", "her",
    "our", "us", "your", "my", "me", "i", "us", "have", "has", "had", "will", "would",
    "could", "should", "may", "might", "than", "then", "thus", "also", "into", "out",
    "up", "down", "over", "under", "any", "all", "some", "more", "most", "such",
    "only", "own", "same", "other", "those", "these", "there", "here", "when", "where",
    "what", "which", "who", "whom", "how", "why", "let", "now", "yes",
}


def _clean_word(tok_text: str) -> str | None:
    """Filter BPE/special characters → return lowercase English word or None."""
    if tok_text is None:
        return None
    # strip GPT-2 BPE markers
    txt = tok_text.replace("Ġ", "").replace("Ċ", "").replace("▁", "")
    txt = txt.strip()
    if not txt:
        return None
    # only pure letters, length ≥ 2
    if not re.fullmatch(r"[A-Za-z]{2,}", txt):
        return None
    txt = txt.lower()
    if txt in EN_STOPWORDS:
        return None
    return txt


def stage4_figures_tables(args, df: pd.DataFrame, partition: dict, grading: dict, threshold: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.scale import SymmetricalLogTransform  # noqa: F401
    from transformers import AutoTokenizer
    try:
        from wordcloud import WordCloud
    except Exception:
        WordCloud = None

    out = Path(args.out_dir)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(exist_ok=True, parents=True)
    tab_dir.mkdir(exist_ok=True, parents=True)

    tok = AutoTokenizer.from_pretrained(args.ckpt)

    # rollout-level table (one row per rollout)
    roll = pd.DataFrame(partition["rollouts"])
    roll["is_correct"] = roll["rollout_id"].map(
        {it["rollout_id"]: it["is_correct"] for it in grading["items"]}
    )

    def _group_of(r):
        if not r["is_trunc"]:
            return "no_trunc"
        return "correct" if r["is_correct"] else "incorrect"

    roll["group"] = roll.apply(_group_of, axis=1)

    n_total = len(roll)
    counts_per_group = roll["group"].value_counts().to_dict()
    n_correct_total = int(roll["is_correct"].sum())
    n_incorrect_total = int(n_total - n_correct_total)

    # ─────── T1: token & rollout counts per (group × stage), + no_trunc row ───────
    t1_rows = []
    for g in GROUPS:
        for s in STAGES:
            sub = df[(df["group"] == g) & (df["stage"] == s)]
            n_tokens = len(sub)
            n_rollouts = sub["rollout_id"].nunique()
            mean_len = (sub.groupby("rollout_id").size().mean()
                        if n_rollouts > 0 else 0.0)
            t1_rows.append({
                "group": g, "stage": s,
                "n_tokens": int(n_tokens),
                "n_rollouts_with_stage": int(n_rollouts),
                "mean_stage_len_per_rollout": float(mean_len),
            })
    nt = df[df["group"] == "no_trunc"]
    t1_rows.append({
        "group": "no_trunc", "stage": "(entire)",
        "n_tokens": int(len(nt)),
        "n_rollouts_with_stage": int(nt["rollout_id"].nunique()),
        "mean_stage_len_per_rollout": float(nt.groupby("rollout_id").size().mean()
                                            if len(nt) > 0 else 0.0),
    })
    T1 = pd.DataFrame(t1_rows)
    T1.to_csv(tab_dir / "T1_counts.csv", index=False)
    (tab_dir / "T1_counts.md").write_text(T1.to_markdown(index=False), encoding="utf-8")

    # ─────── T2: per (group × stage) stats for kl, H_s, H_t ───────
    metrics = ["kl", "H_s", "H_t"]
    t2_rows = []
    for g in GROUPS:
        for s in STAGES:
            sub = df[(df["group"] == g) & (df["stage"] == s)]
            if len(sub) == 0:
                t2_rows.append({"group": g, "stage": s, "n_tokens": 0,
                                **{f"{m}_{stat}": float("nan") for m in metrics for stat in ("mean", "median", "p10", "p90")}})
                continue
            rec = {"group": g, "stage": s, "n_tokens": int(len(sub))}
            for m in metrics:
                v = sub[m].to_numpy()
                rec[f"{m}_mean"] = float(np.mean(v))
                rec[f"{m}_median"] = float(np.median(v))
                rec[f"{m}_p10"] = float(np.percentile(v, 10))
                rec[f"{m}_p90"] = float(np.percentile(v, 90))
            t2_rows.append(rec)
    T2 = pd.DataFrame(t2_rows)
    T2.to_csv(tab_dir / "T2_stage_stats.csv", index=False)
    (tab_dir / "T2_stage_stats.md").write_text(T2.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    # ─────── T3: trigger_norm & stage lengths per group (truncated only) ───────
    t3_rows = []
    for g in GROUPS:
        sub = roll[(roll["is_trunc"]) & (roll["group"] == g)]
        if len(sub) == 0:
            continue
        for col in ["trigger_norm", "trigger_token_pos", "completion_len", "pre_len", "in_len", "after_len"]:
            v = sub[col].to_numpy().astype(np.float64)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            t3_rows.append({
                "group": g, "metric": col,
                "n": int(len(v)),
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
            })
    T3 = pd.DataFrame(t3_rows)
    T3.to_csv(tab_dir / "T3_trigger_norm.csv", index=False)
    (tab_dir / "T3_trigger_norm.md").write_text(T3.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    # ─────── T4: correctness summary ───────
    t4_rows = []
    for g in ["correct", "incorrect", "no_trunc"]:
        sub = roll[roll["group"] == g]
        n = len(sub)
        n_trunc = int(sub["is_trunc"].sum())
        mean_clen = float(sub["completion_len"].mean()) if n > 0 else 0.0
        t4_rows.append({
            "group": g, "n_rollouts": n,
            "frac_total": n / max(1, n_total),
            "n_truncated": n_trunc,
            "frac_trunc_within_group": n_trunc / max(1, n),
            "mean_completion_len": mean_clen,
        })
    t4_rows.append({
        "group": "ALL", "n_rollouts": n_total,
        "frac_total": 1.0,
        "n_truncated": int(roll["is_trunc"].sum()),
        "frac_trunc_within_group": float(roll["is_trunc"].mean()),
        "mean_completion_len": float(roll["completion_len"].mean()),
    })
    T4 = pd.DataFrame(t4_rows)
    T4.to_csv(tab_dir / "T4_correctness_summary.csv", index=False)
    (tab_dir / "T4_correctness_summary.md").write_text(T4.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    # ─────── F1: trigger-norm distribution ───────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, g in zip(axes, GROUPS):
        v = roll[(roll["is_trunc"]) & (roll["group"] == g)]["trigger_norm"].to_numpy()
        if len(v) > 0:
            ax.hist(v, bins=40, range=(0, 1), color=GROUP_COLORS[g], alpha=0.75, density=True)
            ax.axvline(np.mean(v), color="black", ls="--", lw=1, label=f"mean={np.mean(v):.3f}")
            ax.axvline(np.median(v), color="gray", ls=":", lw=1, label=f"median={np.median(v):.3f}")
            ax.legend(fontsize=8)
        ax.set_title(f"{g} (n={len(v)})")
        ax.set_xlabel("(t + i_first) / completion_len")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
    fig.suptitle("F1 · Normalized low-KL trigger position by correctness group")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_trigger_norm_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in GROUPS:
        v = np.sort(roll[(roll["is_trunc"]) & (roll["group"] == g)]["trigger_norm"].to_numpy())
        if len(v) > 0:
            ax.plot(v, np.linspace(0, 1, len(v)), label=f"{g} (n={len(v)})",
                    color=GROUP_COLORS[g], lw=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("(t + i_first) / completion_len")
    ax.set_ylabel("CDF")
    ax.set_title("F1 · Empirical CDF of normalized trigger position")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_trigger_norm_cdf.png", dpi=140)
    plt.close(fig)

    # ─────── F2: entropy violin (student & teacher), group × stage ───────
    def _violin_group_stage(metric_name: str, ylabel: str, fname: str, title: str,
                            log: bool = True, hline: float | None = None,
                            hline_label: str | None = None):
        fig, axes = plt.subplots(1, len(GROUPS), figsize=(5 * len(GROUPS), 4.5),
                                 sharey=True)
        if len(GROUPS) == 1:
            axes = [axes]
        for ax, g in zip(axes, GROUPS):
            present: list[tuple[np.ndarray, str, str]] = []
            for s in STAGES:
                v = df[(df["group"] == g) & (df["stage"] == s)][metric_name].to_numpy()
                if len(v) == 0:
                    continue
                if log:
                    v = np.clip(v, 1e-8, None)
                present.append((v, f"{s}\n(n={len(v)})", STAGE_COLORS[s]))
            if not present:
                ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes, ha="center")
                ax.set_title(g)
                continue
            parts = ax.violinplot([p[0] for p in present],
                                  showmeans=False, showmedians=True, showextrema=False)
            for body, p in zip(parts["bodies"], present):
                body.set_facecolor(p[2])
                body.set_edgecolor("black")
                body.set_alpha(0.7)
            ax.set_xticks(range(1, len(present) + 1))
            ax.set_xticklabels([p[1] for p in present], rotation=15)
            if log:
                ax.set_yscale("log")
            ax.set_ylabel(ylabel)
            ax.set_title(g)
            ax.grid(alpha=0.3, which="both")
            if hline is not None:
                ax.axhline(hline, color="red", linestyle="--", lw=1.2,
                           label=hline_label)
                ax.legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=140)
        plt.close(fig)

    _violin_group_stage("H_s", "Student entropy (nats, log)",
                        "F2_entropy_hist_student.png",
                        "F2 · Student entropy by stage (group × stage)")
    _violin_group_stage("H_t", "Teacher entropy (nats, log)",
                        "F2_entropy_hist_teacher.png",
                        "F2 · Teacher entropy by stage (group × stage)")

    # ─────── F3: KL violin (group × stage), log-y, τ overlay ───────
    if len(df[df["stage"].isin(STAGES)]["kl"]) > 0:
        _violin_group_stage("kl", "Reverse KL per token (log)",
                            "F3_kl_hist.png",
                            f"F3 · Per-token reverse-KL by stage (τ = {threshold:.4g})",
                            log=True, hline=threshold,
                            hline_label=f"τ={threshold:.4g}")

    # ─────── F4: word clouds (group × stage), student & teacher top-1 ───────
    def _build_wordcloud(token_ids_2d: np.ndarray) -> dict[str, float] | None:
        """token_ids_2d: (N, K) array — K ids per position, all counted equally."""
        if token_ids_2d.size == 0:
            return None
        flat = token_ids_2d.ravel()
        ctr: Counter = Counter()
        uniq = np.unique(flat)
        id2word = {}
        for tid in uniq.tolist():
            txt = tok.convert_ids_to_tokens(int(tid))
            id2word[int(tid)] = _clean_word(txt)
        for tid in flat.tolist():
            w = id2word.get(int(tid))
            if w is None:
                continue
            ctr[w] += 1
        if not ctr:
            return None
        return dict(ctr.most_common(200))

    def _wordcloud_grid(source_prefix: str, fname: str, title: str, topk: int = 5):
        if WordCloud is None:
            print("[stage4] wordcloud not installed — skipping", fname)
            return
        cols = [f"{source_prefix}{k}" for k in range(1, topk + 1)]
        # Fall back gracefully if older parquet lacks top2..top5 columns
        cols = [c for c in cols if c in df.columns]
        if not cols:
            print(f"[stage4] no columns {source_prefix}1..{topk} in df — skipping {fname}")
            return
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        for i, g in enumerate(GROUPS):
            for j, s in enumerate(STAGES):
                ax = axes[i, j]
                sub = df[(df["group"] == g) & (df["stage"] == s)]
                if len(sub) == 0:
                    freqs = None
                else:
                    arr = sub[cols].to_numpy()
                    freqs = _build_wordcloud(arr)
                if not freqs:
                    ax.text(0.5, 0.5, "(no English tokens)", transform=ax.transAxes, ha="center")
                    ax.set_title(f"{g} · {s} (empty)", fontsize=11)
                    ax.axis("off")
                    continue
                wc = WordCloud(
                    width=1200, height=800, background_color="white",
                    prefer_horizontal=0.95, max_words=100,
                    collocations=False, min_font_size=8,
                )
                wc.generate_from_frequencies(freqs)
                ax.imshow(wc, interpolation="bilinear")
                ax.set_title(f"{g} · {s} (n_pos={len(sub)}, k={len(cols)}, words={len(freqs)})", fontsize=11)
                ax.axis("off")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=160)
        plt.close(fig)

    _wordcloud_grid("s_top", "F4_wordcloud_student.png",
                    "F4 · Student top-5 word cloud (group × stage, English only)", topk=5)
    _wordcloud_grid("t_top", "F4_wordcloud_teacher.png",
                    "F4 · Teacher top-5 word cloud (group × stage, English only)", topk=5)

    # ─────── debug TSV: sanity check stage assignment on 3 trunc rollouts ───────
    truncated_rids = roll[roll["is_trunc"]]["rollout_id"].to_numpy()
    if len(truncated_rids) >= 3:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(truncated_rids, size=3, replace=False)
        debug_lines = ["rollout_id\tposition\tkl\twin_mean_at(pos-t)\tstage"]
        L = args.L
        for rid in pick:
            g = df[df["rollout_id"] == rid].sort_values("position")
            kl_arr = g["kl"].to_numpy()
            pos_arr = g["position"].to_numpy()
            stage_arr = g["stage"].to_numpy()
            r_info = next(r for r in partition["rollouts"] if r["rollout_id"] == int(rid))
            ifirst = r_info["i_first"]
            ilast = r_info["i_last"]
            lo = max(args.t - 2, 0)
            hi = min(args.t + ilast + L + 10, len(kl_arr))
            for p in range(lo, hi):
                start = p
                end = p + L
                if end <= len(kl_arr):
                    wm = float(kl_arr[start:end].mean())
                else:
                    wm = float("nan")
                debug_lines.append(f"{rid}\t{p}\t{kl_arr[p]:.6g}\t{wm:.6g}\t{stage_arr[p]}")
            debug_lines.append("---")
        (out / "debug_stages.tsv").write_text("\n".join(debug_lines), encoding="utf-8")
        print(f"[stage4] wrote debug_stages.tsv for rids={pick.tolist()}")

    return {
        "T1": T1, "T2": T2, "T3": T3, "T4": T4,
        "n_total": n_total,
        "n_correct": n_correct_total,
        "n_incorrect": n_incorrect_total,
        "counts_per_group": counts_per_group,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Stage 5: Chinese markdown report
# ──────────────────────────────────────────────────────────────────────────────

def stage5_report(args, threshold: float, partition: dict, grading: dict, tabs: dict, bundle: dict) -> Path:
    out = Path(args.out_dir)
    md_path = out / "report.md"

    T1 = tabs["T1"]; T2 = tabs["T2"]; T3 = tabs["T3"]; T4 = tabs["T4"]
    n_total = tabs["n_total"]
    n_corr = tabs["n_correct"]
    n_incorr = tabs["n_incorrect"]
    cpg = tabs["counts_per_group"]
    n_trunc_total = int(sum(1 for r in partition["rollouts"] if r["is_trunc"]))
    finish_reasons = Counter(bundle.get("finish_reasons", []))

    def _g(grp, stg, key):
        sub = T2[(T2["group"] == grp) & (T2["stage"] == stg)]
        if len(sub) == 0 or key not in sub.columns:
            return float("nan")
        return float(sub.iloc[0][key])

    def _fmt(v):
        if isinstance(v, float):
            if v != v:
                return "nan"
            if abs(v) < 1e-3 or abs(v) >= 1e4:
                return f"{v:.2e}"
            return f"{v:.3g}"
        return str(v)

    def _t3(grp, metric, key):
        sub = T3[(T3["group"] == grp) & (T3["metric"] == metric)]
        if len(sub) == 0 or key not in sub.columns:
            return float("nan")
        return float(sub.iloc[0][key])

    md = f"""# KL 截断三阶段 Rollout 机制分析 (Qwen3-1.7B / checkpoint-380)

> Checkpoint: `{args.ckpt}`
> Teacher: `{args.teacher}`
> 阈值 τ = **{threshold:.10g}**（取自 `trainer_state.json["log_history"][step=={args.threshold_step}].kl_trunc/threshold`；step 380 仅有 eval 记录，最近一次训练 step 是 379）
> 总 rollout 数 = {n_total} (= {args.num_prompts} prompts × {args.rollouts_per_prompt} rollouts)
> 触发截断 = {n_trunc_total} (**{n_trunc_total / max(1,n_total):.1%}**)
> 数学打分（math_verify）正确 = {n_corr} (**{n_corr / max(1,n_total):.1%}**)
> 完成原因分布 = {dict(finish_reasons)}

## 1. 实验设置

| 项目 | 值 |
|---|---|
| Student | Qwen3-1.7B-class (hidden=2048), KL-trunc eta=0.3 训练 |
| Teacher | Qwen3-8B |
| Prompt 源 | `{args.prompt_parquet}` 前 {args.num_prompts} 条 |
| 每条 prompt rollout 数 | {args.rollouts_per_prompt} (vLLM `SamplingParams(n={args.rollouts_per_prompt})`, seed={args.seed}) |
| 采样 | T={args.temperature}, top_p={args.top_p}, max_tokens={args.max_completion_length} |
| KL 截断参数 | L={args.L}, t={args.t}, N={args.N}, eta=0.3 |
| KL 信号 | reverse-KL on student top-{args.top_k_loss} + tail bucket, temperature={args.loss_temperature} (与训练 loss 完全同口径) |
| 阈值 τ | {threshold:.10g} (step {args.threshold_step}) |

## 2. 方法

1. **Rollout 生成**：vLLM 上对 200 个 prompt 各采 4 条 completion。
2. **双前向 + per-token KL**：student 与 teacher 分别对 (prompt + completion) prefill，复刻 `generalized_jsd_loss(beta=1, top_k=64, add_tail=True)` 得到每 token reverse-KL；同时记录 student/teacher 全词表熵和各自 top-1 token id。
3. **三阶段切分**：以 stride=1 取 L={args.L} 滑窗 (从 token 位置 t={args.t} 起)，找出首个长度 ≥ N={args.N} 的连续 `win_mean < τ` 段起点 `i_first`，向后扩展到不再低于 τ 的最后窗口 `i_last`：
   - **pre**    : `[0, t + i_first)`
   - **in**     : `[t + i_first, t + i_last + L)`
   - **after**  : `[t + i_last + L, completion_len)`
   未触发的 rollout 整体记为 `no_trunc`，不参与各阶段直方图/词云。
4. **正确性分组**：每条 completion 用 `math500_eval._is_correct`（math_verify + `\\boxed{{}}` fallback）对 `reward_model.ground_truth` 打分。三类桶：
   - `correct`   : 触发截断 且 正确
   - `incorrect` : 触发截断 且 错误
   - `no_trunc`  : 未触发截断（不分对错，单独统计）

## 3. 总览

### T4 · 三类桶规模与基本属性

{T4.to_markdown(index=False, floatfmt=".4g")}

### T1 · token 与 rollout 计数

{T1.to_markdown(index=False, floatfmt=".4g")}

## 4. 低 KL 触发位置分布

![F1 hist](figures/F1_trigger_norm_hist.png)
![F1 CDF](figures/F1_trigger_norm_cdf.png)

### T3 · 触发位置归一化 / 各阶段长度（仅触发 rollout）

{T3.to_markdown(index=False, floatfmt=".4g")}

- correct 组归一化触发位置 mean / median = {_fmt(_t3("correct","trigger_norm","mean"))} / {_fmt(_t3("correct","trigger_norm","median"))}
- incorrect 组归一化触发位置 mean / median = {_fmt(_t3("incorrect","trigger_norm","mean"))} / {_fmt(_t3("incorrect","trigger_norm","median"))}
- correct 组 in 段长度 median = {_fmt(_t3("correct","in_len","median"))}，after 段长度 median = {_fmt(_t3("correct","after_len","median"))}
- incorrect 组 in 段长度 median = {_fmt(_t3("incorrect","in_len","median"))}，after 段长度 median = {_fmt(_t3("incorrect","after_len","median"))}

## 5. 各阶段熵分布

![F2 student](figures/F2_entropy_hist_student.png)
![F2 teacher](figures/F2_entropy_hist_teacher.png)

### T2 · 阶段统计（kl / H_s / H_t mean/median/p10/p90）

{T2.to_markdown(index=False, floatfmt=".4g")}

熵关键比较 (median):

| group | stage | H_s_median | H_t_median |
|---|---|---|---|
| correct | pre | {_fmt(_g("correct","pre","H_s_median"))} | {_fmt(_g("correct","pre","H_t_median"))} |
| correct | in | **{_fmt(_g("correct","in","H_s_median"))}** | **{_fmt(_g("correct","in","H_t_median"))}** |
| correct | after | {_fmt(_g("correct","after","H_s_median"))} | {_fmt(_g("correct","after","H_t_median"))} |
| incorrect | pre | {_fmt(_g("incorrect","pre","H_s_median"))} | {_fmt(_g("incorrect","pre","H_t_median"))} |
| incorrect | in | **{_fmt(_g("incorrect","in","H_s_median"))}** | **{_fmt(_g("incorrect","in","H_t_median"))}** |
| incorrect | after | {_fmt(_g("incorrect","after","H_s_median"))} | {_fmt(_g("incorrect","after","H_t_median"))} |

## 6. 各阶段 KL 分布

![F3](figures/F3_kl_hist.png)

KL 关键比较 (median, 与 τ={threshold:.4g} 对照):

| group | stage | kl_median | 相对 τ |
|---|---|---|---|
| correct | pre | {_fmt(_g("correct","pre","kl_median"))} | {_g("correct","pre","kl_median")/threshold:.2f}× τ |
| correct | in | **{_fmt(_g("correct","in","kl_median"))}** | **{_g("correct","in","kl_median")/threshold:.2f}× τ** |
| correct | after | {_fmt(_g("correct","after","kl_median"))} | {_g("correct","after","kl_median")/threshold:.2f}× τ |
| incorrect | pre | {_fmt(_g("incorrect","pre","kl_median"))} | {_g("incorrect","pre","kl_median")/threshold:.2f}× τ |
| incorrect | in | **{_fmt(_g("incorrect","in","kl_median"))}** | **{_g("incorrect","in","kl_median")/threshold:.2f}× τ** |
| incorrect | after | {_fmt(_g("incorrect","after","kl_median"))} | {_g("incorrect","after","kl_median")/threshold:.2f}× τ |

## 7. 各阶段词云 (Student / Teacher top-5, 仅英文)

![F4 student](figures/F4_wordcloud_student.png)

![F4 teacher](figures/F4_wordcloud_teacher.png)

> 注：词云汇总了每个 token 位置上 student / teacher 各自 logits 的 top-5 预测 token id（每个候选等权计 1 次），过滤所有非纯字母 token（数字、标点、`Ġ/Ċ/▁` 前缀、`ĉĉĉ` 类伪 token、不完整 BPE 片段），并剔除常见英文停用词。top-5 比 top-1 更能反映模型考虑的"候选集合"，尤其在低熵 `in` 段，top-1 几乎独占概率，top-5 揭示了余下 4 个候选指向什么。

## 8. 关键观察

1. **触发位置位置随正确性差异**：correct 组 mean 归一化触发位置 = {_fmt(_t3("correct","trigger_norm","mean"))}，incorrect 组 = {_fmt(_t3("incorrect","trigger_norm","mean"))}。如果 incorrect 组更早 (mean / median 更小)，说明错误 rollout 更早进入退化平台；如果两者接近，说明 collapse 时机与最终答案对错无强单调关系。
2. **in 段是 KL 真正塌缩区**：两组在 in 段的 median KL 都明显低于 τ ({_g("correct","in","kl_median")/threshold:.2f}× / {_g("incorrect","in","kl_median")/threshold:.2f}× τ)，符合「连续 N={args.N} 个滑窗均值 < τ」判据。
3. **熵同步塌缩**：student 与 teacher 在 in 段熵 (中位) 都比 pre 段降低数量级 (见 §5 表)，证明 in 段并非「教师误判」，而是两者都在该 token 序列上确信度极高 → 在该位置上 loss 信号几乎为 0。
4. **after 段的去向**：观察 `after` 段 KL 与熵相对 `in` 段的反弹幅度可以判断 collapse 是否「自恢复」。
5. **词云对比**：词云直接反映 collapse 文本特征——in 段如果出现高频重复词、奇怪片段、非自然词汇，说明确为退化输出；after 段如果重新出现自然推理用词 ('let', 'then', 'so'…)，说明 collapse 是中间瞬态。

## 9. 复现

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python analyze/analyze_stages.py \\
  --ckpt {args.ckpt} \\
  --teacher {args.teacher} \\
  --prompt_parquet {args.prompt_parquet} \\
  --num_prompts {args.num_prompts} --rollouts_per_prompt {args.rollouts_per_prompt} \\
  --threshold_step {args.threshold_step} \\
  --vllm_tensor_parallel_size {args.vllm_tensor_parallel_size} \\
  --out_dir {args.out_dir}
```

中间产物缓存：`rollouts.pt` / `per_token.parquet` / `grading.json` / `partition_summary.json`。
重画图：`--skip_rollouts --skip_prefill --skip_grade`。
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"[stage5] report saved to {md_path}")
    return md_path


# ──────────────────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.threshold is None:
        threshold = load_training_threshold(args.ckpt, args.threshold_step, args.threshold_expected)
        print(f"[main] τ = {threshold:.10g} (from step {args.threshold_step})")
    else:
        threshold = float(args.threshold)
        print(f"[main] τ = {threshold:.10g} (CLI override)")

    # Stage 1
    bundle = stage1_generate_rollouts(args)

    # Stage 2
    per_token_df = stage2_per_token(args, bundle)

    # Stage 3a
    per_token_df, partition = stage3_segment(args, per_token_df, threshold)
    pd.DataFrame(partition["rollouts"]).to_parquet(
        Path(args.out_dir) / "partition_per_rollout.parquet", index=False
    )
    with open(Path(args.out_dir) / "partition_summary.json", "w") as f:
        # exclude verbose 'rollouts' list to keep summary small; keep full list separately
        s = {k: v for k, v in partition.items() if k != "rollouts"}
        s["n_trunc"] = partition["n_trunc"]
        json.dump(s, f, indent=2, ensure_ascii=False)

    # Stage 3b: grade
    grading = stage3b_grade(args, bundle)

    # join (group) and persist combined parquet
    df_full = assemble_groups(per_token_df, partition, grading)
    df_full.to_parquet(Path(args.out_dir) / "per_token_with_stage.parquet", index=False)

    # Stage 4
    tabs = stage4_figures_tables(args, df_full, partition, grading, threshold)

    # Stage 5
    stage5_report(args, threshold, partition, grading, tabs, bundle)

    # Completeness assert
    must = [
        "figures/F1_trigger_norm_hist.png",
        "figures/F1_trigger_norm_cdf.png",
        "figures/F2_entropy_hist_student.png",
        "figures/F2_entropy_hist_teacher.png",
        "figures/F3_kl_hist.png",
        "figures/F4_wordcloud_student.png",
        "figures/F4_wordcloud_teacher.png",
        "tables/T1_counts.csv", "tables/T1_counts.md",
        "tables/T2_stage_stats.csv", "tables/T2_stage_stats.md",
        "tables/T3_trigger_norm.csv", "tables/T3_trigger_norm.md",
        "tables/T4_correctness_summary.csv", "tables/T4_correctness_summary.md",
        "report.md",
    ]
    missing = [m for m in must if not (Path(args.out_dir) / m).exists()]
    if missing:
        print("[main] WARNING missing artifacts:", missing)
    else:
        report_size = (Path(args.out_dir) / "report.md").stat().st_size
        print(f"[done] all artifacts present at {args.out_dir} (report.md {report_size} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
