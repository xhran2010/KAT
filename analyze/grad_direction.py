"""Beneficial / Toxic analysis of OPD rollout segments via checkpoint-displacement
golden direction + SVD dominant subspace projection.

Pipeline:
  A. Re-generate 800 rollouts on theta_ref (ckpt-80) via vLLM, HF prefill (KL/H/topk),
     segment with τ_ref80 = trainer_state.json[step=80].kl_trunc/threshold, math-grade.
  B. Compute d_gold = theta_best - theta_start streamed from safetensors (filtered).
     SVD top-k(d_gold) for each selected 2D matrix, cache to disk.
  C. Load theta_ref as differentiable student (only filtered params requires_grad=True)
     + teacher (no_grad). For each sampled rollout × {pre, in, after}:
        loss_seg = per_token_KL[seg].mean()
        loss_seg.backward()
        u_S = -grad
        stream cosine(u_S, d_gold), proj_ratio on SVD top-k, signed_svd score.
  D. Label segments {beneficial, toxic, neutral, *_on_gold_subspace, off_subspace}.
  E. Figures F1-F8 + Chinese report.md.

Run:
  CUDA_VISIBLE_DEVICES=0,1 \
  python analyze/grad_direction.py \
      --theta_start outputs/1.7b_p2048_eta0.3_s32_ckpt/checkpoint-20 \
      --theta_ref   outputs/1.7b_p2048_eta0.3_s32_ckpt/checkpoint-80 \
      --theta_best  outputs/1.7b_p2048_eta0.3_s32/checkpoint-380 \
      --teacher     Qwen/Qwen3-8B \
      --out_dir     analyze/work_grad_dir
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "analyze") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "analyze"))


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # checkpoints
    p.add_argument("--theta_start", required=True)
    p.add_argument("--theta_ref", required=True)
    p.add_argument("--theta_best", required=True)
    p.add_argument("--teacher", default="Qwen/Qwen3-8B")
    # rollout regen (Stage A)
    p.add_argument("--prompt_parquet",
                   default=str(REPO_ROOT / "datasets/dapo-math-17k.parquet"))
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
    p.add_argument("--threshold_step", type=int, default=80)
    p.add_argument("--threshold_expected", type=float, default=0.1285126330330968)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=2)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45)
    p.add_argument("--vllm_max_model_length", type=int, default=4096)
    p.add_argument("--student_gpu", type=int, default=0)
    p.add_argument("--teacher_gpu", type=int, default=1)
    p.add_argument("--enable_thinking", action="store_true", default=False)
    # selection (Stage B)
    p.add_argument("--scope", default="mid_mlp",
                   choices=["mid_mlp", "mlp", "attn", "mid_attn_mlp", "all_attn_mlp", "custom"])
    p.add_argument("--include_regex", default="")
    p.add_argument("--exclude_regex", default="")
    p.add_argument("--svd_k", type=int, default=32)
    p.add_argument("--svd_oversampling", type=int, default=10)
    p.add_argument("--svd_niter", type=int, default=2)
    # grad pass (Stage C)
    p.add_argument("--num_rollouts", type=int, default=200)
    p.add_argument("--min_seg_tokens", type=int, default=8)
    # labels (Stage D)
    p.add_argument("--eps_pos", type=float, default=0.02)
    p.add_argument("--eps_proj", type=float, default=0.05)
    # misc
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    # cache controls
    p.add_argument("--skip_rollouts", action="store_true")
    p.add_argument("--skip_prefill", action="store_true")
    p.add_argument("--skip_grade", action="store_true")
    p.add_argument("--skip_dgold", action="store_true")
    p.add_argument("--skip_svd", action="store_true")
    p.add_argument("--skip_grad", action="store_true")
    p.add_argument("--skip_figures", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Re-used helpers (copied from analyze/analyze_stages.py to keep self-contained)
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


# ──────────────────────────────────────────────────────────────────────────────
#  Stage A — rollouts on ckpt-80 (reuses analyze_stages stage1/2/3)
# ──────────────────────────────────────────────────────────────────────────────

def stage_A(args) -> tuple[dict, pd.DataFrame, dict, dict, float]:
    """Re-generate rollouts/per_token/partition/grading under {out_dir}/rollouts/.

    Returns (bundle, per_token_df_with_stage, partition, grading, threshold).
    """
    import analyze_stages as AS

    sub_dir = Path(args.out_dir) / "rollouts"
    sub_dir.mkdir(parents=True, exist_ok=True)

    a = SimpleNamespace(
        ckpt=args.theta_ref,
        teacher=args.teacher,
        prompt_parquet=args.prompt_parquet,
        num_prompts=args.num_prompts,
        rollouts_per_prompt=args.rollouts_per_prompt,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        L=args.L, t=args.t, N=args.N,
        top_k_loss=args.top_k_loss,
        loss_temperature=args.loss_temperature,
        threshold=None,
        threshold_step=args.threshold_step,
        threshold_expected=args.threshold_expected,
        out_dir=str(sub_dir),
        seed=args.seed,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        student_gpu=args.student_gpu,
        teacher_gpu=args.teacher_gpu,
        enable_thinking=args.enable_thinking,
        skip_rollouts=args.skip_rollouts,
        skip_prefill=args.skip_prefill,
        skip_grade=args.skip_grade,
    )

    threshold = AS.load_training_threshold(args.theta_ref, args.threshold_step, args.threshold_expected)
    print(f"[stage_A] τ_ref80 = {threshold:.16g} (step {args.threshold_step})")

    bundle = AS.stage1_generate_rollouts(a)
    per_token_df = AS.stage2_per_token(a, bundle)
    per_token_df, partition = AS.stage3_segment(a, per_token_df, threshold)
    pd.DataFrame(partition["rollouts"]).to_parquet(
        sub_dir / "partition_per_rollout.parquet", index=False
    )
    with open(sub_dir / "partition_summary.json", "w") as f:
        s = {k: v for k, v in partition.items() if k != "rollouts"}
        json.dump(s, f, indent=2, ensure_ascii=False)
    grading = AS.stage3b_grade(a, bundle)
    df_full = AS.assemble_groups(per_token_df, partition, grading)
    df_full.to_parquet(sub_dir / "per_token_with_stage.parquet", index=False)
    return bundle, df_full, partition, grading, threshold


# ──────────────────────────────────────────────────────────────────────────────
#  Stage B — parameter selection + d_gold + SVD cache
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_EXCLUDE = re.compile(r"embed_tokens|lm_head|norm|layernorm|rmsnorm|bias")
ATTN_PATS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PATS = ["gate_proj", "up_proj", "down_proj"]


def _layer_idx_from_name(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _module_type_from_name(name: str) -> str | None:
    for p in ATTN_PATS + MLP_PATS:
        if p + ".weight" in name or p + ".bias" in name:
            return p.replace("_proj", "")
    return None


def safe_name(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")


def build_param_filter(args, all_names: list[str], num_layers: int):
    """Return list of param names that pass the scope/include/exclude filters."""
    mid_lo = num_layers // 3
    mid_hi = num_layers - num_layers // 3  # exclusive
    mid_layers = set(range(mid_lo, mid_hi))

    def is_attn(n): return any(p + ".weight" in n for p in ATTN_PATS)
    def is_mlp(n): return any(p + ".weight" in n for p in MLP_PATS)

    if args.scope == "custom":
        if not args.include_regex:
            raise ValueError("--scope custom requires --include_regex")
        inc = re.compile(args.include_regex)
        exc = re.compile(args.exclude_regex) if args.exclude_regex else None
        names = [n for n in all_names if inc.search(n) and (exc is None or not exc.search(n))]
    else:
        scope_filters = {
            "all_attn_mlp": lambda n: is_attn(n) or is_mlp(n),
            "mlp":          lambda n: is_mlp(n),
            "attn":         lambda n: is_attn(n),
            "mid_mlp":      lambda n: is_mlp(n) and _layer_idx_from_name(n) in mid_layers,
            "mid_attn_mlp": lambda n: (is_attn(n) or is_mlp(n)) and _layer_idx_from_name(n) in mid_layers,
        }
        fn = scope_filters[args.scope]
        names = [n for n in all_names if fn(n) and not DEFAULT_EXCLUDE.search(n)]
    return sorted(names), (mid_lo, mid_hi)


def stage_B_d_gold(args, selected_names: list[str]):
    """Stream-load theta_start and theta_best safetensors, write d_gold per param."""
    from safetensors import safe_open
    d_dir = Path(args.out_dir) / "d_gold"
    d_dir.mkdir(parents=True, exist_ok=True)
    stats_rows = []

    start_dir = Path(args.theta_start)
    best_dir = Path(args.theta_best)

    # Determine the safetensors files: single model.safetensors OR sharded
    def _open_index(ckpt_dir: Path):
        if (ckpt_dir / "model.safetensors").exists():
            return [ckpt_dir / "model.safetensors"]
        idx = ckpt_dir / "model.safetensors.index.json"
        if idx.exists():
            with open(idx) as f:
                weight_map = json.load(f)["weight_map"]
            files = sorted(set(ckpt_dir / f for f in weight_map.values()))
            return files
        raise FileNotFoundError(f"No model.safetensors[.index.json] in {ckpt_dir}")

    start_files = _open_index(start_dir)
    best_files = _open_index(best_dir)

    # Open all safetensors files lazily; build name->file map
    def _name_map(files):
        m = {}
        for fp in files:
            with safe_open(str(fp), framework="pt", device="cpu") as f:
                for k in f.keys():
                    m[k] = str(fp)
        return m

    start_map = _name_map(start_files)
    best_map = _name_map(best_files)

    for i, name in enumerate(selected_names):
        out_path = d_dir / f"{safe_name(name)}.pt"
        if args.skip_dgold and out_path.exists():
            t = torch.load(out_path, map_location="cpu", weights_only=True)
            stats_rows.append({
                "name": name,
                "shape": list(t["d_gold"].shape),
                "fro_sq": float(t["fro_sq"]),
                "fro": float(t["fro_sq"]) ** 0.5,
                "layer": _layer_idx_from_name(name),
                "module_type": _module_type_from_name(name),
            })
            continue
        if name not in start_map or name not in best_map:
            print(f"[stage_B] WARN: {name} missing in start/best, skip")
            continue
        with safe_open(start_map[name], framework="pt", device="cpu") as f:
            w_start = f.get_tensor(name).float()
        with safe_open(best_map[name], framework="pt", device="cpu") as f:
            w_best = f.get_tensor(name).float()
        d = w_best - w_start
        fro_sq = float((d * d).sum().item())
        torch.save({"d_gold": d, "fro_sq": fro_sq, "name": name}, out_path)
        stats_rows.append({
            "name": name,
            "shape": list(d.shape),
            "fro_sq": fro_sq,
            "fro": fro_sq ** 0.5,
            "layer": _layer_idx_from_name(name),
            "module_type": _module_type_from_name(name),
        })
        if (i + 1) % 10 == 0:
            print(f"[stage_B] d_gold {i+1}/{len(selected_names)}")

    return stats_rows


def stage_B_svd(args, selected_names: list[str], svd_device: str = "cuda:0"):
    """SVD top-k cache for each selected 2D matrix's d_gold."""
    svd_dir = Path(args.out_dir) / "svd_cache"
    svd_dir.mkdir(parents=True, exist_ok=True)
    d_dir = Path(args.out_dir) / "d_gold"
    k = args.svd_k
    explained = {}

    for i, name in enumerate(selected_names):
        sf = safe_name(name)
        d_path = d_dir / f"{sf}.pt"
        out_path = svd_dir / f"{sf}.pt"
        if args.skip_svd and out_path.exists():
            obj = torch.load(out_path, map_location="cpu", weights_only=True)
            explained[name] = float(obj.get("top_k_explained", float("nan")))
            continue
        if not d_path.exists():
            continue
        d = torch.load(d_path, map_location="cpu", weights_only=True)["d_gold"]
        if d.dim() != 2:
            continue
        out_dim, in_dim = d.shape
        min_dim = min(out_dim, in_dim)
        kk = min(k, min_dim)

        d_gpu = d.to(svd_device, dtype=torch.float32)
        if min_dim <= 256:
            U, S, Vh = torch.linalg.svd(d_gpu, full_matrices=False)
            U_k = U[:, :kk].contiguous()
            S_k = S[:kk].contiguous()
            V_k = Vh.transpose(-2, -1)[:, :kk].contiguous()
            full_fro_sq = float((S ** 2).sum().item())
        else:
            q = min(kk + args.svd_oversampling, min_dim)
            U, S, V = torch.svd_lowrank(d_gpu, q=q, niter=args.svd_niter)
            U_k = U[:, :kk].contiguous()
            S_k = S[:kk].contiguous()
            V_k = V[:, :kk].contiguous()
            # For low-rank approximation, use full Frobenius norm of ΔW for the ratio
            full_fro_sq = float((d_gpu * d_gpu).sum().item())

        top_k_explained = float((S_k ** 2).sum().item() / max(full_fro_sq, 1e-30))
        torch.save({
            "U_k": U_k.cpu(),
            "S_k": S_k.cpu(),
            "V_k": V_k.cpu(),
            "fro_sq": full_fro_sq,
            "top_k_explained": top_k_explained,
            "name": name, "k": kk,
        }, out_path)
        explained[name] = top_k_explained
        del d_gpu, U, S
        if (i + 1) % 10 == 0:
            print(f"[stage_B] SVD {i+1}/{len(selected_names)}  (mean explained so far: {np.mean(list(explained.values())):.3f})")
        torch.cuda.empty_cache()

    return explained


# ──────────────────────────────────────────────────────────────────────────────
#  Stage C — per-segment gradient on theta_ref
# ──────────────────────────────────────────────────────────────────────────────

STAGES = ("pre", "in", "after")


def _stage_ranges(row: pd.Series, args) -> dict[str, tuple[int, int]]:
    """Reproduce analyze_stages.py:381-383 boundaries.

       pre_end = min(t + i_first, S)
       in_end  = min(t + i_last + L, S)
    """
    S = int(row["completion_len"])
    if not bool(row["is_trunc"]):
        return {"pre": (0, 0), "in": (0, 0), "after": (0, 0)}
    i_first = int(row["i_first"]); i_last = int(row["i_last"])
    pre_end = min(args.t + i_first, S)
    in_end = min(args.t + i_last + args.L, S)
    return {"pre": (0, pre_end), "in": (pre_end, in_end), "after": (in_end, S)}


def sample_rollouts(args, partition: dict, grading: dict) -> list[dict]:
    """Stratified sample: half correct, half incorrect, all is_trunc=True."""
    correct_map = {it["rollout_id"]: bool(it["is_correct"]) for it in grading["items"]}
    rolls = pd.DataFrame(partition["rollouts"])
    rolls["is_correct"] = rolls["rollout_id"].map(correct_map).fillna(False)
    triggered = rolls[rolls["is_trunc"]].copy()
    correct = triggered[triggered["is_correct"]]
    incorrect = triggered[~triggered["is_correct"]]
    target = args.num_rollouts // 2

    rng = np.random.default_rng(args.seed)
    def _sample(df, n):
        if len(df) <= n:
            print(f"[stage_C] WARN sample only {len(df)}/{n}")
            return df.copy()
        idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
        return df.loc[idx].copy()

    selected = pd.concat([_sample(correct, target), _sample(incorrect, target)]).sort_values("rollout_id")
    print(f"[stage_C] sampled {len(selected)} rollouts: "
          f"correct={int(selected['is_correct'].sum())} / incorrect={int((~selected['is_correct']).sum())}")
    return selected.to_dict("records")


def _build_aggregator(selected_names: list[str]):
    """Returns dict-of-running-sums for global/mlp/attn/layer/module aggregates."""
    return {
        "global": {"dot": 0.0, "u_sq": 0.0, "d_sq": 0.0},
        "mlp":    {"dot": 0.0, "u_sq": 0.0, "d_sq": 0.0},
        "attn":   {"dot": 0.0, "u_sq": 0.0, "d_sq": 0.0},
        "layer":  {},        # layer_idx -> dict
        "module": {},        # module_type -> dict
        "perparam": {},      # name -> dict (for module-wise cosine)
    }


def _accum(agg: dict, key_tuple, dot, u_sq, d_sq):
    """key_tuple is a chain like ('layer', 9) or ('module', 'gate') or ('global',)."""
    bucket = agg
    for i, key in enumerate(key_tuple):
        if i == len(key_tuple) - 1:
            if key not in bucket:
                bucket[key] = {"dot": 0.0, "u_sq": 0.0, "d_sq": 0.0}
            bucket[key]["dot"] += dot
            bucket[key]["u_sq"] += u_sq
            bucket[key]["d_sq"] += d_sq
        else:
            if key not in bucket:
                bucket[key] = {}
            bucket = bucket[key]


def _cos_of(d):
    if d["u_sq"] <= 0 or d["d_sq"] <= 0:
        return float("nan")
    return d["dot"] / ((d["u_sq"] ** 0.5) * (d["d_sq"] ** 0.5) + 1e-12)


def stage_C_gradients(args, partition: dict, grading: dict, bundle: dict, selected_names: list[str]):
    """For each sampled (rollout, stage): forward+backward, stream cosines + SVD projection."""
    from transformers import AutoModelForCausalLM

    s_dev = torch.device(f"cuda:{args.student_gpu}")
    t_dev = torch.device(f"cuda:{args.teacher_gpu}") if torch.cuda.device_count() >= 2 else s_dev

    print(f"[stage_C] loading student from {args.theta_ref} → {s_dev}")
    student = AutoModelForCausalLM.from_pretrained(
        args.theta_ref, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
    ).to(s_dev)
    student.train()
    for p in student.parameters():
        p.requires_grad_(False)
    name_to_param = dict(student.named_parameters())
    selected_present = [n for n in selected_names if n in name_to_param]
    missing = set(selected_names) - set(selected_present)
    if missing:
        print(f"[stage_C] WARN: {len(missing)} names missing from student model: {list(missing)[:3]}…")
    for n in selected_present:
        name_to_param[n].requires_grad_(True)
    print(f"[stage_C] enabled grad on {len(selected_present)} params")

    print(f"[stage_C] loading teacher from {args.teacher} → {t_dev}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", low_cpu_mem_usage=True,
    ).to(t_dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # pre-load d_gold + SVD cache onto cuda:0 lazily, cache the small ones
    d_gold_cache: dict[str, torch.Tensor] = {}
    svd_cache: dict[str, dict[str, torch.Tensor]] = {}
    d_dir = Path(args.out_dir) / "d_gold"
    svd_dir = Path(args.out_dir) / "svd_cache"

    def _get_dgold(name) -> torch.Tensor:
        if name not in d_gold_cache:
            obj = torch.load(d_dir / f"{safe_name(name)}.pt", map_location="cpu", weights_only=True)
            d_gold_cache[name] = obj["d_gold"].to(s_dev, dtype=torch.float32)
        return d_gold_cache[name]

    def _get_svd(name):
        if name not in svd_cache:
            p_ = svd_dir / f"{safe_name(name)}.pt"
            if not p_.exists():
                svd_cache[name] = None
                return None
            obj = torch.load(p_, map_location="cpu", weights_only=True)
            svd_cache[name] = {
                "U_k": obj["U_k"].to(s_dev, dtype=torch.float32),
                "S_k": obj["S_k"].to(s_dev, dtype=torch.float32),
                "V_k": obj["V_k"].to(s_dev, dtype=torch.float32),
            }
        return svd_cache[name]

    # sample rollouts
    selected_rolls = sample_rollouts(args, partition, grading)
    if not selected_rolls:
        raise RuntimeError("No triggered+graded rollouts available; cannot run stage C.")

    seg_global_rows = []
    seg_module_rows = []

    n_done = 0
    for rec in selected_rolls:
        rid = int(rec["rollout_id"])
        prompt = bundle["prompt_ids"][rid]
        completion = bundle["completion_ids"][rid]
        Lp, Lc = len(prompt), len(completion)
        if Lc < 4:
            continue
        ranges = _stage_ranges(pd.Series(rec), args)

        full_ids = torch.tensor(prompt + completion, dtype=torch.long).unsqueeze(0)
        full_ids_s = full_ids.to(s_dev)
        full_ids_t = full_ids.to(t_dev)
        Ltot = full_ids.size(1)

        # student forward with grad; teacher forward no_grad
        s_out = student(full_ids_s, use_cache=False)
        s_logits_full = s_out.logits  # bf16
        s_logits = s_logits_full[:, Lp - 1 : Ltot - 1, :].float()  # (1, Lc, V) requires_grad

        with torch.no_grad():
            t_out = teacher(full_ids_t, use_cache=False)
            t_logits = t_out.logits[:, Lp - 1 : Ltot - 1, :].to(s_dev).float()
        del t_out

        per_tok_kl = per_token_reverse_kl_topk_tail(
            s_logits, t_logits, top_k=args.top_k_loss, temperature=args.loss_temperature
        )  # (1, Lc), grad-attached

        seg_list = [(s, ranges[s]) for s in STAGES]
        seg_list = [(s, (lo, hi)) for s, (lo, hi) in seg_list if (hi - lo) >= args.min_seg_tokens]

        for sidx, (stage_name, (lo, hi)) in enumerate(seg_list):
            student.zero_grad(set_to_none=True)
            loss_seg = per_tok_kl[0, lo:hi].mean()
            kl_mean_val = float(loss_seg.item())
            retain = (sidx < len(seg_list) - 1)
            loss_seg.backward(retain_graph=retain)

            agg = _build_aggregator(selected_present)
            for name in selected_present:
                p = name_to_param[name]
                if p.grad is None:
                    continue
                u = (-p.grad).float()
                d = _get_dgold(name)
                # cosine streaming
                dot = float((u * d).sum().item())
                u_sq = float((u * u).sum().item())
                d_sq = float((d * d).sum().item())
                _accum(agg, ("global",), dot, u_sq, d_sq)
                mt = _module_type_from_name(name)
                if mt in ("gate", "up", "down"):
                    _accum(agg, ("mlp",), dot, u_sq, d_sq)
                if mt in ("q", "k", "v", "o"):
                    _accum(agg, ("attn",), dot, u_sq, d_sq)
                Lidx = _layer_idx_from_name(name)
                if Lidx is not None:
                    _accum(agg, ("layer", Lidx), dot, u_sq, d_sq)
                if mt is not None:
                    _accum(agg, ("module", mt), dot, u_sq, d_sq)
                _accum(agg, ("perparam", name), dot, u_sq, d_sq)

                # SVD projection (only for 2D weight matrices)
                svd = _get_svd(name) if u.dim() == 2 else None
                proj_ratio = float("nan")
                signed_svd = float("nan")
                if svd is not None and u.dim() == 2:
                    U_k = svd["U_k"]; S_k = svd["S_k"]; V_k = svd["V_k"]
                    M = U_k.T @ u @ V_k        # (k, k)
                    coeffs = torch.diagonal(M)  # (k,)
                    proj_norm_sq = float((coeffs * coeffs).sum().item())
                    total_norm_sq = u_sq
                    proj_ratio = proj_norm_sq / max(total_norm_sq, 1e-30)
                    num = float((coeffs * S_k).sum().item())
                    den = (float((coeffs * coeffs).sum().item()) ** 0.5) * (float((S_k * S_k).sum().item()) ** 0.5)
                    signed_svd = num / (den + 1e-12)

                seg_module_rows.append({
                    "rollout_id": rid,
                    "stage": stage_name,
                    "is_correct": bool(rec["is_correct"]),
                    "name": name,
                    "layer": Lidx,
                    "module_type": mt,
                    "u_fro": u_sq ** 0.5,
                    "d_fro": d_sq ** 0.5,
                    "cos": (dot / (u_sq ** 0.5 * d_sq ** 0.5 + 1e-12)) if (u_sq > 0 and d_sq > 0) else float("nan"),
                    "proj_ratio": proj_ratio,
                    "signed_svd": signed_svd,
                })

            # SVD on global combined u/d (sum of per-param projections) — we use mean of per-module proj_ratio weighted by ||u||^2
            # Compute global proj ratio across all selected 2D matrices
            global_u_sq = agg["global"]["u_sq"]
            sum_proj_sq = 0.0
            sum_signed_num = 0.0
            sum_signed_den_coeffs_sq = 0.0
            sum_signed_den_S_sq = 0.0
            for name in selected_present:
                p = name_to_param[name]
                if p.grad is None or p.dim() != 2:
                    continue
                svd = _get_svd(name)
                if svd is None:
                    continue
                u = (-p.grad).float()
                U_k = svd["U_k"]; S_k = svd["S_k"]; V_k = svd["V_k"]
                M = U_k.T @ u @ V_k
                coeffs = torch.diagonal(M)
                proj_norm_sq = float((coeffs * coeffs).sum().item())
                sum_proj_sq += proj_norm_sq
                sum_signed_num += float((coeffs * S_k).sum().item())
                sum_signed_den_coeffs_sq += float((coeffs * coeffs).sum().item())
                sum_signed_den_S_sq += float((S_k * S_k).sum().item())
            ratio_global = sum_proj_sq / max(global_u_sq, 1e-30)
            signed_svd_global = sum_signed_num / ((sum_signed_den_coeffs_sq ** 0.5) * (sum_signed_den_S_sq ** 0.5) + 1e-12) if sum_signed_den_S_sq > 0 else float("nan")

            cos_global = _cos_of(agg["global"])
            cos_mlp = _cos_of(agg["mlp"])
            cos_attn = _cos_of(agg["attn"])

            label = _basic_label(cos_global, args.eps_pos)
            svd_label = _svd_label(cos_global, ratio_global, args.eps_pos, args.eps_proj)

            seg_global_rows.append({
                "rollout_id": rid,
                "stage": stage_name,
                "is_correct": bool(rec["is_correct"]),
                "group": "correct" if rec["is_correct"] else "incorrect",
                "completion_len": Lc,
                "seg_lo": lo, "seg_hi": hi, "n_tokens": hi - lo,
                "kl_mean": kl_mean_val,
                "cos_global": cos_global,
                "cos_mlp": cos_mlp,
                "cos_attn": cos_attn,
                "ratio_global": ratio_global,
                "signed_svd_global": signed_svd_global,
                "label": label,
                "svd_label": svd_label,
            })

        # release this rollout's graph
        del s_out, s_logits_full, s_logits, t_logits, per_tok_kl
        student.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        n_done += 1
        if n_done % 10 == 0:
            print(f"[stage_C] processed {n_done}/{len(selected_rolls)} rollouts; "
                  f"global_rows={len(seg_global_rows)} module_rows={len(seg_module_rows)}", flush=True)

    seg_global = pd.DataFrame(seg_global_rows)
    seg_module = pd.DataFrame(seg_module_rows)
    seg_global.to_parquet(Path(args.out_dir) / "segment_global.parquet", index=False)
    seg_module.to_parquet(Path(args.out_dir) / "segment_module.parquet", index=False)
    print(f"[stage_C] saved segment_global.parquet ({len(seg_global)} rows), segment_module.parquet ({len(seg_module)} rows)")

    del student, teacher
    gc.collect()
    torch.cuda.empty_cache()
    return seg_global, seg_module


# ──────────────────────────────────────────────────────────────────────────────
#  Stage D — labels (used inline in Stage C already)
# ──────────────────────────────────────────────────────────────────────────────

def _basic_label(cos: float, eps: float) -> str:
    if cos != cos:  # nan
        return "nan"
    if cos > eps:
        return "beneficial"
    if cos < -eps:
        return "toxic"
    return "neutral"


def _svd_label(cos: float, ratio: float, eps_pos: float, eps_proj: float) -> str:
    if cos != cos or ratio != ratio:
        return "nan"
    if ratio < eps_proj:
        return "off_subspace_or_low_value"
    if cos > eps_pos:
        return "beneficial_on_gold_subspace"
    if cos < -eps_pos:
        return "toxic_on_gold_subspace"
    return "orthogonal_but_gold_subspace"


# ──────────────────────────────────────────────────────────────────────────────
#  Stage E — figures + Chinese report
# ──────────────────────────────────────────────────────────────────────────────

STAGE_COLORS = {"pre": "#1f77b4", "in": "#d62728", "after": "#2ca02c"}
GROUP_COLORS = {"correct": "#1f77b4", "incorrect": "#d62728"}
LABEL_COLORS = {
    "beneficial": "#2ca02c",
    "toxic": "#d62728",
    "neutral": "#7f7f7f",
    "beneficial_on_gold_subspace": "#1f7a1f",
    "toxic_on_gold_subspace": "#a01010",
    "orthogonal_but_gold_subspace": "#ff7f0e",
    "off_subspace_or_low_value": "#bfbfbf",
    "nan": "#000000",
}


def stage_E_figures(args, seg_global: pd.DataFrame, seg_module: pd.DataFrame,
                    d_gold_stats: pd.DataFrame, partition: dict, grading: dict,
                    threshold: float, svd_explained: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(args.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stages = list(STAGES)
    groups = ["correct", "incorrect"]

    # auto x-range for cosine plots (driven by data, symmetric around 0)
    all_cos = seg_global["cos_global"].dropna().to_numpy()
    cos_xmax = float(np.percentile(np.abs(all_cos), 99.5)) * 1.2 if len(all_cos) else 1.0
    cos_xmax = max(cos_xmax, 0.01)
    all_signed = seg_global["signed_svd_global"].dropna().to_numpy()
    signed_xmax = float(np.percentile(np.abs(all_signed), 99.5)) * 1.1 if len(all_signed) else 1.0
    signed_xmax = max(signed_xmax, 0.1)

    # ── F1: cos_global hist by stage ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, s in zip(axes, stages):
        v = seg_global[seg_global["stage"] == s]["cos_global"].dropna().to_numpy()
        if len(v):
            ax.hist(v, bins=40, range=(-cos_xmax, cos_xmax), color=STAGE_COLORS[s], alpha=0.8, density=True)
            ax.axvline(0, color="black", lw=1)
            ax.axvline(v.mean(), color="black", ls="--", lw=1, label=f"mean={v.mean():.4f}")
            ax.axvline(np.median(v), color="gray", ls=":", lw=1, label=f"median={np.median(v):.4f}")
            ax.legend(fontsize=8)
        ax.set_xlim(-cos_xmax, cos_xmax)
        ax.set_title(f"{s} (n={len(v)})")
        ax.set_xlabel("cos(u_S, d_gold)")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
    fig.suptitle("F1 · Global cosine(u_S, d_gold) by stage")
    fig.tight_layout()
    fig.savefig(fig_dir / "F1_cos_global_hist_by_stage.png", dpi=140)
    plt.close(fig)

    # ── F2: cos_global by (group × stage) ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
    for i, g in enumerate(groups):
        for j, s in enumerate(stages):
            ax = axes[i, j]
            v = seg_global[(seg_global["stage"] == s) & (seg_global["group"] == g)]["cos_global"].dropna().to_numpy()
            if len(v):
                ax.hist(v, bins=30, range=(-cos_xmax, cos_xmax), color=GROUP_COLORS[g], alpha=0.75, density=True)
                ax.axvline(0, color="black", lw=1)
                ax.axvline(v.mean(), color="black", ls="--", lw=1,
                           label=f"μ={v.mean():.4f}\nm={np.median(v):.4f}")
                ax.legend(fontsize=8, loc="upper left")
            ax.set_xlim(-cos_xmax, cos_xmax)
            ax.set_title(f"{g} · {s} (n={len(v)})")
            if i == 1:
                ax.set_xlabel("cos_global")
            if j == 0:
                ax.set_ylabel("density")
            ax.grid(alpha=0.3)
    fig.suptitle("F2 · cos_global by correctness × stage")
    fig.tight_layout()
    fig.savefig(fig_dir / "F2_cos_global_by_group_stage.png", dpi=140)
    plt.close(fig)

    # ── F3: proj_ratio hist by stage ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, s in zip(axes, stages):
        v = seg_global[seg_global["stage"] == s]["ratio_global"].dropna().to_numpy()
        if len(v):
            ax.hist(np.clip(v, 0, 1), bins=40, range=(0, 1), color=STAGE_COLORS[s], alpha=0.8, density=True)
            ax.axvline(args.eps_proj, color="black", ls=":", lw=1, label=f"eps={args.eps_proj}")
            ax.axvline(v.mean(), color="black", ls="--", lw=1, label=f"mean={v.mean():.3f}")
            ax.legend(fontsize=8)
        ax.set_title(f"{s} (n={len(v)})")
        ax.set_xlabel("proj_ratio (top-k SVD)")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
    fig.suptitle(f"F3 · Top-k SVD projection ratio (k={args.svd_k})")
    fig.tight_layout()
    fig.savefig(fig_dir / "F3_proj_ratio_by_stage.png", dpi=140)
    plt.close(fig)

    # ── F4: cos vs proj_ratio scatter (1×3) ───────────────────────────────
    all_ratio = seg_global["ratio_global"].dropna().to_numpy()
    ratio_ymax = float(np.percentile(all_ratio, 99)) * 1.2 if len(all_ratio) else 1.0
    ratio_ymax = max(ratio_ymax, 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, sharex=True)
    for ax, s in zip(axes, stages):
        sub = seg_global[seg_global["stage"] == s]
        if len(sub):
            for g in groups:
                ss = sub[sub["group"] == g]
                ax.scatter(ss["cos_global"], ss["ratio_global"],
                           c=GROUP_COLORS[g], alpha=0.5, s=14, label=f"{g} (n={len(ss)})")
            ax.axhline(args.eps_proj, color="black", ls=":", lw=0.8)
            ax.axvline(args.eps_pos, color="black", ls=":", lw=0.8, label=f"±eps_pos={args.eps_pos}")
            ax.axvline(-args.eps_pos, color="black", ls=":", lw=0.8)
            ax.axvline(0, color="gray", lw=0.5)
            ax.legend(fontsize=7)
        ax.set_title(f"stage = {s}")
        ax.set_xlim(-cos_xmax, cos_xmax)
        ax.set_ylim(0, ratio_ymax)
        ax.set_xlabel("cos_global")
        ax.set_ylabel("proj_ratio")
        ax.grid(alpha=0.3)
    fig.suptitle("F4 · cosine vs SVD projection ratio")
    fig.tight_layout()
    fig.savefig(fig_dir / "F4_cos_vs_proj_scatter.png", dpi=140)
    plt.close(fig)

    # ── F5: layer × stage heatmap of mean cosine (per-module aggregate) ──
    seg_module_nonan = seg_module.dropna(subset=["layer", "cos"])
    layers_sorted = sorted(seg_module_nonan["layer"].dropna().astype(int).unique().tolist())

    def _heat(mat, xticks, yticks, vmin, vmax, cmap, label, title, fname, figsize, fmt=".2f"):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(xticks))); ax.set_xticklabels(xticks, rotation=0)
        ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks)
        thresh = (vmin + vmax) / 2 if cmap == "viridis" else (vmax * 0.5)
        for j in range(mat.shape[0]):
            for i in range(mat.shape[1]):
                if np.isfinite(mat[j, i]):
                    val = mat[j, i]
                    if cmap == "coolwarm":
                        text_color = "white" if abs(val) > thresh else "black"
                    else:
                        text_color = "white" if val < thresh else "black"
                    ax.text(i, j, format(val, fmt), ha="center", va="center", fontsize=8, color=text_color)
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=140)
        plt.close(fig)

    if layers_sorted:
        mat = np.full((len(stages), len(layers_sorted)), np.nan)
        for j, s in enumerate(stages):
            for i, L in enumerate(layers_sorted):
                v = seg_module_nonan[(seg_module_nonan["stage"] == s) & (seg_module_nonan["layer"].astype(int) == L)]["cos"]
                if len(v):
                    mat[j, i] = float(v.mean())
        vmax = max(float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.1, 0.05)
        _heat(mat, [f"L{l}" for l in layers_sorted], list(stages),
              -vmax, vmax, "coolwarm", "mean module cos",
              "F5 · Layer × Stage mean module cosine",
              "F5_layer_cos_heatmap.png",
              (max(8, len(layers_sorted) * 0.7), 3.4))

    # ── F6: module-type × stage heatmap ──────────────────────────────────
    mtypes_order = ["q", "k", "v", "o", "gate", "up", "down"]
    mtypes_present = [m for m in mtypes_order if m in seg_module_nonan["module_type"].unique()]
    if mtypes_present:
        mat = np.full((len(stages), len(mtypes_present)), np.nan)
        for j, s in enumerate(stages):
            for i, mt in enumerate(mtypes_present):
                v = seg_module_nonan[(seg_module_nonan["stage"] == s) & (seg_module_nonan["module_type"] == mt)]["cos"]
                if len(v):
                    mat[j, i] = float(v.mean())
        vmax = max(float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.1, 0.05)
        _heat(mat, mtypes_present, list(stages),
              -vmax, vmax, "coolwarm", "mean cos",
              "F6 · Module-type × Stage mean cosine",
              "F6_module_cos_heatmap.png",
              (max(5, len(mtypes_present) * 0.9), 3.2))

    # ── F5b: layer × stage heatmap of mean proj_ratio ────────────────────
    seg_module_proj = seg_module.dropna(subset=["layer", "proj_ratio"])
    if layers_sorted and len(seg_module_proj):
        mat = np.full((len(stages), len(layers_sorted)), np.nan)
        for j, s in enumerate(stages):
            for i, L in enumerate(layers_sorted):
                v = seg_module_proj[(seg_module_proj["stage"] == s) & (seg_module_proj["layer"].astype(int) == L)]["proj_ratio"]
                if len(v):
                    mat[j, i] = float(v.mean())
        pmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 0.1
        pmax = max(pmax, 1e-3)
        _heat(mat, [f"L{l}" for l in layers_sorted], list(stages),
              0, pmax, "viridis", "mean proj_ratio (top-k SVD)",
              "F5b · Layer × Stage mean proj_ratio",
              "F5b_layer_proj_ratio_heatmap.png",
              (max(8, len(layers_sorted) * 0.7), 3.4),
              fmt=".3f")

    # ── F6b: module-type × stage heatmap of mean proj_ratio ──────────────
    if mtypes_present and len(seg_module_proj):
        mat = np.full((len(stages), len(mtypes_present)), np.nan)
        for j, s in enumerate(stages):
            for i, mt in enumerate(mtypes_present):
                v = seg_module_proj[(seg_module_proj["stage"] == s) & (seg_module_proj["module_type"] == mt)]["proj_ratio"]
                if len(v):
                    mat[j, i] = float(v.mean())
        pmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 0.1
        pmax = max(pmax, 1e-3)
        _heat(mat, mtypes_present, list(stages),
              0, pmax, "viridis", "mean proj_ratio",
              "F6b · Module-type × Stage mean proj_ratio",
              "F6b_module_proj_ratio_heatmap.png",
              (max(5, len(mtypes_present) * 0.9), 3.2),
              fmt=".3f")

    # ── F7: label distribution by stage (stacked bars) ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label_col, title in zip(axes, ["label", "svd_label"], ["basic label", "SVD-aware label"]):
        all_labels = list(LABEL_COLORS.keys())
        df_lab = seg_global.copy()
        ct = pd.crosstab(df_lab["stage"], df_lab[label_col]).reindex(index=stages, fill_value=0)
        for lab in all_labels:
            if lab not in ct.columns:
                ct[lab] = 0
        ct = ct[all_labels]
        ct_pct = ct.div(ct.sum(axis=1), axis=0).fillna(0.0)
        bottom = np.zeros(len(stages))
        for lab in all_labels:
            vals = ct_pct[lab].to_numpy()
            ax.bar(stages, vals, bottom=bottom, color=LABEL_COLORS[lab], label=lab, edgecolor="white", lw=0.5)
            for i, v in enumerate(vals):
                if v > 0.04:
                    ax.text(i, bottom[i] + v / 2, f"{int(v*100)}%", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.1 else "black")
            bottom += vals
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction")
        ax.set_title(f"F7 · {title} by stage")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(fig_dir / "F7_label_bar_by_stage.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── F8: signed SVD score histogram by stage ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, s in zip(axes, stages):
        v = seg_global[seg_global["stage"] == s]["signed_svd_global"].dropna().to_numpy()
        if len(v):
            ax.hist(v, bins=40, range=(-signed_xmax, signed_xmax), color=STAGE_COLORS[s], alpha=0.8, density=True)
            ax.axvline(0, color="black", lw=1)
            ax.axvline(v.mean(), color="black", ls="--", lw=1, label=f"mean={v.mean():.3f}")
            ax.legend(fontsize=8)
        ax.set_xlim(-signed_xmax, signed_xmax)
        ax.set_title(f"{s} (n={len(v)})")
        ax.set_xlabel("signed SVD score")
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)
    fig.suptitle("F8 · Signed SVD subspace score by stage")
    fig.tight_layout()
    fig.savefig(fig_dir / "F8_signed_svd_hist.png", dpi=140)
    plt.close(fig)

    print(f"[stage_E] wrote 8 figures to {fig_dir}")


# ──────────────────────────────────────────────────────────────────────────────
#  Chinese report
# ──────────────────────────────────────────────────────────────────────────────

def stage_E_report(args, seg_global: pd.DataFrame, seg_module: pd.DataFrame,
                   d_gold_stats: pd.DataFrame, partition: dict, grading: dict,
                   threshold: float, svd_explained: dict, selected_names: list[str]):
    out = Path(args.out_dir)
    rows_overall = []
    for s in STAGES:
        sub = seg_global[seg_global["stage"] == s]
        for col in ["cos_global", "cos_mlp", "cos_attn", "ratio_global", "signed_svd_global"]:
            v = sub[col].dropna().to_numpy()
            if len(v) == 0:
                continue
            rows_overall.append({
                "stage": s, "metric": col, "n": len(v),
                "mean": float(np.mean(v)), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            })
    T_overall = pd.DataFrame(rows_overall)

    rows_grp = []
    for g in ["correct", "incorrect"]:
        for s in STAGES:
            sub = seg_global[(seg_global["group"] == g) & (seg_global["stage"] == s)]
            for col in ["cos_global", "ratio_global", "signed_svd_global"]:
                v = sub[col].dropna().to_numpy()
                if len(v) == 0:
                    continue
                rows_grp.append({
                    "group": g, "stage": s, "metric": col, "n": len(v),
                    "mean": float(np.mean(v)), "median": float(np.median(v)),
                    "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                })
    T_grp = pd.DataFrame(rows_grp)

    label_counts = pd.crosstab(seg_global["stage"], seg_global["label"]).reindex(index=list(STAGES), fill_value=0)
    svd_label_counts = pd.crosstab(seg_global["stage"], seg_global["svd_label"]).reindex(index=list(STAGES), fill_value=0)

    T_overall.to_csv(out / "T_overall_stats.csv", index=False)
    T_grp.to_csv(out / "T_group_stats.csv", index=False)
    label_counts.to_csv(out / "T_label_counts.csv")
    svd_label_counts.to_csv(out / "T_svd_label_counts.csv")

    n_rollouts_used = seg_global["rollout_id"].nunique()
    n_correct_used = seg_global[seg_global["group"] == "correct"]["rollout_id"].nunique()
    n_incorrect_used = seg_global[seg_global["group"] == "incorrect"]["rollout_id"].nunique()
    n_trunc_total = int(sum(1 for r in partition["rollouts"] if r["is_trunc"]))

    mean_explained = float(np.mean([v for v in svd_explained.values() if np.isfinite(v)])) if svd_explained else float("nan")

    md = []
    md.append(f"# OPD Beneficial / Toxic 分析：Checkpoint-displacement Golden Direction + SVD Subspace Projection")
    md.append("")
    md.append(f"> θ_start = `{args.theta_start}`")
    md.append(f"> θ_ref   = `{args.theta_ref}`")
    md.append(f"> θ_best  = `{args.theta_best}`")
    md.append(f"> Teacher = `{args.teacher}`")
    md.append(f"> τ_ref80 = **{threshold:.10g}** (from `trainer_state.json` step {args.threshold_step})")
    md.append(f"> Scope = **{args.scope}**, selected params = {len(selected_names)}, SVD k = {args.svd_k}, mean top-k explained = {mean_explained:.3f}")
    md.append(f"> Sampled rollouts = {n_rollouts_used}  (correct={n_correct_used} / incorrect={n_incorrect_used}; all is_trunc=True)")
    md.append(f"> 触发截断率 = {n_trunc_total}/{len(partition['rollouts'])} ({n_trunc_total/max(1,len(partition['rollouts'])):.1%})  ← 注意是在 ckpt-80 上重新 sample 的")
    md.append("")
    md.append("## 1. 实验设置")
    md.append("")
    md.append("- Student = Qwen3-1.7B (hidden=2048, 28 layers); Teacher = Qwen3-8B")
    md.append(f"- Rollouts 重新在 θ_ref=ckpt-80 采样: 200 prompts × 4 rollouts = 800, T={args.temperature}, top_p={args.top_p}, max_tokens={args.max_completion_length}, seed={args.seed}.")
    md.append(f"- KL 信号: reverse-KL on student top-{args.top_k_loss} + tail bucket, T={args.loss_temperature} (与训练 loss 完全同口径).")
    md.append(f"- 三阶段切分: L={args.L}, t={args.t}, N={args.N}, τ=τ_ref80; 公式 pre=[0,t+i_first), in=[t+i_first,t+i_last+L), after=[t+i_last+L,Lc).")
    md.append(f"- 参数过滤: scope=`{args.scope}` → {len(selected_names)} 个二维权重矩阵.")
    md.append(f"- SVD: 对每个 ΔW=W_best-W_start, top-k={args.svd_k} (oversampling={args.svd_oversampling}, niter={args.svd_niter}).")
    md.append(f"- 采样: 每条 rollout 三段 pre/in/after, 仅在触发截断的 rollout 上各抽 correct/incorrect 各 {args.num_rollouts//2} 条, segment < {args.min_seg_tokens} token 跳过.")
    md.append("")
    md.append("## 2. 方法")
    md.append("")
    md.append("**Golden direction**:  d_gold[W] = W_best - W_start, 全部 fp32 离线一次性算完缓存到磁盘.")
    md.append("")
    md.append("**Segment update direction**:  在 θ_ref 上对 segment range [s_lo, s_hi) 上的 per-token reverse-KL 取均值得到 loss_seg, autograd 一次后")
    md.append("u_S[W] = - ∂L_seg/∂W. 三个 segment 通过 retain_graph 复用同一次 forward, zero_grad 在 segment 之间清空.")
    md.append("")
    md.append("**Cosine** (streaming, 不 flatten 拼大向量):")
    md.append("```")
    md.append("dot   = Σ_W <u_S[W], d_gold[W]>")
    md.append("u_sq  = Σ_W ||u_S[W]||_F^2")
    md.append("d_sq  = Σ_W ||d_gold[W]||_F^2")
    md.append("cos   = dot / sqrt(u_sq * d_sq)")
    md.append("```")
    md.append("同时按 mlp / attn / layer / module-type 维度独立聚合得到 cos_mlp / cos_attn / layer-cos / module-cos.")
    md.append("")
    md.append("**SVD subspace projection** (per 2D 矩阵):  ΔW = U Σ V^T (top-k), `coeff_i = u_i^T G v_i`, `proj_ratio = Σ coeff_i^2 / ||G||_F^2`, `signed_svd = Σ coeff_i σ_i / (||coeff|| · ||σ||)`. 全局 `ratio_global` 与 `signed_svd_global` 在 selected 参数集合上求加权和后归一化.")
    md.append("")
    md.append(f"**标签规则** (eps_pos={args.eps_pos}, eps_proj={args.eps_proj}):")
    md.append("- `cos > eps_pos` → `beneficial` ；`cos < -eps_pos` → `toxic` ；否则 `neutral`")
    md.append("- 若 `ratio_global ≥ eps_proj` 与 cos 联立: `beneficial_on_gold_subspace` / `toxic_on_gold_subspace` / `orthogonal_but_gold_subspace`；`ratio_global < eps_proj` 一律 `off_subspace_or_low_value`")
    md.append("")
    md.append("## 3. 总体统计 (per stage, all groups)")
    md.append("")
    md.append(T_overall.to_markdown(index=False, floatfmt=".4g"))
    md.append("")
    md.append("![F1 cos hist](figures/F1_cos_global_hist_by_stage.png)")
    md.append("![F3 proj ratio](figures/F3_proj_ratio_by_stage.png)")
    md.append("![F8 signed SVD](figures/F8_signed_svd_hist.png)")
    md.append("")
    md.append("## 4. correct vs incorrect 拆分")
    md.append("")
    md.append(T_grp.to_markdown(index=False, floatfmt=".4g"))
    md.append("")
    md.append("![F2 cos by group×stage](figures/F2_cos_global_by_group_stage.png)")
    md.append("")
    md.append("## 5. 层 & 模块分布")
    md.append("")
    md.append("**Cosine (alignment with d_gold):**")
    md.append("")
    md.append("![F5 layer cos heat](figures/F5_layer_cos_heatmap.png)")
    md.append("![F6 module cos heat](figures/F6_module_cos_heatmap.png)")
    md.append("")
    md.append("**Projection ratio (energy in top-k SVD subspace of d_gold):**")
    md.append("")
    md.append("![F5b layer proj heat](figures/F5b_layer_proj_ratio_heatmap.png)")
    md.append("![F6b module proj heat](figures/F6b_module_proj_ratio_heatmap.png)")
    md.append("")
    md.append("## 6. 联合视角 (cosine × proj_ratio)")
    md.append("")
    md.append("![F4 scatter](figures/F4_cos_vs_proj_scatter.png)")
    md.append("")
    md.append("## 7. 标签分布")
    md.append("")
    md.append("**Basic label counts (per stage):**")
    md.append("")
    md.append(label_counts.to_markdown())
    md.append("")
    md.append("**SVD-aware label counts (per stage):**")
    md.append("")
    md.append(svd_label_counts.to_markdown())
    md.append("")
    md.append("![F7 labels](figures/F7_label_bar_by_stage.png)")
    md.append("")

    # ── 8. 自动观察
    def _mean_of(stage, col, grp=None):
        sub = seg_global[seg_global["stage"] == stage]
        if grp is not None:
            sub = sub[sub["group"] == grp]
        v = sub[col].dropna().to_numpy()
        return float(np.mean(v)) if len(v) else float("nan")

    md.append("## 8. 关键观察")
    md.append("")
    md.append(f"- **pre / in / after 段平均 cos_global** = {_mean_of('pre','cos_global'):.3f} / {_mean_of('in','cos_global'):.3f} / {_mean_of('after','cos_global'):.3f}")
    md.append(f"  - 正值 ⇒ 该 segment 的训练梯度反方向更接近 θ_best-θ_start 的方向, 即朝最终好模型推进 (beneficial).")
    md.append(f"  - 负值 ⇒ 与最终方向背道 (toxic).")
    md.append(f"- **pre / in / after 平均 proj_ratio** = {_mean_of('pre','ratio_global'):.3f} / {_mean_of('in','ratio_global'):.3f} / {_mean_of('after','ratio_global'):.3f}")
    md.append(f"  - 高 ratio ⇒ 梯度能量集中在最终位移的 top-{args.svd_k} subspace; 低 ratio ⇒ 与最终训练效益方向正交 (浪费).")
    md.append(f"- **correct vs incorrect 在 in 段平均 cos**: correct={_mean_of('in','cos_global','correct'):.3f}  incorrect={_mean_of('in','cos_global','incorrect'):.3f}")
    md.append(f"- **基本标签整体占比 (across stages)**: " + ", ".join([f"{lab}={int(seg_global['label'].value_counts().get(lab,0))}" for lab in ['beneficial','toxic','neutral']]))
    md.append(f"- **SVD-aware 标签整体占比**: " + ", ".join([f"{lab}={int(seg_global['svd_label'].value_counts().get(lab,0))}" for lab in ['beneficial_on_gold_subspace','toxic_on_gold_subspace','orthogonal_but_gold_subspace','off_subspace_or_low_value']]))
    md.append("")
    md.append("## 9. 复现命令")
    md.append("")
    md.append("```bash")
    md.append(f"CUDA_VISIBLE_DEVICES=0,1 python analyze/grad_direction.py \\")
    md.append(f"  --theta_start {args.theta_start} --theta_ref {args.theta_ref} --theta_best {args.theta_best} \\")
    md.append(f"  --teacher {args.teacher} --scope {args.scope} --svd_k {args.svd_k} \\")
    md.append(f"  --num_rollouts {args.num_rollouts} --out_dir {args.out_dir}")
    md.append("```")
    md.append("")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[stage_E] report.md written ({len((out / 'report.md').read_text())} chars)")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ── Stage A ──────────────────────────────────────────────────────────
    bundle, df_full, partition, grading, threshold = stage_A(args)

    # ── Stage B: param filter ────────────────────────────────────────────
    # We need full list of param names; load student config + named_parameters once cheaply via safetensors keys
    from safetensors import safe_open
    if (Path(args.theta_ref) / "model.safetensors").exists():
        with safe_open(str(Path(args.theta_ref) / "model.safetensors"), framework="pt", device="cpu") as f:
            all_names = list(f.keys())
    else:
        idx = Path(args.theta_ref) / "model.safetensors.index.json"
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        all_names = sorted(wm.keys())

    # determine num_layers from config
    cfg = json.load(open(Path(args.theta_ref) / "config.json"))
    num_layers = int(cfg.get("num_hidden_layers", 28))

    selected_names, (mid_lo, mid_hi) = build_param_filter(args, all_names, num_layers)
    print(f"[stage_B] scope={args.scope} mid_layers=[{mid_lo},{mid_hi}) → {len(selected_names)} params")
    sel_info = {
        "scope": args.scope, "mid_layers": [mid_lo, mid_hi],
        "num_layers": num_layers, "n_selected": len(selected_names),
        "include_regex": args.include_regex, "exclude_regex": args.exclude_regex,
        "names": selected_names,
    }
    with open(Path(args.out_dir) / "selected_params.json", "w") as f:
        json.dump(sel_info, f, indent=2)

    d_gold_stats = stage_B_d_gold(args, selected_names)
    pd.DataFrame(d_gold_stats).to_parquet(Path(args.out_dir) / "d_gold_stats.parquet", index=False)
    svd_explained = stage_B_svd(args, selected_names)

    # update d_gold_stats with explained
    dgs = pd.read_parquet(Path(args.out_dir) / "d_gold_stats.parquet")
    dgs["top_k_explained"] = dgs["name"].map(svd_explained).astype(float)
    dgs.to_parquet(Path(args.out_dir) / "d_gold_stats.parquet", index=False)

    # ── Stage C ──────────────────────────────────────────────────────────
    if args.skip_grad and (Path(args.out_dir) / "segment_global.parquet").exists():
        seg_global = pd.read_parquet(Path(args.out_dir) / "segment_global.parquet")
        seg_module = pd.read_parquet(Path(args.out_dir) / "segment_module.parquet")
        print(f"[stage_C] reused {(Path(args.out_dir) / 'segment_global.parquet')}")
    else:
        seg_global, seg_module = stage_C_gradients(args, partition, grading, bundle, selected_names)

    # ── Stage E ──────────────────────────────────────────────────────────
    if not args.skip_figures:
        stage_E_figures(args, seg_global, seg_module, dgs, partition, grading, threshold, svd_explained)
    stage_E_report(args, seg_global, seg_module, dgs, partition, grading, threshold, svd_explained, selected_names)
    print(f"[done] artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
