"""Take the 100-rollout subset built by heatmap_normalized.py and further split
the heatmap by final math correctness (math_verify against DAPO ground truth).

Reads (under --out_dir):
  rollouts.pt
  per_token_kl.parquet
  trigger_info.parquet
  subset_ids.json
  stats.json

Writes (under --out_dir):
  figures/heatmap_sustained_lowkl_normalized_by_correctness.png
  subset_grading.json
  stats_subset_by_correctness.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prompt_parquet",
                   default="data/dapo-math-17k/dapo_clean.parquet")
    p.add_argument("--n_bins", type=int, default=200)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--L0", type=int, default=32)
    p.add_argument("--colormap", default="magma")
    return p.parse_args()


def _z_series(kl: np.ndarray, W: int) -> np.ndarray:
    """Partial-window warmup so every position has a defined value."""
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


def _import_is_correct():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from math500_eval import _is_correct
    return _is_correct


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    print(f"[main] loading from {out}")

    with open(out / "stats.json") as f:
        stats = json.load(f)
    threshold = float(stats["threshold"])

    with open(out / "subset_ids.json") as f:
        subset = json.load(f)
    row_order: list[int] = subset["row_order"]
    triggered_ids = set(int(x) for x in subset["triggered"])

    bundle = torch.load(out / "rollouts.pt", weights_only=False)
    completion_text = bundle["completion_text"]
    prompt_idx_list = bundle["prompt_idx"]

    df_dapo = pd.read_parquet(args.prompt_parquet)
    # ground truth lookup by row index = prompt_idx
    def _gt(rid: int) -> str:
        pi = int(prompt_idx_list[rid])
        rm = df_dapo.iloc[pi]["reward_model"]
        return str(rm["ground_truth"]) if isinstance(rm, dict) else ""

    is_correct = _import_is_correct()
    info = pd.read_parquet(out / "trigger_info.parquet").set_index("rollout_id")
    per_tok = pd.read_parquet(out / "per_token_kl.parquet")
    grouped = per_tok.groupby("rollout_id", sort=False)

    # Grade the 100 subset
    grading: list[dict] = []
    for rid in row_order:
        ct = completion_text[int(rid)]
        gt = _gt(rid)
        ok = bool(is_correct(ct, gt))
        grading.append({
            "rollout_id": int(rid),
            "is_correct": ok,
            "ground_truth": gt,
            "is_trunc": bool(rid in triggered_ids),
            "trigger_pos": int(info.loc[int(rid), "trigger_pos"]),
            "completion_len": int(info.loc[int(rid), "completion_len"]),
        })
    grade_df = pd.DataFrame(grading)
    with open(out / "subset_grading.json", "w") as f:
        json.dump({"items": grading,
                   "n_correct": int(grade_df["is_correct"].sum()),
                   "n_incorrect": int((~grade_df["is_correct"]).sum())},
                  f, indent=2, ensure_ascii=False)
    print(f"[main] graded: correct={int(grade_df['is_correct'].sum())} / 100")

    # Split into two groups; within each, keep the original (triggered then non-triggered) ordering.
    def _ordered_group(group_df: pd.DataFrame) -> list[int]:
        # restore row_order ordering inside this group
        order_map = {int(rid): i for i, rid in enumerate(row_order)}
        return sorted(group_df["rollout_id"].tolist(), key=lambda x: order_map[int(x)])

    correct_ids = _ordered_group(grade_df[grade_df["is_correct"]])
    incorrect_ids = _ordered_group(grade_df[~grade_df["is_correct"]])

    def _build_matrix(ids: list[int]) -> np.ndarray:
        if not ids:
            return np.zeros((0, args.n_bins), dtype=np.float32)
        mat = np.full((len(ids), args.n_bins), np.nan, dtype=np.float32)
        for i, rid in enumerate(ids):
            g = grouped.get_group(int(rid)).sort_values("position")
            kl = g["kl"].to_numpy()
            z = _z_series(kl, args.W)
            mat[i, :] = _resample_to_grid(z, args.n_bins) / threshold
        return mat

    mat_c = _build_matrix(correct_ids)
    mat_i = _build_matrix(incorrect_ids)

    # Render side-by-side heatmaps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    all_finite = np.concatenate([mat_c[np.isfinite(mat_c)], mat_i[np.isfinite(mat_i)]])
    if all_finite.size > 0:
        vmin = max(float(np.percentile(all_finite, 0.5)), 1e-4)
        vmax = min(float(np.percentile(all_finite, 99.5)), 50.0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    cmap = plt.get_cmap(args.colormap).copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1, 1]})

    def _draw(ax, mat: np.ndarray, ids: list[int], title: str):
        R = mat.shape[0]
        if R == 0:
            ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes, ha="center")
            ax.set_title(title)
            return
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", origin="upper",
                       cmap=cmap, norm=norm, interpolation="nearest",
                       extent=[0.0, 1.0, R - 0.5, -0.5])
        # τ markers for triggered rows
        gd = grade_df.set_index("rollout_id")
        rid_to_row = {rid: i for i, rid in enumerate(ids)}
        xs, ys = [], []
        n_trig_in_group = 0
        for rid in ids:
            r = gd.loc[int(rid)]
            if bool(r["is_trunc"]):
                xs.append(r["trigger_pos"] / r["completion_len"])
                ys.append(rid_to_row[int(rid)])
                n_trig_in_group += 1
        if xs:
            ax.scatter(xs, ys, s=14, c="black", marker="|", linewidths=0.9,
                       label=f"τ trigger (n={n_trig_in_group})")
            ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
        # red dashed divider between triggered and non-triggered within this column
        if 0 < n_trig_in_group < R:
            ax.axhline(n_trig_in_group - 0.5, color="red", lw=0.6, ls="--", alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("normalized rollout position  (token_index / completion_len)")
        ax.set_ylabel("rollouts (triggered ↑ early → late, non-triggered ↓)")
        ax.set_title(title)
        return im

    n_c, n_i = len(correct_ids), len(incorrect_ids)
    n_c_trig = int(grade_df[grade_df["is_correct"] & grade_df["is_trunc"]].shape[0])
    n_i_trig = int(grade_df[(~grade_df["is_correct"]) & grade_df["is_trunc"]].shape[0])
    ckpt_tag = Path(stats.get("ckpt", out.parent.name)).name if False else None  # placeholder
    im_c = _draw(axes[0], mat_c, correct_ids,
                 f"Correct  (n={n_c}, triggered={n_c_trig} → {n_c_trig/max(1,n_c):.1%})")
    im_i = _draw(axes[1], mat_i, incorrect_ids,
                 f"Incorrect  (n={n_i}, triggered={n_i_trig} → {n_i_trig/max(1,n_i):.1%})")

    # Shared colorbar
    im_for_cbar = im_c if im_c is not None else im_i
    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, ax=axes, pad=0.015, fraction=0.025)
        cbar.set_label("normalized sliding-window KL  z_t / s_k")
        try:
            cbar.ax.plot([0, 1], [1.0, 1.0], color="red", lw=1.2,
                         transform=cbar.ax.get_yaxis_transform())
        except Exception:
            pass

    fig.suptitle(
        f"Sustained low-KL regions split by final correctness   |   subset N=100 "
        f"(τ={threshold:.4g})",
        fontsize=13
    )
    fig_path = out / "figures" / "heatmap_sustained_lowkl_normalized_by_correctness.png"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[main] saved {fig_path}")

    # Group-level stats
    def _g_stats(g_df: pd.DataFrame) -> dict:
        n = len(g_df)
        n_tr = int(g_df["is_trunc"].sum())
        if n_tr > 0:
            sub = g_df[g_df["is_trunc"]].copy()
            sub["tau_norm"] = sub["trigger_pos"] / sub["completion_len"]
            sub["post_frac"] = (sub["completion_len"] - sub["trigger_pos"]) / sub["completion_len"]
            return {
                "n": n, "n_triggered": n_tr,
                "trigger_rate": n_tr / max(1, n),
                "mean_tau_norm": float(sub["tau_norm"].mean()),
                "median_tau_norm": float(sub["tau_norm"].median()),
                "mean_post_trigger_fraction": float(sub["post_frac"].mean()),
                "mean_completion_len": float(g_df["completion_len"].mean()),
            }
        return {"n": n, "n_triggered": 0, "trigger_rate": 0.0,
                "mean_tau_norm": float("nan"),
                "median_tau_norm": float("nan"),
                "mean_post_trigger_fraction": float("nan"),
                "mean_completion_len": float(g_df["completion_len"].mean()) if n else 0.0}

    cbc = {
        "correct": _g_stats(grade_df[grade_df["is_correct"]]),
        "incorrect": _g_stats(grade_df[~grade_df["is_correct"]]),
        "threshold": threshold,
    }
    with open(out / "stats_subset_by_correctness.json", "w") as f:
        json.dump(cbc, f, indent=2)
    print("[main] stats by correctness:")
    print(json.dumps(cbc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
