"""Companion to heatmap_lowkl.py: proportional-sample 100 rollouts from the
existing 1000 (triggered : non-triggered preserved at 66 : 34 in this run),
normalize each rollout's token position by its own completion_len, resample
onto a fixed [0,1] grid, and render the heatmap with no NaN padding on the
right.

Reads:
  analyze/work_heatmap/per_token_kl.parquet
  analyze/work_heatmap/trigger_info.parquet
  analyze/work_heatmap/stats.json

Writes:
  analyze/work_heatmap/figures/heatmap_sustained_lowkl_normalized.png
  analyze/work_heatmap/stats_subset.json
  analyze/work_heatmap/subset_ids.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default="analyze/work_heatmap")
    p.add_argument("--n_subset", type=int, default=100)
    p.add_argument("--n_bins", type=int, default=200,
                   help="Number of normalized position bins in [0,1].")
    p.add_argument("--W", type=int, default=64, help="Sliding window size used for z_t.")
    p.add_argument("--L0", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--colormap", default="magma")
    return p.parse_args()


def _z_series(kl: np.ndarray, W: int) -> np.ndarray:
    """Right-aligned rolling mean with partial-window warmup so every position has
    a defined value: z[t] = mean(kl[max(0, t-W+1) .. t]).
    For t >= W-1 this is the standard window mean; for t < W-1 it is the running
    mean of the available prefix (used only for visualization — the trigger
    detection itself still uses the full-window definition)."""
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
    """Resample z (length L) to n_bins positions on [0,1] using nearest-neighbour
    on the right-aligned grid x_t = (t + 1) / L. NaNs in z stay NaN in output."""
    out = np.full(n_bins, np.nan, dtype=np.float32)
    L = z.shape[0]
    if L < 1:
        return out
    src_x = (np.arange(L, dtype=np.float64) + 1.0) / float(L)  # right-edge of token t
    # For each grid bin centre, find the nearest src_x ≤ that centre
    tgt_x = (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
    # Use searchsorted on src_x sorted ascending (it is)
    idx = np.searchsorted(src_x, tgt_x, side="right") - 1
    idx = np.clip(idx, 0, L - 1)
    out[:] = z[idx]
    return out


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    per_tok = pd.read_parquet(out / "per_token_kl.parquet")
    info = pd.read_parquet(out / "trigger_info.parquet")
    with open(out / "stats.json") as f:
        stats = json.load(f)
    threshold = float(stats["threshold"])

    n_total = len(info)
    n_trig = int(info["is_trunc"].sum())
    n_notrig = n_total - n_trig
    # proportional split
    n_sub_trig = int(round(args.n_subset * n_trig / n_total))
    n_sub_notrig = args.n_subset - n_sub_trig

    rng = np.random.default_rng(args.seed)
    trig_ids = info[info["is_trunc"]]["rollout_id"].to_numpy()
    notrig_ids = info[~info["is_trunc"]]["rollout_id"].to_numpy()
    pick_trig = rng.choice(trig_ids, size=min(n_sub_trig, len(trig_ids)), replace=False)
    pick_notrig = rng.choice(notrig_ids, size=min(n_sub_notrig, len(notrig_ids)), replace=False)
    print(f"[main] sampled {len(pick_trig)} triggered + {len(pick_notrig)} non-triggered "
          f"(target {n_sub_trig}/{n_sub_notrig} from population {n_trig}/{n_notrig})")

    info_idx = info.set_index("rollout_id")

    # Sort triggered by trigger_norm = trigger_pos / completion_len ascending; non-triggered by completion_len asc
    trig_meta = info_idx.loc[pick_trig].copy()
    trig_meta["trigger_norm"] = trig_meta["trigger_pos"] / trig_meta["completion_len"]
    trig_meta = trig_meta.sort_values("trigger_norm")

    notrig_meta = info_idx.loc[pick_notrig].copy()
    notrig_meta = notrig_meta.sort_values("completion_len")

    row_ids = list(trig_meta.index) + list(notrig_meta.index)

    # Build matrix (R, n_bins)
    grouped = per_tok.groupby("rollout_id", sort=False)
    R = len(row_ids)
    mat = np.full((R, args.n_bins), np.nan, dtype=np.float32)
    for i, rid in enumerate(row_ids):
        g = grouped.get_group(int(rid)).sort_values("position")
        kl = g["kl"].to_numpy()
        z = _z_series(kl, args.W)
        mat[i, :] = _resample_to_grid(z, args.n_bins) / threshold

    # Render heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(12, 7.5))
    finite = mat[np.isfinite(mat)]
    if finite.size > 0:
        vmin = max(float(np.percentile(finite, 0.5)), 1e-4)
        vmax = min(float(np.percentile(finite, 99.5)), 50.0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None
    # Custom diverging colormap: low → high
    custom_colors = [
        "#053061", "#134b87", "#327db7", "#6fafd2", "#c7e0ed",
        "#fbd2bc", "#feab88", "#b71c2c", "#8b0824", "#6a0624",
    ]
    cmap = LinearSegmentedColormap.from_list("custom_div", custom_colors, N=256).copy()
    cmap.set_bad(color="white")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest",
                   extent=[0.0, 1.0, R - 0.5, -0.5])

    # trigger markers (normalized)
    if len(trig_meta) > 0:
        # row index in heatmap == position in row_ids
        ys = np.arange(len(trig_meta))
        xs = trig_meta["trigger_norm"].to_numpy()
        ax.scatter(xs, ys, s=14, c="black", marker="|", linewidths=0.9,
                   label=f"τ trigger (n={len(trig_meta)})")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    n_trig_rows = len(trig_meta)
    if 0 < n_trig_rows < R:
        ax.axhline(n_trig_rows - 0.5, color="red", lw=0.6, ls="--", alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_xlabel("normalized rollout position  (token_index / completion_len)")
    ax.set_ylabel("rollouts sorted by normalized trigger position")
    ax.set_title(
        "Rollout-specific sustained low-KL regions (length-normalized)\n"
        f"subset N={R} (triggered={n_trig_rows}, non-triggered={R - n_trig_rows}), "
        f"proportional sample from population (trigger_rate={stats['trigger_rate']:.1%}), "
        f"τ={threshold:.4g}"
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("normalized sliding-window KL  z_t / s_k")
    # reference line for y=1 in colorbar
    try:
        # cbar.ax.axhline at value 1.0 in log space
        cbar.ax.plot([0, 1], [1.0, 1.0], color="red", lw=1.2, transform=cbar.ax.get_yaxis_transform())
    except Exception:
        pass

    fig.tight_layout()
    fig_path = out / "figures" / "heatmap_sustained_lowkl_normalized.png"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[main] saved {fig_path}")

    # subset stats
    sub_info = info_idx.loc[row_ids].reset_index()
    sub_trig = sub_info[sub_info["is_trunc"]]
    n_max_len = int((sub_info["completion_len"] >= stats.get("max_completion_length", 2048)).sum())
    if len(sub_trig) > 0:
        mean_tau_norm = float((sub_trig["trigger_pos"] / sub_trig["completion_len"]).mean())
        median_tau_norm = float((sub_trig["trigger_pos"] / sub_trig["completion_len"]).median())
        mean_post_frac = float(((sub_trig["completion_len"] - sub_trig["trigger_pos"])
                                / sub_trig["completion_len"]).mean())
    else:
        mean_tau_norm = median_tau_norm = mean_post_frac = float("nan")
    subset_stats = {
        "num_rollouts_subset": int(R),
        "triggered_count_subset": int(n_trig_rows),
        "trigger_rate_subset": n_trig_rows / max(1, R),
        "mean_tau_norm": mean_tau_norm,
        "median_tau_norm": median_tau_norm,
        "mean_post_trigger_fraction": mean_post_frac,
        "fraction_reaching_max_length_subset": n_max_len / max(1, R),
        "mean_rollout_length_subset": float(sub_info["completion_len"].mean()),
        "population_trigger_rate": stats["trigger_rate"],
        "population_num_rollouts": stats["num_rollouts"],
        "n_bins": args.n_bins, "W": args.W, "L0": args.L0,
        "threshold": threshold,
    }
    with open(out / "stats_subset.json", "w") as f:
        json.dump(subset_stats, f, indent=2)
    print("[main] subset stats:")
    for k, v in subset_stats.items():
        print(f"  {k} = {v}")

    with open(out / "subset_ids.json", "w") as f:
        json.dump({
            "triggered": [int(x) for x in trig_meta.index.tolist()],
            "non_triggered": [int(x) for x in notrig_meta.index.tolist()],
            "row_order": [int(x) for x in row_ids],
        }, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
