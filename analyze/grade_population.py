"""Grade every rollout in a work dir's rollouts.pt against DAPO ground truth.
Writes population_grading.json with {n_correct, n_incorrect, items[]}.

Run:
  python analyze/grade_population.py \
    --out_dir analyze/work_heatmap         # ckpt-380
  python analyze/grade_population.py \
    --out_dir analyze/work_heatmap_ckpt60  # ckpt-60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prompt_parquet",
                   default="data/dapo-math-17k/dapo_clean.parquet")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    bundle = torch.load(out / "rollouts.pt", weights_only=False)
    n = len(bundle["completion_text"])
    df = pd.read_parquet(args.prompt_parquet)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from math500_eval import _is_correct

    items = []
    for rid in range(n):
        pi = int(bundle["prompt_idx"][rid])
        rm = df.iloc[pi]["reward_model"]
        gt = str(rm["ground_truth"]) if isinstance(rm, dict) else ""
        ok = bool(_is_correct(bundle["completion_text"][rid], gt))
        items.append({"rollout_id": rid, "prompt_idx": pi, "ground_truth": gt, "is_correct": ok})

    n_correct = sum(1 for x in items if x["is_correct"])
    print(f"[main] {out}: {n_correct}/{n} correct ({n_correct/max(1,n):.2%})")
    with open(out / "population_grading.json", "w") as f:
        json.dump({"n_correct": n_correct, "n_incorrect": n - n_correct, "items": items},
                  f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
