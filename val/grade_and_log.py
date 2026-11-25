"""Grade existing eval_outputs JSONLs and push to wandb.

CPU-only recovery script for the case where run_eval_trl.py finished
generation but grading crashed (e.g. missing pylatexenc). Reuses OPD's
run_grading + wandb_log_metrics so the output is identical to a
full-pipeline run.

Usage:
    python val/grade_and_log.py \
        --eval-dir val/eval_outputs/prefix_512__checkpoint-340 \
        --ckpt outputs/prefix_512/checkpoint-340 \
        --wandb-project trl_gkd_eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VAL_DIR = Path(__file__).resolve().parent
if str(VAL_DIR) not in sys.path:
    sys.path.insert(0, str(VAL_DIR))

import run_eval as opd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True,
                   help="Path to eval_outputs/<run_name>/ containing the JSONLs.")
    p.add_argument("--ckpt", required=True,
                   help="Original checkpoint path (recorded into wandb config).")
    p.add_argument("--run-name", default=None,
                   help="Wandb run name. Defaults to eval-dir basename.")
    p.add_argument("--wandb-project", default="trl_gkd_eval")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = (REPO_ROOT / eval_dir).resolve()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = (REPO_ROOT / ckpt).resolve()
    run_name = args.run_name or eval_dir.name

    if not eval_dir.is_dir():
        raise SystemExit(f"eval-dir not found: {eval_dir}")
    jsonls = sorted(eval_dir.glob("*.jsonl"))
    if not jsonls:
        raise SystemExit(f"No JSONL files under {eval_dir}")
    print(f"Grading {len(jsonls)} JSONLs in {eval_dir}")

    results = opd.run_grading(eval_dir)
    for r in results:
        task = r["hyperparameters"].get("task_name", "?")
        n = r["hyperparameters"].get("n", "?")
        print(f"  [grade] {task}: avg={r.get('mean_score', 0):.4f} "
              f"pass@{n}={r.get('best_score', 0):.4f}")

    if not args.no_wandb:
        opd.wandb_log_metrics(args.wandb_project, run_name, ckpt, results, eval_dir)
        print(f"[wandb] pushed run '{run_name}' to project '{args.wandb_project}'")


if __name__ == "__main__":
    main()
