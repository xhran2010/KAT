#!/usr/bin/env bash
# Evaluate trl_gkd HF checkpoints (already HF-format, no FSDP merge) on
# AIME24/25 (N=32), AMC23/Minerva (N=8), MATH-500 (N=2), and push to wandb.
# Reuses OPD's run_eval pipeline via val/run_eval_trl.py.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

GPUS="${GPUS:-0,1}"
WANDB_PROJECT="${WANDB_PROJECT:-trl_gkd_eval}"
CKPT_FILE="${CKPT_FILE:-val/eval_ckpts.txt}"

export WANDB_PROJECT
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p logs/eval
LOG="logs/eval/run_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

python val/run_eval_trl.py \
    --checkpoints-file "$CKPT_FILE" \
    --gpus "$GPUS" \
    --wandb-project "$WANDB_PROJECT" \
    2>&1 | tee "$LOG"
