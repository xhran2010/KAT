#!/usr/bin/env bash
# Thin wrapper for analyze/analyze_stages.py.
# Usage:
#   CKPT=/path/to/checkpoint-X bash run_analyze_kl_trunc.sh
set -euo pipefail

CKPT="${CKPT:?must set CKPT=/path/to/checkpoint}"
TEACHER="${TEACHER:-Qwen/Qwen3-8B}"
PROMPT_PARQUET="${PROMPT_PARQUET:-data/dapo-math-17k/dapo_clean.parquet}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-256}"
OUT_DIR="${OUT_DIR:-${CKPT%/}/stage_analysis_step$(basename "${CKPT}" | sed 's/^checkpoint-//')}"
EXTRA_ARGS=("${@}")

cd "$(dirname "$0")"

python analyze/analyze_stages.py \
  --ckpt "${CKPT}" \
  --teacher "${TEACHER}" \
  --prompt_parquet "${PROMPT_PARQUET}" \
  --num_rollouts "${NUM_ROLLOUTS}" \
  --out_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
