#!/usr/bin/env bash
# Example launcher for on-policy distillation (Qwen3-8B -> Qwen3-0.6B-Base) on DAPO-Math-17k.
# Usage:
#   NUM_GPUS=2 KL_TRUNC_ENABLE=1 bash run_train.sh
#
# The student is trained pure on-policy (lmbda=1.0) with reverse KL (beta=1.0) and
# top-64 loss support. MATH-500 accuracy@1 is reported every --eval_steps with its own
# rollout length (--eval_max_completion_length), independent of training rollouts.
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"
EXP_NAME="${EXP_NAME:-qwen3_06b_distill}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./outputs}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-0.6B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
TRAIN_PARQUET="${TRAIN_PARQUET:-data/dapo-math-17k/dapo_clean.parquet}"
EVAL_PARQUET="${EVAL_PARQUET:-data/MATH-500/math500.parquet}"

# KAT rollout termination (set KL_TRUNC_ENABLE=1 to turn on)
KL_TRUNC_ENABLE="${KL_TRUNC_ENABLE:-0}"
KL_TRUNC_WINDOW_SIZE="${KL_TRUNC_WINDOW_SIZE:-64}"
KL_TRUNC_START_POS="${KL_TRUNC_START_POS:-32}"
KL_TRUNC_ETA="${KL_TRUNC_ETA:-0.3}"
KL_TRUNC_N_CONSEC="${KL_TRUNC_N_CONSEC:-4}"
KL_TRUNC_WARMUP_STEPS="${KL_TRUNC_WARMUP_STEPS:-50}"
KL_TRUNC_BUFFER_SIZE="${KL_TRUNC_BUFFER_SIZE:-4096}"
KL_TRUNC_MIN_SAMPLES="${KL_TRUNC_MIN_SAMPLES:-512}"
# offline (default: full rollout + loss mask) | online (early stopping during rollout)
KL_TRUNC_MODE="${KL_TRUNC_MODE:-offline}"
KL_TRUNC_ONLINE_CHUNK="${KL_TRUNC_ONLINE_CHUNK:-64}"
KL_TRUNC_ONLINE_SCORE_BS="${KL_TRUNC_ONLINE_SCORE_BS:-32}"
KL_TRUNC_ONLINE_MAX_CONCURRENT="${KL_TRUNC_ONLINE_MAX_CONCURRENT:-0}"

KL_TRUNC_ARGS=()
if [[ "${KL_TRUNC_ENABLE}" == "1" || "${KL_TRUNC_ENABLE,,}" == "true" ]]; then
  KL_TRUNC_ARGS=(
    --kl_trunc_enable
    --kl_trunc_window_size "${KL_TRUNC_WINDOW_SIZE}"
    --kl_trunc_start_pos "${KL_TRUNC_START_POS}"
    --kl_trunc_eta "${KL_TRUNC_ETA}"
    --kl_trunc_n_consec "${KL_TRUNC_N_CONSEC}"
    --kl_trunc_warmup_steps "${KL_TRUNC_WARMUP_STEPS}"
    --kl_trunc_buffer_size "${KL_TRUNC_BUFFER_SIZE}"
    --kl_trunc_min_samples "${KL_TRUNC_MIN_SAMPLES}"
    --kl_trunc_mode "${KL_TRUNC_MODE}"
    --kl_trunc_online_chunk "${KL_TRUNC_ONLINE_CHUNK}"
    --kl_trunc_online_score_bs "${KL_TRUNC_ONLINE_SCORE_BS}"
    --kl_trunc_online_max_concurrent "${KL_TRUNC_ONLINE_MAX_CONCURRENT}"
  )
fi

# Random per-rollout response truncation (set RANDOM_TRUNC_ENABLE=1 to turn on)
RANDOM_TRUNC_ENABLE="${RANDOM_TRUNC_ENABLE:-0}"
RANDOM_TRUNC_MIN_KEEP="${RANDOM_TRUNC_MIN_KEEP:-1}"

RANDOM_TRUNC_ARGS=()
if [[ "${RANDOM_TRUNC_ENABLE}" == "1" || "${RANDOM_TRUNC_ENABLE,,}" == "true" ]]; then
  RANDOM_TRUNC_ARGS=(
    --random_trunc_enable
    --random_trunc_min_keep "${RANDOM_TRUNC_MIN_KEEP}"
  )
fi

# Resume from checkpoint:
#   RESUME_FROM_CHECKPOINT=auto                              -> auto-pick latest in outputs/$EXP_NAME
#   RESUME_FROM_CHECKPOINT=/abs/path/to/checkpoint-1000      -> resume from that exact path
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

cd "$(dirname "$0")"

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --gpu_ids all \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  distill.py \
    --student_model "${STUDENT_MODEL}" \
    --teacher_model "${TEACHER_MODEL}" \
    --train_parquet "${TRAIN_PARQUET}" \
    --eval_parquet "${EVAL_PARQUET}" \
    --exp_name "${EXP_NAME}" \
    --base_output_dir "${BASE_OUTPUT_DIR}" \
    --report_to wandb \
    --lmbda 1.0 \
    --beta 1.0 \
    --temperature 1.0 \
    --top_p 0.95 \
    --loss_top_k 64 \
    --max_prompt_length 1024 \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --eval_max_completion_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --eval_steps 20 \
    --eval_batch_size 8 \
    --save_steps 20 \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.3 \
    --bf16 \
    --gradient_checkpointing \
    "${KL_TRUNC_ARGS[@]}" \
    "${RANDOM_TRUNC_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    ${EXTRA_ARGS} \
    "$@"
