# Escaping the KL Agreement Trap in On-Policy Distillation

Official implementation of **KAT (KL Agreement Trap Termination)** from our EMNLP 2026 paper:

> **Escaping the KL Agreement Trap in On-Policy Distillation**  
> Haoran Xin, Anhao Zhao, Ying Sun, Jin Li, Xiaoyu Shen, Hui Xiong  
> *Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026)*  
> [[Paper (ACL Anthology)](https://aclanthology.org/2026.emnlp-main.1817.pdf)]

## Overview

On-policy distillation (OPD) provides dense token-level supervision by having a teacher model score student-generated rollouts. However, when a student drifts into an unrecoverable prefix, the teacher can locally agree with the degraded continuation—yielding low reverse KL divergence while offering little to no corrective signal. We identify this persistent failure mode as the **low-KL agreement trap**.

**KAT (KL Agreement Trap Termination)** mitigates this issue by monitoring a sliding window of per-token reverse KL divergence against a dynamic, training-adaptive quantile threshold maintained over a FIFO buffer. Once $N$ consecutive windows fall below the threshold, KAT terminates the degenerate suffix, filtering out uninformative agreement and improving reasoning accuracy while reducing effective rollout length.

## Repository Structure

```text
kat-opd/
├── distill.py                 # Main entry point for on-policy distillation (OPD & KAT-OPD)
├── math500_eval.py            # Periodic MATH-500 evaluation callback during training
├── run_train.sh               # Launcher script (e.g., Qwen3-8B -> Qwen3-0.6B-Base / 1.7B-Base)
├── run_train_4b.sh            # Multi-GPU DeepSpeed ZeRO-2 launcher (e.g., Qwen3-8B -> Qwen3-4B-Base)
├── configs/
│   ├── zero2.yaml             # Accelerate + DeepSpeed ZeRO-2 config
│   └── zero2_offload.yaml     # Accelerate + DeepSpeed ZeRO-2 (CPU optimizer offload) config
├── distillation/
│   ├── distillation_config.py # DistillationConfig
│   ├── distillation_trainer.py# DistillationTrainer (extended from TRL)
│   └── kl_truncation.py       # KAT sliding-window KL termination manager
├── data/                      # Training prompts and evaluation benchmarks
├── val/                       # Multi-benchmark evaluation pipeline (AIME, AMC, MATH-500, Minerva, etc.)
└── analyze/                   # Analysis scripts (stage analysis, teacher recovery, gradient direction, heatmaps)
```

## Installation

We recommend Python 3.10+ with CUDA and PyTorch 2.x.

```bash
pip install torch transformers trl accelerate deepspeed vllm datasets pandas tqdm wandb
pip install flash-attn --no-build-isolation
pip install math-verify
```

## Data

The `data/` directory contains the training prompts and evaluation benchmarks used in our experiments:

- **Training prompts**: `data/dapo-math-17k/dapo_clean.parquet` (DAPO-Math-17k), `data/OpenThoughts3/`, `data/competition_math/`, `data/gsm8k/`
- **Evaluation benchmarks**: `data/MATH-500/math500.parquet`, `data/AIME_2024/`, `data/aime25/`, `data/amc23/`, `data/minervamath/`

## Training

### 1. On-Policy Distillation with KAT

By default, `run_train.sh` distills `Qwen/Qwen3-8B` into `Qwen/Qwen3-0.6B-Base` on `data/dapo-math-17k/dapo_clean.parquet` using vLLM colocated rollouts and periodic MATH-500 validation:

```bash
NUM_GPUS=2 \
EXP_NAME=qwen3_06b_kat \
KL_TRUNC_ENABLE=1 \
bash run_train.sh
```

To run standard on-policy distillation (without KAT):

```bash
NUM_GPUS=2 \
EXP_NAME=qwen3_06b_opd \
KL_TRUNC_ENABLE=0 \
bash run_train.sh
```

You can override model paths, dataset paths, or hyperparameters via environment variables or extra CLI flags:

```bash
NUM_GPUS=2 \
STUDENT_MODEL=/path/to/Qwen3-1.7B-Base \
TEACHER_MODEL=/path/to/Qwen3-8B \
EXP_NAME=qwen3_1.7b_kat \
KL_TRUNC_ENABLE=1 \
KL_TRUNC_ETA=0.3 \
bash run_train.sh --report_to none
```

### 2. Multi-GPU Training with DeepSpeed ZeRO-2 (e.g., 4B Student)

For larger student models such as `Qwen3-4B-Base`, use `run_train_4b.sh` with `configs/zero2.yaml`:

```bash
NUM_GPUS=4 \
EXP_NAME=qwen3_4b_kat \
KL_TRUNC_ENABLE=1 \
bash run_train_4b.sh
```

### 3. Key KAT Hyperparameters

| Argument | Env Var in `run_train*.sh` | Default | Description |
| :--- | :--- | :---: | :--- |
| `--kl_trunc_enable` | `KL_TRUNC_ENABLE` | `False` | Enable KAT rollout termination. |
| `--kl_trunc_window_size` | `KL_TRUNC_WINDOW_SIZE` | `64` | Sliding window width $L$ over response tokens. |
| `--kl_trunc_start_pos` | `KL_TRUNC_START_POS` | `16` (`32` in shell) | Offset $t$ skipped at the beginning of each response. |
| `--kl_trunc_eta` | `KL_TRUNC_ETA` | `0.3` | Quantile parameter $\eta \in (0, 1)$; threshold is the $(1-\eta)$-quantile of the buffer. |
| `--kl_trunc_n_consec` | `KL_TRUNC_N_CONSEC` | `4` | Consecutive low-KL windows $N$ required to trigger termination. |
| `--kl_trunc_warmup_steps` | `KL_TRUNC_WARMUP_STEPS` | `50` | Initial steps $K$ used only to populate the FIFO buffer. |
| `--kl_trunc_buffer_size` | `KL_TRUNC_BUFFER_SIZE` | `4096` | FIFO buffer capacity $M$. |
| `--kl_trunc_min_samples` | `KL_TRUNC_MIN_SAMPLES` | `512` | Minimum buffered rollouts before termination activates. |

## Evaluation

During training, `math500_eval.py` evaluates MATH-500 accuracy@1 every `--eval_steps` steps.

For post-training multi-benchmark evaluation across saved HuggingFace checkpoints, list checkpoint directories in `val/eval_ckpts.txt` (one path per line) and run:

```bash
GPUS=0,1 bash val/run_eval.sh
```

## Analysis

Scripts under `analyze/` reproduce the empirical analyses in the paper (stage breakdown, low-KL heatmaps, gradient direction alignment, and teacher recoverability).

## Citation

If you find this repository helpful in your research, please cite our paper:

```bibtex
@inproceedings{xin-etal-2026-escaping,
    title = "Escaping the {KL} Agreement Trap in On-Policy Distillation",
    author = "Xin, Haoran and Zhao, Anhao and Sun, Ying and Li, Jin and Shen, Xiaoyu and Xiong, Hui",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    url = "https://aclanthology.org/2026.emnlp-main.1817/",
    doi = "10.18653/2026.emnlp-main.1817",
    pages = "41418--41431",
}
```

## Acknowledgements

This codebase builds upon [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm), and [verl](https://github.com/volcengine/verl).
