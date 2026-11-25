"""Pure on-policy distillation: Qwen3-8B → Qwen3-0.6B-Base on dapo-math-17k.

Built on top of ``trl.experimental.distillation.DistillationTrainer``. The teacher runs
locally (no teacher server), the student rolls out completions either with
``model.generate`` or vLLM (colocate), and MATH-500 accuracy@1 is reported on a
configurable schedule with its own independent rollout length.

Launch with ``accelerate`` / ``torchrun`` to set the number of GPUs.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
import torch  # noqa: F401  (kept so accelerate sees torch imported before its launcher hooks)
from datasets import Dataset
from transformers import AutoTokenizer

from math500_eval import MATH500Callback, MATH500EvalArgs


DEFAULT_TEACHER = "Qwen/Qwen3-8B"
DEFAULT_STUDENT = "Qwen/Qwen3-0.6B-Base"
DEFAULT_TRAIN_PARQUET = "data/dapo-math-17k/dapo_clean.parquet"
DEFAULT_EVAL_PARQUET = "data/MATH-500/math500.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Models
    p.add_argument("--student_model", default=DEFAULT_STUDENT)
    p.add_argument("--teacher_model", default=DEFAULT_TEACHER)

    # Datasets
    p.add_argument("--train_parquet", default=DEFAULT_TRAIN_PARQUET)
    p.add_argument("--eval_parquet", default=DEFAULT_EVAL_PARQUET)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    # Lengths
    p.add_argument("--max_length", type=int, default=4096, help="Total prompt+completion budget during training.")
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=2048, help="Train-time rollout max length.")
    p.add_argument("--eval_max_completion_length", type=int, default=4096, help="Validation-time rollout max length.")

    # Distillation / on-policy
    p.add_argument("--lmbda", type=float, default=1.0, help="On-policy fraction (1.0 = pure on-policy).")
    p.add_argument("--beta", type=float, default=1.0, help="0=forward KL, 1=reverse KL, in-between=JSD.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k_sampling", type=int, default=0, help="vLLM/HF generation top-k. 0 disables top-k.")
    p.add_argument("--loss_top_k", type=int, default=128)
    p.add_argument("--reverse_kl_top_1_mode", default="sampled", choices=["sampled", "argmax"])
    p.add_argument("--num_generations", type=int, default=1)

    # KL-trunc rollout truncation (sliding-window mask, ported from verl)
    p.add_argument("--kl_trunc_enable", action="store_true", default=False,
                   help="Enable sliding-window KL-based rollout truncation.")
    p.add_argument("--kl_trunc_window_size", type=int, default=64,
                   help="L: sliding window width over response tokens.")
    p.add_argument("--kl_trunc_start_pos", type=int, default=16,
                   help="t: skip first t tokens of response before opening windows.")
    p.add_argument("--kl_trunc_eta", type=float, default=0.3,
                   help="eta in (0,1): threshold = quantile(buffer, 1-eta).")
    p.add_argument("--kl_trunc_n_consec", type=int, default=4,
                   help="N: number of consecutive low windows that trigger truncation.")
    p.add_argument("--kl_trunc_warmup_steps", type=int, default=50,
                   help="K: initial steps that only fill the buffer, no truncation.")
    p.add_argument("--kl_trunc_buffer_size", type=int, default=4096,
                   help="M: FIFO buffer capacity.")
    p.add_argument("--kl_trunc_min_samples", type=int, default=512,
                   help="Buffer must hold this many samples before truncation activates.")
    p.add_argument("--kl_trunc_mode", default="offline", choices=["offline", "online"],
                   help="Termination mode: 'offline' (default) performs full rollout with post-hoc KL loss masking; "
                        "'online' scores incrementally during rollout and stops early at the truncation point.")
    p.add_argument("--kl_trunc_online_chunk", type=int, default=64,
                   help="Online mode: tokens generated per rollout between teacher-scoring rounds.")
    p.add_argument("--kl_trunc_online_score_bs", type=int, default=32,
                   help="Online mode: rollouts per teacher/student scoring sub-batch.")
    p.add_argument("--kl_trunc_online_max_concurrent", type=int, default=0,
                   help="Online mode: max rollouts in flight per rank (0 = all); lower to bound KV-cache memory.")

    # Random per-rollout response truncation (independent of kl_trunc)
    p.add_argument("--random_trunc_enable", action="store_true", default=False,
                   help="Enable per-rollout uniformly-random response truncation.")
    p.add_argument("--random_trunc_min_keep", type=int, default=1,
                   help="Minimum response tokens to keep per row (clamped to row length).")

    # Optim / schedule
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_train_batch_size", "--mini_batch_size", dest="per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", dest="bf16", action="store_false")
    p.add_argument("--seed", type=int, default=42)

    # vLLM (student rollouts)
    p.add_argument("--use_vllm", action="store_true", default=False)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--vllm_max_model_length", type=int, default=None)

    # Eval
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_temperature", type=float, default=0.0)
    p.add_argument("--eval_top_p", type=float, default=1.0)
    p.add_argument("--eval_at_start", action="store_true")
    p.add_argument("--eval_log_completions", type=int, default=2)
    p.add_argument(
        "--eval_use_vllm",
        action="store_true",
        default=None,
        help="Use trainer.vllm_generation for MATH-500 generation. Defaults to whatever --use_vllm is.",
    )
    p.add_argument("--no_eval_use_vllm", dest="eval_use_vllm", action="store_false")

    # Logging / output
    p.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name. Determines output_dir (base_output_dir/exp_name), wandb run_name, and wandb run id.",
    )
    p.add_argument("--base_output_dir", default="./outputs", help="Parent directory; final output_dir = base/exp_name.")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--run_name", default=None, help="Defaults to --exp_name.")
    p.add_argument("--report_to", default="wandb", choices=["none", "wandb", "trackio"])
    p.add_argument("--wandb_project", default="trl_gkd_distill", help="WANDB_PROJECT env var.")
    p.add_argument("--log_completions", action="store_true")
    p.add_argument("--log_completions_steps", type=int, default=100)

    # Misc
    p.add_argument("--num_gpus", type=int, default=None, help="Informational. Real GPU count comes from the launcher.")
    p.add_argument("--enable_thinking", action="store_true", default=False)

    # Resume
    p.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help=(
            "Resume training from a checkpoint. Either a path to a checkpoint directory "
            "(e.g. outputs/exp/checkpoint-200) or one of 'auto'/'latest'/'true' to auto-pick "
            "the latest checkpoint inside --base_output_dir/--exp_name. Omit for fresh training."
        ),
    )

    return p.parse_args()


def build_train_dataset(parquet_path: str, max_samples: int | None) -> Dataset:
    """Convert dapo-math-17k into a prompt-only ``messages`` dataset.

    Pure on-policy distillation only needs prompts — the student generates the completion
    and the teacher scores it. We strip ground-truth answers entirely.
    """
    df = pd.read_parquet(parquet_path)
    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        prompt = record["prompt"]
        if isinstance(prompt, (list, tuple)) or (hasattr(prompt, "__iter__") and not isinstance(prompt, str)):
            messages = [dict(m) for m in list(prompt)]
        else:
            messages = [{"role": "user", "content": str(prompt)}]
        # Drop assistant turns if any (force pure on-policy by ignoring off-policy completions).
        messages = [m for m in messages if m.get("role") != "assistant"]
        rows.append({"messages": messages})
    if max_samples is not None:
        rows = rows[:max_samples]
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    args.max_length = args.max_prompt_length + args.max_completion_length

    # Defer trl imports until after argparse so --help is fast.
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    from distillation import DistillationConfig, DistillationTrainer
    from distillation.distillation_trainer import _DistillationCollator

    # ── Resolve output_dir and wandb identity from exp_name ──
    args.output_dir = os.path.join(args.base_output_dir, args.exp_name)
    if args.run_name is None:
        args.run_name = args.exp_name

    # ── Resolve resume_from_checkpoint ──
    resume_arg: bool | str | None = None
    if args.resume_from_checkpoint is not None:
        token = str(args.resume_from_checkpoint).strip()
        if token.lower() in {"auto", "latest", "true", "1", "yes"}:
            # HF Trainer with resume_from_checkpoint=True scans output_dir for "checkpoint-*".
            # Validate eagerly so we fail loudly instead of silently starting fresh.
            ck_root = args.output_dir
            ck_dirs = []
            if os.path.isdir(ck_root):
                ck_dirs = sorted(
                    (d for d in os.listdir(ck_root) if d.startswith("checkpoint-")),
                    key=lambda d: int(d.split("-")[-1]) if d.split("-")[-1].isdigit() else -1,
                )
            if not ck_dirs:
                raise FileNotFoundError(
                    f"--resume_from_checkpoint=auto but no 'checkpoint-*' found in {ck_root}. "
                    f"Either point to a path explicitly or check EXP_NAME / base_output_dir."
                )
            resume_arg = True
            print(f"[resume] auto-detected latest checkpoint: {ck_root}/{ck_dirs[-1]}", flush=True)
        elif token.lower() in {"false", "0", "no", "none", ""}:
            resume_arg = None
        else:
            if not os.path.isdir(token):
                raise FileNotFoundError(f"--resume_from_checkpoint path does not exist: {token}")
            if not os.path.isfile(os.path.join(token, "trainer_state.json")):
                raise FileNotFoundError(
                    f"--resume_from_checkpoint path missing trainer_state.json: {token}"
                )
            resume_arg = token
            print(f"[resume] resuming from explicit path: {token}", flush=True)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_NAME", args.run_name)
        if resume_arg is not None:
            # Pin run_id to exp_name ONLY when resuming, so HF Trainer reattaches
            # to the previous wandb run instead of starting a fresh curve.
            os.environ.setdefault("WANDB_RUN_ID", args.exp_name)
            os.environ["WANDB_RESUME"] = "allow"
            print(
                f"[resume] WANDB_RESUME=allow, WANDB_RUN_ID={os.environ['WANDB_RUN_ID']}",
                flush=True,
            )

    # ── Build the train dataset ──
    train_dataset = build_train_dataset(args.train_parquet, args.max_train_samples)

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Custom collator that honours enable_thinking=False on Qwen3 ──
    enable_thinking = args.enable_thinking

    class _CollatorWithChatKwargs(_DistillationCollator):
        def __call__(self, examples):  # type: ignore[override]
            original_apply = self.tokenizer.apply_chat_template

            def _apply(messages, **kwargs):
                kwargs.setdefault("enable_thinking", enable_thinking)
                try:
                    return original_apply(messages, **kwargs)
                except TypeError:
                    kwargs.pop("enable_thinking", None)
                    return original_apply(messages, **kwargs)

            self.tokenizer.apply_chat_template = _apply  # type: ignore[assignment]
            try:
                return super().__call__(examples)
            finally:
                self.tokenizer.apply_chat_template = original_apply  # type: ignore[assignment]

    collator = _CollatorWithChatKwargs(
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    # ── DistillationConfig ──
    dtype_str = "bfloat16" if args.bf16 else "float32"
    model_init_kwargs = {
        "dtype": dtype_str,
        "use_cache": not args.gradient_checkpointing,
        "attn_implementation": "flash_attention_2",
        "device_map": None,  # DDP cannot run on device_map='auto' (trl default)
    }
    teacher_init_kwargs = {
        "dtype": dtype_str,
        "use_cache": True,
        "attn_implementation": "flash_attention_2",
        "device_map": None,
    }

    report_to = "none" if args.report_to in (None, "none", "") else [args.report_to]

    cfg = DistillationConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        # Model
        model_init_kwargs=model_init_kwargs,
        teacher_model_name_or_path=args.teacher_model,
        teacher_model_init_kwargs=teacher_init_kwargs,
        # Lengths
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # Distillation
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k_sampling,
        loss_top_k=args.loss_top_k,
        reverse_kl_top_1_mode=args.reverse_kl_top_1_mode,
        num_generations=args.num_generations,
        # KL-trunc
        kl_trunc_enable=args.kl_trunc_enable,
        kl_trunc_window_size=args.kl_trunc_window_size,
        kl_trunc_start_pos=args.kl_trunc_start_pos,
        kl_trunc_eta=args.kl_trunc_eta,
        kl_trunc_n_consec=args.kl_trunc_n_consec,
        kl_trunc_warmup_steps=args.kl_trunc_warmup_steps,
        kl_trunc_buffer_size=args.kl_trunc_buffer_size,
        kl_trunc_min_samples=args.kl_trunc_min_samples,
        kl_trunc_mode=args.kl_trunc_mode,
        kl_trunc_online_chunk=args.kl_trunc_online_chunk,
        kl_trunc_online_score_bs=args.kl_trunc_online_score_bs,
        kl_trunc_online_max_concurrent=args.kl_trunc_online_max_concurrent,
        # Random rollout truncation
        random_trunc_enable=args.random_trunc_enable,
        random_trunc_min_keep=args.random_trunc_min_keep,
        use_teacher_server=False,
        disable_dropout=True,
        # Optim
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        # vLLM
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_length=args.vllm_max_model_length,
        # Logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",  # we run our own MATH-500 eval via callback
        report_to=report_to,
        log_completions=args.log_completions,
        log_completions_steps=args.log_completions_steps,
    )

    # ── Trainer ──
    trainer = DistillationTrainer(
        model=args.student_model,
        teacher_model=args.teacher_model,
        args=cfg,
        data_collator=collator,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # ── MATH-500 callback (independent eval rollout length) ──
    eval_use_vllm = args.use_vllm if args.eval_use_vllm is None else args.eval_use_vllm
    eval_callback = MATH500Callback(
        MATH500EvalArgs(
            parquet_path=args.eval_parquet,
            max_new_tokens=args.eval_max_completion_length,
            batch_size=args.eval_batch_size,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            enable_thinking=args.enable_thinking,
            max_samples=args.max_eval_samples,
            log_completions=args.eval_log_completions,
            use_vllm=eval_use_vllm,
        )
    )
    eval_callback.attach(trainer, eval_steps=args.eval_steps, eval_at_start=args.eval_at_start)
    trainer.add_callback(eval_callback)

    # ── Train ──
    trainer.train(resume_from_checkpoint=resume_arg)

    # ── Save ──
    trainer.save_model(args.output_dir)
    if trainer.accelerator.is_main_process:
        try:
            tokenizer.save_pretrained(args.output_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
