"""MATH-500 accuracy@1 evaluator + TrainerCallback for on-policy distillation."""

from __future__ import annotations

import gc
import json
import math
import os
import time as _ptime
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from transformers import TrainerCallback

from distillation.distillation_trainer import _phase_log


def _load_math500(parquet_path: str) -> list[dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    samples: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        prompt_field = row["problem"]
        if isinstance(prompt_field, (list, tuple)) or hasattr(prompt_field, "__iter__") and not isinstance(
            prompt_field, str
        ):
            messages = [dict(m) for m in list(prompt_field)]
        else:
            messages = [{"role": "user", "content": str(prompt_field)}]
        samples.append(
            {
                "messages": messages,
                "answer": str(row["answer"]),
            }
        )
    return samples


def _is_correct(prediction: str, ground_truth: str) -> bool:
    """Verify a model's prediction against ground truth using math_verify when available."""
    try:
        from math_verify import parse, verify

        gt_parsed = parse(f"${ground_truth}$") or parse(f"\\boxed{{{ground_truth}}}") or parse(ground_truth)
        pred_parsed = parse(prediction)
        if gt_parsed and pred_parsed:
            return bool(verify(gt_parsed, pred_parsed))
    except Exception:
        pass

    # Fallback: extract last \boxed{...} and compare as strings.
    def _extract_boxed(text: str) -> str | None:
        idx = text.rfind("\\boxed")
        if idx == -1:
            return None
        i = idx + len("\\boxed")
        while i < len(text) and text[i] == " ":
            i += 1
        if i >= len(text) or text[i] != "{":
            return None
        depth = 1
        i += 1
        start = i
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            return None
        return text[start : i - 1].strip()

    pred_boxed = _extract_boxed(prediction)
    return pred_boxed is not None and pred_boxed.strip() == ground_truth.strip()


@dataclass
class MATH500EvalArgs:
    parquet_path: str
    max_new_tokens: int = 2048
    batch_size: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False
    max_samples: int | None = None
    log_completions: int = 0  # how many predictions to print per eval
    use_vllm: bool = False  # use trainer's vllm engine when available


class MATH500Callback(TrainerCallback):
    """Run greedy MATH-500 inference with the current student weights and log accuracy@1.

    Evaluation uses ``model.generate`` on each rank's local model copy and gathers the
    per-sample correctness across ranks before reducing to a single scalar. The student's
    *training* rollout length and the *eval* rollout length are kept independent — this
    callback receives its own ``max_new_tokens``.
    """

    def __init__(self, eval_args: MATH500EvalArgs):
        self.eval_args = eval_args
        self._samples: list[dict[str, Any]] | None = None

    def _ensure_loaded(self) -> list[dict[str, Any]]:
        if self._samples is None:
            samples = _load_math500(self.eval_args.parquet_path)
            if self.eval_args.max_samples is not None:
                samples = samples[: self.eval_args.max_samples]
            self._samples = samples
        return self._samples

    def _render_prompts(self, tokenizer, samples: list[dict[str, Any]], indices: list[int]) -> list[str]:
        chat_kwargs: dict[str, Any] = {}
        if self.eval_args.enable_thinking is not None:
            chat_kwargs["enable_thinking"] = self.eval_args.enable_thinking
        rendered: list[str] = []
        for idx in indices:
            try:
                text = tokenizer.apply_chat_template(
                    samples[idx]["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_kwargs,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    samples[idx]["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            rendered.append(text)
        return rendered

    def _can_use_vllm(self, trainer) -> bool:
        vllm_gen = getattr(self, "_vllm_generation", None) or getattr(trainer, "vllm_generation", None)
        if vllm_gen is None or not self.eval_args.use_vllm:
            return False
        if getattr(vllm_gen, "mode", "colocate") != "colocate":
            # The server mode would need an HTTP path; only colocate is supported for eval here.
            return False
        return True

    @torch.no_grad()
    def _run_eval_vllm(self, trainer) -> dict[str, float]:
        """Generate eval completions through trainer.vllm_generation.llm with eval-specific sampling params."""
        from vllm import SamplingParams

        samples = self._ensure_loaded()
        tokenizer = trainer.processing_class
        accelerator = trainer.accelerator
        device = accelerator.device
        vllm_gen = getattr(self, "_vllm_generation", None) or trainer.vllm_generation

        # Ensure vLLM has fresh weights — collective, must be called on every rank.
        last_sync = getattr(trainer, "_last_vllm_sync_step", None)
        if last_sync is None or last_sync != trainer.state.global_step:
            vllm_gen.sync_weights()
            trainer._last_vllm_sync_step = trainer.state.global_step

        if vllm_gen.enable_sleep_mode:
            vllm_gen.llm.wake_up(tags=["kv_cache"])

        world_size = accelerator.num_processes
        rank = accelerator.process_index
        tp_size = max(1, int(getattr(vllm_gen, "tensor_parallel_size", 1)))
        dp_size = world_size // tp_size  # data-parallel groups
        dp_rank = rank // tp_size
        # Round-robin across DP groups (every rank in a TP group sees the same prompts).
        local_indices = list(range(dp_rank, len(samples), dp_size)) if dp_size > 0 else list(range(len(samples)))
        prompts_text = self._render_prompts(tokenizer, samples, local_indices)
        prompt_token_ids = [tokenizer.encode(t, add_special_tokens=False) for t in prompts_text]

        sampling = SamplingParams(
            n=1,
            temperature=self.eval_args.temperature,
            top_p=self.eval_args.top_p if self.eval_args.temperature > 0.0 else 1.0,
            max_tokens=self.eval_args.max_new_tokens,
        )

        # TP>1 inside colocate: every rank in the TP group must call generate with the same inputs.
        if tp_size > 1:
            import torch.distributed as _dist

            tp_group = getattr(vllm_gen, "tp_group", None)
            gathered = [None for _ in range(tp_size)]
            _dist.all_gather_object(gathered, prompt_token_ids, group=tp_group)
            tp_prompts = [p for sub in gathered for p in sub]
        else:
            tp_prompts = prompt_token_ids

        vllm_inputs = [{"prompt_token_ids": ids} for ids in tp_prompts]
        # Only rank 0 shows the bar — every other rank would print a duplicate to the same terminal.
        show_bar = accelerator.is_main_process
        outputs = vllm_gen.llm.generate(vllm_inputs, sampling_params=sampling, use_tqdm=show_bar)

        if tp_size > 1:
            # Each rank ran the full TP-group batch; keep only its own slice.
            local_in_group = rank % tp_size
            sl = slice(local_in_group * len(prompt_token_ids), (local_in_group + 1) * len(prompt_token_ids))
            outputs = outputs[sl]

        local_predictions: list[str] = []
        local_correct: list[int] = []
        for global_idx, out in zip(local_indices, outputs, strict=True):
            text = out.outputs[0].text
            ok = _is_correct(text, samples[global_idx]["answer"])
            local_predictions.append(text)
            local_correct.append(1 if ok else 0)

        if vllm_gen.enable_sleep_mode:
            vllm_gen.llm.sleep(level=2)

        return self._reduce_and_report(accelerator, device, local_indices, local_predictions, local_correct, samples)

    @torch.no_grad()
    def _run_eval(self, trainer) -> dict[str, float]:
        if self._can_use_vllm(trainer):
            return self._run_eval_vllm(trainer)
        samples = self._ensure_loaded()
        tokenizer = trainer.processing_class
        model = trainer.model
        accelerator = trainer.accelerator
        device = accelerator.device
        was_training = model.training
        model.eval()

        world_size = accelerator.num_processes
        rank = accelerator.process_index
        local_indices = list(range(rank, len(samples), world_size))
        prompts_text = self._render_prompts(tokenizer, samples, local_indices)

        local_predictions: list[str] = []
        local_correct: list[int] = []
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id

        do_sample = self.eval_args.temperature > 0.0
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=self.eval_args.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_id,
        )
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if do_sample:
            gen_kwargs["temperature"] = self.eval_args.temperature
            gen_kwargs["top_p"] = self.eval_args.top_p

        batch_size = max(1, self.eval_args.batch_size)
        for start in range(0, len(prompts_text), batch_size):
            chunk_prompts = prompts_text[start : start + batch_size]
            chunk_idx = local_indices[start : start + batch_size]
            old_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            try:
                tokenized = tokenizer(
                    chunk_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=False,
                ).to(device)
            finally:
                tokenizer.padding_side = old_side
            prompt_len = tokenized["input_ids"].shape[1]
            unwrapped = accelerator.unwrap_model(model)
            outputs = unwrapped.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                **gen_kwargs,
            )
            completion_ids = outputs[:, prompt_len:]
            decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            for global_idx, text in zip(chunk_idx, decoded, strict=True):
                gt = samples[global_idx]["answer"]
                ok = _is_correct(text, gt)
                local_predictions.append(text)
                local_correct.append(1 if ok else 0)

        if was_training:
            model.train()

        return self._reduce_and_report(accelerator, device, local_indices, local_predictions, local_correct, samples)

    def _reduce_and_report(
        self, accelerator, device, local_indices, local_predictions, local_correct, samples
    ) -> dict[str, float]:
        correct_tensor = torch.tensor([sum(local_correct), len(local_correct)], device=device, dtype=torch.long)
        if (
            dist.is_available()
            and dist.is_initialized()
            and accelerator.num_processes > 1
        ):
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = int(correct_tensor[0].item())
        total = int(correct_tensor[1].item())
        accuracy = total_correct / total if total > 0 else 0.0

        if self.eval_args.log_completions > 0 and accelerator.is_main_process and local_predictions:
            n_print = min(self.eval_args.log_completions, len(local_predictions))
            for j in range(n_print):
                global_idx = local_indices[j]
                gt = samples[global_idx]["answer"]
                pred = local_predictions[j]
                snippet = pred[:600].replace("\n", " ⏎ ")
                print(
                    f"[MATH-500] sample {global_idx}  gt={gt!r}  correct={bool(local_correct[j])}\n  pred: {snippet}",
                    flush=True,
                )

        self._dump_predictions(accelerator, local_indices, local_predictions, local_correct, samples)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "math500_accuracy@1": accuracy,
            "math500_total": total,
            "math500_correct": total_correct,
        }

    def _dump_predictions(self, accelerator, local_indices, local_predictions, local_correct, samples) -> None:
        """Gather per-rank predictions to rank 0 and write JSONL with prompt/gt/pred per sample."""
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return

        if (
            dist.is_available()
            and dist.is_initialized()
            and accelerator.num_processes > 1
        ):
            gathered_indices: list[Any] = [None] * accelerator.num_processes
            gathered_preds: list[Any] = [None] * accelerator.num_processes
            gathered_correct: list[Any] = [None] * accelerator.num_processes
            dist.all_gather_object(gathered_indices, list(local_indices))
            dist.all_gather_object(gathered_preds, list(local_predictions))
            dist.all_gather_object(gathered_correct, list(local_correct))
        else:
            gathered_indices = [list(local_indices)]
            gathered_preds = [list(local_predictions)]
            gathered_correct = [list(local_correct)]

        if not accelerator.is_main_process:
            return

        step = int(getattr(self, "_current_eval_step", -1))
        output_dir = getattr(trainer.args, "output_dir", None) or "."
        save_dir = os.path.join(output_dir, "math500_eval")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"step_{step}.jsonl")

        # Dedup by sample index (TP>1 in vLLM mode yields duplicates across TP peers).
        records: dict[int, dict[str, Any]] = {}
        for idxs, preds, corrs in zip(gathered_indices, gathered_preds, gathered_correct):
            if not idxs:
                continue
            for i, p, c in zip(idxs, preds, corrs):
                gi = int(i)
                records[gi] = {
                    "index": gi,
                    "prompt": samples[gi]["messages"],
                    "gt": samples[gi]["answer"],
                    "prediction": p,
                    "correct": bool(c),
                }

        with open(out_path, "w", encoding="utf-8") as f:
            for gi in sorted(records):
                f.write(json.dumps(records[gi], ensure_ascii=False) + "\n")
        print(f"[MATH-500] dumped {len(records)} predictions -> {out_path}", flush=True)

    def _maybe_eval(self, state):
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return
        backend = "vLLM" if self._can_use_vllm(trainer) else "model.generate"
        n_samples = len(self._ensure_loaded())
        _phase_log(
            state.global_step,
            "MATH-500 evaluation start",
            f"{n_samples} samples, backend={backend}, max_new_tokens={self.eval_args.max_new_tokens}",
        )
        self._eval_resume_pending = True
        self._current_eval_step = int(state.global_step)
        _eval_t0 = _ptime.time()
        metrics = self._run_eval(trainer)
        log_metrics = {f"eval_{k}": v for k, v in metrics.items()}
        # trainer.log() runs dist.all_reduce when model.training is True (DistillationTrainer
        # aggregates on/off-policy losses across ranks). Gating it to rank 0 deadlocks the
        # other ranks. Flip the student into eval() so log() takes its no-collective branch,
        # and invoke it on every rank.
        model = trainer.model
        was_training = model.training
        if was_training:
            model.eval()
        try:
            trainer.log(log_metrics)
        except Exception:
            if trainer.accelerator.is_main_process:
                print(f"[MATH-500 @ step {state.global_step}] {log_metrics}", flush=True)
        finally:
            if was_training:
                model.train()
        _phase_log(
            state.global_step,
            "MATH-500 evaluation done",
            f"acc@1={metrics.get('math500_accuracy@1', 0.0):.4f} "
            f"({metrics.get('math500_correct', 0)}/{metrics.get('math500_total', 0)}), "
            f"{_ptime.time() - _eval_t0:.2f}s",
        )

        # Mark this checkpoint as best when accuracy improves. HF Trainer's
        # _sorted_checkpoints reorders state.best_model_checkpoint to the end of the
        # rotation list, so save_total_limit will preserve it on top of the most-recent N.
        acc = metrics.get("math500_accuracy@1")
        if acc is not None:
            prev_best = getattr(state, "best_metric", None)
            if prev_best is None or float(acc) > float(prev_best):
                state.best_metric = float(acc)
                state.best_model_checkpoint = os.path.join(
                    trainer.args.output_dir,
                    f"checkpoint-{state.global_step}",
                )
                if trainer.accelerator.is_main_process:
                    _phase_log(
                        state.global_step,
                        "MATH-500 new best",
                        f"acc@1={float(acc):.4f} -> {state.best_model_checkpoint}",
                    )
        return metrics

    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
        # HF Trainer fires this when its built-in eval loop runs. We just piggy-back to log MATH-500.
        self._maybe_eval(state)
        return control

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        # Trigger our own eval cadence so we are not bound to the trainer's eval dataloader.
        if self.eval_steps is None or self.eval_steps <= 0:
            return control
        if state.global_step > 0 and state.global_step != self._last_eval_step and state.global_step % self.eval_steps == 0:
            self._last_eval_step = state.global_step
            self._maybe_eval(state)
        return control

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        if self.eval_at_start:
            self._last_eval_step = state.global_step
            self._maybe_eval(state)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if state.global_step != self._last_eval_step:
            self._last_eval_step = state.global_step
            self._maybe_eval(state)
        _phase_log(state.global_step, "Training run finished")
        return control

    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        _phase_log(state.global_step, "Saving checkpoint", ckpt_path)
        self._save_announced_step = state.global_step
        return control

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        # Announce resumption after an eval or save interrupted the train loop.
        if getattr(self, "_eval_resume_pending", False):
            _phase_log(state.global_step, "Resuming training after evaluation")
            self._eval_resume_pending = False
        if getattr(self, "_save_announced_step", None) is not None:
            _phase_log(state.global_step, "Resuming training after checkpoint save")
            self._save_announced_step = None
        return control

    def attach(self, trainer, eval_steps: int | None = None, eval_at_start: bool = False):
        """Bind the callback to a trainer and configure its own eval cadence."""
        self._trainer = trainer
        self._vllm_generation = getattr(trainer, "vllm_generation", None)
        self.eval_steps = eval_steps
        self.eval_at_start = eval_at_start
        self._last_eval_step = -1
        self._eval_resume_pending = False
        self._save_announced_step = None
