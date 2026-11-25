"""Thin wrapper around OPD's scripts/val/run_eval.py for trl_gkd HF checkpoints.

trl_gkd checkpoints are already HF-format (model.safetensors at the top of
checkpoint-N/), so we skip OPD's FSDP -> HF merge step and feed the checkpoint
directory directly into the vLLM rollout + grading pipeline.

Also adds explicit per-benchmark progress logs (start + finish) on top of
OPD's defaults, so the log stream clearly says which task is running.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

import run_eval as opd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "val" / "eval_outputs"


def derive_run_name(ckpt_path: Path) -> str:
    # /.../outputs/prefix_512/checkpoint-340 -> prefix_512__checkpoint-340
    return f"{ckpt_path.parent.name}__{ckpt_path.name}"


# ---------------- Worker with per-task progress logs ---------------- #
def worker_process(args_tuple):
    """Same as opd.worker_process but prints a clear start-of-task line
    so the log shows which benchmark is running on each GPU."""
    model_path, task_jobs, gpu_id, enable_thinking, show_tqdm = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    import gc as _gc
    import torch
    from vllm import LLM, SamplingParams

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError:
        destroy_distributed_environment = None
        destroy_model_parallel = None

    out_per_task: dict[str, list[dict]] = {}
    llm = None
    try:
        print(f"[GPU {gpu_id}] loading {model_path}", flush=True)
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
        )
        tokenizer = llm.get_tokenizer()

        stop_token_ids = []
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            try:
                enc = tokenizer.encode(tok, add_special_tokens=False)
                if enc:
                    stop_token_ids.append(enc[0])
            except Exception:
                pass

        active_jobs = [j for j in task_jobs if j["rollout_ids"]]
        total_active = len(active_jobs)
        active_idx = 0
        for job in task_jobs:
            name = job["name"]
            samples = job["samples"]
            rollout_ids = job["rollout_ids"]
            max_tokens = job["max_tokens"]
            if not rollout_ids:
                out_per_task[name] = []
                continue

            active_idx += 1
            n_prompts = len(samples)
            n_rollouts = len(rollout_ids) * n_prompts
            print(
                f"[GPU {gpu_id}] >>> START task {name} "
                f"({active_idx}/{total_active}) "
                f"[{len(rollout_ids)} seeds x {n_prompts} prompts = {n_rollouts} gens, "
                f"max_tokens={max_tokens}]",
                flush=True,
            )
            t0 = time.time()

            sampling = SamplingParams(
                temperature=opd.TEMPERATURE,
                top_p=opd.TOP_P,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids if stop_token_ids else None,
            )

            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for s in samples
            ]
            results = []
            for r_i, rid in enumerate(rollout_ids, 1):
                print(
                    f"[GPU {gpu_id}]     {name} seed {r_i}/{len(rollout_ids)} "
                    f"(rid={rid})",
                    flush=True,
                )
                outs = llm.generate(formatted, sampling, use_tqdm=show_tqdm)
                for s, o in zip(samples, outs):
                    results.append({
                        "example_id": s["example_id"],
                        "prompt": s["prompt"],
                        "answer": s["answer"],
                        "seed": rid,
                        "response": o.outputs[0].text,
                    })
            out_per_task[name] = results
            dt = time.time() - t0
            print(
                f"[GPU {gpu_id}] <<< DONE  task {name} "
                f"({active_idx}/{total_active}) "
                f"{len(results)} rows in {dt:.1f}s",
                flush=True,
            )
    except Exception as e:
        print(f"[GPU {gpu_id}] CRITICAL: {e}", flush=True)
        raise
    finally:
        print(f"[GPU {gpu_id}] cleanup", flush=True)
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try: destroy_model_parallel()
            except Exception: pass
        if destroy_distributed_environment is not None:
            try: destroy_distributed_environment()
            except Exception: pass
        _gc.collect()
        torch.cuda.empty_cache()
    return out_per_task


# ---------------- Generation orchestrator with task plan log ---------------- #
def run_generation(model_path: Path, out_dir: Path, gpu_workers: list[str], enable_thinking: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    cached = []
    for task in opd.TASKS:
        path = opd.task_out_path(out_dir, task["name"], task["N"], task["max_tokens"])
        if path.exists():
            cached.append(task["name"])
            print(f"  [gen] skip {task['name']} -> {path.name} exists")
            continue
        samples = opd.load_samples(task["path"])
        for s in samples:
            s["prompt"] = opd.PROMPT_TEMPLATE.format(problem=s["prompt"])
        pending.append({"task": task, "samples": samples, "out_path": path})

    print(
        f"  [gen] plan: {len(pending)} pending / {len(cached)} cached "
        f"(pending tasks: {[p['task']['name'] for p in pending]})",
        flush=True,
    )
    if not pending:
        print("  [gen] all tasks already complete.")
        return

    num_workers = len(gpu_workers)
    worker_jobs: list[list[dict]] = [[] for _ in range(num_workers)]
    for p in pending:
        chunks = opd.split_rollout_ids(list(range(p["task"]["N"])), num_workers)
        for w in range(num_workers):
            worker_jobs[w].append({
                "name": p["task"]["name"],
                "samples": p["samples"],
                "rollout_ids": chunks[w],
                "max_tokens": p["task"]["max_tokens"],
            })

    args_list = [
        (str(model_path), worker_jobs[w], gpu_workers[w], enable_thinking, w == 0)
        for w in range(num_workers)
    ]

    ctx = multiprocessing.get_context("spawn")
    worker_outputs: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futures = [ex.submit(worker_process, a) for a in args_list]
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="GPU workers"):
            worker_outputs.append(fut.result())

    for p in pending:
        name = p["task"]["name"]
        rows = []
        for wo in worker_outputs:
            rows.extend(wo.get(name, []))
        if not rows:
            print(f"  [gen] WARN: no rows produced for {name}")
            continue
        with p["out_path"].open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [gen] saved {len(rows)} rows -> {p['out_path']}")


# ---------------- Per-checkpoint driver ---------------- #
def process_one(ckpt_path: Path, args, ckpt_idx: int, ckpt_total: int) -> None:
    print(f"\n{'='*70}\n[ckpt {ckpt_idx}/{ckpt_total}] Checkpoint: {ckpt_path}\n{'='*70}")
    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {ckpt_path}")
    if not (ckpt_path / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json under {ckpt_path}")

    run_name = derive_run_name(ckpt_path)
    out_dir = (Path(args.output_root) / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_workers = [g.strip() for g in args.gpus.split(",") if g.strip()]
    run_generation(ckpt_path, out_dir, gpu_workers, opd.ENABLE_THINKING)

    print(f"  [grade] starting on {out_dir}", flush=True)
    results = opd.run_grading(out_dir)
    for r in results:
        task = r["hyperparameters"].get("task_name", "?")
        print(
            f"  [grade] {task}: avg={r.get('mean_score', 0):.4f} "
            f"pass@{r['hyperparameters'].get('n', '?')}={r.get('best_score', 0):.4f}",
            flush=True,
        )

    if not args.no_wandb:
        try:
            opd.wandb_log_metrics(args.wandb_project, run_name, ckpt_path, results, out_dir)
        except Exception as e:
            print(f"  [wandb] WARN: {e}")
    print(f"  [ckpt {ckpt_idx}/{ckpt_total}] done: {run_name}")


def read_checkpoints(args) -> list[Path]:
    if args.checkpoints_file:
        with open(args.checkpoints_file) as f:
            paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        paths = list(args.checkpoints)
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = REPO_ROOT / pp
        out.append(pp)
    return out


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoints-file", type=str)
    src.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--wandb-project", default="trl_gkd_eval")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    ckpts = read_checkpoints(args)
    print(f"Got {len(ckpts)} checkpoint(s) to evaluate. GPUs: {args.gpus}")

    failures: list[tuple[Path, str]] = []
    for i, ckpt in enumerate(ckpts, 1):
        print(f"\n[{i}/{len(ckpts)}] {ckpt}")
        try:
            process_one(ckpt, args, i, len(ckpts))
        except Exception as e:
            print(f"  FAILED on {ckpt}: {e}")
            failures.append((ckpt, str(e)))
            continue
        gc.collect()

    if failures:
        print("\nFailures:")
        for c, msg in failures:
            print(f"  - {c}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
