"""End-to-end evaluation driver: merge FSDP -> vLLM rollout -> rule grading -> wandb.

Designed for 4 GPUs. Tasks/N:
  AIME24 / AIME25  -> N = 32  (avg + pass@32)
  AMC23  / Minerva -> N = 8   (avg + pass@8)
  MATH-500         -> N = 2   (avg + pass@2)

Each checkpoint goes through the pipeline sequentially. Within a checkpoint the
4 GPUs are saturated by spawning 4 worker processes (TP=1 each). Each worker
loads the merged HF model once and iterates over all 5 tasks, so we pay model
load cost only once per checkpoint instead of once per task.

Resumption: rollout JSONLs and merged HF weights are persistent. A run is
restartable - already-merged checkpoints and already-generated tasks are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import multiprocessing
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets" / "test_data"
EVAL_DIR = Path(__file__).resolve().parents[2] / "OPD" / "scripts" / "val" / "eval"
DEFAULT_OUT_ROOT = REPO_ROOT / "val" / "eval_outputs"

sys.path.insert(0, str(EVAL_DIR))

# ---------------- Eval config ---------------- #
# TASKS = [
#     {"name": "AIME24",   "path": DATA_DIR / "AIME24/test.parquet",   "N": 32, "max_tokens": 8192},
#     {"name": "AIME25",   "path": DATA_DIR / "AIME25/test.parquet",   "N": 32, "max_tokens": 8192},
#     {"name": "AMC23",    "path": DATA_DIR / "AMC23/test.parquet",    "N": 8,  "max_tokens": 4096},
#     {"name": "MATH-500", "path": DATA_DIR / "MATH-500/test.parquet", "N": 8,  "max_tokens": 4096},
#     {"name": "Minerva",  "path": DATA_DIR / "Minerva/test.parquet",  "N": 8,  "max_tokens": 4096},
# ]
TASKS = [
    {"name": "GPQA",           "path": DATA_DIR / "GPQA/test.parquet",           "N": 4, "max_tokens": 4096},
    {"name": "Olympiad-Bench", "path": DATA_DIR / "Olympiad-Bench/test.parquet", "N": 2, "max_tokens": 8192},
]

PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
TEMPERATURE = 0.7
TOP_P = 0.95
ENABLE_THINKING = False  # OPD students here are non-thinking base models.


# ---------------- Helpers ---------------- #
def task_out_path(out_dir: Path, task_name: str, n: int, max_tokens: int) -> Path:
    return out_dir / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{n}-MNT{max_tokens}.jsonl"


def load_samples(parquet_path: Path) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    return [
        {
            "example_id": i,
            "prompt": df.at[i, "prompt"][0]["content"].strip(),
            "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
        }
        for i in range(len(df))
    ]


def split_rollout_ids(rollout_ids: list[int], num_workers: int) -> list[list[int]]:
    chunks: list[list[int]] = [[] for _ in range(num_workers)]
    for i, rid in enumerate(rollout_ids):
        chunks[i % num_workers].append(rid)
    return chunks


def is_already_merged(hf_dir: Path) -> bool:
    if not hf_dir.exists():
        return False
    for f in hf_dir.iterdir():
        if f.suffix == ".safetensors":
            return True
        if f.name == "pytorch_model.bin":
            return True
        if f.name == "model.safetensors.index.json":
            return True
    return False


def merge_fsdp_to_hf(actor_dir: Path) -> Path:
    """Run `python -m verl.model_merger merge --backend fsdp` if not already merged.

    target_dir == actor/huggingface (already populated with config + tokenizer).
    """
    hf_dir = actor_dir / "huggingface"
    if is_already_merged(hf_dir):
        print(f"  [merge] already merged: {hf_dir}")
        return hf_dir

    hf_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(actor_dir),
        "--target_dir", str(hf_dir),
    ]
    print(f"  [merge] {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU merge to keep GPUs free
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    if not is_already_merged(hf_dir):
        raise RuntimeError(f"Merge did not produce weights in {hf_dir}")
    return hf_dir


def derive_run_name(ckpt_path: Path) -> str:
    """e.g. .../thu_prefix/resp256_..._2026-04-30_12-47-10/global_step_278
    -> resp256_..._2026-04-30_12-47-10__global_step_278
    """
    parts = ckpt_path.parts
    step = parts[-1]               # global_step_NNN
    run_dir = parts[-2]            # the long experiment directory
    return f"{run_dir}__{step}"


# ---------------- Worker (one per GPU, persistent across all 5 tasks) ---------------- #
def worker_process(args_tuple):
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

        for job in task_jobs:
            name = job["name"]
            samples = job["samples"]
            rollout_ids = job["rollout_ids"]
            max_tokens = job["max_tokens"]
            if not rollout_ids:
                out_per_task[name] = []
                continue

            sampling = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
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
            for rid in rollout_ids:
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
            print(f"[GPU {gpu_id}] task {name} done ({len(results)} rows)", flush=True)
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


def run_generation(model_path: Path, out_dir: Path, gpu_workers: list[str], enable_thinking: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for task in TASKS:
        path = task_out_path(out_dir, task["name"], task["N"], task["max_tokens"])
        if path.exists():
            print(f"  [gen] skip {task['name']} -> {path.name} exists")
            continue
        samples = load_samples(task["path"])
        for s in samples:
            s["prompt"] = PROMPT_TEMPLATE.format(problem=s["prompt"])
        pending.append({"task": task, "samples": samples, "out_path": path})

    if not pending:
        print("  [gen] all tasks already complete.")
        return

    num_workers = len(gpu_workers)
    worker_jobs: list[list[dict]] = [[] for _ in range(num_workers)]
    for p in pending:
        chunks = split_rollout_ids(list(range(p["task"]["N"])), num_workers)
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


# ---------------- Grading ---------------- #
def run_grading(out_dir: Path) -> list[dict]:
    """Rule-based grading via grade.py. Returns the list of per-task results."""
    cwd = os.getcwd()
    try:
        os.chdir(str(EVAL_DIR))
        from grade import grade_file  # type: ignore  noqa: E402
    finally:
        os.chdir(cwd)

    results = []
    for fp in sorted(out_dir.glob("*.jsonl")):
        # grade.py uses relative paths internally; run from eval/ dir.
        cwd = os.getcwd()
        try:
            os.chdir(str(EVAL_DIR))
            r = grade_file(Path(fp), use_model_verifier=False)
        finally:
            os.chdir(cwd)
        if r:
            results.append(r)
    grading_path = out_dir / "grading_results.json"
    with grading_path.open("w") as f:
        json.dump(results, f, indent=4)
    print(f"  [grade] -> {grading_path}")
    return results


# ---------------- WandB ---------------- #
def wandb_log_metrics(project: str, run_name: str, ckpt_path: Path,
                      results: list[dict], out_dir: Path):
    import wandb

    config = {
        "checkpoint": str(ckpt_path),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens_per_task": {t["name"]: t["max_tokens"] for t in TASKS},
        "enable_thinking": ENABLE_THINKING,
    }
    summary: dict[str, float] = {}
    table_rows = []
    for r in results:
        task = r["hyperparameters"].get("task_name", "?")
        n = int(r["hyperparameters"].get("n", 0))
        avg = r.get("mean_score", 0.0)
        passk = r.get("best_score", 0.0)
        summary[f"{task}/avg"] = avg
        summary[f"{task}/pass@{n}"] = passk
        summary[f"{task}/avg_output_length"] = r.get("avg_output_length", 0.0)
        summary[f"{task}/format_error_rollouts"] = r.get("format_error_rollouts", 0)
        summary[f"{task}/distinct_4gram"] = r.get("distinct_4gram", 0.0)
        table_rows.append([task, n, avg, passk,
                           r.get("avg_output_length", 0.0),
                           r.get("format_error_rollouts", 0)])

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        reinit=True,
        dir=str(out_dir),
    )
    try:
        run.log(summary)
        for k, v in summary.items():
            run.summary[k] = v
        cols = ["task", "n", "avg", "pass@n", "avg_output_length", "format_errors"]
        run.log({"per_task": wandb.Table(columns=cols, data=table_rows)})
    finally:
        run.finish()


# ---------------- Driver ---------------- #
def read_checkpoints(args) -> list[Path]:
    if args.checkpoints_file:
        with open(args.checkpoints_file) as f:
            paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        paths = list(args.checkpoints)
    out = []
    for p in paths:
        pp = Path(p)
        if not pp.is_absolute():
            pp = REPO_ROOT / pp
        out.append(pp)
    return out


def process_one(ckpt_path: Path, args) -> None:
    print(f"\n{'='*70}\nCheckpoint: {ckpt_path}\n{'='*70}")
    actor_dir = ckpt_path / "actor"
    if not actor_dir.is_dir():
        raise FileNotFoundError(f"No actor/ under {ckpt_path}")

    hf_dir = merge_fsdp_to_hf(actor_dir)

    run_name = derive_run_name(ckpt_path)
    out_dir = (Path(args.output_root) / run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_workers = [g.strip() for g in args.gpus.split(",") if g.strip()]
    run_generation(hf_dir, out_dir, gpu_workers, ENABLE_THINKING)

    results = run_grading(out_dir)

    if not args.no_wandb:
        try:
            wandb_log_metrics(args.wandb_project, run_name, ckpt_path, results, out_dir)
        except Exception as e:
            print(f"  [wandb] WARN: {e}")
    print(f"  done: {run_name}")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoints-file", type=str)
    src.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--wandb-project", default="opd_eval")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    ckpts = read_checkpoints(args)
    print(f"Got {len(ckpts)} checkpoint(s) to evaluate. GPUs: {args.gpus}")

    failures: list[tuple[Path, str]] = []
    for i, ckpt in enumerate(ckpts, 1):
        print(f"\n[{i}/{len(ckpts)}] {ckpt}")
        try:
            process_one(ckpt, args)
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
