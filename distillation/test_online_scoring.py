"""GPU consistency test for online KAT scoring (single GPU).

Generates student rollouts with the chunked online rollout (vLLM + incremental KV-cache scoring) and
compares, per token, against a full-sequence forward of teacher and student on prompt+completion
(batch size 1, exactly as a training micro-batch) followed by the loss-path
`DistillationTrainer.generalized_jsd_loss`:
  * per-token KL           (online chunked vs. full forward + generalized_jsd_loss)
  * teacher support log-probs reused by the loss (gathered on the same support ids)
  * student top-k support agreement
  * truncation decision of `find_truncation` on both KL sequences (same threshold)

Usage (from the repo root):
  python -m distillation.test_online_scoring --student ... --teacher ... --parquet ... [--n 24]
"""

from __future__ import annotations

import argparse
import os
import time

# In-process vLLM engine (as in trl's colocate mode); a forked engine core cannot re-initialise CUDA.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import pandas as pd
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--configs", default="64:32:0,17:5:10", help="chunk:score_bs:max_concurrent,...")
    args = ap.parse_args()

    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM

    from distillation.distillation_trainer import DistillationTrainer, _jsd_divergence
    from distillation.kl_truncation import find_truncation
    from distillation.online_rollout import OnlineKATRollout

    torch.manual_seed(0)
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    df = pd.read_parquet(args.parquet)
    prompts = []
    for rec in df.head(args.n).to_dict("records"):
        pr = rec["prompt"]
        if isinstance(pr, str):
            msgs = [{"role": "user", "content": pr}]
        else:
            msgs = [dict(m) for m in list(pr) if dict(m).get("role") != "assistant"]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        prompts.append(list(ids))

    llm = LLM(model=args.student, gpu_memory_utilization=0.3, max_num_seqs=32, max_num_batched_tokens=4096, seed=0,
              logprobs_mode="processed_logprobs")
    acc = Accelerator(mixed_precision="bf16")
    kw = dict(dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    student = AutoModelForCausalLM.from_pretrained(args.student, use_cache=False, **kw).to(dev)
    student.gradient_checkpointing_enable()
    student.train()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, use_cache=True, **kw).to(dev)
    student = acc.prepare_model(student, evaluation_mode=True)  # same autocast + fp32-output wrapper as training
    teacher = acc.prepare_model(teacher, evaluation_mode=True)

    L, t, N = 64, 32, 4
    sampling = dict(n=1, temperature=1.0, top_p=0.95, top_k=0, min_p=0.0, repetition_penalty=1.0, logprobs=None)

    for cfg in args.configs.split(","):
        chunk, score_bs, max_conc = (int(x) for x in cfg.split(":"))
        ro = OnlineKATRollout(
            llm=llm, sampling_kwargs=sampling, teacher=teacher, student=student, pad_token_id=tok.pad_token_id,
            max_completion_length=args.max_completion_length, chunk=chunk, score_bs=score_bs,
            max_concurrent=max_conc or 10**9, temperature=1.0, top_k=args.top_k, beta=1.0, add_tail=True,
            L=L, t=t, N=N, jsd_fn=_jsd_divergence, device=dev,
        )
        # threshold=-inf: nothing truncates -> full rollouts, every token scored online
        thr = torch.tensor(float("-inf"), device=dev)
        t0 = time.perf_counter()
        res = ro.run(prompts, thr)
        dt = time.perf_counter() - t0
        assert student.training, "student train() mode not restored"

        max_abs_kl, max_abs_tlp, sum_abs_kl, n_tok, sup_agree, sup_tot = 0.0, 0.0, 0.0, 0, 0, 0
        max_rel_kl = 0.0
        full_kls = []
        with torch.no_grad():
            for i, p in enumerate(prompts):
                comp = res.completion_ids[i]
                if not comp:
                    continue
                # Same layout as a training micro-batch: batch 1, completion right-padded to max length.
                n_pad = args.max_completion_length - len(comp)
                ids = torch.tensor([p + comp + [tok.pad_token_id] * n_pad], device=dev)
                am = torch.tensor([[1] * (len(p) + len(comp)) + [0] * n_pad], device=dev)
                end = len(p) + len(comp) - 1
                student.eval()
                s_logits = student(input_ids=ids, attention_mask=am).logits[:, len(p) - 1 : end]
                student.train()
                t_logits = teacher(input_ids=ids, attention_mask=am).logits[:, len(p) - 1 : end]
                jsd = DistillationTrainer.generalized_jsd_loss(
                    None, s_logits, t_logits, labels=None, beta=1.0, temperature=1.0, reduction="none",
                    top_k=args.top_k, add_tail=True,
                )
                kl_full = jsd.sum(-1).float()[0]
                full_kls.append(kl_full)
                kl_on = res.kl[i]
                d = (kl_full - kl_on).abs()
                max_abs_kl = max(max_abs_kl, float(d.max()))
                max_rel_kl = max(max_rel_kl, float((d / kl_full.abs().clamp_min(1e-3)).max()))
                sum_abs_kl += float(d.sum())
                n_tok += d.numel()
                t_lp_full = F.log_softmax(t_logits, dim=-1)[0].gather(-1, res.support_ids[i])
                max_abs_tlp = max(max_abs_tlp, float((t_lp_full - res.teacher_support_logps[i].float()).abs().max()))
                sup_full = s_logits[0].topk(args.top_k, dim=-1).indices
                sup_agree += int((sup_full.sort(-1).values == res.support_ids[i].sort(-1).values).all(-1).sum())
                sup_tot += sup_full.shape[0]

        # truncation decisions on both KL signals, with a realistic threshold (30% quantile-like)
        lens = torch.tensor([k.numel() for k in full_kls], device=dev)
        S = int(lens.max())
        A = torch.zeros(len(full_kls), S, device=dev)
        Bk = torch.zeros(len(full_kls), S, device=dev)
        for i, (kf, ko) in enumerate(zip(full_kls, [k for k in res.kl if k.numel() > 0])):
            A[i, : kf.numel()] = kf
            Bk[i, : ko.numel()] = ko
        valid = (torch.arange(S, device=dev).unsqueeze(0) < lens.unsqueeze(1)).float()
        ra = find_truncation(A, valid, None, L=L, t=t, N=N)
        rmins = ra["row_min"][ra["row_has_kept"]]
        thr2 = torch.quantile(rmins, 0.7) if rmins.numel() > 0 else torch.tensor(0.1, device=dev)
        ra = find_truncation(A, valid, thr2, L=L, t=t, N=N)
        rb = find_truncation(Bk, valid, thr2, L=L, t=t, N=N)
        same = int(((ra["is_trunc"] == rb["is_trunc"]) & (~ra["is_trunc"] | (ra["mask_start"] == rb["mask_start"]))).sum())

        # online run with that threshold: rollouts must stop early
        res2 = ro.run(prompts, thr2)
        print(
            f"[online scoring] chunk={chunk} score_bs={score_bs} max_concurrent={max_conc or 'all'}: "
            f"{len(prompts)} rollouts, {n_tok} tokens, rollout+scoring {dt:.1f}s ({res.timing})\n"
            f"  per-token KL |online - full|: max={max_abs_kl:.3e} mean={sum_abs_kl / max(n_tok, 1):.3e} "
            f"max_rel={max_rel_kl:.3e}\n"
            f"  teacher support log-prob |cached - full|: max={max_abs_tlp:.3e}\n"
            f"  student top-{args.top_k} support identical (as sets) at {sup_agree}/{sup_tot} positions\n"
            f"  truncation decision (thr={float(thr2):.4f}) identical for {same}/{len(full_kls)} rollouts "
            f"(full-forward KL vs online KL)\n"
            f"  online run with thr: {int(res2.is_trunc.sum())}/{len(prompts)} truncated, generated "
            f"{int(res2.gen_len.sum())} tokens, kept {int(res2.kept_len.sum())} ({res2.timing})",
            flush=True,
        )


if __name__ == "__main__":
    main()
