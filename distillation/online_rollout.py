# Online KAT rollout: chunked student generation (vLLM colocate) + streaming teacher/student scoring.
#
# Every round the still-active rollouts are extended by `chunk` tokens with vLLM, the new tokens are
# scored incrementally by the teacher and by a no-grad student forward that both keep a KV cache,
# the per-token KL (same formula as the training loss: student top-K support + tail bucket) is
# appended, and `find_truncation` is evaluated on the accumulated KL. A rollout that triggers is cut
# at `mask_start` and stops generating; the rest continue until EOS / max length.
#
# The teacher log-probs on the support set (and teacher entropy) gathered while scoring are returned
# so the loss can reuse them instead of running a second teacher forward. That reuse is exact in
# distribution because the rollout and every micro-batch of the optimizer step use the same student
# weights (the buffer is filled once per step, before any update).
#
# Two documented differences from the loss-mask path:
#   * the threshold is fixed for the whole optimizer step (loss-mask re-reads it per micro-batch);
#   * the support set comes from the no-grad scoring forward of the student (same weights; only
#     bf16-level numeric differences from the training forward).

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .kl_truncation import find_truncation


def _add_tail_bucket(log_probs, valid_mask):
    # identical to distillation_trainer._add_tail_bucket (kept local to avoid a circular import)
    log_sum = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail = torch.log(-torch.expm1(log_sum))
    tail_mask = torch.ones_like(valid_mask[..., :1], dtype=torch.bool)
    return torch.cat([log_probs, tail], dim=-1), torch.cat([valid_mask, tail_mask], dim=-1)


def sparse_divergence_from_support(student_support_logps, teacher_support_logps, beta, add_tail, jsd_fn):
    """Per-token divergence on a fixed student-selected support (+ optional tail bucket).

    Mirrors the `top_k > 0, beta == 1` branch of `DistillationTrainer.generalized_jsd_loss` op for op,
    starting after the gather. Returns the (..., K[+1]) tensor *before* the support-dim sum.
    """
    neg_inf = torch.full((), float("-inf"), dtype=student_support_logps.dtype, device=student_support_logps.device)
    support_mask = torch.ones_like(student_support_logps, dtype=torch.bool)
    student_topk_logps = torch.where(support_mask, student_support_logps, neg_inf)
    teacher_topk_logps = torch.where(support_mask, teacher_support_logps, neg_inf)
    if add_tail:
        base_support_mask = support_mask
        student_log_probs, support_mask = _add_tail_bucket(student_topk_logps, base_support_mask)
        teacher_log_probs, _ = _add_tail_bucket(teacher_topk_logps, base_support_mask)
    else:
        student_log_probs = student_topk_logps - torch.logsumexp(student_topk_logps, dim=-1, keepdim=True)
        teacher_log_probs = teacher_topk_logps - torch.logsumexp(teacher_topk_logps, dim=-1, keepdim=True)
    return jsd_fn(student_log_probs, teacher_log_probs, beta, support_mask)


@torch.no_grad()
def score_logits(student_logits, teacher_logits, temperature, top_k, beta, add_tail, jsd_fn, want_entropy=True):
    """From (b, q, V) logits -> per-token KL (b, q) fp32, support ids (b, q, K),
    teacher support log-probs (b, q, K) and teacher entropy (b, q) fp32."""
    # x / 1.0 is bit-identical to x; skipping it avoids two extra (b, q, V) copies.
    s = student_logits / temperature if temperature != 1.0 else student_logits
    t = teacher_logits / temperature if temperature != 1.0 else teacher_logits
    s_lp = F.log_softmax(s, dim=-1)
    _, support = s.topk(top_k, dim=-1)
    del s
    s_sup = s_lp.gather(-1, support)
    del s_lp
    t_lp = F.log_softmax(t, dim=-1)
    del t
    t_sup = t_lp.gather(-1, support)
    kl = sparse_divergence_from_support(s_sup, t_sup, beta, add_tail, jsd_fn).sum(dim=-1).float()
    ent = None
    if want_entropy:
        # same as _record_distill_metrics: entropy of softmax(teacher_logits.float()) (no temperature)
        if temperature == 1.0 and t_lp.dtype == torch.float32:
            t_logp = t_lp
        else:
            del t_lp
            t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
        p = t_logp.exp()
        p.mul_(t_logp)
        del t_logp
        ent = -p.sum(dim=-1)
        del p
    return kl, support, t_sup, ent


@dataclass
class _Group:
    rows: list[int]                       # global sequence indices, in cache-row order
    attn: torch.Tensor                    # (b, prompt_width-1 + fed) attention mask
    pending: torch.Tensor                 # (b,) next token to feed (last prompt / last generated token)
    t_cache: Any = None
    s_cache: Any = None


@dataclass
class OnlineRolloutResult:
    completion_ids: list[list[int]]       # kept completion token ids per rollout
    support_ids: list[torch.Tensor]       # (kept, K) int64
    teacher_support_logps: list[torch.Tensor]  # (kept, K)
    teacher_entropy: list[torch.Tensor]   # (kept,) fp32
    kl: list[torch.Tensor]                # (kept,) fp32 per-token KL used for the decision
    gen_len: torch.Tensor                 # (B,) generated completion tokens
    kept_len: torch.Tensor                # (B,)
    is_trunc: torch.Tensor                # (B,) bool
    mask_start: torch.Tensor              # (B,)
    push_vals: torch.Tensor               # values to push into the FIFO buffer (one per rollout with a kept window)
    win_stats: tuple[float, float, float, float]
    timing: dict[str, float] = field(default_factory=dict)


class OnlineKATRollout:
    def __init__(
        self,
        *,
        llm,
        sampling_kwargs: dict,
        teacher,
        student,
        pad_token_id: int,
        max_completion_length: int,
        chunk: int,
        score_bs: int,
        max_concurrent: int,
        temperature: float,
        top_k: int,
        beta: float,
        add_tail: bool,
        L: int,
        t: int,
        N: int,
        jsd_fn,
        device,
    ):
        self.llm = llm
        self.sampling_kwargs = dict(sampling_kwargs)
        self.teacher = teacher
        self.student = student
        self.pad_token_id = int(pad_token_id)
        self.max_len = int(max_completion_length)
        self.chunk = int(chunk)
        self.score_bs = int(score_bs)
        self.max_concurrent = int(max_concurrent)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.beta = beta
        self.add_tail = add_tail
        self.L, self.t, self.N = int(L), int(t), int(N)
        self.jsd_fn = jsd_fn
        self.device = device

    # ── model helpers ──
    def _fwd(self, model, input_ids, attn, cache, logits_to_keep=0):
        # Positions count real tokens only (left padding does not shift RoPE), so every row sees the same
        # positions as in the unpadded batch-size-1 training forward.
        position_ids = (attn.cumsum(dim=-1) - 1).clamp_min(0)[:, -input_ids.shape[1]:]
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        return out.logits, out.past_key_values

    def _prefill(self, prompts: list[list[int]], rows: list[int]) -> _Group:
        ctx = [prompts[r][:-1] for r in rows]
        # +1: every row keeps at least one masked (pad) column. With an all-ones mask HF drops the mask and
        # FlashAttention takes its non-varlen kernel, whose q<kv numerics differ from the varlen kernel used
        # by the (right-padded) training micro-batch; the pad column keeps both on the same kernel.
        width = max(len(c) for c in ctx) + 1
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(rows), width), dtype=torch.long)
        for i, c in enumerate(ctx):
            if c:
                ids[i, width - len(c):] = torch.tensor(c, dtype=torch.long)
                attn[i, width - len(c):] = 1
        ids = ids.to(self.device)
        attn = attn.to(self.device)
        pending = torch.tensor([prompts[r][-1] for r in rows], dtype=torch.long, device=self.device)
        g = _Group(rows=list(rows), attn=attn, pending=pending)
        _, g.t_cache = self._fwd(self.teacher, ids, attn, None, logits_to_keep=1)
        _, g.s_cache = self._fwd(self.student, ids, attn, None, logits_to_keep=1)
        return g

    # ── main entry ──
    @torch.no_grad()
    def run(self, prompts: list[list[int]], threshold: torch.Tensor) -> OnlineRolloutResult:
        from vllm import SamplingParams

        dev = self.device
        B = len(prompts)
        K = self.top_k
        C = self.chunk
        T = self.max_len
        for p in prompts:
            if len(p) < 1:
                raise ValueError("online KAT rollout needs non-empty prompts")

        gen_tokens: list[list[int]] = [[] for _ in range(B)]
        KL = torch.zeros(B, T, dtype=torch.float32, device=dev)
        SUP = torch.zeros(B, T, K, dtype=torch.long, device=dev)
        TLP = None  # allocated lazily with the logits dtype
        TENT = torch.zeros(B, T, dtype=torch.float32, device=dev)
        scored = [0] * B
        done = [False] * B
        trunc_at = [-1] * B

        t_vllm = 0.0
        t_score = 0.0
        t_detect = 0.0
        n_rounds = 0
        n_vllm_calls = 0

        was_training_s = self.student.training
        was_training_t = self.teacher.training
        self.student.eval()
        self.teacher.eval()
        try:
            for w0 in range(0, B, self.max_concurrent):
                wave = list(range(w0, min(B, w0 + self.max_concurrent)))
                # prefill prompts (minus last token) into teacher/student caches, per scoring sub-batch
                _t0 = time.perf_counter()
                groups = [self._prefill(prompts, wave[i:i + self.score_bs]) for i in range(0, len(wave), self.score_bs)]
                torch.cuda.synchronize(dev)
                t_score += time.perf_counter() - _t0

                while groups:
                    n_rounds += 1
                    active = [r for g in groups for r in g.rows]
                    # 1) student continues every active rollout by up to C tokens
                    _t0 = time.perf_counter()
                    reqs, sps = [], []
                    for r in active:
                        remaining = T - len(gen_tokens[r])
                        reqs.append({"prompt_token_ids": prompts[r] + gen_tokens[r]})
                        sps.append(SamplingParams(**{**self.sampling_kwargs, "max_tokens": min(C, remaining)}))
                    outs = self.llm.generate(reqs, sampling_params=sps, use_tqdm=False)
                    n_vllm_calls += 1
                    finished = set()
                    new_tok: dict[int, list[int]] = {}
                    for r, o in zip(active, outs):
                        toks = list(o.outputs[0].token_ids)
                        new_tok[r] = toks
                        gen_tokens[r].extend(toks)
                        if o.outputs[0].finish_reason == "stop" or len(gen_tokens[r]) >= T or len(toks) == 0:
                            finished.add(r)
                    t_vllm += time.perf_counter() - _t0

                    # 2) incremental scoring of the new tokens
                    _t0 = time.perf_counter()
                    for g in groups:
                        q = max(len(new_tok[r]) for r in g.rows)
                        b = len(g.rows)
                        ids = torch.full((b, q), self.pad_token_id, dtype=torch.long)
                        qmask = torch.zeros((b, q), dtype=torch.long)
                        for i, r in enumerate(g.rows):
                            nt = new_tok[r]
                            if len(nt) > 1:
                                ids[i, 1:len(nt)] = torch.tensor(nt[:-1], dtype=torch.long)
                            qmask[i, :len(nt)] = 1
                        ids = ids.to(dev)
                        ids[:, 0] = g.pending
                        qmask = qmask.to(dev)
                        g.attn = torch.cat([g.attn, qmask], dim=1)
                        t_logits, g.t_cache = self._fwd(self.teacher, ids, g.attn, g.t_cache)
                        s_logits, g.s_cache = self._fwd(self.student, ids, g.attn, g.s_cache)
                        kl, sup, tsup, ent = score_logits(
                            s_logits, t_logits, self.temperature, K, self.beta, self.add_tail, self.jsd_fn
                        )
                        del t_logits, s_logits
                        if TLP is None:
                            TLP = torch.zeros(B, T, K, dtype=tsup.dtype, device=dev)
                        new_pending = g.pending.clone()
                        for i, r in enumerate(g.rows):
                            n = len(new_tok[r])
                            if n == 0:
                                continue
                            a = scored[r]
                            KL[r, a:a + n] = kl[i, :n]
                            SUP[r, a:a + n] = sup[i, :n]
                            TLP[r, a:a + n] = tsup[i, :n]
                            TENT[r, a:a + n] = ent[i, :n]
                            scored[r] = a + n
                            new_pending[i] = new_tok[r][-1]
                        g.pending = new_pending
                        del kl, sup, tsup, ent
                    torch.cuda.synchronize(dev)
                    t_score += time.perf_counter() - _t0

                    # 3) truncation decision on the accumulated KL of the active rows
                    _t0 = time.perf_counter()
                    n = max(scored[r] for r in active)
                    idx = torch.tensor(active, device=dev)
                    lens = torch.tensor([scored[r] for r in active], device=dev)
                    valid = (torch.arange(n, device=dev).unsqueeze(0) < lens.unsqueeze(1)).float()
                    res = find_truncation(KL[idx, :n] * valid, valid, threshold, L=self.L, t=self.t, N=self.N)
                    is_tr = res["is_trunc"].tolist()
                    ms = res["mask_start"].tolist()
                    for r, tr, m in zip(active, is_tr, ms):
                        if tr:
                            trunc_at[r] = int(m)
                            done[r] = True
                        elif r in finished:
                            done[r] = True
                    # 4) drop finished rollouts from the caches
                    new_groups = []
                    for g in groups:
                        keep = [i for i, r in enumerate(g.rows) if not done[r]]
                        if not keep:
                            g.t_cache = g.s_cache = None
                            continue
                        if len(keep) < len(g.rows):
                            sel = torch.tensor(keep, device=dev)
                            g.t_cache.batch_select_indices(sel)
                            g.s_cache.batch_select_indices(sel)
                            g.attn = g.attn[sel]
                            g.pending = g.pending[sel]
                            g.rows = [g.rows[i] for i in keep]
                        new_groups.append(g)
                    groups = new_groups
                    t_detect += time.perf_counter() - _t0
        finally:
            if was_training_s:
                self.student.train()
            if was_training_t:
                self.teacher.train()

        # Final pass over every rollout's generated prefix: buffer values + consistency check.
        gen_len = torch.tensor([len(x) for x in gen_tokens], device=dev)
        n = int(gen_len.max())
        valid = (torch.arange(n, device=dev).unsqueeze(0) < gen_len.unsqueeze(1)).float()
        res = find_truncation(KL[:, :n] * valid, valid, threshold, L=self.L, t=self.t, N=self.N)
        is_trunc = res["is_trunc"]
        mask_start = res["mask_start"]
        online_trunc = torch.tensor([x >= 0 for x in trunc_at], device=dev)
        if not torch.equal(online_trunc, is_trunc) or not torch.equal(
            mask_start[is_trunc], torch.tensor(trunc_at, device=dev)[is_trunc]
        ):
            raise RuntimeError("online KAT: chunked truncation decision disagrees with the full-prefix decision")
        kept_len = torch.where(is_trunc, mask_start, gen_len)
        if res["win_mean"] is not None:
            wv = res["win_valid"]
            wm = res["win_mean"]
            cnt = float(wv.sum())
            win_stats = (
                float((wm * wv.float()).sum()),
                cnt,
                float(torch.where(wv, wm, torch.full_like(wm, float("inf"))).min()) if cnt > 0 else float("nan"),
                float(torch.where(wv, wm, torch.full_like(wm, float("-inf"))).max()) if cnt > 0 else float("nan"),
            )
        else:
            win_stats = (0.0, 0.0, float("nan"), float("nan"))

        kept = kept_len.tolist()
        return OnlineRolloutResult(
            completion_ids=[gen_tokens[i][: kept[i]] for i in range(B)],
            support_ids=[SUP[i, : kept[i]] for i in range(B)],
            teacher_support_logps=[TLP[i, : kept[i]] for i in range(B)],
            teacher_entropy=[TENT[i, : kept[i]] for i in range(B)],
            kl=[KL[i, : kept[i]] for i in range(B)],
            gen_len=gen_len,
            kept_len=kept_len,
            is_trunc=is_trunc,
            mask_start=mask_start,
            push_vals=res["row_min"][res["row_has_kept"]],
            win_stats=win_stats,
            timing={
                "vllm_s": t_vllm,
                "score_s": t_score,
                "detect_s": t_detect,
                "rounds": float(n_rounds),
                "vllm_calls": float(n_vllm_calls),
            },
        )
