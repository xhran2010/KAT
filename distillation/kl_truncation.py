# Adapted from verl (Bytedance Ltd.), Apache-2.0 License.
# Source: OPD commit 18c677218af3e68e956ec54f814a49b7d4b696b9
#         verl/verl/trainer/ppo/kl_truncation.py
#
# Sliding-window KL-based rollout truncation: maintain a global FIFO buffer of per-rollout
# *minimum* window-mean reverse-KL values, derive a percentile threshold once enough samples
# are buffered, and produce a per-row truncation mask once >= N consecutive windows fall
# below the threshold. Mask returned is 1=keep, 0=mask.

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def find_truncation(
    kl: torch.Tensor,                 # (B, S) float32 per-token KL
    valid: torch.Tensor,              # (B, S) float32 1=real response token
    threshold: torch.Tensor | None,   # scalar threshold, or None when truncation is inactive
    L: int,
    t: int,
    N: int,
) -> dict[str, Any]:
    """Pure truncation decision shared by the loss-mask and the online paths.

    window means (length L, starting at offset t) -> low = mean < threshold on fully-valid
    windows -> first run of N consecutive low windows i_first -> mask_start = t + i_first + L
    (keep the first low window, cut at its end). Rows without a hit get mask_start = S.
    row_min = min window mean over fully-valid windows that end at or before mask_start
    (the value pushed into the FIFO buffer); row_has_kept says whether such a window exists.

    The decision is prefix-consistent: evaluated on any prefix that contains window
    i_first + N - 1, it returns the same i_first as on the full sequence, which is what lets
    the online rollout call it on the KL accumulated so far.
    """
    device = kl.device
    B, S = kl.shape
    S_long = torch.full((B,), S, device=device, dtype=torch.int64)
    W = S - t - L + 1
    if W <= 0:
        return {
            "mask_start": S_long,
            "is_trunc": torch.zeros(B, device=device, dtype=torch.bool),
            "row_min": torch.full((B,), float("inf"), device=device, dtype=torch.float32),
            "row_has_kept": torch.zeros(B, device=device, dtype=torch.bool),
            "win_mean": None,
            "win_valid": None,
        }

    kl_slice = kl[:, t:].unsqueeze(1)
    win_mean = F.avg_pool1d(kl_slice, kernel_size=L, stride=1).squeeze(1)  # (B, W)

    mask_slice = valid[:, t:].unsqueeze(1)
    win_valid_frac = F.avg_pool1d(mask_slice, kernel_size=L, stride=1).squeeze(1)
    win_valid = win_valid_frac >= (1.0 - 1e-6)

    is_trunc = torch.zeros(B, device=device, dtype=torch.bool)
    i_first = torch.zeros(B, device=device, dtype=torch.int64)

    if threshold is not None and W >= N:
        low = (win_mean < threshold) & win_valid
        kernel = torch.ones(1, 1, N, device=device, dtype=torch.float32)
        conv_out = F.conv1d(low.float().unsqueeze(1), kernel).squeeze(1)
        hits = conv_out >= (N - 0.5)
        any_hit = hits.any(dim=-1)
        i_first = hits.to(torch.int64).argmax(dim=-1)
        is_trunc = any_hit

    mask_start = torch.where(is_trunc, (t + i_first + L).to(torch.int64), S_long)

    last_pos_plus_one = t + torch.arange(W, device=device, dtype=torch.int64) + L  # (W,)
    kept_window = (last_pos_plus_one.unsqueeze(0) <= mask_start.unsqueeze(1)) & win_valid

    inf_t = torch.full_like(win_mean, float("inf"))
    win_for_min = torch.where(kept_window, win_mean, inf_t)
    row_min, _ = win_for_min.min(dim=1)
    row_has_kept = kept_window.any(dim=1)

    return {
        "mask_start": mask_start,
        "is_trunc": is_trunc,
        "row_min": row_min,
        "row_has_kept": row_has_kept,
        "win_mean": win_mean,
        "win_valid": win_valid,
    }


class KLTruncationManager:
    def __init__(self, config: Any, device: Any):
        self.enabled = bool(config.get("enable", False))
        self.K = int(config.get("warmup_steps", 50))
        self.L = int(config.get("window_size", 64))
        self.t = int(config.get("start_position", 16))
        self.M = int(config.get("buffer_size", 4096))
        self.eta = float(config.get("eta", 0.30))
        self.N = int(config.get("consecutive_count", 4))
        self.min_samples = int(config.get("min_samples_for_threshold", 512))

        assert 0.0 < self.eta < 1.0, f"eta must be in (0,1), got {self.eta}"
        assert self.L >= 1 and self.t >= 0 and self.N >= 1
        assert self.min_samples <= self.M, "min_samples_for_threshold must be <= buffer_size"

        self.device = device
        self.buf = torch.zeros(self.M, device=device, dtype=torch.float32)
        self.buf_ptr = 0
        self.buf_size = 0

    # ── shared helpers (used by both loss-mask `step()` and the online rollout) ──

    def _buf_view(self) -> torch.Tensor:
        return self.buf if self.buf_size >= self.M else self.buf[: self.buf_size]

    def _current_threshold(self) -> torch.Tensor:
        return torch.quantile(self._buf_view(), 1.0 - self.eta)

    @torch.no_grad()
    def push(self, push_vals: torch.Tensor) -> None:
        """FIFO-write one value per rollout (min kept window mean) into the buffer."""
        n_push = push_vals.numel()
        if n_push == 0:
            return
        device = self.buf.device
        push_vals = push_vals.to(device)
        if n_push > self.M:
            push_vals = push_vals[-self.M:]
            n_push = self.M
        indices = (self.buf_ptr + torch.arange(n_push, device=device, dtype=torch.int64)) % self.M
        self.buf.index_copy_(0, indices, push_vals.detach().to(torch.float32))
        self.buf_ptr = (self.buf_ptr + n_push) % self.M
        self.buf_size = min(self.buf_size + n_push, self.M)

    # ── online (rollout-time) API ──

    @torch.no_grad()
    def begin_step(self, global_step: int, device: Any = None) -> torch.Tensor | None:
        """Threshold for this optimizer step, or None when truncation is not active yet.

        None means warmup (`global_step <= K`) or the buffer holds fewer than `min_samples`
        values; the caller then falls back to the loss-mask path, whose `step()` keeps filling
        the buffer exactly as before. The threshold is fixed for the whole step (the
        loss-mask path re-reads it after every micro-batch).
        """
        if not self.enabled:
            return None
        if device is not None and self.buf.device != torch.device(device):
            self.buf = self.buf.to(device)
        if global_step <= self.K or self.buf_size < self.min_samples:
            return None
        return self._current_threshold().float()

    @torch.no_grad()
    def online_metrics(
        self,
        threshold: torch.Tensor,
        kept_len: torch.Tensor,       # (B,) kept completion tokens per rollout
        gen_len: torch.Tensor,        # (B,) completion tokens actually generated per rollout
        is_trunc: torch.Tensor,       # (B,) bool
        mask_start: torch.Tensor,     # (B,) truncation start (only meaningful where is_trunc)
        win_stats: tuple[float, float, float, float],  # (sum, count, min, max) of valid window means
        max_len: int,
        global_step: int,
    ) -> dict[str, float]:
        """Same `kl_trunc/*` keys as `step()`, plus `kl_trunc/generated_length_mean`.

        In online mode the untruncated length of a cut rollout is unknown (generation stopped),
        so `original_length_mean` / `tokens_masked_*` refer to the *generated* tokens; the
        discarded tokens are the overshoot generated past the cut point.
        """
        nan = float("nan")
        kept_f = kept_len.float()
        gen_f = gen_len.float()
        trunc_f = is_trunc.float()
        masked = (gen_f - kept_f).sum()
        n_trunc = trunc_f.sum()
        start_global = torch.where(is_trunc, mask_start.float(), torch.full_like(kept_f, float(max_len)))
        w_sum, w_cnt, w_min, w_max = win_stats
        if self.buf_size > 0:
            bv = self._buf_view()
            qs = torch.quantile(bv, torch.tensor([0.1, 0.5, 0.9], device=bv.device, dtype=torch.float32)).tolist()
            buf_mean = float(bv.mean())
        else:
            qs = [nan, nan, nan]
            buf_mean = nan
        m = {
            "kl_trunc/threshold": float(threshold),
            "kl_trunc/frac_seqs_truncated": float(trunc_f.mean()),
            "kl_trunc/avg_trunc_start_pos": float((mask_start.float() * trunc_f).sum() / n_trunc) if n_trunc > 0 else nan,
            "kl_trunc/avg_trunc_start_pos_global": float(start_global.mean()),
            "kl_trunc/tokens_masked_total": float(masked),
            "kl_trunc/tokens_masked_frac": float(masked / (gen_f.sum() + 1e-9)),
            "kl_trunc/effective_length_mean": float(kept_f.mean()),
            "kl_trunc/original_length_mean": float(gen_f.mean()),
            "kl_trunc/generated_length_mean": float(gen_f.mean()),
            "kl_trunc/window_mean_mean": (w_sum / w_cnt) if w_cnt > 0 else nan,
            "kl_trunc/window_mean_min": w_min if w_cnt > 0 else nan,
            "kl_trunc/window_mean_max": w_max if w_cnt > 0 else nan,
            "kl_trunc/buffer_mean": buf_mean,
            "kl_trunc/buffer_p10": qs[0],
            "kl_trunc/buffer_p50": qs[1],
            "kl_trunc/buffer_p90": qs[2],
            "kl_trunc/buffer_fill": float(self.buf_size) / float(self.M),
            "kl_trunc/is_warmup": float(global_step <= self.K),
            "kl_trunc/active": 1.0,
            "kl_trunc/n_windows": float(max(max_len - self.t - self.L + 1, 0)),
            "kl_trunc/buffer_size_curr": float(self.buf_size),
        }
        return m

    @torch.no_grad()
    def step(
        self,
        per_token_kl: torch.Tensor,   # (B, S) any float dtype
        response_mask: torch.Tensor,  # (B, S) bool or float
        global_step: int,
    ):
        device = per_token_kl.device
        dtype = per_token_kl.dtype
        B, S = per_token_kl.shape

        if not self.enabled:
            return torch.ones(B, S, device=device, dtype=dtype), {}, None

        kl_f32 = per_token_kl.float()
        rmask_f32 = response_mask.float() if response_mask.dtype != torch.float32 else response_mask

        L = self.L
        t = self.t
        N = self.N
        is_warmup = global_step <= self.K

        if S - t < L:
            return (
                torch.ones(B, S, device=device, dtype=dtype),
                self._dummy_metrics(global_step=global_step, is_warmup=is_warmup, B=B, S=S, device=device),
                None,
            )

        W = S - t - L + 1
        active = (not is_warmup) and (self.buf_size >= self.min_samples) and (W >= N)

        threshold_t = torch.full((), float("nan"), device=device, dtype=torch.float32)
        if active:
            threshold_t = self._current_threshold()

        res = find_truncation(kl_f32, rmask_f32, threshold_t if active else None, L=L, t=t, N=N)
        win_mean = res["win_mean"]
        win_valid = res["win_valid"]
        is_trunc = res["is_trunc"]
        mask_start = res["mask_start"]

        arange_S = torch.arange(S, device=device).unsqueeze(0)
        keep_bool = arange_S < mask_start.unsqueeze(1)
        trunc_mask = keep_bool.to(dtype)

        trunc_start_f = mask_start.to(torch.float32)

        # Buffer update: one value per rollout = min of window means over the kept region.
        self.push(res["row_min"][res["row_has_kept"]])

        # Metrics
        kept_per_row = (trunc_mask.float() * rmask_f32).sum(dim=-1)
        orig_per_row = rmask_f32.sum(dim=-1)
        valid_resp_tokens = orig_per_row.sum()
        kept_tokens = kept_per_row.sum()
        masked_tokens = valid_resp_tokens - kept_tokens
        effective_length_mean = kept_per_row.mean()
        original_length_mean = orig_per_row.mean()

        frac_seqs_trunc = is_trunc.float().mean()
        any_trunc = is_trunc.any()
        avg_start_trunc_only = torch.where(
            any_trunc,
            (trunc_start_f * is_trunc.float()).sum() / is_trunc.float().sum().clamp(min=1.0),
            torch.full((), float("nan"), device=device, dtype=torch.float32),
        )
        avg_start_global = trunc_start_f.mean()

        win_valid_f = win_valid.float()
        n_valid_w = win_valid_f.sum()
        win_mean_sum = (win_mean * win_valid_f).sum()
        win_mean_for_min = torch.where(win_valid, win_mean, torch.full_like(win_mean, float("inf")))
        win_mean_for_max = torch.where(win_valid, win_mean, torch.full_like(win_mean, float("-inf")))
        nan_t = torch.full((), float("nan"), device=device, dtype=torch.float32)
        win_mean_mean = torch.where(n_valid_w > 0, win_mean_sum / n_valid_w.clamp(min=1.0), nan_t)
        win_mean_min = torch.where(n_valid_w > 0, win_mean_for_min.min(), nan_t)
        win_mean_max = torch.where(n_valid_w > 0, win_mean_for_max.max(), nan_t)

        if self.buf_size > 0:
            buf_view2 = self.buf if self.buf_size >= self.M else self.buf[: self.buf_size]
            buf_mean = buf_view2.mean()
            buf_qs = torch.quantile(buf_view2, torch.tensor([0.1, 0.5, 0.9], device=device, dtype=torch.float32))
            buf_p10, buf_p50, buf_p90 = buf_qs[0], buf_qs[1], buf_qs[2]
        else:
            buf_mean = nan_t
            buf_p10 = nan_t
            buf_p50 = nan_t
            buf_p90 = nan_t

        scalar_keys = [
            "threshold",
            "frac_seqs_truncated",
            "avg_trunc_start_pos",
            "avg_trunc_start_pos_global",
            "tokens_masked_total",
            "tokens_masked_frac",
            "effective_length_mean",
            "original_length_mean",
            "window_mean_mean",
            "window_mean_min",
            "window_mean_max",
            "buffer_mean",
            "buffer_p10",
            "buffer_p50",
            "buffer_p90",
        ]
        scalar_tensors = [
            threshold_t.float().reshape(()),
            frac_seqs_trunc.reshape(()),
            avg_start_trunc_only.reshape(()),
            avg_start_global.reshape(()),
            masked_tokens.float().reshape(()),
            (masked_tokens / (valid_resp_tokens + 1e-9)).float().reshape(()),
            effective_length_mean.float().reshape(()),
            original_length_mean.float().reshape(()),
            win_mean_mean.reshape(()),
            win_mean_min.reshape(()),
            win_mean_max.reshape(()),
            buf_mean.reshape(()),
            buf_p10.reshape(()),
            buf_p50.reshape(()),
            buf_p90.reshape(()),
        ]
        host_vals = torch.stack(scalar_tensors).cpu().tolist()
        metrics = {f"kl_trunc/{k}": v for k, v in zip(scalar_keys, host_vals)}
        metrics["kl_trunc/buffer_fill"] = float(self.buf_size) / float(self.M)
        metrics["kl_trunc/is_warmup"] = float(is_warmup)
        metrics["kl_trunc/active"] = float(active)
        metrics["kl_trunc/n_windows"] = float(W)
        metrics["kl_trunc/buffer_size_curr"] = float(self.buf_size)

        debug = {
            "window_mean": win_mean,
            "win_valid": win_valid,
            "trunc_start": trunc_start_f,
            "is_trunc": is_trunc,
            "threshold": threshold_t,
        }
        return trunc_mask, metrics, debug

    def _dummy_metrics(self, global_step, is_warmup, B, S, device):
        return {
            "kl_trunc/threshold": float("nan"),
            "kl_trunc/frac_seqs_truncated": 0.0,
            "kl_trunc/avg_trunc_start_pos": float("nan"),
            "kl_trunc/avg_trunc_start_pos_global": float(S),
            "kl_trunc/tokens_masked_total": 0.0,
            "kl_trunc/tokens_masked_frac": 0.0,
            "kl_trunc/effective_length_mean": float(S),
            "kl_trunc/original_length_mean": float(S),
            "kl_trunc/window_mean_mean": float("nan"),
            "kl_trunc/window_mean_min": float("nan"),
            "kl_trunc/window_mean_max": float("nan"),
            "kl_trunc/buffer_mean": float("nan"),
            "kl_trunc/buffer_p10": float("nan"),
            "kl_trunc/buffer_p50": float("nan"),
            "kl_trunc/buffer_p90": float("nan"),
            "kl_trunc/buffer_fill": float(self.buf_size) / float(self.M),
            "kl_trunc/is_warmup": float(is_warmup),
            "kl_trunc/active": 0.0,
            "kl_trunc/n_windows": 0.0,
            "kl_trunc/buffer_size_curr": float(self.buf_size),
        }


def _self_test():
    """Synthetic correctness check: engineered low-KL region triggers expected truncation."""
    class _Cfg(dict):
        def get(self, k, d=None):
            return self[k] if k in self else d

    L = 4
    t = 2
    N = 3
    M = 64
    cfg = _Cfg(
        enable=True,
        warmup_steps=0,
        window_size=L,
        start_position=t,
        buffer_size=M,
        eta=0.30,
        consecutive_count=N,
        min_samples_for_threshold=4,
    )
    device = "cpu"
    mgr = KLTruncationManager(cfg, device=device)

    mgr.buf[:10] = torch.tensor([1.0] * 10, dtype=torch.float32)
    mgr.buf_size = 10
    mgr.buf_ptr = 10

    B, S = 2, 60
    kl = torch.full((B, S), 1.0)
    low_start = 30
    low_len = (N + 4) + L
    kl[0, low_start: low_start + low_len] = 0.0
    response_mask = torch.ones(B, S, dtype=torch.bool)

    trunc_mask, metrics, _ = mgr.step(kl, response_mask, global_step=1)
    i_first_expected = low_start - t - L + 1
    mask_start_expected = t + i_first_expected + L

    assert trunc_mask[0, mask_start_expected:].sum().item() == 0.0
    assert trunc_mask[0, :mask_start_expected].sum().item() == float(mask_start_expected)
    assert trunc_mask[1].sum().item() == float(S)

    print("[kl_truncation self-test] PASSED")
    print(f"  metrics: {metrics}")


def _online_equivalence_test(n_trials: int = 200, seed: int = 0):
    """Chunk-by-chunk online detection == one-shot `step()` (same mask_start, same buffer pushes).

    Each trial: random batch of KL sequences with random lengths and planted low-KL stretches,
    a pre-filled buffer, and a random chunk size. The loss-mask manager runs `step()` once on the
    right-padded batch; the online manager reveals tokens chunk by chunk, calls `find_truncation`
    on the accumulated prefix after every chunk, stops a row as soon as it triggers or ends, and
    pushes the final row_min values after the whole batch (threshold fixed for the step).
    """
    class _Cfg(dict):
        def get(self, k, d=None):
            return self[k] if k in self else d

    g = torch.Generator().manual_seed(seed)
    n_trunc_total = 0
    n_rows_total = 0
    for trial in range(n_trials):
        L = int(torch.randint(2, 17, (1,), generator=g))
        t = int(torch.randint(0, 9, (1,), generator=g))
        N = int(torch.randint(1, 6, (1,), generator=g))
        M = 256
        cfg = _Cfg(enable=True, warmup_steps=0, window_size=L, start_position=t, buffer_size=M,
                   eta=float(torch.empty(1).uniform_(0.1, 0.9, generator=g)), consecutive_count=N,
                   min_samples_for_threshold=8)
        mgr_a = KLTruncationManager(cfg, device="cpu")
        mgr_b = KLTruncationManager(cfg, device="cpu")
        n0 = int(torch.randint(8, 200, (1,), generator=g))
        init = torch.rand(n0, generator=g) * 0.5
        for m in (mgr_a, mgr_b):
            m.push(init.clone())

        B = int(torch.randint(1, 9, (1,), generator=g))
        S = int(torch.randint(1, 160, (1,), generator=g))
        lens = torch.randint(1, S + 1, (B,), generator=g)
        kl = torch.rand(B, S, generator=g) * 0.6
        for b in range(B):
            if torch.rand(1, generator=g) < 0.6:
                s0 = int(torch.randint(0, S, (1,), generator=g))
                s1 = min(S, s0 + int(torch.randint(1, 60, (1,), generator=g)))
                kl[b, s0:s1] *= 0.02
        valid = (torch.arange(S).unsqueeze(0) < lens.unsqueeze(1)).float()
        kl = kl * valid  # padded positions carry 0 KL, as in the loss path

        # (a) loss-mask path: one step() on the padded batch
        trunc_mask, _, dbg = mgr_a.step(kl, valid.bool(), global_step=1)
        if dbg is None:  # S - t < L: no windows at all
            is_a = torch.zeros(B, dtype=torch.bool)
            ms_a = torch.full((B,), S, dtype=torch.long)
        else:
            is_a = dbg["is_trunc"]
            ms_a = dbg["trunc_start"].long()

        # (b) online path
        thr = mgr_b.begin_step(1)
        assert thr is not None
        C = int(torch.randint(1, 40, (1,), generator=g))
        gen = torch.zeros(B, dtype=torch.long)
        active = torch.ones(B, dtype=torch.bool)
        ms_b = torch.full((B,), -1, dtype=torch.long)
        while active.any():
            gen = torch.where(active, torch.minimum(gen + C, lens), gen)
            n = int(gen.max())
            v = (torch.arange(n).unsqueeze(0) < gen.unsqueeze(1)).float()
            res = find_truncation(kl[:, :n] * v, v, thr, L=L, t=t, N=N)
            for b in range(B):
                if not active[b]:
                    continue
                if bool(res["is_trunc"][b]):
                    ms_b[b] = int(res["mask_start"][b])
                    assert ms_b[b] <= gen[b]
                    active[b] = False
                elif gen[b] >= lens[b]:
                    active[b] = False
        is_b = ms_b >= 0
        n_max = int(gen.max())
        v = (torch.arange(n_max).unsqueeze(0) < gen.unsqueeze(1)).float()
        res = find_truncation(kl[:, :n_max] * v, v, thr, L=L, t=t, N=N)
        assert torch.equal(res["is_trunc"], is_b), (trial, res["is_trunc"], is_b)
        mgr_b.push(res["row_min"][res["row_has_kept"]])

        assert torch.equal(is_a, is_b), (trial, is_a, is_b)
        assert torch.equal(ms_a[is_a], ms_b[is_b]), (trial, ms_a, ms_b)
        assert mgr_a.buf_size == mgr_b.buf_size and mgr_a.buf_ptr == mgr_b.buf_ptr, trial
        assert torch.equal(mgr_a.buf, mgr_b.buf), (trial, (mgr_a.buf - mgr_b.buf).abs().max())
        n_trunc_total += int(is_a.sum())
        n_rows_total += B
    print(f"[kl_truncation online-equivalence] PASSED — {n_trials} trials, "
          f"{n_rows_total} rollouts, {n_trunc_total} truncated")


if __name__ == "__main__":
    _self_test()
    _online_equivalence_test()
