# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal conv1d for kernel widths the fused kernels do not unroll (LFM3 uses 7)."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

_PAD = tl.constexpr(PAD_SLOT_ID)
_NULL = tl.constexpr(NULL_BLOCK_ID)


@triton.jit(do_not_specialize=["N", "R"])
def _convolve(
    X,
    W,
    Bias,
    State,
    Slots,
    Starts,
    Initial,
    Y,
    N,
    D: tl.constexpr,
    K: tl.constexpr,
    R,
    SS: tl.constexpr,
    SD: tl.constexpr,
    ST: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    DECODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    if DECODE:
        req = t
        start = t
        initial = tl.full((BT,), True, tl.int1)
    else:
        # Map packed tokens to requests without a host sync.
        lo = tl.full((BT,), 0, tl.int32)
        hi = tl.full((BT,), 0, tl.int32) + R
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            end = tl.load(Starts + mid + 1, mid < R, N)
            right = end <= t
            active = lo < hi
            lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
        req = lo
        start = tl.load(Starts + req, req < R, 0)
        initial = tl.load(Initial + req, req < R, 0)
    # int64: slot * SS indexes a strided view of the whole state pool and passes
    # 2^31 once high block ids are in use
    slot = tl.load(Slots + req * SLOT_STRIDE, (t < N) & (req < R), _PAD).to(tl.int64)
    valid = (t < N) & (req < R) & (slot != _PAD) & (slot != _NULL)
    acc = tl.full((BT, BD), 0, tl.float32)
    if HAS_BIAS:
        acc += tl.load(Bias + d, d < D, 0)[None, :].to(tl.float32)
    for k in tl.static_range(K):
        src = t - (K - 1) + k
        x = tl.load(
            X + src[:, None] * D + d[None, :],
            valid[:, None] & (src >= start)[:, None] & (d < D)[None, :],
            0,
        )
        prev = tl.load(
            State
            + slot[:, None] * SS
            + d[None, :] * SD
            + (K - 1 + src - start)[:, None] * ST,
            valid[:, None]
            & initial[:, None]
            & (src < start)[:, None]
            & (d < D)[None, :],
            0,
        )
        x = tl.where((src >= start)[:, None], x, prev).to(tl.float32)
        w = tl.load(W + d * K + k, d < D, 0).to(tl.float32)
        acc += x * w[None, :]
    tl.store(Y + t[:, None] * D + d[None, :], acc, (t < N)[:, None] & (d < D)[None, :])


@triton.jit(do_not_specialize=["N"])
def _update_state(
    X,
    State,
    Slots,
    Starts,
    Initial,
    N,
    D: tl.constexpr,
    K: tl.constexpr,
    SS: tl.constexpr,
    SD: tl.constexpr,
    ST: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    DECODE: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    req = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    k = tl.arange(0, BK)
    slot = tl.load(Slots + req * SLOT_STRIDE).to(tl.int64)
    if slot == _PAD or slot == _NULL:  # noqa: SIM109 (Triton scalar predicates)
        return
    if DECODE:
        start = req
        end = req + 1
        initial = True
    else:
        start = tl.load(Starts + req)
        end = tl.load(Starts + req + 1)
        initial = tl.load(Initial + req)
    src = end - (K - 1) + k
    mask = (k < K - 1)[:, None] & (d < D)[None, :]
    x = tl.load(X + src[:, None] * D + d[None, :], mask & (src >= start)[:, None], 0)
    old = tl.load(
        State + slot * SS + d[None, :] * SD + (K - 1 + src - start)[:, None] * ST,
        mask & (src < start)[:, None] & initial,
        0,
    )
    value = tl.where((src >= start)[:, None], x, old)
    # All old state is loaded before any thread overwrites it.
    tl.debug_barrier()
    tl.store(State + slot * SS + d[None, :] * SD + k[:, None] * ST, value, mask)


def causal_conv1d_wide(x, weight, bias, state, slots, starts=None, initial=None):
    """x: [tokens, channels]; state holds the K-1 inputs preceding each request.

    Decode (one token per request) when `starts` is None.
    """
    x = x.contiguous()
    n, d = x.shape
    k = weight.shape[-1]
    decode = starts is None
    requests = n if decode else starts.numel() - 1
    output = torch.empty_like(x)
    if not n:
        return output
    common = dict(
        N=n,
        D=d,
        K=k,
        SS=state.stride(0),
        SD=state.stride(1),
        ST=state.stride(2),
        SLOT_STRIDE=slots.stride(0),
        DECODE=decode,
    )
    _convolve[(triton.cdiv(n, 16), triton.cdiv(d, 64))](
        x,
        weight,
        bias,
        state,
        slots,
        starts,
        initial,
        output,
        R=requests,
        HAS_BIAS=bias is not None,
        BT=16,
        BD=64,
        **common,
        enable_fp_fusion=False,
    )
    if k > 1:
        _update_state[(requests, triton.cdiv(d, 64))](
            x,
            state,
            slots,
            starts,
            initial,
            BK=triton.next_power_of_2(k - 1),
            BD=64,
            **common,
        )
    return output
