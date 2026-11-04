# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d_wide import causal_conv1d_wide
from vllm.platforms import current_platform


def reference(x: torch.Tensor, weight: torch.Tensor, bias, state: torch.Tensor):
    """x: (tokens, dim); state: (dim, K-1) preceding inputs -> output, new state."""
    padded = torch.cat([state.T, x], dim=0).T.unsqueeze(0).float()
    bias = None if bias is None else bias.float()
    out = F.conv1d(padded, weight.float().unsqueeze(1), bias, groups=weight.shape[0])
    return out[0].T.to(x.dtype), padded[0, :, -(weight.shape[1] - 1) :].to(x.dtype)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Triton kernel")
@pytest.mark.parametrize("width", [7, 9])
@pytest.mark.parametrize("has_bias", [False, True])
@torch.inference_mode()
def test_prefill_and_decode_match_reference(width: int, has_bias: bool):
    torch.manual_seed(0)
    device, dtype, dim = "cuda", torch.bfloat16, 64
    lengths = [5, 13, 1]
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None
    # slot 0 is the null block; request 0 continues from a cached state, the
    # others start from silence
    slots = torch.tensor([3, 1, 7], device=device, dtype=torch.int32)
    initial = torch.tensor([True, False, False], device=device)
    state = torch.randn(8, dim, width - 1, device=device, dtype=dtype)
    ref_states = [
        state[3].clone(),
        torch.zeros(dim, width - 1, device=device, dtype=dtype),
        torch.zeros(dim, width - 1, device=device, dtype=dtype),
    ]
    x = torch.randn(sum(lengths), dim, device=device, dtype=dtype)
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32
    )

    out = causal_conv1d_wide(x, weight, bias, state, slots, starts, initial)

    offset = 0
    for request, length in enumerate(lengths):
        expected, new_state = reference(
            x[offset : offset + length], weight, bias, ref_states[request]
        )
        torch.testing.assert_close(
            out[offset : offset + length], expected, atol=0, rtol=0
        )
        torch.testing.assert_close(state[slots[request]], new_state, atol=0, rtol=0)
        ref_states[request] = new_state
        offset += length

    # one decode token per request reads and advances the cached state
    x_d = torch.randn(len(lengths), dim, device=device, dtype=dtype)
    out_d = causal_conv1d_wide(x_d, weight, bias, state, slots)
    for request in range(len(lengths)):
        expected, new_state = reference(
            x_d[request : request + 1], weight, bias, ref_states[request]
        )
        torch.testing.assert_close(
            out_d[request : request + 1], expected, atol=0, rtol=0
        )
        torch.testing.assert_close(state[slots[request]], new_state, atol=0, rtol=0)
