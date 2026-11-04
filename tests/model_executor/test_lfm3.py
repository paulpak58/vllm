# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

modeling_lfm3 = pytest.importorskip("transformers.models.lfm3.modeling_lfm3")

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.models.lfm3 import get_rotary_embedding  # noqa: E402


@pytest.fixture
def cpu_vllm_config():
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        yield


def _config(layer_type: str, rope_parameters: dict):
    return modeling_lfm3.Lfm3Config(
        hidden_size=64,
        head_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        layer_types=[layer_type],
        max_position_embeddings=128,
        rope_parameters={layer_type: rope_parameters},
    )


@pytest.mark.parametrize("layer_type", ["full_attention", "sliding_attention"])
@pytest.mark.parametrize(
    "rope_type,partial_rotary_factor",
    [("default", 1.0), ("proportional", 1.0), ("proportional", 0.5)],
)
def test_rotary_embedding_matches_transformers(
    cpu_vllm_config, layer_type: str, rope_type: str, partial_rotary_factor: float
):
    config = _config(
        layer_type,
        {
            "rope_type": rope_type,
            "rope_theta": 10000.0,
            "partial_rotary_factor": partial_rotary_factor,
        },
    )
    rotary_emb = get_rotary_embedding(config, layer_type)

    torch.manual_seed(0)
    query = torch.randn(4, 2, 32)
    key = torch.randn(4, 1, 32)
    positions = torch.tensor([0, 1, 63, 127])
    actual_q, actual_k = rotary_emb.forward_native(
        positions, query.flatten(1).clone(), key.flatten(1).clone()
    )
    cos, sin = modeling_lfm3.Lfm3RotaryEmbedding(config)(
        query, positions[None], layer_type
    )
    expected_q, expected_k = modeling_lfm3.apply_rotary_pos_emb(
        query[None], key[None], cos, sin, unsqueeze_dim=2
    )
    torch.testing.assert_close(actual_q, expected_q[0].flatten(1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_k, expected_k[0].flatten(1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "rope_parameters", [{}, {"rope_type": "linear", "factor": 2.0}]
)
def test_partial_rotation_needs_proportional_rope(cpu_vllm_config, rope_parameters):
    config = _config(
        "full_attention",
        {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, **rope_parameters},
    )
    with pytest.raises(NotImplementedError, match="partial rotation"):
        get_rotary_embedding(config, "full_attention")
