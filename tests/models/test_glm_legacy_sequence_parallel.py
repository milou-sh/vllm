# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch import nn

from vllm.config import ParallelConfig
from vllm.model_executor.models import deepseek_mtp, deepseek_v2
from vllm.platforms import current_platform


class _IdentityNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(self, *args, **kwargs) -> torch.Tensor:
        hidden_states = kwargs.get(
            "hidden_states", args[1] if len(args) > 1 else args[0]
        )
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


class _RecordingProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states[:, :2]


class _SequenceParallelMTPBlock:
    use_sequence_parallel = True
    fuse_attention_allreduce_rms = False

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        assert residual is None
        return hidden_states * 2, hidden_states * 3


def _mock_collectives(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "sp_reduce_scatter",
        lambda tensor: tensor.chunk(2, dim=0)[0],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )


def test_legacy_glm_tp_only_sequence_parallel_is_opt_in(monkeypatch):
    monkeypatch.setattr(current_platform, "device_count", lambda: 4)
    monkeypatch.setattr(deepseek_v2.envs, "VLLM_GLM_LEGACY_TP_MOE_SP", False)
    parallel_config = ParallelConfig(
        tensor_parallel_size=4,
        data_parallel_size=1,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )

    assert not parallel_config.use_sequence_parallel_moe

    monkeypatch.setattr(deepseek_v2.envs, "VLLM_GLM_LEGACY_TP_MOE_SP", True)

    assert parallel_config.use_sequence_parallel_moe


def test_legacy_decoder_keeps_dense_states_sharded(monkeypatch):
    layer = object.__new__(deepseek_v2.DeepseekV2DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.use_mha = False
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingModule()
    layer.mlp = _RecordingModule()
    layer.routed_scaling_factor = 1.0

    _mock_collectives(monkeypatch, deepseek_v2)

    positions = torch.arange(3)
    hidden_states = deepseek_v2.sp_shard(
        torch.arange(6, dtype=torch.float32).view(3, 2)
    )
    hidden_states, residual = layer(positions, hidden_states, residual=None)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 3
    assert layer.mlp.num_tokens == 2

    hidden_states, residual = layer(positions, hidden_states, residual)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 3
    assert layer.mlp.num_tokens == 2


def test_legacy_mtp_projects_shard_and_gathers_both_states(monkeypatch):
    layer = object.__new__(deepseek_mtp.DeepSeekMultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = _IdentityNorm()
    layer.hnorm = _IdentityNorm()
    layer.eh_proj = _RecordingProjection()
    object.__setattr__(layer, "mtp_block", _SequenceParallelMTPBlock())
    layer.shared_head = _IdentityNorm()

    _mock_collectives(monkeypatch, deepseek_mtp)
    monkeypatch.setattr(deepseek_mtp.envs, "VLLM_MOE_SKIP_PADDING", False)

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states, recycled_hidden_states = layer(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(1, 4),
        previous_hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
    )

    sharded_states = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, 1))[:2]
    expected = torch.cat([sharded_states * 5, sharded_states * 5])[:3]
    assert layer.eh_proj.num_tokens == 2
    torch.testing.assert_close(hidden_states, expected)
    torch.testing.assert_close(recycled_hidden_states, expected)
