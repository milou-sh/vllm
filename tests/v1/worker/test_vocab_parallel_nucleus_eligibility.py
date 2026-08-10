# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import envs
from vllm.v1.worker.gpu.sample.vocab_parallel_nucleus import (
    _gather_high_histogram,
    _gather_low_histogram,
)
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_sampler_module
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


class _FakeGroup:
    world_size = 2
    rank_in_group = 0

    def __init__(self, outputs: list[torch.Tensor]):
        self.outputs = iter(outputs)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        del input_, dim
        return next(self.outputs)


def _make_sampler() -> tuple[RejectionSampler, SimpleNamespace]:
    vocab_size = 154880
    states = SimpleNamespace(
        vocab_size=vocab_size,
        temperature=SimpleNamespace(np=np.ones(4, dtype=np.float32)),
        top_p=SimpleNamespace(np=np.full(4, 0.95, dtype=np.float32)),
        top_k=SimpleNamespace(np=np.full(4, vocab_size, dtype=np.int32)),
        min_p=SimpleNamespace(np=np.zeros(4, dtype=np.float32)),
        max_num_logprobs=lambda _: -1,
    )
    sampler_state = SimpleNamespace(
        use_fp64_gumbel=False,
        compute_nans=False,
        sampling_states=states,
        logprob_token_ids_state=SimpleNamespace(max_num_token_ids=lambda _: 0),
        logit_bias_state=SimpleNamespace(use_logit_bias=np.zeros(4, dtype=bool)),
        penalties_state=SimpleNamespace(use_penalty=np.zeros(4, dtype=bool)),
        bad_words_state=SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.zeros(4, dtype=np.int32))
        ),
    )
    rejection_sampler = RejectionSampler.__new__(RejectionSampler)
    rejection_sampler.sampler = sampler_state
    rejection_sampler.use_block_verification = False
    rejection_sampler.synthetic_conditional_rates = None
    input_batch = SimpleNamespace(idx_mapping_np=np.arange(4, dtype=np.int32))
    return rejection_sampler, input_batch


def test_production_sampling_shape_is_eligible():
    sampler, input_batch = _make_sampler()
    assert sampler.can_vocab_parallel_nucleus(input_batch, draft_logits=None)


def test_unsupported_sampling_features_fall_back():
    sampler, input_batch = _make_sampler()
    states = sampler.sampler.sampling_states

    states.temperature.np[0] = 0.0
    assert not sampler.can_vocab_parallel_nucleus(input_batch, None)
    states.temperature.np[0] = 1.0

    states.top_p.np[0] = 1.0
    assert not sampler.can_vocab_parallel_nucleus(input_batch, None)
    states.top_p.np[0] = 0.95

    states.top_k.np[0] = 64
    assert not sampler.can_vocab_parallel_nucleus(input_batch, None)
    states.top_k.np[0] = states.vocab_size

    states.min_p.np[0] = 0.05
    assert not sampler.can_vocab_parallel_nucleus(input_batch, None)
    states.min_p.np[0] = 0.0

    sampler.sampler.logit_bias_state.use_logit_bias[0] = True
    assert not sampler.can_vocab_parallel_nucleus(input_batch, None)
    sampler.sampler.logit_bias_state.use_logit_bias[0] = False

    sampler.sampler.penalties_state.use_penalty[0] = True
    assert not sampler.can_vocab_parallel_nucleus(input_batch, None)
    sampler.sampler.penalties_state.use_penalty[0] = False

    sampler.sampler.bad_words_state.num_bad_words.np[0] = 1
    assert not sampler.can_vocab_parallel_nucleus(input_batch, None)
    sampler.sampler.bad_words_state.num_bad_words.np[0] = 0

    assert not sampler.can_vocab_parallel_nucleus(input_batch, draft_logits=object())


def test_capture_schedule_saves_replayable_full_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sampler, _ = _make_sampler()
    sampler._nucleus_capture_step = 0
    sampler._nucleus_capture_count = 0
    full_logits = torch.arange(24, dtype=torch.bfloat16).view(2, 12)
    group = _FakeGroup([full_logits, full_logits])
    monkeypatch.setattr(rejection_sampler_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        envs, "VLLM_GLM52_NUCLEUS_CAPTURE_DIR", str(tmp_path), raising=False
    )
    monkeypatch.setattr(envs, "VLLM_GLM52_NUCLEUS_CAPTURE_EVERY", 2, raising=False)
    monkeypatch.setattr(envs, "VLLM_GLM52_NUCLEUS_CAPTURE_LIMIT", 2, raising=False)
    local_logits = full_logits[:, :6].contiguous()
    top_p = torch.full((2,), 0.95)
    positions = torch.tensor([131000, 131001])
    cu_num_logits = torch.tensor([0, 2])

    for _ in range(4):
        sampler._capture_nucleus_logits(
            local_logits, top_p, positions, cu_num_logits, vocab_size=10
        )

    captures = sorted(tmp_path.glob("*.pt"))
    assert [path.name for path in captures] == [
        "target-logits-000-step-00000000.pt",
        "target-logits-001-step-00000002.pt",
    ]
    payload = torch.load(captures[1], weights_only=True)
    torch.testing.assert_close(payload["logits"], full_logits[:, :10])
    torch.testing.assert_close(payload["positions"], positions)
    assert payload["step"] == 2


def test_histogram_gathers_rebase_rank_local_mass():
    rank_max = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    high = torch.zeros(2, 2, 256)
    high[0, :, 7] = torch.tensor([3.0, 5.0])
    high[1, :, 7] = torch.tensor([11.0, 13.0])
    high_packed = torch.cat((rank_max.unsqueeze(-1), high), dim=-1)
    group = _FakeGroup([high_packed.flatten(0, 1)])

    gathered_high, global_max, gathered_rank_max = _gather_high_histogram(
        high[0], rank_max[0], group
    )
    expected_scale = torch.exp(rank_max - rank_max.amax(dim=0).unsqueeze(0))
    expected_high = (high * expected_scale.unsqueeze(-1)).sum(dim=0)
    torch.testing.assert_close(global_max, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(gathered_rank_max, rank_max)
    torch.testing.assert_close(gathered_high, expected_high)

    low = torch.zeros(2, 2, 512)
    low[0, :, 9] = torch.tensor([17.0, 19.0])
    low[1, :, 9] = torch.tensor([23.0, 29.0])
    low[0, :, 256 + 9] = torch.tensor([2.0, 3.0])
    low[1, :, 256 + 9] = torch.tensor([5.0, 7.0])
    group = _FakeGroup([low.flatten(0, 1)])
    gathered_low, rank_low = _gather_low_histogram(
        low[0], gathered_rank_max, global_max, group
    )
    expected_mass = (low[:, :, :256] * expected_scale.unsqueeze(-1)).sum(dim=0)
    expected_counts = low[:, :, 256:].sum(dim=0)
    torch.testing.assert_close(rank_low, low)
    torch.testing.assert_close(gathered_low[:, :256], expected_mass)
    torch.testing.assert_close(gathered_low[:, 256:], expected_counts)
