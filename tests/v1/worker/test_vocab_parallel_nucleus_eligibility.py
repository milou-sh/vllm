# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


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
