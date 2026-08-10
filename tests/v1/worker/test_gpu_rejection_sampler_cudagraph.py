# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sample_cudagraph_utils import (
    RejectionSamplerCudaGraphManager,
)


def _descriptor(
    *,
    mode: CUDAGraphMode = CUDAGraphMode.FULL,
    uniform_token_count: int | None = 4,
    num_reqs: int | None = 8,
    num_active_loras: int = 0,
) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=mode,
        num_tokens=32,
        num_reqs=num_reqs,
        uniform_token_count=uniform_token_count,
        num_active_loras=num_active_loras,
    )


def _manager_and_batch():
    desc = _descriptor()
    states = SimpleNamespace(
        vocab_size=100,
        top_k=SimpleNamespace(np=np.full(16, 100)),
        top_p=SimpleNamespace(np=np.full(16, 0.95)),
        max_num_logprobs=lambda idx: NO_LOGPROBS,
    )
    sampler = SimpleNamespace(
        sampling_states=states,
        logprob_token_ids_state=SimpleNamespace(max_num_token_ids=lambda idx: 0),
        penalties_state=SimpleNamespace(use_penalty=np.zeros(16, dtype=bool)),
        bad_words_state=SimpleNamespace(
            num_bad_words=SimpleNamespace(np=np.zeros(16, dtype=np.int32))
        ),
        logit_bias_state=SimpleNamespace(use_logit_bias=np.zeros(16, dtype=bool)),
    )
    manager = object.__new__(RejectionSamplerCudaGraphManager)
    manager.decode_query_len = 4
    manager.vocab_parallel_nucleus = False
    manager.graphs = {desc: object()}
    manager.rejection_sampler = SimpleNamespace(sampler=sampler)
    manager.draft_logits = None
    batch = SimpleNamespace(
        num_draft_tokens=24,
        is_prefilling_np=np.zeros(8, dtype=bool),
        idx_mapping_np=np.arange(8),
    )
    return manager, desc, batch


@pytest.mark.parametrize(
    ("desc", "expected"),
    [
        (_descriptor(), True),
        (_descriptor(mode=CUDAGraphMode.PIECEWISE), False),
        (_descriptor(uniform_token_count=1), False),
        (_descriptor(num_reqs=None), False),
        (_descriptor(num_active_loras=1), False),
    ],
)
def test_capturable_descriptors(desc, expected):
    manager = object.__new__(RejectionSamplerCudaGraphManager)
    manager.decode_query_len = 4
    assert manager._is_capturable_desc(desc) is expected


def test_production_top_p_batch_can_run():
    manager, desc, batch = _manager_and_batch()
    assert manager.can_run(desc, batch, has_grammar=False)


def test_distributed_nucleus_batch_can_run():
    manager, desc, batch = _manager_and_batch()
    manager.vocab_parallel_nucleus = True
    manager.rejection_sampler.can_vocab_parallel_nucleus = (
        lambda input_batch, draft_logits: input_batch is batch and draft_logits is None
    )
    assert manager.can_run(desc, batch, has_grammar=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "grammar",
        "missing_graph",
        "no_draft",
        "prefill",
        "logprobs",
        "per_token_logprobs",
        "top_k",
        "top_p_one",
        "penalty",
        "bad_words",
        "logit_bias",
    ],
)
def test_unsupported_sampling_falls_back(mutation):
    manager, desc, batch = _manager_and_batch()
    sampler = manager.rejection_sampler.sampler
    has_grammar = False

    if mutation == "grammar":
        has_grammar = True
    elif mutation == "missing_graph":
        manager.graphs.clear()
    elif mutation == "no_draft":
        batch.num_draft_tokens = 0
    elif mutation == "prefill":
        batch.is_prefilling_np[0] = True
    elif mutation == "logprobs":
        sampler.sampling_states.max_num_logprobs = lambda idx: 1
    elif mutation == "per_token_logprobs":
        sampler.logprob_token_ids_state.max_num_token_ids = lambda idx: 1
    elif mutation == "top_k":
        sampler.sampling_states.top_k.np[0] = 50
    elif mutation == "top_p_one":
        sampler.sampling_states.top_p.np.fill(1.0)
    elif mutation == "penalty":
        sampler.penalties_state.use_penalty[0] = True
    elif mutation == "bad_words":
        sampler.bad_words_state.num_bad_words.np[0] = 1
    elif mutation == "logit_bias":
        sampler.logit_bias_state.use_logit_bias[0] = True

    assert not manager.can_run(desc, batch, has_grammar)
