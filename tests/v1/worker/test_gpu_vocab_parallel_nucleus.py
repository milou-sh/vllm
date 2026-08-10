# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.vocab_parallel_nucleus import (
    distributed_bf16_top_p_cutoff,
    distributed_nucleus_candidates,
    distributed_one_hot_rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ordered_key(logits: torch.Tensor) -> torch.Tensor:
    bits = logits.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)


def _reference_keep_mask(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)
    probs = logits.float().softmax(dim=-1)
    sorted_probs = probs.gather(-1, order)
    cumulative = sorted_probs.cumsum(dim=-1)
    counts = (cumulative < top_p.unsqueeze(-1)).sum(dim=-1) + 1
    positions = torch.arange(logits.shape[1], device=logits.device)
    sorted_keep = positions.unsqueeze(0) < counts.unsqueeze(-1)
    keep = torch.zeros_like(sorted_keep)
    return keep.scatter(-1, order, sorted_keep)


@pytest.mark.parametrize("num_rows", [1, 8, 40])
@pytest.mark.parametrize("distribution", ["normal", "peaked", "masked"])
def test_radix_cutoff_matches_stable_top_p(num_rows: int, distribution: str):
    torch.manual_seed(11)
    vocab_size = 4096
    logits = torch.randn(num_rows, vocab_size, device="cuda", dtype=torch.bfloat16)
    if distribution == "peaked":
        logits[:, 0] += 13.0
        logits[:, 1:8] += 8.0
    elif distribution == "masked":
        logits[:, 512:] = -float("inf")
    top_p = torch.linspace(0.8, 0.99, num_rows, device="cuda")
    cutoff = distributed_bf16_top_p_cutoff(
        logits,
        top_p,
        vocab_start=0,
        org_vocab_size=vocab_size,
        tp_group=None,
    )

    keys = _ordered_key(logits)
    token_ids = torch.arange(vocab_size, device="cuda")
    actual = (keys > cutoff.ordered_key.unsqueeze(-1)) | (
        (keys == cutoff.ordered_key.unsqueeze(-1))
        & (token_ids.unsqueeze(0) <= cutoff.last_token_id.unsqueeze(-1))
    )
    expected = _reference_keep_mask(logits, top_p)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    expected_mass = (
        torch.exp(logits.float() - cutoff.global_max.unsqueeze(-1)) * expected
    ).sum(dim=-1)
    torch.testing.assert_close(
        cutoff.retained_mass, expected_mass, rtol=2e-5, atol=2e-5
    )


def test_distributed_candidates_match_full_vocab_gumbel():
    torch.manual_seed(19)
    num_rows = 12
    vocab_size = 4096
    logits = torch.randn(num_rows, vocab_size, device="cuda", dtype=torch.bfloat16)
    top_p = torch.full((num_rows,), 0.95, device="cuda")
    idx_mapping = torch.arange(num_rows, dtype=torch.int32, device="cuda")
    seeds = torch.arange(100, 100 + num_rows, dtype=torch.int64, device="cuda")
    positions = torch.arange(300, 300 + num_rows, dtype=torch.int64, device="cuda")
    excluded = torch.arange(num_rows, dtype=torch.int64, device="cuda") * 7
    cutoff = distributed_bf16_top_p_cutoff(
        logits,
        top_p,
        vocab_start=0,
        org_vocab_size=vocab_size,
        tp_group=None,
    )
    actual = distributed_nucleus_candidates(
        logits,
        cutoff.ordered_key,
        cutoff.last_token_id,
        idx_mapping,
        seeds,
        positions,
        excluded,
        vocab_start=0,
        org_vocab_size=vocab_size,
        tp_group=None,
    )

    keep = _reference_keep_mask(logits, top_p)
    processed = logits.float().masked_fill(~keep, -float("inf"))
    temperature = torch.ones(num_rows, dtype=torch.float32, device="cuda")
    expected = gumbel_sample(
        processed,
        idx_mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=False,
    )
    without_excluded = processed.clone()
    without_excluded.scatter_(1, excluded.unsqueeze(-1), -float("inf"))
    expected_without_excluded = gumbel_sample(
        without_excluded,
        idx_mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=False,
    )
    torch.testing.assert_close(actual.sampled, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.sampled_without_excluded, expected_without_excluded, rtol=0, atol=0
    )


def test_one_hot_rejection_matches_full_vocab_kernel():
    torch.manual_seed(23)
    num_reqs = 8
    num_speculative_steps = 3
    logits_per_req = num_speculative_steps + 1
    num_rows = num_reqs * logits_per_req
    vocab_size = 2048
    logits = torch.randn(num_rows, vocab_size, device="cuda", dtype=torch.bfloat16)
    top_p = torch.full((num_rows,), 0.95, device="cuda")
    cu_num_logits = torch.arange(
        0,
        num_rows + 1,
        logits_per_req,
        dtype=torch.int32,
        device="cuda",
    )
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device="cuda")
    expanded_idx_mapping = idx_mapping.repeat_interleave(logits_per_req)
    expanded_local_pos = torch.arange(
        logits_per_req, dtype=torch.int32, device="cuda"
    ).repeat(num_reqs)
    positions = (
        torch.arange(logits_per_req, dtype=torch.int64, device="cuda")
        .repeat(num_reqs)
        .add(
            torch.arange(num_reqs, device="cuda").repeat_interleave(logits_per_req)
            * 100
        )
    )
    draft_sampled = torch.randint(
        0, vocab_size, (num_rows,), dtype=torch.int32, device="cuda"
    )
    seeds = torch.arange(800, 800 + num_reqs, dtype=torch.int64, device="cuda")
    temperature = torch.ones(num_reqs, dtype=torch.float32, device="cuda")
    cutoff = distributed_bf16_top_p_cutoff(
        logits,
        top_p,
        vocab_start=0,
        org_vocab_size=vocab_size,
        tp_group=None,
    )
    actual = distributed_one_hot_rejection_sample(
        logits,
        cutoff,
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        expanded_idx_mapping,
        seeds,
        positions,
        num_speculative_steps,
        vocab_start=0,
        org_vocab_size=vocab_size,
        tp_group=None,
    )

    keep = _reference_keep_mask(logits, top_p)
    processed = logits.float().masked_fill(~keep, -float("inf"))
    expected_sampled, expected_num_sampled = rejection_sample(
        processed,
        None,
        draft_sampled,
        cu_num_logits,
        positions,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seeds,
        num_speculative_steps,
    )
    torch.testing.assert_close(actual.num_sampled, expected_num_sampled, rtol=0, atol=0)
    steps = torch.arange(num_speculative_steps + 1, device="cuda")
    valid = steps.unsqueeze(0) < expected_num_sampled.unsqueeze(1)
    torch.testing.assert_close(
        actual.sampled[valid], expected_sampled[valid], rtol=0, atol=0
    )
