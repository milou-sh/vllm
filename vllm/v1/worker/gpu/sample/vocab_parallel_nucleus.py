# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Protocol

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32

_NUM_RADIX_BINS = 256


class TensorParallelGroup(Protocol):
    world_size: int
    rank_in_group: int

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...


@dataclass(frozen=True)
class NucleusCutoff:
    ordered_key: torch.Tensor
    last_token_id: torch.Tensor
    global_max: torch.Tensor
    retained_mass: torch.Tensor


@dataclass(frozen=True)
class NucleusSamples:
    sampled: torch.Tensor
    sampled_without_excluded: torch.Tensor
    excluded_logit: torch.Tensor


@dataclass(frozen=True)
class OneHotRejectionOutput:
    sampled: torch.Tensor
    num_sampled: torch.Tensor


@triton.jit
def _ordered_bf16_key(value):
    bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
    return tl.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)


@triton.jit
def _radix_high_histogram_kernel(
    logits_ptr,
    logits_stride,
    histogram_ptr,
    histogram_stride,
    local_max_ptr,
    vocab_size,
    vocab_start,
    org_vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < vocab_size) & (offsets + vocab_start < org_vocab_size)
    logits = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    keys = _ordered_bf16_key(logits)
    weights = tl.exp(logits.to(tl.float32) - tl.load(local_max_ptr + row))
    tl.atomic_add(
        histogram_ptr + row * histogram_stride + (keys >> 8),
        weights,
        mask=mask,
    )


@triton.jit
def _select_high_radix_kernel(
    histogram_ptr,
    histogram_stride,
    top_p_ptr,
    high_key_ptr,
    mass_above_ptr,
    total_mass_ptr,
    NUM_BINS: tl.constexpr,
):
    row = tl.program_id(0)
    descending = NUM_BINS - 1 - tl.arange(0, NUM_BINS)
    mass = tl.load(histogram_ptr + row * histogram_stride + descending)
    cumulative = tl.cumsum(mass)
    total = tl.sum(mass)
    selected = tl.argmax(
        (cumulative >= tl.load(top_p_ptr + row) * total).to(tl.int32), axis=0
    )
    high_key = NUM_BINS - 1 - selected
    tl.store(high_key_ptr + row, high_key)
    tl.store(mass_above_ptr + row, tl.sum(tl.where(descending > high_key, mass, 0.0)))
    tl.store(total_mass_ptr + row, total)


@triton.jit
def _radix_low_histogram_kernel(
    logits_ptr,
    logits_stride,
    histogram_ptr,
    histogram_stride,
    local_max_ptr,
    high_key_ptr,
    vocab_size,
    vocab_start,
    org_vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    row = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < vocab_size) & (offsets + vocab_start < org_vocab_size)
    logits = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    keys = _ordered_bf16_key(logits)
    mask &= (keys >> 8) == tl.load(high_key_ptr + row)
    weights = tl.exp(logits.to(tl.float32) - tl.load(local_max_ptr + row))
    tl.atomic_add(
        histogram_ptr + row * histogram_stride + (keys & 0xFF),
        weights,
        mask=mask,
    )
    tl.atomic_add(
        histogram_ptr + row * histogram_stride + NUM_BINS + (keys & 0xFF),
        1.0,
        mask=mask,
    )


@triton.jit
def _select_low_radix_kernel(
    histogram_ptr,
    histogram_stride,
    top_p_ptr,
    high_key_ptr,
    mass_above_ptr,
    total_mass_ptr,
    cutoff_ptr,
    cutoff_keep_count_ptr,
    retained_mass_ptr,
    NUM_BINS: tl.constexpr,
):
    row = tl.program_id(0)
    descending = NUM_BINS - 1 - tl.arange(0, NUM_BINS)
    mass = tl.load(histogram_ptr + row * histogram_stride + descending)
    cumulative = tl.cumsum(mass)
    target = tl.load(top_p_ptr + row) * tl.load(total_mass_ptr + row) - tl.load(
        mass_above_ptr + row
    )
    selected = tl.argmax((cumulative >= target).to(tl.int32), axis=0)
    low_key = NUM_BINS - 1 - selected
    high_key = tl.load(high_key_ptr + row)
    tl.store(cutoff_ptr + row, (high_key << 8) | low_key)
    cutoff_count = tl.load(histogram_ptr + row * histogram_stride + NUM_BINS + low_key)
    cutoff_mass = tl.load(histogram_ptr + row * histogram_stride + low_key)
    unit_mass = cutoff_mass / cutoff_count
    keep_count = tl.ceil(target / unit_mass).to(tl.int32)
    keep_count = tl.maximum(1, tl.minimum(keep_count, cutoff_count.to(tl.int32)))
    tl.store(cutoff_keep_count_ptr + row, keep_count)
    tl.store(
        retained_mass_ptr + row,
        tl.load(mass_above_ptr + row) + keep_count * unit_mass,
    )


@triton.jit
def _count_cutoff_blocks_kernel(
    logits_ptr,
    logits_stride,
    cutoff_ptr,
    counts_ptr,
    counts_stride,
    vocab_size,
    vocab_start,
    org_vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < vocab_size) & (offsets + vocab_start < org_vocab_size)
    logits = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    count = tl.sum(mask & (_ordered_bf16_key(logits) == tl.load(cutoff_ptr + row)))
    tl.store(counts_ptr + row * counts_stride + block_idx, count)


@triton.jit
def _select_cutoff_token_kernel(
    logits_ptr,
    logits_stride,
    cutoff_ptr,
    block_idx_ptr,
    ordinal_ptr,
    is_owner_ptr,
    output_ptr,
    vocab_size,
    vocab_start,
    org_vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.load(block_idx_ptr + row)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    global_tokens = offsets + vocab_start
    mask = (offsets < vocab_size) & (global_tokens < org_vocab_size)
    logits = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    matches = mask & (_ordered_bf16_key(logits) == tl.load(cutoff_ptr + row))
    selected = tl.argmax(
        (tl.cumsum(matches) >= tl.load(ordinal_ptr + row)).to(tl.int32), axis=0
    )
    token = block_idx * BLOCK_SIZE + selected + vocab_start
    tl.store(
        output_ptr + row,
        token,
        mask=tl.load(is_owner_ptr + row),
    )


@triton.jit
def _local_nucleus_candidates_kernel(
    logits_ptr,
    logits_stride,
    cutoff_ptr,
    last_cutoff_token_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    positions_ptr,
    excluded_token_ptr,
    local_value_ptr,
    local_value_stride,
    local_token_ptr,
    local_token_stride,
    excluded_value_ptr,
    excluded_value_stride,
    excluded_candidate_ptr,
    excluded_candidate_stride,
    vocab_size,
    vocab_start,
    org_vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    global_tokens = offsets + vocab_start
    mask = (offsets < vocab_size) & (global_tokens < org_vocab_size)
    logits = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    keys = _ordered_bf16_key(logits)
    cutoff = tl.load(cutoff_ptr + row)
    mask &= (keys > cutoff) | (
        (keys == cutoff) & (global_tokens <= tl.load(last_cutoff_token_ptr + row))
    )

    req_state_idx = tl.load(expanded_idx_mapping_ptr + row)
    seed = tl.load(seeds_ptr + req_state_idx)
    position = tl.load(positions_ptr + row)
    gumbel_seed = tl.randint(seed, position)
    uniform = tl_rand32(gumbel_seed, global_tokens, includes_zero=False)
    noise = -tl.log(-tldevice.log1p(-uniform))
    noisy_logits = tl.where(mask, logits.to(tl.float32) + noise, -float("inf"))

    value, index = tl.max(noisy_logits, axis=0, return_indices=True)
    token = block_idx * BLOCK_SIZE + index + vocab_start
    tl.store(local_value_ptr + row * local_value_stride + block_idx, value)
    tl.store(local_token_ptr + row * local_token_stride + block_idx, token)

    excluded_token = tl.load(excluded_token_ptr + row)
    without_excluded = tl.where(
        mask & (global_tokens != excluded_token), noisy_logits, -float("inf")
    )
    excluded_value, excluded_index = tl.max(
        without_excluded, axis=0, return_indices=True
    )
    excluded_candidate = block_idx * BLOCK_SIZE + excluded_index + vocab_start
    tl.store(
        excluded_value_ptr + row * excluded_value_stride + block_idx,
        excluded_value,
    )
    tl.store(
        excluded_candidate_ptr + row * excluded_candidate_stride + block_idx,
        excluded_candidate,
    )


@triton.jit
def _build_excluded_tokens_kernel(
    excluded_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start = tl.load(cu_num_logits_ptr + req_idx)
    end = tl.load(cu_num_logits_ptr + req_idx + 1)
    for row in tl.range(start, end):
        tl.store(
            excluded_ptr + row,
            tl.load(draft_sampled_ptr + row + 1, mask=row + 1 < end, other=-1),
        )


@triton.jit
def _lookup_local_excluded_logits_kernel(
    logits_ptr,
    logits_stride,
    cutoff_ptr,
    last_cutoff_token_ptr,
    excluded_ptr,
    output_ptr,
    vocab_size,
    vocab_start,
):
    row = tl.program_id(0)
    token = tl.load(excluded_ptr + row)
    local_token = token - vocab_start
    valid = (local_token >= 0) & (local_token < vocab_size)
    logit = tl.load(
        logits_ptr + row * logits_stride + local_token,
        mask=valid,
        other=-float("inf"),
    )
    key = _ordered_bf16_key(logit)
    cutoff = tl.load(cutoff_ptr + row)
    valid &= (key > cutoff) | (
        (key == cutoff) & (token <= tl.load(last_cutoff_token_ptr + row))
    )
    tl.store(output_ptr + row, tl.where(valid, logit, -float("inf")))


@triton.jit
def _one_hot_rejection_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    draft_probs_ptr,
    draft_sampled_ptr,
    regular_samples_ptr,
    excluded_samples_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    seeds_ptr,
    positions_ptr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    seed = tl.load(seeds_ptr + req_state_idx)
    start = tl.load(cu_num_logits_ptr + req_idx)
    end = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end - start - 1
    accepted = tl.zeros((), tl.int32)
    verifying = True
    for i in tl.range(num_draft_tokens):
        row = start + i
        draft_token = tl.load(draft_sampled_ptr + row + 1)
        uniform = tl_rand32(seed, tl.load(positions_ptr + row), includes_zero=False)
        take_draft = (draft_token >= 0) & (uniform < tl.load(draft_probs_ptr + row))
        if verifying:
            tl.store(
                sampled_ptr + req_idx * sampled_stride + i,
                tl.where(take_draft, draft_token, tl.load(excluded_samples_ptr + row)),
            )
            accepted += take_draft
        verifying &= take_draft

    if verifying:
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_draft_tokens,
            tl.load(regular_samples_ptr + end - 1),
        )
        tl.store(num_sampled_ptr + req_idx, num_draft_tokens + 1)
    else:
        tl.store(num_sampled_ptr + req_idx, accepted + 1)


def _gather_high_histogram(
    local_histogram: torch.Tensor,
    local_max: torch.Tensor,
    tp_group: TensorParallelGroup | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tp_group is None or tp_group.world_size == 1:
        return local_histogram, local_max, local_max.unsqueeze(0)
    num_rows = local_histogram.shape[0]
    packed = torch.cat((local_max.unsqueeze(-1), local_histogram), dim=-1)
    gathered = tp_group.all_gather(packed, dim=0).view(
        tp_group.world_size, num_rows, packed.shape[-1]
    )
    rank_max = gathered[:, :, 0]
    global_max = rank_max.amax(dim=0)
    scale = torch.exp(rank_max - global_max.unsqueeze(0)).unsqueeze(-1)
    histogram = (gathered[:, :, 1:] * scale).sum(dim=0)
    return histogram, global_max, rank_max


def _gather_low_histogram(
    local_histogram: torch.Tensor,
    rank_max: torch.Tensor,
    global_max: torch.Tensor,
    tp_group: TensorParallelGroup | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tp_group is None or tp_group.world_size == 1:
        rank_histogram = local_histogram.unsqueeze(0)
    else:
        rank_histogram = tp_group.all_gather(local_histogram, dim=0).view(
            tp_group.world_size, local_histogram.shape[0], local_histogram.shape[1]
        )
    scale = torch.exp(rank_max - global_max.unsqueeze(0)).unsqueeze(-1)
    mass = (rank_histogram[:, :, :_NUM_RADIX_BINS] * scale).sum(dim=0)
    counts = rank_histogram[:, :, _NUM_RADIX_BINS:].sum(dim=0)
    return torch.cat((mass, counts), dim=-1), rank_histogram


def _last_cutoff_token(
    local_logits: torch.Tensor,
    cutoff: torch.Tensor,
    keep_count: torch.Tensor,
    rank_counts: torch.Tensor,
    *,
    vocab_start: int,
    org_vocab_size: int,
    tp_group: TensorParallelGroup | None,
) -> torch.Tensor:
    num_rows, local_vocab_size = local_logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(local_vocab_size, block_size)
    block_counts = torch.empty(
        (num_rows, num_blocks), dtype=torch.int32, device=local_logits.device
    )
    _count_cutoff_blocks_kernel[(num_rows, num_blocks)](
        local_logits,
        local_logits.stride(0),
        cutoff,
        block_counts,
        block_counts.stride(0),
        local_vocab_size,
        vocab_start,
        org_vocab_size,
        BLOCK_SIZE=block_size,
    )
    rank = 0 if tp_group is None or tp_group.world_size == 1 else tp_group.rank_in_group
    rank_prefix = rank_counts.cumsum(dim=0)
    owner = (rank_prefix >= keep_count.unsqueeze(0)).to(torch.int32).argmax(dim=0)
    owner_before_idx = (owner - 1).clamp_min(0).unsqueeze(0)
    owner_before = rank_prefix.gather(0, owner_before_idx).squeeze(0)
    owner_before = torch.where(owner == 0, 0, owner_before)
    local_ordinal = keep_count - owner_before
    is_owner = owner == rank

    block_prefix = block_counts.cumsum(dim=-1)
    block_idx = (
        (block_prefix >= local_ordinal.unsqueeze(-1)).to(torch.int32).argmax(dim=-1)
    )
    block_before_idx = (block_idx - 1).clamp_min(0).unsqueeze(-1)
    block_before = block_prefix.gather(-1, block_before_idx).squeeze(-1)
    block_before = torch.where(block_idx == 0, 0, block_before)
    ordinal_in_block = local_ordinal - block_before

    local_token = torch.full(
        (num_rows,), org_vocab_size, dtype=torch.int64, device=local_logits.device
    )
    _select_cutoff_token_kernel[(num_rows,)](
        local_logits,
        local_logits.stride(0),
        cutoff,
        block_idx,
        ordinal_in_block,
        is_owner,
        local_token,
        local_vocab_size,
        vocab_start,
        org_vocab_size,
        BLOCK_SIZE=block_size,
    )
    if tp_group is None or tp_group.world_size == 1:
        return local_token
    gathered = tp_group.all_gather(local_token, dim=0)
    return gathered.view(tp_group.world_size, num_rows).amin(dim=0)


def distributed_bf16_top_p_cutoff(
    local_logits: torch.Tensor,
    top_p: torch.Tensor,
    *,
    vocab_start: int,
    org_vocab_size: int,
    tp_group: TensorParallelGroup | None,
) -> NucleusCutoff:
    if local_logits.dtype != torch.bfloat16:
        raise ValueError("The radix nucleus path requires BF16 logits.")
    if local_logits.ndim != 2 or local_logits.stride(-1) != 1:
        raise ValueError("Expected contiguous [num_tokens, local_vocab] logits.")
    if top_p.shape != (local_logits.shape[0],):
        raise ValueError("top_p must contain one value per logits row.")

    num_rows, local_vocab_size = local_logits.shape
    local_max = local_logits.amax(dim=-1).float()
    histogram = torch.zeros(
        (num_rows, _NUM_RADIX_BINS), dtype=torch.float32, device=local_logits.device
    )
    block_size = 256
    grid = (num_rows, triton.cdiv(local_vocab_size, block_size))
    _radix_high_histogram_kernel[grid](
        local_logits,
        local_logits.stride(0),
        histogram,
        histogram.stride(0),
        local_max,
        local_vocab_size,
        vocab_start,
        org_vocab_size,
        BLOCK_SIZE=block_size,
    )
    histogram, global_max, rank_max = _gather_high_histogram(
        histogram, local_max, tp_group
    )

    high_key = torch.empty(num_rows, dtype=torch.int32, device=local_logits.device)
    mass_above = torch.empty(num_rows, dtype=torch.float32, device=local_logits.device)
    total_mass = torch.empty_like(mass_above)
    _select_high_radix_kernel[(num_rows,)](
        histogram,
        histogram.stride(0),
        top_p,
        high_key,
        mass_above,
        total_mass,
        NUM_BINS=_NUM_RADIX_BINS,
        num_warps=8,
    )

    histogram = torch.zeros(
        (num_rows, _NUM_RADIX_BINS * 2),
        dtype=torch.float32,
        device=local_logits.device,
    )
    _radix_low_histogram_kernel[grid](
        local_logits,
        local_logits.stride(0),
        histogram,
        histogram.stride(0),
        local_max,
        high_key,
        local_vocab_size,
        vocab_start,
        org_vocab_size,
        BLOCK_SIZE=block_size,
        NUM_BINS=_NUM_RADIX_BINS,
    )
    histogram, rank_histogram = _gather_low_histogram(
        histogram, rank_max, global_max, tp_group
    )

    cutoff = torch.empty_like(high_key)
    cutoff_keep_count = torch.empty_like(high_key)
    retained_mass = torch.empty_like(total_mass)
    _select_low_radix_kernel[(num_rows,)](
        histogram,
        histogram.stride(0),
        top_p,
        high_key,
        mass_above,
        total_mass,
        cutoff,
        cutoff_keep_count,
        retained_mass,
        NUM_BINS=_NUM_RADIX_BINS,
        num_warps=8,
    )
    low_key = (cutoff & 0xFF).long()
    rank_counts = (
        rank_histogram[:, :, _NUM_RADIX_BINS:]
        .gather(
            2,
            low_key.view(1, num_rows, 1).expand(rank_histogram.shape[0], -1, -1),
        )
        .squeeze(-1)
        .to(torch.int32)
    )
    last_token = _last_cutoff_token(
        local_logits,
        cutoff,
        cutoff_keep_count,
        rank_counts,
        vocab_start=vocab_start,
        org_vocab_size=org_vocab_size,
        tp_group=tp_group,
    )
    return NucleusCutoff(cutoff, last_token, global_max, retained_mass)


def _reduce_local_candidates(
    values: torch.Tensor,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block = values.argmax(dim=-1, keepdim=True)
    return values.gather(-1, block).squeeze(-1), tokens.gather(-1, block).squeeze(-1)


def distributed_nucleus_candidates(
    local_logits: torch.Tensor,
    cutoff: torch.Tensor,
    last_cutoff_token: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    excluded_tokens: torch.Tensor,
    *,
    vocab_start: int,
    org_vocab_size: int,
    tp_group: TensorParallelGroup | None,
) -> NucleusSamples:
    num_rows, local_vocab_size = local_logits.shape
    if cutoff.shape != (num_rows,) or excluded_tokens.shape != (num_rows,):
        raise ValueError("Expected one cutoff and excluded token per logits row.")

    block_size = 1024
    num_blocks = triton.cdiv(local_vocab_size, block_size)
    values = torch.empty(
        (num_rows, num_blocks), dtype=torch.float32, device=local_logits.device
    )
    tokens = torch.empty(
        (num_rows, num_blocks), dtype=torch.int64, device=local_logits.device
    )
    excluded_values = torch.empty_like(values)
    excluded_candidates = torch.empty_like(tokens)
    _local_nucleus_candidates_kernel[(num_rows, num_blocks)](
        local_logits,
        local_logits.stride(0),
        cutoff,
        last_cutoff_token,
        expanded_idx_mapping,
        seeds,
        positions,
        excluded_tokens,
        values,
        values.stride(0),
        tokens,
        tokens.stride(0),
        excluded_values,
        excluded_values.stride(0),
        excluded_candidates,
        excluded_candidates.stride(0),
        local_vocab_size,
        vocab_start,
        org_vocab_size,
        BLOCK_SIZE=block_size,
    )
    value, token = _reduce_local_candidates(values, tokens)
    excluded_value, excluded_token = _reduce_local_candidates(
        excluded_values, excluded_candidates
    )
    local_excluded_logits = torch.empty(
        num_rows, dtype=torch.float32, device=local_logits.device
    )
    _lookup_local_excluded_logits_kernel[(num_rows,)](
        local_logits,
        local_logits.stride(0),
        cutoff,
        last_cutoff_token,
        excluded_tokens,
        local_excluded_logits,
        local_vocab_size,
        vocab_start,
    )

    if tp_group is None or tp_group.world_size == 1:
        return NucleusSamples(token, excluded_token, local_excluded_logits)

    packed = torch.stack(
        (
            value,
            token.float(),
            excluded_value,
            excluded_token.float(),
            local_excluded_logits,
        ),
        dim=-1,
    )
    gathered = tp_group.all_gather(packed, dim=0).view(tp_group.world_size, num_rows, 5)
    best_rank = gathered[:, :, 0].argmax(dim=0, keepdim=True)
    sampled = gathered[:, :, 1].gather(0, best_rank).squeeze(0).long()
    best_excluded_rank = gathered[:, :, 2].argmax(dim=0, keepdim=True)
    sampled_without_excluded = (
        gathered[:, :, 3].gather(0, best_excluded_rank).squeeze(0).long()
    )
    excluded_logits = gathered[:, :, 4].amax(dim=0)
    return NucleusSamples(sampled, sampled_without_excluded, excluded_logits)


def distributed_one_hot_rejection_sample(
    local_logits: torch.Tensor,
    cutoff: NucleusCutoff,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    num_speculative_steps: int,
    *,
    vocab_start: int,
    org_vocab_size: int,
    tp_group: TensorParallelGroup | None,
) -> OneHotRejectionOutput:
    num_rows = local_logits.shape[0]
    num_reqs = cu_num_logits.shape[0] - 1
    excluded = torch.empty(num_rows, dtype=torch.int64, device=local_logits.device)
    _build_excluded_tokens_kernel[(num_reqs,)](
        excluded,
        draft_sampled,
        cu_num_logits,
        num_warps=1,
    )
    candidates = distributed_nucleus_candidates(
        local_logits,
        cutoff.ordered_key,
        cutoff.last_token_id,
        expanded_idx_mapping,
        seeds,
        positions,
        excluded,
        vocab_start=vocab_start,
        org_vocab_size=org_vocab_size,
        tp_group=tp_group,
    )

    draft_probs = (
        torch.exp(candidates.excluded_logit - cutoff.global_max) / cutoff.retained_mass
    )

    sampled = torch.full(
        (num_reqs, num_speculative_steps + 1),
        -1,
        dtype=torch.int64,
        device=local_logits.device,
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=local_logits.device)
    _one_hot_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        draft_probs,
        draft_sampled,
        candidates.sampled,
        candidates.sampled_without_excluded,
        cu_num_logits,
        idx_mapping,
        seeds,
        positions,
        num_warps=1,
    )
    return OneHotRejectionOutput(sampled, num_sampled)
