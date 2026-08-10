# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.distributed import get_tp_group
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.sample.thinking_budget import apply_thinking_budget
from vllm.v1.worker.gpu.sample.vocab_parallel_nucleus import (
    distributed_bf16_top_p_cutoff,
    distributed_one_hot_rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

# Cap on the FP32 target-logits buffer materialized by apply_sampling_params.
# TODO(mgoin): Chunking is a workaround. The rejection kernels already upcast
# per vocab block on load and apply ops like temperature and gumbel, so folding
# sampling-param application into those kernels would remove this buffer and
# its traffic entirely.
MAX_CHUNK_BYTES = 2**30  # 1GB
_FP32_BYTES = 4


def _iter_request_chunks(
    cu_num_logits: np.ndarray, max_chunk_logits: int
) -> Iterator[tuple[int, int]]:
    """Yield maximally packed request ranges without splitting requests."""
    assert max_chunk_logits > 0
    num_reqs = cu_num_logits.size - 1
    start = 0
    while start < num_reqs:
        max_logit = int(cu_num_logits[start]) + max_chunk_logits
        end = int(np.searchsorted(cu_num_logits, max_logit, side="right") - 1)
        end = min(num_reqs, max(start + 1, end))
        yield start, end
        start = end


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    def can_vocab_parallel_nucleus(
        self,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
    ) -> bool:
        if (
            draft_logits is not None
            or self.use_block_verification
            or self.synthetic_conditional_rates is not None
            or self.sampler.use_fp64_gumbel
            or self.sampler.compute_nans
        ):
            return False
        idx = input_batch.idx_mapping_np
        states = self.sampler.sampling_states
        if states.max_num_logprobs(idx) != NO_LOGPROBS:
            return False
        if self.sampler.logprob_token_ids_state.max_num_token_ids(idx) > 0:
            return False
        if not np.all(states.temperature.np[idx] == 1.0):
            return False
        top_p = states.top_p.np[idx]
        if not np.all((top_p > 0.0) & (top_p < 1.0)):
            return False
        if not np.all(states.top_k.np[idx] == states.vocab_size):
            return False
        if not np.all(states.min_p.np[idx] == 0.0):
            return False
        if np.any(self.sampler.logit_bias_state.use_logit_bias[idx]):
            return False
        if np.any(self.sampler.penalties_state.use_penalty[idx]):
            return False
        return not np.any(self.sampler.bad_words_state.num_bad_words.np[idx] > 0)

    def vocab_parallel_nucleus(
        self,
        local_logits: torch.Tensor,
        vocab_start: int,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        if local_logits.dtype != torch.bfloat16:
            raise ValueError("Distributed nucleus sampling requires BF16 logits.")

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        self.sampler.thinking_budget_state.apply(
            local_logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            draft_sampled,
            input_batch.expanded_local_pos,
            vocab_start=vocab_start,
        )
        states = self.sampler.sampling_states
        top_p = states.top_p.gpu[input_batch.expanded_idx_mapping]
        tp_group = get_tp_group()
        cutoff = distributed_bf16_top_p_cutoff(
            local_logits,
            top_p,
            vocab_start=vocab_start,
            org_vocab_size=states.vocab_size,
            tp_group=tp_group,
        )
        output = distributed_one_hot_rejection_sample(
            local_logits,
            cutoff,
            draft_sampled,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            states.seeds.gpu,
            pos,
            self.num_speculative_steps,
            vocab_start=vocab_start,
            org_vocab_size=states.vocab_size,
            tp_group=tp_group,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            output.num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=output.sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def forward_vocab_parallel_graph(
        self,
        local_logits: torch.Tensor,
        vocab_start: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        top_p: torch.Tensor,
        seeds: torch.Tensor,
    ) -> SamplerOutput:
        if local_logits.dtype != torch.bfloat16:
            raise ValueError("Distributed nucleus sampling requires BF16 logits.")

        draft_sampled = input_ids[logits_indices]
        pos = positions[logits_indices]
        thinking_budget = self.sampler.thinking_budget_state
        if thinking_budget.enabled:
            apply_thinking_budget(
                local_logits,
                idx_mapping,
                expanded_idx_mapping,
                thinking_budget.thinking_token_budget.gpu,
                thinking_budget.req_states.all_token_ids.gpu,
                thinking_budget.req_states.total_len.gpu,
                draft_sampled,
                expanded_local_pos,
                thinking_budget.cached_last_start,
                thinking_budget.cached_last_end,
                thinking_budget.cached_scan_pos,
                thinking_budget.reasoning_start_token_ids,
                thinking_budget.natural_reasoning_end_token_ids,
                thinking_budget.reasoning_end_token_ids,
                vocab_start,
            )

        states = self.sampler.sampling_states
        cutoff = distributed_bf16_top_p_cutoff(
            local_logits,
            top_p[expanded_idx_mapping],
            vocab_start=vocab_start,
            org_vocab_size=states.vocab_size,
            tp_group=get_tp_group(),
        )
        output = distributed_one_hot_rejection_sample(
            local_logits,
            cutoff,
            draft_sampled,
            cu_num_logits,
            idx_mapping,
            expanded_idx_mapping,
            seeds,
            pos,
            self.num_speculative_steps,
            vocab_start=vocab_start,
            org_vocab_size=states.vocab_size,
            tp_group=get_tp_group(),
        )
        return SamplerOutput(
            sampled_token_ids=output.sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=output.num_sampled,
        )

    def _get_logprobs_tensors(
        self,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
        cu_num_logits: torch.Tensor,
        cu_num_logits_np: np.ndarray,
        max_num_logprobs: int,
    ) -> LogprobsTensors | None:
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != num_reqs
        return compute_topk_scores(
            logits,
            max_num_logprobs,
            flat_sampled,
            cu_num_logits_np.tolist() if expanded_logits else None,
            logits_mode=self.sampler.logprobs_mode
            in ("raw_logits", "processed_logits"),
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        return processed_logits, sampled, num_sampled

    def _verify_in_chunks(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        max_chunk_logits: int,
        max_num_logprobs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, LogprobsTensors | None]:
        cu_num_logits_np = input_batch.cu_num_logits_np
        use_processed_logits = self.sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        sampled_chunks: list[torch.Tensor] = []
        num_sampled_chunks: list[torch.Tensor] = []
        logprobs_chunks: list[LogprobsTensors] = []

        for start, end in _iter_request_chunks(cu_num_logits_np, max_chunk_logits):
            lo = int(cu_num_logits_np[start])
            hi = int(cu_num_logits_np[end])
            chunk_cu_num_logits_np = cu_num_logits_np[start : end + 1] - lo
            chunk_cu_num_logits = input_batch.cu_num_logits[start : end + 1] - lo
            # draft_logits uses persistent request-state indices and stays global.
            processed_logits, sampled, num_sampled = self._verify(
                logits[lo:hi],
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                chunk_cu_num_logits,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                input_batch.expanded_idx_mapping[lo:hi],
                input_batch.expanded_local_pos[lo:hi],
            )
            chunk_logprobs = self._get_logprobs_tensors(
                sampled,
                num_sampled,
                processed_logits if use_processed_logits else logits[lo:hi],
                chunk_cu_num_logits,
                chunk_cu_num_logits_np,
                max_num_logprobs,
            )
            if chunk_logprobs is not None:
                logprobs_chunks.append(chunk_logprobs)
            del processed_logits
            sampled_chunks.append(sampled)
            num_sampled_chunks.append(num_sampled)

        if len(sampled_chunks) == 1:
            logprobs_tensors = logprobs_chunks[0] if logprobs_chunks else None
            return sampled_chunks[0], num_sampled_chunks[0], logprobs_tensors

        logprobs_tensors = None
        if logprobs_chunks:
            expanded_logits = logits.shape[0] != input_batch.num_reqs
            logprobs_tensors = LogprobsTensors.cat(
                logprobs_chunks,
                cu_num_generated_tokens=(
                    cu_num_logits_np.tolist() if expanded_logits else None
                ),
            )

        sampled = torch.cat(sampled_chunks)
        num_sampled = torch.cat(num_sampled_chunks)
        return sampled, num_sampled, logprobs_tensors

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]

        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        max_chunk_logits = max(1, MAX_CHUNK_BYTES // (logits.shape[1] * _FP32_BYTES))
        sampled, num_sampled, logprobs_tensors = self._verify_in_chunks(
            logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            max_chunk_logits,
            max_num_logprobs,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def forward_graph(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        temperature: torch.Tensor,
        min_p: torch.Tensor,
        top_p: torch.Tensor,
        seeds: torch.Tensor,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_ids[logits_indices]
        draft_positions = positions[logits_indices]

        processed_logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        thinking_budget = self.sampler.thinking_budget_state
        if thinking_budget.enabled:
            apply_thinking_budget(
                processed_logits,
                idx_mapping,
                expanded_idx_mapping,
                thinking_budget.thinking_token_budget.gpu,
                thinking_budget.req_states.all_token_ids.gpu,
                thinking_budget.req_states.total_len.gpu,
                draft_sampled,
                expanded_local_pos,
                thinking_budget.cached_last_start,
                thinking_budget.cached_last_end,
                thinking_budget.cached_scan_pos,
                thinking_budget.reasoning_start_token_ids,
                thinking_budget.natural_reasoning_end_token_ids,
                thinking_budget.reasoning_end_token_ids,
            )
        apply_temperature(processed_logits, expanded_idx_mapping, temperature)
        apply_min_p(processed_logits, expanded_idx_mapping, min_p)
        processed_logits = apply_top_k_top_p(
            processed_logits,
            None,
            top_p[expanded_idx_mapping],
        )

        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            draft_positions,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seeds,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            # Output logprobs are not supported during cudagraph capture.
            logprobs_tensors=None,
            num_nans=num_nans,
            num_sampled=num_sampled,
        )
