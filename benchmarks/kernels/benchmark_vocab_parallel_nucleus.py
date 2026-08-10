# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
from vllm.distributed.parallel_state import (
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.vocab_parallel_nucleus import (
    distributed_bf16_top_p_cutoff,
    distributed_one_hot_rejection_sample,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


def _time_ms(fn, warmup: int, iterations: int) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    cuda_timings = []
    wall_timings = []
    for _ in range(iterations):
        dist.barrier()
        torch.accelerator.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall_timings.append((time.perf_counter() - wall_start) * 1000)
        cuda_timings.append(start.elapsed_time(end))
    return cuda_timings, wall_timings


def _capture_cuda_graph(
    fn, device: torch.device
) -> tuple[torch.cuda.CUDAGraph, object]:
    with graph_capture(device=device):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn()
    return graph, output


def _ordered_bf16_key(logits: torch.Tensor) -> torch.Tensor:
    bits = logits.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)


def _stable_top_p_mask(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)
    probs = logits.float().softmax(dim=-1)
    cumulative = probs.gather(-1, order).cumsum(dim=-1)
    counts = (cumulative < top_p.unsqueeze(-1)).sum(dim=-1) + 1
    positions = torch.arange(logits.shape[-1], device=logits.device)
    sorted_keep = positions.unsqueeze(0) < counts.unsqueeze(-1)
    return torch.zeros_like(sorted_keep).scatter(-1, order, sorted_keep)


def _make_logits(args, num_rows: int, device: torch.device) -> torch.Tensor:
    if args.distribution == "captured":
        if args.logits_file is None:
            raise ValueError("--logits-file is required for captured logits")
        captured = torch.load(args.logits_file, map_location="cpu", weights_only=True)
        if isinstance(captured, dict):
            captured = captured.get("logits")
        if not isinstance(captured, torch.Tensor) or captured.ndim != 2:
            raise ValueError("captured logits must be a tensor shaped [rows, vocab]")
        if captured.shape[0] < num_rows or captured.shape[1] != args.vocab_size:
            raise ValueError(
                f"captured logits shape {tuple(captured.shape)} cannot provide "
                f"({num_rows}, {args.vocab_size})"
            )
        return captured[:num_rows].to(device=device, dtype=torch.bfloat16)

    logits = torch.randn(num_rows, args.vocab_size, dtype=torch.bfloat16, device=device)
    if args.distribution == "peaked":
        logits[:, 0] += 13.0
        logits[:, 1:8] += 8.0
    elif args.distribution == "masked":
        logits[:, args.valid_vocab :] = -float("inf")
    return logits


def _load_tensor(
    path: str,
    key: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        value = value.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} does not contain tensor key {key!r}")
    return value.to(device=device, dtype=dtype)


def _make_fp8_lm_head_runner(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_type: str,
):
    if weight_scale_type == "tensor":
        weight_fp8, weight_scale = ops.scaled_fp8_quant(weight)
    else:
        weight_fp8, weight_scale = ops.scaled_fp8_quant(
            weight, use_per_token_if_dynamic=True
        )
        weight_scale = weight_scale.t().contiguous()
    weight_fp8 = weight_fp8.t().contiguous()

    def run():
        hidden_fp8, hidden_scale = ops.scaled_fp8_quant(
            hidden_states, use_per_token_if_dynamic=True
        )
        return ops.cutlass_scaled_mm(
            hidden_fp8,
            weight_fp8,
            hidden_scale,
            weight_scale,
            torch.bfloat16,
        )

    return run


def _fp8_lm_head_quality(
    reference_local: torch.Tensor,
    candidate_local: torch.Tensor,
    top_p: torch.Tensor,
    tp_group,
) -> dict[str, float]:
    reference = tp_group.all_gather(reference_local, dim=-1).float()
    candidate = tp_group.all_gather(candidate_local, dim=-1).float()
    reference_log_probs = reference.log_softmax(dim=-1)
    candidate_log_probs = candidate.log_softmax(dim=-1)
    reference_probs = reference_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    reference_keep = _stable_top_p_mask(reference, top_p)
    candidate_keep = _stable_top_p_mask(candidate, top_p)
    intersection = (reference_keep & candidate_keep).sum(dim=-1)
    union = (reference_keep | candidate_keep).sum(dim=-1)
    relative_l2 = (candidate - reference).norm(dim=-1) / reference.norm(dim=-1)
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    total_variation = 0.5 * (reference_probs - candidate_probs).abs().sum(dim=-1)
    kl_divergence = (reference_probs * (reference_log_probs - candidate_log_probs)).sum(
        dim=-1
    )
    return {
        "argmax_agreement": float(
            (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean()
        ),
        "cosine_mean": float(cosine.mean()),
        "kl_mean": float(kl_divergence.mean()),
        "max_abs_error": float((candidate - reference).abs().amax()),
        "relative_l2_mean": float(relative_l2.mean()),
        "top_p_jaccard_mean": float((intersection / union).float().mean()),
        "top_p_rows_exact_fraction": float(
            (reference_keep == candidate_keep).all(dim=-1).float().mean()
        ),
        "total_variation_mean": float(total_variation.mean()),
        "total_variation_max": float(total_variation.amax()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--speculative-steps", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--model-hidden-size", type=int, default=6144)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--distribution",
        choices=("normal", "peaked", "masked", "captured"),
        default="normal",
    )
    parser.add_argument("--valid-vocab", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--logits-file")
    parser.add_argument("--hidden-states-file")
    parser.add_argument("--lm-head-weight-file")
    parser.add_argument("--measure-lm-head", action="store_true")
    parser.add_argument("--measure-fp8-lm-head", action="store_true")
    parser.add_argument("--measure-cudagraph", action="store_true")
    parser.add_argument("--require-output-equality", action="store_true")
    args = parser.parse_args()
    if args.distribution == "masked" and not 0 < args.valid_vocab <= args.vocab_size:
        raise ValueError("valid-vocab must be in [1, vocab-size]")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    set_custom_all_reduce(False)
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    tp_group = get_tp_group()
    rank = tp_group.rank_in_group
    world_size = tp_group.world_size
    if args.vocab_size % world_size != 0:
        raise ValueError("vocab-size must be divisible by world size")

    device = torch.device("cuda", local_rank)
    logits_per_req = args.speculative_steps + 1
    num_rows = args.num_requests * logits_per_req
    local_vocab = args.vocab_size // world_size
    vocab_start = rank * local_vocab
    torch.manual_seed(args.seed)
    global_logits = _make_logits(args, num_rows, device)
    local_logits = global_logits[
        :, vocab_start : vocab_start + local_vocab
    ].contiguous()
    del global_logits

    cu_num_logits = torch.arange(
        0,
        num_rows + 1,
        logits_per_req,
        dtype=torch.int32,
        device=device,
    )
    idx_mapping = torch.arange(args.num_requests, dtype=torch.int32, device=device)
    expanded_idx_mapping = idx_mapping.repeat_interleave(logits_per_req)
    expanded_local_pos = torch.arange(
        logits_per_req, dtype=torch.int32, device=device
    ).repeat(args.num_requests)
    positions = (
        torch.arange(logits_per_req, dtype=torch.int64, device=device)
        .repeat(args.num_requests)
        .add(
            torch.arange(args.num_requests, device=device).repeat_interleave(
                logits_per_req
            )
            * 1000
        )
    )
    draft_sampled = torch.randint(
        0, args.vocab_size, (num_rows,), dtype=torch.int32, device=device
    )
    seeds = torch.arange(
        5000, 5000 + args.num_requests, dtype=torch.int64, device=device
    )
    temperature = torch.ones(args.num_requests, dtype=torch.float32, device=device)
    top_p = torch.full((num_rows,), args.top_p, dtype=torch.float32, device=device)
    tp_group.all_reduce(
        torch.zeros(
            (num_rows, args.model_hidden_size), dtype=torch.bfloat16, device=device
        )
    )
    dist.barrier()

    def distributed_path():
        cutoff = distributed_bf16_top_p_cutoff(
            local_logits,
            top_p,
            vocab_start=vocab_start,
            org_vocab_size=args.vocab_size,
            tp_group=tp_group,
        )
        return distributed_one_hot_rejection_sample(
            local_logits,
            cutoff,
            draft_sampled,
            cu_num_logits,
            idx_mapping,
            expanded_idx_mapping,
            seeds,
            positions,
            args.speculative_steps,
            vocab_start=vocab_start,
            org_vocab_size=args.vocab_size,
            tp_group=tp_group,
        )

    def full_path():
        full = tp_group.all_gather(local_logits, dim=-1).float()
        apply_top_k_top_p(full, None, top_p)
        return rejection_sample(
            full,
            None,
            draft_sampled,
            cu_num_logits,
            positions,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seeds,
            args.speculative_steps,
        )

    initial_cutoff = distributed_bf16_top_p_cutoff(
        local_logits,
        top_p,
        vocab_start=vocab_start,
        org_vocab_size=args.vocab_size,
        tp_group=tp_group,
    )
    local_tokens = torch.arange(local_vocab, device=device) + vocab_start
    local_keys = _ordered_bf16_key(local_logits)
    local_keep = (local_keys > initial_cutoff.ordered_key.unsqueeze(-1)) | (
        (local_keys == initial_cutoff.ordered_key.unsqueeze(-1))
        & (local_tokens.unsqueeze(0) <= initial_cutoff.last_token_id.unsqueeze(-1))
    )
    distributed_keep = tp_group.all_gather(local_keep.to(torch.uint8), dim=-1).bool()
    full_logits = tp_group.all_gather(local_logits, dim=-1)
    stable_keep = _stable_top_p_mask(full_logits, top_p)
    torch.testing.assert_close(distributed_keep, stable_keep, rtol=0, atol=0)
    expected_retained_mass = (
        torch.exp(full_logits.float() - initial_cutoff.global_max.unsqueeze(-1))
        * stable_keep
    ).sum(dim=-1)
    retained_mass_relative_error = (
        (initial_cutoff.retained_mass - expected_retained_mass).abs()
        / expected_retained_mass
    ).amax()

    triton_logits = full_logits.float()
    apply_top_k_top_p(triton_logits, None, top_p)
    triton_keep = torch.isfinite(triton_logits)
    triton_mask_agreement = (triton_keep == stable_keep).float().mean()
    triton_rows_exact = (triton_keep == stable_keep).all(dim=-1).float().mean()
    del full_logits, triton_logits

    distributed_output = distributed_path()
    full_sampled, full_num_sampled = full_path()
    counts_equal = torch.equal(distributed_output.num_sampled, full_num_sampled)
    steps = torch.arange(args.speculative_steps + 1, device=device).unsqueeze(0)
    common_valid = steps < torch.minimum(
        distributed_output.num_sampled, full_num_sampled
    ).unsqueeze(1)
    comparable_tokens = int(common_valid.sum().item())
    matching_tokens = int(
        (distributed_output.sampled[common_valid] == full_sampled[common_valid])
        .sum()
        .item()
    )
    output_token_agreement = (
        matching_tokens / comparable_tokens if comparable_tokens else 1.0
    )
    outputs_equal = counts_equal and matching_tokens == comparable_tokens
    if args.require_output_equality and not outputs_equal:
        raise AssertionError(
            "Distributed stable top-p output differs from the current Triton "
            "top-p output; inspect the reported agreement metrics."
        )

    distributed_cuda_ms, distributed_wall_ms = _time_ms(
        distributed_path, args.warmup, args.iterations
    )
    full_cuda_ms, full_wall_ms = _time_ms(full_path, args.warmup, args.iterations)
    distributed_graph_cuda_ms = None
    distributed_graph_wall_ms = None
    full_graph_cuda_ms = None
    full_graph_wall_ms = None
    if args.measure_cudagraph:
        distributed_graph, _distributed_graph_output = _capture_cuda_graph(
            distributed_path, device
        )
        full_graph, _full_graph_output = _capture_cuda_graph(full_path, device)
        distributed_graph_cuda_ms, distributed_graph_wall_ms = _time_ms(
            distributed_graph.replay, args.warmup, args.iterations
        )
        full_graph_cuda_ms, full_graph_wall_ms = _time_ms(
            full_graph.replay, args.warmup, args.iterations
        )
    lm_head_cuda_ms = None
    lm_head_wall_ms = None
    lm_head_graph_cuda_ms = None
    lm_head_graph_wall_ms = None
    fp8_lm_head_results = None
    if args.measure_lm_head or args.measure_fp8_lm_head:
        if args.hidden_states_file:
            hidden_states = _load_tensor(
                args.hidden_states_file.format(rank=rank),
                "hidden_states",
                device,
                torch.bfloat16,
            )
            if hidden_states.ndim != 2 or hidden_states.shape[0] < num_rows:
                raise ValueError(
                    "captured hidden states must have shape "
                    f"[at least {num_rows}, {args.model_hidden_size}]"
                )
            hidden_states = hidden_states[:num_rows]
        else:
            torch.manual_seed(args.seed + 1000)
            hidden_states = torch.randn(
                num_rows,
                args.model_hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
        if hidden_states.shape[1] != args.model_hidden_size:
            raise ValueError(
                f"hidden size {hidden_states.shape[1]} does not match "
                f"--model-hidden-size {args.model_hidden_size}"
            )

        if args.lm_head_weight_file:
            lm_head_weight = _load_tensor(
                args.lm_head_weight_file.format(rank=rank),
                "weight",
                torch.device("cpu"),
                torch.bfloat16,
            )
            if lm_head_weight.shape[0] == args.vocab_size:
                lm_head_weight = lm_head_weight[
                    vocab_start : vocab_start + local_vocab
                ].contiguous()
            if lm_head_weight.shape != (local_vocab, args.model_hidden_size):
                raise ValueError(
                    "LM-head weight must be global [vocab, hidden] or local "
                    f"[{local_vocab}, {args.model_hidden_size}], got "
                    f"{tuple(lm_head_weight.shape)}"
                )
            lm_head_weight = lm_head_weight.to(device=device)
        else:
            torch.manual_seed(args.seed + 2000 + rank)
            lm_head_weight = torch.randn(
                local_vocab,
                args.model_hidden_size,
                dtype=torch.bfloat16,
                device=device,
            ).mul_(args.model_hidden_size**-0.5)
        lm_head_cuda_ms, lm_head_wall_ms = _time_ms(
            lambda: F.linear(hidden_states, lm_head_weight),
            args.warmup,
            args.iterations,
        )
        if args.measure_cudagraph:
            lm_head_graph, _lm_head_graph_output = _capture_cuda_graph(
                lambda: F.linear(hidden_states, lm_head_weight), device
            )
            lm_head_graph_cuda_ms, lm_head_graph_wall_ms = _time_ms(
                lm_head_graph.replay,
                args.warmup,
                args.iterations,
            )
        if args.measure_fp8_lm_head:
            reference_local = F.linear(hidden_states, lm_head_weight)
            fp8_lm_head_results = {}
            for weight_scale_type in ("tensor", "channel"):
                runner = _make_fp8_lm_head_runner(
                    hidden_states, lm_head_weight, weight_scale_type
                )
                candidate_local = runner()
                cuda_ms, wall_ms = _time_ms(
                    runner,
                    args.warmup,
                    args.iterations,
                )
                result = {
                    "cuda_median_ms": statistics.median(cuda_ms),
                    "wall_median_ms": statistics.median(wall_ms),
                    "quality": _fp8_lm_head_quality(
                        reference_local, candidate_local, top_p, tp_group
                    ),
                }
                if args.measure_cudagraph:
                    graph, _graph_output = _capture_cuda_graph(runner, device)
                    graph_cuda_ms, graph_wall_ms = _time_ms(
                        graph.replay,
                        args.warmup,
                        args.iterations,
                    )
                    result.update(
                        graph_cuda_median_ms=statistics.median(graph_cuda_ms),
                        graph_wall_median_ms=statistics.median(graph_wall_ms),
                    )
                fp8_lm_head_results[weight_scale_type] = result
    if rank == 0:
        distributed_cuda_median = statistics.median(distributed_cuda_ms)
        distributed_wall_median = statistics.median(distributed_wall_ms)
        full_cuda_median = statistics.median(full_cuda_ms)
        full_wall_median = statistics.median(full_wall_ms)
        lm_head_cuda_median = (
            statistics.median(lm_head_cuda_ms) if lm_head_cuda_ms else None
        )
        lm_head_wall_median = (
            statistics.median(lm_head_wall_ms) if lm_head_wall_ms else None
        )
        lm_head_graph_cuda_median = (
            statistics.median(lm_head_graph_cuda_ms) if lm_head_graph_cuda_ms else None
        )
        lm_head_graph_wall_median = (
            statistics.median(lm_head_graph_wall_ms) if lm_head_graph_wall_ms else None
        )
        if fp8_lm_head_results is not None:
            for result in fp8_lm_head_results.values():
                result["cuda_speedup_vs_bf16"] = (
                    lm_head_cuda_median / result["cuda_median_ms"]
                )
                result["wall_speedup_vs_bf16"] = (
                    lm_head_wall_median / result["wall_median_ms"]
                )
                if lm_head_graph_cuda_median is not None:
                    result["graph_cuda_speedup_vs_bf16"] = (
                        lm_head_graph_cuda_median / result["graph_cuda_median_ms"]
                    )
                    result["graph_wall_speedup_vs_bf16"] = (
                        lm_head_graph_wall_median / result["graph_wall_median_ms"]
                    )
        distributed_graph_cuda_median = (
            statistics.median(distributed_graph_cuda_ms)
            if distributed_graph_cuda_ms
            else None
        )
        distributed_graph_wall_median = (
            statistics.median(distributed_graph_wall_ms)
            if distributed_graph_wall_ms
            else None
        )
        full_graph_cuda_median = (
            statistics.median(full_graph_cuda_ms) if full_graph_cuda_ms else None
        )
        full_graph_wall_median = (
            statistics.median(full_graph_wall_ms) if full_graph_wall_ms else None
        )
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "num_requests": args.num_requests,
                    "num_rows": num_rows,
                    "vocab_size": args.vocab_size,
                    "top_p": args.top_p,
                    "distribution": args.distribution,
                    "seed": args.seed,
                    "valid_vocab": (
                        args.valid_vocab if args.distribution == "masked" else None
                    ),
                    "logits_file": args.logits_file,
                    "allreduce_flashinfer": os.getenv(
                        "VLLM_ALLREDUCE_USE_FLASHINFER", "0"
                    ),
                    "flashinfer_backend": os.getenv(
                        "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "auto"
                    ),
                    "distributed_cuda_median_ms": distributed_cuda_median,
                    "distributed_wall_median_ms": distributed_wall_median,
                    "full_cuda_median_ms": full_cuda_median,
                    "full_wall_median_ms": full_wall_median,
                    "cuda_speedup": full_cuda_median / distributed_cuda_median,
                    "wall_speedup": full_wall_median / distributed_wall_median,
                    "distributed_graph_cuda_median_ms": (distributed_graph_cuda_median),
                    "distributed_graph_wall_median_ms": (distributed_graph_wall_median),
                    "distributed_graph_cuda_speedup": (
                        distributed_cuda_median / distributed_graph_cuda_median
                        if distributed_graph_cuda_median is not None
                        else None
                    ),
                    "distributed_graph_wall_speedup": (
                        distributed_wall_median / distributed_graph_wall_median
                        if distributed_graph_wall_median is not None
                        else None
                    ),
                    "full_graph_cuda_median_ms": full_graph_cuda_median,
                    "full_graph_wall_median_ms": full_graph_wall_median,
                    "lm_head_cuda_median_ms": lm_head_cuda_median,
                    "lm_head_wall_median_ms": lm_head_wall_median,
                    "hidden_states_file": args.hidden_states_file,
                    "lm_head_weight_file": args.lm_head_weight_file,
                    "lm_head_graph_cuda_median_ms": lm_head_graph_cuda_median,
                    "lm_head_graph_wall_median_ms": lm_head_graph_wall_median,
                    "fp8_lm_head": fp8_lm_head_results,
                    "pipeline_cuda_speedup": (
                        (lm_head_cuda_median + full_cuda_median)
                        / (lm_head_cuda_median + distributed_cuda_median)
                        if lm_head_cuda_median is not None
                        else None
                    ),
                    "pipeline_wall_speedup": (
                        (lm_head_wall_median + full_wall_median)
                        / (lm_head_wall_median + distributed_wall_median)
                        if lm_head_wall_median is not None
                        else None
                    ),
                    "stable_mask_exact": True,
                    "retained_mass_max_relative_error": float(
                        retained_mass_relative_error.item()
                    ),
                    "triton_mask_element_agreement": float(
                        triton_mask_agreement.item()
                    ),
                    "triton_mask_rows_exact_fraction": float(triton_rows_exact.item()),
                    "output_counts_equal": counts_equal,
                    "output_token_agreement": output_token_agreement,
                    "outputs_equal": outputs_equal,
                },
                sort_keys=True,
            )
        )
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
