# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
from vllm.distributed.parallel_state import (
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
    torch.cuda.synchronize()
    cuda_timings = []
    wall_timings = []
    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--speculative-steps", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=154880)
    parser.add_argument("--model-hidden-size", type=int, default=6144)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
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
    torch.manual_seed(20260810)
    global_logits = torch.randn(
        num_rows, args.vocab_size, dtype=torch.bfloat16, device=device
    )
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

    distributed_output = distributed_path()
    full_sampled, full_num_sampled = full_path()
    torch.testing.assert_close(
        distributed_output.num_sampled, full_num_sampled, rtol=0, atol=0
    )
    steps = torch.arange(args.speculative_steps + 1, device=device)
    valid = steps.unsqueeze(0) < full_num_sampled.unsqueeze(1)
    torch.testing.assert_close(
        distributed_output.sampled[valid], full_sampled[valid], rtol=0, atol=0
    )

    distributed_cuda_ms, distributed_wall_ms = _time_ms(
        distributed_path, args.warmup, args.iterations
    )
    full_cuda_ms, full_wall_ms = _time_ms(full_path, args.warmup, args.iterations)
    if rank == 0:
        distributed_cuda_median = statistics.median(distributed_cuda_ms)
        distributed_wall_median = statistics.median(distributed_wall_ms)
        full_cuda_median = statistics.median(full_cuda_ms)
        full_wall_median = statistics.median(full_wall_ms)
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "num_requests": args.num_requests,
                    "num_rows": num_rows,
                    "vocab_size": args.vocab_size,
                    "top_p": args.top_p,
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
                    "outputs_equal": True,
                },
                sort_keys=True,
            )
        )
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
