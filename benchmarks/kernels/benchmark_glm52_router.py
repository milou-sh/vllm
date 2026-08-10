# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from collections.abc import Callable
from functools import cache
from statistics import median

import torch

import vllm._custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
    LLBf16Gemm,
    _stream,
)

HIDDEN_SIZE = 6144
NUM_EXPERTS = 168
NUM_MOE_LAYERS = 75


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=NUM_MOE_LAYERS)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--include-dsv3", action="store_true")
    return parser.parse_args()


@cache
def compile_cute_arm(key: LLBf16Gemm.CompileKey):
    gemm = LLBf16Gemm()
    gemm.compile(key)
    if key.backend == "dotprod":
        compiled = gemm._compiled_cache[(key.M, key.K, key.bs)]

        def run(x, weight):
            output = torch.empty(
                x.shape[0], NUM_EXPERTS, dtype=torch.float32, device=x.device
            )
            compiled(x, weight, output, NUM_EXPERTS, _stream())
            return output

        return run

    compiled = gemm._splitk_cache[(key.split_k, key.num_stages)]

    def run(x, weight):
        output = torch.empty(
            x.shape[0], NUM_EXPERTS, dtype=torch.float32, device=x.device
        )
        compiled(x, weight, output, _stream(), 1.0)
        return output

    return run


def capture_graph(
    operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    weights: list[torch.Tensor],
):
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            for weight in weights:
                operation(x, weight)
        capture_stream.synchronize()
        with torch.cuda.graph(graph):
            for weight in weights:
                operation(x, weight)
    return graph


def measure_graph(graph, *, layers: int, replays: int, samples: int):
    durations = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end) * 1000 / replays / layers)
    return durations


def validate(operation, x, weight):
    output = operation(x, weight)
    torch.cuda.synchronize()
    reference = torch.mm(x, weight.T, out_dtype=torch.float32)
    delta = output - reference
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)
    actual_top8 = output.topk(8, dim=-1).indices.sort(dim=-1).values
    reference_top8 = reference.topk(8, dim=-1).indices.sort(dim=-1).values
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": relative_l2.item(),
        "top8_equal": torch.equal(actual_top8, reference_top8),
    }


def benchmark_arm(name, operation, x, weights, args):
    accuracy = validate(operation, x, weights[0])
    graph = capture_graph(operation, x, weights)
    durations = measure_graph(
        graph,
        layers=len(weights),
        replays=args.replays,
        samples=args.samples,
    )
    result = {
        "arm": name,
        "m": x.shape[0],
        "median_us": median(durations),
        "range_us": [min(durations), max(durations)],
        **accuracy,
    }
    print(json.dumps(result), flush=True)


def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(1234)
    weights = [
        torch.randn(
            NUM_EXPERTS,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / HIDDEN_SIZE**0.5
        for _ in range(args.layers)
    ]

    def cublas(x, weight):
        return torch.mm(x, weight.T, out_dtype=torch.float32)

    def dsv3(x, weight):
        return ops.dsv3_router_gemm(x, weight, torch.float32)

    for m in range(1, 17):
        x = torch.randn(m, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        benchmark_arm("cublas", cublas, x, weights, args)
        if args.include_dsv3:
            benchmark_arm("dsv3", dsv3, x, weights, args)

        if m <= 4:
            for block_size in (64, 128, 256, 384, 512, 768):
                key = LLBf16Gemm.CompileKey(
                    backend="dotprod", M=m, K=HIDDEN_SIZE, bs=block_size
                )
                benchmark_arm(
                    f"cute-dot-bs{block_size}",
                    compile_cute_arm(key),
                    x,
                    weights,
                    args,
                )
        else:
            for split_k in (4, 5, 6, 7, 8):
                for num_stages in (2, 3, 4, 5):
                    key = LLBf16Gemm.CompileKey(
                        backend="splitk",
                        split_k=split_k,
                        num_stages=num_stages,
                    )
                    benchmark_arm(
                        f"cute-splitk{split_k}-stages{num_stages}",
                        compile_cute_arm(key),
                        x,
                        weights,
                        args,
                    )


if __name__ == "__main__":
    main()
