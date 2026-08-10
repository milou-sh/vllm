# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
from functools import partial

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.models import deepseek_v2  # noqa: F401
from vllm.triton_utils import triton

os.environ.setdefault("VLLM_GLM52_SM90_FUSED_A_GEMM", "1")

PROJECTIONS = {
    "fused_qkv_a_proj": (6144, 2624),
    "q_b_proj_tp4": (2048, 8192),
    "kv_b_proj_tp4": (512, 14336),
    "o_proj_tp4": (8192, 6144),
}


def _bench(fn) -> tuple[float, float, float]:
    median, low, high = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
    return float(median), float(low), float(high)


def _torch_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight)


def _custom_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.min_latency_fused_qkv_a_proj(x, weight)


def _direct_op(
    output: torch.Tensor,
    x: torch.Tensor,
    weight_t: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    ops.dsv3_fused_a_gemm(output, x, weight_t, enable_pdl=enable_pdl)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        default=",".join(str(i) for i in (*range(1, 17), 24, 32, 40, 48, 56, 64)),
    )
    parser.add_argument("--projections", default=",".join(PROJECTIONS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"This benchmark targets H100/H200 SM90, got SM{capability}")

    torch.manual_seed(0)
    results = []
    projections = args.projections.split(",")
    unknown = set(projections) - PROJECTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown projections: {sorted(unknown)}")
    for projection in projections:
        k, n = PROJECTIONS[projection]
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        weight_t = weight.t()
        for tokens in (int(value) for value in args.tokens.split(",")):
            x = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16)
            output = torch.empty((tokens, n), device="cuda", dtype=torch.bfloat16)
            torch_fn = partial(_torch_linear, x, weight)
            custom_fn = partial(_custom_op, x, weight)
            direct_fn = partial(_direct_op, output, x, weight_t, False)
            pdl_fn = partial(_direct_op, output, x, weight_t, True)

            torch_ms = _bench(torch_fn)
            custom_ms = _bench(custom_fn)
            direct_ms = _bench(direct_fn)
            pdl_ms = _bench(pdl_fn)
            expected = torch_fn().float()
            actual = custom_fn().float()
            results.append(
                {
                    "projection": projection,
                    "tokens": tokens,
                    "k": k,
                    "n": n,
                    "torch_ms": torch_ms[0],
                    "custom_op_ms": custom_ms[0],
                    "direct_ms": direct_ms[0],
                    "direct_pdl_ms": pdl_ms[0],
                    "custom_op_speedup": torch_ms[0] / custom_ms[0],
                    "direct_speedup": torch_ms[0] / direct_ms[0],
                    "direct_pdl_speedup": torch_ms[0] / pdl_ms[0],
                    "cosine_similarity": torch.nn.functional.cosine_similarity(
                        actual.flatten(), expected.flatten(), dim=0
                    ).item(),
                    "p20_p80_ms": {
                        "torch": [torch_ms[1], torch_ms[2]],
                        "custom_op": [custom_ms[1], custom_ms[2]],
                        "direct": [direct_ms[1], direct_ms[2]],
                        "direct_pdl": [pdl_ms[1], pdl_ms[2]],
                    },
                },
            )
        del weight, weight_t
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "projections": projections,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
