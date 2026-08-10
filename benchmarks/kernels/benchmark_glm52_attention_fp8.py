# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from functools import partial

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton

TP4_ATTENTION_SHAPES = {
    "fused_qkv_a_proj": (6144, 2624),
    "q_b_proj": (2048, 8192),
    "kv_b_proj": (512, 14336),
    "o_proj": (8192, 6144),
}


def _bench(fn) -> tuple[float, float, float]:
    median, low, high = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
    return float(median), float(low), float(high)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = actual_f - expected_f
    denominator = expected_f.abs().clamp_min(1e-5)
    return {
        "mean_absolute_error": difference.abs().mean().item(),
        "mean_relative_error": (difference.abs() / denominator).mean().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            actual_f.flatten(), expected_f.flatten(), dim=0
        ).item(),
    }


def _bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight)


def _fp8_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return ops.cutlass_scaled_mm(x, weight, x_scale, weight_scale, torch.bfloat16)


def _fp8_e2e(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token: bool,
) -> torch.Tensor:
    x_quantized, x_scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=per_token)
    return _fp8_mm(x_quantized, weight, x_scale, weight_scale)


def benchmark_shape(name: str, k: int, n: int, tokens: list[int]) -> list[dict]:
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    weight_fp8, weight_scale = ops.scaled_fp8_quant(weight)
    weight_fp8 = weight_fp8.t()
    results = []

    for m in tokens:
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        x_fp8_token, x_scale_token = ops.scaled_fp8_quant(
            x, use_per_token_if_dynamic=True
        )
        x_fp8_tensor, x_scale_tensor = ops.scaled_fp8_quant(x)

        bf16 = partial(_bf16_linear, x, weight)
        fp8_token_mm = partial(
            _fp8_mm, x_fp8_token, weight_fp8, x_scale_token, weight_scale
        )
        fp8_token_e2e = partial(_fp8_e2e, x, weight_fp8, weight_scale, True)
        fp8_tensor_e2e = partial(_fp8_e2e, x, weight_fp8, weight_scale, False)

        bf16_ms = _bench(bf16)
        token_mm_ms = _bench(fp8_token_mm)
        token_e2e_ms = _bench(fp8_token_e2e)
        tensor_e2e_ms = _bench(fp8_tensor_e2e)
        expected = bf16()
        token_output = fp8_token_mm()
        tensor_output = _fp8_mm(x_fp8_tensor, weight_fp8, x_scale_tensor, weight_scale)
        results.append(
            {
                "projection": name,
                "m": m,
                "k": k,
                "n": n,
                "bf16_ms": bf16_ms[0],
                "fp8_token_mm_ms": token_mm_ms[0],
                "fp8_token_e2e_ms": token_e2e_ms[0],
                "fp8_tensor_e2e_ms": tensor_e2e_ms[0],
                "fp8_token_e2e_speedup": bf16_ms[0] / token_e2e_ms[0],
                "fp8_tensor_e2e_speedup": bf16_ms[0] / tensor_e2e_ms[0],
                "activation_quantization_share": 1.0 - token_mm_ms[0] / token_e2e_ms[0],
                "fp8_token_accuracy": _metrics(token_output, expected),
                "fp8_tensor_accuracy": _metrics(tensor_output, expected),
                "p20_p80_ms": {
                    "bf16": [bf16_ms[1], bf16_ms[2]],
                    "fp8_token_mm": [token_mm_ms[1], token_mm_ms[2]],
                    "fp8_token_e2e": [token_e2e_ms[1], token_e2e_ms[2]],
                    "fp8_tensor_e2e": [tensor_e2e_ms[1], tensor_e2e_ms[2]],
                },
            }
        )

    del weight, weight_fp8, weight_scale
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,10,16,32,64,128,256,512,1024")
    parser.add_argument("--projections", default=",".join(TP4_ATTENTION_SHAPES.keys()))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"This benchmark targets H100/H200 SM90, got SM{capability}")

    tokens = [int(value) for value in args.tokens.split(",")]
    projections = args.projections.split(",")
    unknown = set(projections) - TP4_ATTENTION_SHAPES.keys()
    if unknown:
        raise ValueError(f"Unknown projections: {sorted(unknown)}")

    torch.manual_seed(0)
    output = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "tensor_parallel_size": 4,
        "results": [],
    }
    for projection in projections:
        k, n = TP4_ATTENTION_SHAPES[projection]
        output["results"].extend(benchmark_shape(projection, k, n, tokens))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
