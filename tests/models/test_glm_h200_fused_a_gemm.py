# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.models import deepseek_v2


def _weight(n: int, k: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty((n, k), device="meta", dtype=dtype)


def test_glm_fused_a_is_opt_in_on_sm90(monkeypatch) -> None:
    monkeypatch.setattr(deepseek_v2.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        deepseek_v2.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        deepseek_v2.current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    monkeypatch.setattr(envs, "VLLM_GLM52_SM90_FUSED_A_GEMM", False)
    assert not deepseek_v2._supports_min_latency_fused_qkv_a(_weight(2624, 6144))
    assert deepseek_v2._min_latency_fused_a_max_tokens(_weight(2624, 6144)) == 16

    monkeypatch.setattr(envs, "VLLM_GLM52_SM90_FUSED_A_GEMM", True)
    assert deepseek_v2._supports_min_latency_fused_qkv_a(_weight(2624, 6144))
    assert deepseek_v2._supports_min_latency_fused_qkv_a(_weight(8192, 2048))
    assert deepseek_v2._supports_min_latency_fused_qkv_a(_weight(14336, 512))
    assert deepseek_v2._supports_min_latency_fused_qkv_a(_weight(6144, 8192))
    assert deepseek_v2._min_latency_fused_a_max_tokens(_weight(2624, 6144)) == 64
    assert not deepseek_v2._supports_min_latency_fused_qkv_a(_weight(2625, 6144))
    assert not deepseek_v2._supports_min_latency_fused_qkv_a(
        _weight(2624, 6144, torch.float16)
    )


def test_existing_deepseek_shape_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(deepseek_v2.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        deepseek_v2.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        deepseek_v2.current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    monkeypatch.setattr(envs, "VLLM_GLM52_SM90_FUSED_A_GEMM", False)
    assert deepseek_v2._supports_min_latency_fused_qkv_a(_weight(2112, 7168))


def test_q_b_swap_requires_exact_unquantized_shape(monkeypatch) -> None:
    class FakeLinear(nn.Module):
        def __init__(self, shape, quant_method) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(shape, device="meta", dtype=torch.bfloat16)
            )
            self.quant_method = quant_method

    monkeypatch.setattr(deepseek_v2.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        deepseek_v2.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(envs, "VLLM_GLM52_SM90_FUSED_A_GEMM", True)

    layer = FakeLinear((8192, 2048), deepseek_v2.UnquantizedLinearMethod())
    deepseek_v2._enable_glm52_sm90_fused_a(layer)
    assert isinstance(layer.quant_method, deepseek_v2.Glm52SM90FusedALinearMethod)

    quantized_method = object()
    quantized = FakeLinear((8192, 2048), quantized_method)
    deepseek_v2._enable_glm52_sm90_fused_a(quantized)
    assert quantized.quant_method is quantized_method

    wrong_shape = FakeLinear((2048, 2048), deepseek_v2.UnquantizedLinearMethod())
    deepseek_v2._enable_glm52_sm90_fused_a(wrong_shape)
    assert type(wrong_shape.quant_method) is deepseek_v2.UnquantizedLinearMethod


def test_tp4_q_b_kernel_shape_is_instantiated() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc/libtorch_stable/dsv3_fused_a_gemm.cu"
    ).read_text()
    assert "DISPATCH_DSV3_SHAPE(2048, 8192)" in source
    assert "DISPATCH_DSV3_SHAPE(512, 14336)" in source
    assert "DISPATCH_DSV3_SHAPE(8192, 6144)" in source
    assert "gemm_n - cta_n_idx" in source
    assert "num_tokens <= 64" in source
