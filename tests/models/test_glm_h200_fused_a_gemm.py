# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

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

    monkeypatch.setattr(envs, "VLLM_GLM52_SM90_FUSED_A_GEMM", True)
    assert deepseek_v2._supports_min_latency_fused_qkv_a(_weight(2624, 6144))
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
