# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm._custom_ops import dsv3_router_gemm

HIDDEN_SIZE = 6144
NUM_EXPERTS = 168


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_glm52_router_matches_cublas(num_tokens: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("dsv3_router_gemm requires SM90+")

    torch.manual_seed(1234 + num_tokens)
    hidden_states = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    router_weight = (
        torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        / HIDDEN_SIZE**0.5
    )

    actual = dsv3_router_gemm(hidden_states, router_weight, torch.float32)
    reference = torch.mm(hidden_states, router_weight.T, out_dtype=torch.float32)

    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=2e-2)
    actual_top8 = actual.topk(8, dim=-1).indices.sort(dim=-1).values
    reference_top8 = reference.topk(8, dim=-1).indices.sort(dim=-1).values
    torch.testing.assert_close(actual_top8, reference_top8, rtol=0, atol=0)
