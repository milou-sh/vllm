# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16


@pytest.mark.parametrize(
    "num_tokens,expected_backend,expected_config",
    [
        (1, "dotprod", 256),
        (2, "dotprod", 512),
        (3, "dotprod", 512),
        (4, "dotprod", 512),
        *[(m, "splitk", (8, 3)) for m in range(5, 17)],
    ],
)
def test_glm52_e168_hopper_tuned_dispatch(
    monkeypatch, num_tokens, expected_backend, expected_config
):
    monkeypatch.setattr(
        ll_bf16,
        "_arch_tuned_configs",
        lambda: (
            ll_bf16._SM90_TUNED_DOTPROD_BS,
            ll_bf16._SM90_TUNED_SPLITK_CONFIGS,
        ),
    )

    key = ll_bf16.LLBf16Gemm().dispatch(M=num_tokens, K=6144, N=168)

    assert key.backend == expected_backend
    if expected_backend == "dotprod":
        assert key.bs == expected_config
    else:
        assert (key.split_k, key.num_stages) == expected_config
