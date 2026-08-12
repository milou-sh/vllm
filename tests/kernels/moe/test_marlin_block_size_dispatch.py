# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe.experts import marlin_moe


def select(
    monkeypatch,
    num_tokens: int,
    *,
    experts: int = 168,
    topk: int = 8,
    hidden_size: int = 6144,
    intermediate_size: int = 512,
) -> int:
    monkeypatch.setattr(
        marlin_moe.current_platform, "is_device_capability", lambda capability: True
    )
    return marlin_moe._select_marlin_moe_block_size(
        num_tokens=num_tokens,
        topk=topk,
        local_num_experts=experts,
        global_num_experts=experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        input_dtype=None,
    )


def test_glm52_tp4_hopper_crossover(monkeypatch):
    assert select(monkeypatch, 80) == 8
    assert select(monkeypatch, 88) == 16
    assert select(monkeypatch, 128) == 16
    assert select(monkeypatch, 151) == 16
    assert select(monkeypatch, 152) == 16


def test_non_glm_shapes_keep_generic_heuristic(monkeypatch):
    assert select(monkeypatch, 128, experts=256) == 8
    assert select(monkeypatch, 128, intermediate_size=1024) == 8
    assert select(monkeypatch, 128, hidden_size=7168) == 8
