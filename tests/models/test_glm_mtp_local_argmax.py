# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.model_executor.models.deepseek_mtp import GlmMoeDsaMTP
from vllm.model_executor.models.glm4_moe_mtp import Glm4MoeMultiTokenPredictor


def _local_argmax_state():
    shared_head = MagicMock()
    shared_head.head = object()
    normalized = object()
    shared_head.return_value = normalized
    logits_processor = MagicMock()
    top_tokens = object()
    logits_processor.get_top_tokens.return_value = top_tokens
    return shared_head, normalized, logits_processor, top_tokens


def test_glm_moe_dsa_uses_current_mtp_head_for_local_argmax():
    shared_head, normalized, logits_processor, top_tokens = _local_argmax_state()
    hidden_states = object()
    instance = SimpleNamespace(
        model=SimpleNamespace(
            num_mtp_layers=3,
            mtp_start_layer_idx=10,
            layers={"11": SimpleNamespace(shared_head=shared_head)},
            logits_processor=logits_processor,
        )
    )

    result = GlmMoeDsaMTP.get_top_tokens(instance, hidden_states, spec_step_idx=4)

    shared_head.assert_called_once_with(hidden_states)
    logits_processor.get_top_tokens.assert_called_once_with(
        shared_head.head, normalized
    )
    assert result is top_tokens


def test_glm4_moe_uses_current_mtp_head_for_local_argmax():
    shared_head, normalized, logits_processor, top_tokens = _local_argmax_state()
    hidden_states = object()
    instance = SimpleNamespace(
        num_mtp_layers=3,
        mtp_start_layer_idx=10,
        layers={"11": SimpleNamespace(shared_head=shared_head)},
        logits_processor=logits_processor,
    )

    result = Glm4MoeMultiTokenPredictor.get_top_tokens(
        instance, hidden_states, spec_step_idx=4
    )

    shared_head.assert_called_once_with(hidden_states)
    logits_processor.get_top_tokens.assert_called_once_with(
        shared_head.head, normalized
    )
    assert result is top_tokens
