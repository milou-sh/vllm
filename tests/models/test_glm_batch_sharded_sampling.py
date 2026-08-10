# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import torch

from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM


def test_glm_family_computes_local_logits_without_gather():
    model = object.__new__(DeepseekV2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.logits_processor = Mock(return_value=torch.empty(2, 8))
    model.lm_head = Mock()
    hidden_states = torch.empty(2, 16)

    result = model.compute_logits_local(hidden_states)

    model.logits_processor.assert_called_once_with(
        model.lm_head, hidden_states, skip_gather=True
    )
    assert result.shape == (2, 8)
