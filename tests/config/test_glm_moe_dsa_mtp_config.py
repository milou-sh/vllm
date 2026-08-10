# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig


@pytest.mark.parametrize(
    "model_type,architecture,expected_architecture",
    [
        ("glm_moe_dsa", "GlmMoeDsaForCausalLM", "GlmMoeDsaMTPModel"),
        ("deepseek_v3", "DeepseekV3ForCausalLM", "DeepSeekMTPModel"),
        ("deepseek_v32", "DeepseekV32ForCausalLM", "DeepSeekMTPModel"),
    ],
)
def test_mtp_override_selects_model_specific_architecture(
    model_type: str,
    architecture: str,
    expected_architecture: str,
):
    config = PretrainedConfig(
        architectures=[architecture],
        num_nextn_predict_layers=3,
    )
    config.model_type = model_type

    overridden = SpeculativeConfig.hf_config_override(config)

    assert overridden.model_type == "deepseek_mtp"
    assert overridden.architectures == [expected_architecture]
    assert overridden.n_predict == 3
