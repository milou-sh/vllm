# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
    FI_ALLREDUCE_FUSION_MAX_SIZE_MB,
)
from vllm.model_executor.models.deepseek_v2 import (
    _compiler_sequence_parallel_enabled,
)


def test_sm90_tp4_fusion_limit_matches_h200_profile():
    assert FI_ALLREDUCE_FUSION_MAX_SIZE_MB[90][4] == 14


def test_compiler_sequence_parallel_dispatch():
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=True)
        )
    )

    assert _compiler_sequence_parallel_enabled(config)
    config.compilation_config.pass_config.enable_sp = False
    assert not _compiler_sequence_parallel_enabled(config)
