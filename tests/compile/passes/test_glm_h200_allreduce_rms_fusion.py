# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
    FI_ALLREDUCE_FUSION_MAX_SIZE_MB,
)


def test_sm90_tp4_fusion_limit_matches_h200_profile():
    assert FI_ALLREDUCE_FUSION_MAX_SIZE_MB[90][4] == 14
