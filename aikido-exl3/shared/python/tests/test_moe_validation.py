import pytest
import torch

from aikido_exl3.vllm_glue.moe import _validate_expert_tensors


def expert_tensors():
    return {
        "trellis": {0: torch.empty((16, 8, 48), dtype=torch.int16)},
        "suh": {0: torch.empty((256,), dtype=torch.float16)},
        "svh": {0: torch.empty((128,), dtype=torch.float16)},
        "mcg": {0: torch.empty((), dtype=torch.int32)},
        "mul1": {},
    }


def test_accepts_expected_expert_tensor_structure():
    _validate_expert_tensors("layer", "gate", 0, expert_tensors(), (256, 128, 6, "mcg", False, False), 256, 128)


@pytest.mark.parametrize("suffix,tensor,match", [
    ("trellis", torch.empty((16, 7, 48), dtype=torch.int16), "expected I16 trellis"),
    ("trellis", torch.empty((16, 8, 48), dtype=torch.int32), "expected I16 trellis"),
    ("suh", torch.empty((255,), dtype=torch.float16), "expected F16 suh"),
    ("svh", torch.empty((128,), dtype=torch.bfloat16), "expected F16 svh"),
])
def test_rejects_malformed_expert_tensors(suffix, tensor, match):
    tensors = expert_tensors()
    tensors[suffix][0] = tensor
    with pytest.raises(ValueError, match=match):
        _validate_expert_tensors("layer", "gate", 0, tensors, (256, 128, 6, "mcg", False, False), 256, 128)


def test_rejects_conflicting_codebook_markers():
    tensors = expert_tensors()
    tensors["mul1"][0] = torch.empty((), dtype=torch.int32)
    with pytest.raises(ValueError, match="both MCG and MUL1"):
        _validate_expert_tensors("layer", "gate", 0, tensors, (256, 128, 6, "mcg", False, False), 256, 128)


def test_rejects_header_shape_mismatch():
    with pytest.raises(ValueError, match="does not match the vLLM layer shape"):
        _validate_expert_tensors("layer", "gate", 0, expert_tensors(),
                                 (128, 128, 6, "mcg", False, False), 256, 128)


def test_rejects_header_codebook_mismatch():
    with pytest.raises(ValueError, match="does not match the header codebook"):
        _validate_expert_tensors("layer", "gate", 0, expert_tensors(),
                                 (256, 128, 6, "mul1", False, False), 256, 128)
