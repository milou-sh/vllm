from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4LinearMethod,
)


def _config(exclude_modules: list[str]) -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=exclude_modules,
    )


def test_matching_excluded_linear_uses_online_fp8(monkeypatch):
    config = _config(["model.layers.*.self_attn*"])
    layer = MagicMock(spec=LinearBase)
    sentinel = object()
    monkeypatch.setenv(
        "VLLM_MODELOPT_ONLINE_FP8_PATTERNS",
        "model.layers.*.self_attn.o_proj",
    )

    with patch(
        "vllm.model_executor.layers.quantization.online.fp8."
        "Fp8PerTensorOnlineLinearMethod",
        return_value=sentinel,
    ):
        method = config.get_quant_method(layer, "model.layers.12.self_attn.o_proj")

    assert method is sentinel


def test_nonmatching_excluded_linear_stays_unquantized(monkeypatch):
    config = _config(["model.layers.*.self_attn*"])
    layer = MagicMock(spec=LinearBase)
    monkeypatch.setenv(
        "VLLM_MODELOPT_ONLINE_FP8_PATTERNS",
        "model.layers.*.self_attn.o_proj",
    )

    method = config.get_quant_method(layer, "model.layers.12.self_attn.q_b_proj")

    assert isinstance(method, UnquantizedLinearMethod)


def test_online_fp8_does_not_override_serialized_layer(monkeypatch):
    config = _config([])
    layer = MagicMock(spec=LinearBase)
    monkeypatch.setenv("VLLM_MODELOPT_ONLINE_FP8_PATTERNS", "*")

    with patch(
        "vllm.model_executor.layers.quantization.modelopt.init_nvfp4_linear_kernel"
    ):
        method = config.get_quant_method(layer, "model.layers.12.mlp.experts")

    assert isinstance(method, ModelOptNvFp4LinearMethod)
