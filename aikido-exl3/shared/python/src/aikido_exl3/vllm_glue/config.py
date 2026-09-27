"""Exl3Config: what is quantized, per tensor, from safetensors headers (never from the JSON hints)."""
from __future__ import annotations

from typing import Any

import torch

from ..format import build_manifest
from ..format.safetensors_header import ShardHeader
from ..format.spec import TensorInfo
from . import compat

logger = compat.init_logger(__name__)


class Exl3Config(compat.QuantizationConfig):
    def __init__(self, declared: dict[str, Any] | None = None):
        super().__init__()
        self.declared = dict(declared or {})
        # module key -> (k, n, 2*K, codebook, has_bias, legacy_signs); plain data: this object is pickled to workers
        self.hf_modules: dict[str, tuple] = {}
        self.modules: dict[str, tuple] = {}       # same, in vLLM's module namespace (accumulated per mapper)

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # Stock EXL3 declares quant_method "exl3". TR3 checkpoints (GLM-5.3 rank-sliced experts, format/tr3.py) carry
        # a modelopt/NVFP4 dispatch shim in quantization_config for a vLLM fork; the real declaration is
        # hf_config.hybrid_tr3_tail. Claim those, and only those, so `--quantization exl3` is accepted (vLLM
        # otherwise rejects the request as "does not match the quantization method in the model config").
        tail = getattr(hf_config, "hybrid_tr3_tail", None) if hf_config is not None else None
        if isinstance(tail, dict) and tail.get("format") == "exl3-trellis" and user_quant in (None, "exl3"):
            return "exl3"
        return None

    def maybe_update_config(self, model_name: str, hf_config=None, revision: str | None = None):
        if self.hf_modules:
            return
        revision = revision or getattr(hf_config, "_commit_hash", None)   # EXL3 repos are branch-per-bitrate
        meta = compat.get_safetensors_params_metadata(model_name, revision=revision)
        # vLLM passes shard `__metadata__` entries through (TR3 shards carry one); only real tensors have a dtype
        tensors = {name: TensorInfo(name, m["dtype"], tuple(m["shape"])) for name, m in meta.items()
                   if isinstance(m, dict) and "dtype" in m and "shape" in m}
        manifest = build_manifest([ShardHeader("", {}, tensors)], self.declared)
        self.hf_modules = {k: (m.k, m.n, m.bits.twice, m.codebook.value, m.has_bias, m.legacy_signs)
                           for k, m in manifest.matrices.items()}
        self.modules = dict(self.hf_modules)
        for w in manifest.warnings:
            logger.warning("EXL3 checkpoint: %s", w)
        s = manifest.summary()
        logger.info("EXL3 checkpoint: %d quantized linears %s, %d other tensors",
                    s["quantized_linears"], s["by_bits_codebook"], s["unquantized_tensors"])
        if hf_config is not None and any(k.endswith("lm_head") for k in self.hf_modules):
            # vLLM only unties when it sees lm_head.weight; a quantized head is stored as lm_head.trellis.
            for cfg in {id(c): c for c in (hf_config, hf_config.get_text_config())}.values():
                if getattr(cfg, "tie_word_embeddings", False):
                    cfg.tie_word_embeddings = False
                    logger.info("EXL3 checkpoint has its own quantized lm_head: untied from embed_tokens")

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        # Called once per model component (outer wrapper, language model, MTP draft), each with its own
        # mapper; always map from the immutable HF names so repeated calls only add spellings.
        # Mapper rules are written for tensor names ("lm_head." -> "language_model.lm_head."), so map
        # "<module key>." and strip the dot again; a bare "lm_head" would not match its own rule.
        mapped = hf_to_vllm_mapper.apply_dict({k + ".": v for k, v in self.hf_modules.items()})
        self.modules.update({k.rstrip("."): v for k, v in mapped.items()})

    def _sources(self, prefix: str) -> list[str]:
        parent, _, leaf = prefix.rpartition(".")
        packed = self.packed_modules_mapping.get(leaf)
        return [f"{parent}.{s}" if parent else s for s in packed] if packed else [prefix]

    def get_quant_method(self, layer, prefix: str):
        from .linear import Exl3LinearMethod
        if isinstance(layer, compat.RoutedExperts):
            from .moe import Exl3MoEMethod
            experts = {k: v for k, v in self.modules.items() if k.startswith(prefix + ".")}
            return Exl3MoEMethod(prefix, experts, layer.moe_config) if experts else None
        if not isinstance(layer, (compat.LinearBase, compat.ParallelLMHead)):
            return None   # attention, input embeddings, anything else: not ours
        infos = [self.modules.get(p) for p in self._sources(prefix)]
        if not any(infos):
            return compat.UnquantizedLinearMethod() if isinstance(layer, compat.LinearBase) else None
        if not all(infos):
            raise ValueError(f"{prefix}: fused module mixes EXL3 and unquantized sources {self._sources(prefix)}")
        return Exl3LinearMethod(prefix, infos)
