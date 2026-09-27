"""Every vLLM import in this package goes through here (tests/test_layering.py enforces it).
Tested against vLLM 0.29.0. When vLLM moves something, this is the only file that changes."""
from __future__ import annotations

from vllm import __version__ as VLLM_VERSION
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.fused_moe import fused_experts  # MoE front: original-basis large-batch tier
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # MoE front (hopper_moe)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase, UnquantizedLinearMethod,
                                               register_weight_loader_v2_supported_method)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.transformers_utils.config import get_safetensors_params_metadata

__all__ = ["VLLM_VERSION", "BasevLLMParameter", "FusedMoEMethodBase", "RoutedExperts", "LinearBase", "LinearMethodBase", "ParallelLMHead",
           "QuantizationConfig", "QuantizeMethodBase", "UnquantizedLinearMethod", "VocabParallelEmbedding",
           "fused_experts", "get_safetensors_params_metadata", "init_logger", "moe_align_block_size", "register_quantization_config",
           "register_weight_loader_v2_supported_method"]


def deepseek_v32_attention_cls():
    """vLLM 0.29.0 GLM-5.x / DeepSeek-V3.2 attention (source-patched by vllm_glue/skinny.py when opted in)."""
    from vllm.models.deepseek_v32.attention import DeepseekV32Attention
    return DeepseekV32Attention


def flashattn_mla_sparse_parts():
    """vLLM 0.29.0 FA3 sparse-MLA backend pieces (vllm_glue/attn_reuse.py re-implements its forward_mqa with them)."""
    from vllm.v1.attention.backends.mla import flashattn_mla_sparse as m
    return m.FlashAttnMLASparseImpl, m.flat_kv_row_view, m.triton_convert_req_index_to_global_index, \
        m.flash_attn_varlen_func


def fa3_get_scheduler_metadata():
    from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata
    return get_scheduler_metadata
