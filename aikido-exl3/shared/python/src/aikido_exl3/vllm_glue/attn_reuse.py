"""Opt-in reuse of the sparse-MLA index conversion across layers (AIKIDO_NONMOE_ATTN_REUSE=1, default off).

vLLM 0.29.0 FlashAttnMLASparseImpl.forward_mqa runs, on EVERY layer, triton_convert_req_index_to_global_index (top-k
token indices -> global KV rows + valid counts) and a torch.arange for cu_seqlens_q. On GLM-5.3 only 21 of 78 layers
compute a fresh top-k (the others read the shared topk_indices_buffer the last indexer layer wrote), and the block
table, req ids and KV row stride are the same for every layer of a forward. So the converted indices of a skip layer
are bit-identical to those of the last fresh layer: we compute them on fresh layers only and hand the same tensors to
the skip layers (inside one captured CUDA graph this is plain data reuse). Cache key: the metadata object of this
forward (identity; target and MTP drafter have different ones), buffer / block-table addresses, token count and row stride; a
fresh layer always recomputes. Everything else in forward_mqa is vLLM's code, unchanged.
"""
from __future__ import annotations

import os

import torch

from . import compat

ENABLED = os.environ.get("AIKIDO_NONMOE_ATTN_REUSE", "0") == "1"
# also precompute FA3's scheduler metadata once per fresh top-k (vLLM's dense FA3 backend does this once per step and
# shares it across layers); without it FA3 runs its prepare_varlen_num_blocks kernel inside every layer's call
SCHED = os.environ.get("AIKIDO_NONMOE_ATTN_SCHED", "0") == "1"
_installed = False
logger = compat.init_logger(__name__)


def install() -> None:
    global _installed
    if _installed or not ENABLED:
        return
    try:
        cls, flat_kv_row_view, convert, flash_attn_varlen_func = compat.flashattn_mla_sparse_parts()
        get_scheduler_metadata = compat.fa3_get_scheduler_metadata()
    except ImportError as e:   # e.g. a CPU-only process (API server): nothing to patch there
        logger.info("AIKIDO_NONMOE_ATTN_REUSE: FA3 sparse-MLA backend not importable here (%s)", e)
        return
    cache: dict = {}

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if not isinstance(q, tuple):
            raise NotImplementedError("FlashAttnMLASparseImpl expects split (q_nope, q_rope) input.")
        q_nope, q_rope = q
        num_actual_toks = q_rope.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        kv_rows, block_stride_rows = flat_kv_row_view(kv_c_and_k_pe_cache, attn_metadata.block_size)
        fresh = getattr(layer, "indexer", None) is not None and not getattr(layer, "skip_topk", False)
        key = (self.topk_indices_buffer.data_ptr(), attn_metadata.block_table.data_ptr(),
               num_actual_toks, block_stride_rows, attn_metadata.block_size, topk_indices.shape[1])
        hit = None if fresh else cache.get("entry")
        # the entry holds the metadata object itself (identity check, and it cannot be freed and its id recycled)
        if hit is not None and hit[0] is attn_metadata and hit[1] == key:
            _, _, topk_global, valid_counts, cu_seqlens_q, sched = hit
        else:
            topk_global, valid_counts = convert(
                attn_metadata.req_id_per_token[:num_actual_toks], attn_metadata.block_table, topk_indices,
                BLOCK_SIZE=attn_metadata.block_size, BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1], return_valid_counts=True)
            cu_seqlens_q = torch.arange(0, num_actual_toks + 1, dtype=torch.int32, device=q_rope.device)
            sched = None
            if SCHED:
                sched = get_scheduler_metadata(
                    num_actual_toks, 1, topk_global.shape[1], q_rope.shape[1], 1, q_rope.shape[-1], valid_counts,
                    qkv_dtype=q_rope.dtype, headdim_v=q_nope.shape[-1], cu_seqlens_q=cu_seqlens_q, page_size=1,
                    causal=True)
            cache["entry"] = (attn_metadata, key, topk_global, valid_counts, cu_seqlens_q, sched)
        k_cache = kv_rows[:, self.kv_lora_rank:].unsqueeze(1).unsqueeze(1)
        v_cache = kv_rows[:, : self.kv_lora_rank].unsqueeze(1).unsqueeze(1)
        out = flash_attn_varlen_func(
            q=q_rope, k=k_cache, v=v_cache, q_v=q_nope, max_seqlen_q=1, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_global.shape[1], seqused_k=valid_counts, block_table=topk_global,
            softmax_scale=self.scale, causal=True, fa_version=3, scheduler_metadata=sched)
        return out, None

    cls.forward_mqa = forward_mqa
    _installed = True
    logger.info("aikido: sparse-MLA index conversion reused across skip-top-k layers (%s)", cls.__name__)
