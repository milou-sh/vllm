"""Exl3LinearMethod: also serves ParallelLMHead (same create_weights / apply slots)."""
from __future__ import annotations

import os

import torch

from ..kernels import hopper, reference
from ..runtime import ops
from . import compat
from .params import Exl3Collector

logger = compat.init_logger(__name__)
_SUFFIXES = ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1")
_CODEBOOK_ID = {"3inst": 0, "mcg": 1, "mul1": 2}
_QKV = {"q": 0, "k": 1, "v": 2}
# Keep a decoded fp16 copy of every matrix on the GPU (4x the 4-bit size) so multi-row decode steps use cuBLAS.
# A memory-for-concurrency trade: on an H200 the 27B costs 54 GB of the KV pool. Becomes a runtime setting.
_DENSE_CACHE = os.environ.get("AIKIDO_EXL3_DENSE_CACHE", "0") == "1"
_DENSE_CACHE_MAX_N = 65536      # not for lm_head: 2.5 GB for a layer that only ever sees one row per sequence


def _partitions(shard_id) -> tuple[int, ...]:
    """vLLM logical output partitions covered by one checkpoint tensor."""
    if shard_id is None:
        return ()
    if isinstance(shard_id, tuple):     # one tensor fused on disk across several partitions (GDN in_proj_qkv)
        return shard_id
    return (_QKV.get(shard_id, shard_id),)


def _unpack_signs(packed: torch.Tensor) -> torch.Tensor:
    bits = (packed.to(torch.int32).unsqueeze(1) >> torch.arange(16, device=packed.device)) & 1
    return (1.0 - 2.0 * bits.flatten()).to(torch.float16)


@compat.register_weight_loader_v2_supported_method
class Exl3LinearMethod(compat.LinearMethodBase):
    def __init__(self, prefix: str, infos: list[tuple]):
        self.prefix = prefix
        self.infos = infos          # per source matrix, checkpoint order: (k, n, 2K, codebook, has_bias, legacy)

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
            raise NotImplementedError(f"{self.prefix}: tensor parallel EXL3 is not implemented yet (M4)")
        if any(info[4] for info in self.infos):
            raise NotImplementedError(f"{self.prefix}: EXL3 linears with bias are not implemented yet")
        layer.exl3_out_sizes = list(getattr(layer, "output_sizes", None) or [output_size])
        device = torch.device("cuda", torch.cuda.current_device())
        for suffix in _SUFFIXES:
            layer.register_parameter(suffix, Exl3Collector(device))

    def process_weights_after_loading(self, layer) -> None:
        got = {s: getattr(layer, s).shards for s in _SUFFIXES}
        ids = sorted(got["trellis"], key=lambda sid: _partitions(sid) or (0,))
        if len(ids) != len(self.infos):
            raise ValueError(f"{self.prefix}: expected {len(self.infos)} EXL3 matrices, checkpoint delivered "
                             f"{len(ids)} ({ids}); vLLM does not report missing tensors for quantized models")
        codebooks, widths = [], []
        for i, sid in enumerate(ids):
            suh = got["suh"].get(sid)
            svh = got["svh"].get(sid)
            if suh is None and sid in got["su"]:
                suh, svh = _unpack_signs(got["su"][sid]), _unpack_signs(got["sv"][sid])
            if suh is None or svh is None:
                raise ValueError(f"{self.prefix}: shard {sid!r} has a trellis but no scale vectors")
            codebook = "mcg" if sid in got["mcg"] else "mul1" if sid in got["mul1"] else "3inst"
            trellis = got["trellis"][sid].contiguous()
            k, n, twice, declared = self.infos[i][:4]
            if (trellis.shape[0] * 16, trellis.shape[1] * 16, trellis.shape[2] // 8, codebook) != (k, n, twice, declared):
                raise ValueError(f"{self.prefix}: shard {sid!r} does not match the header scan")
            parts = _partitions(sid)
            widths.append(sum(layer.exl3_out_sizes[p] for p in parts) if parts else sum(layer.exl3_out_sizes))
            codebooks.append(_CODEBOOK_ID[codebook])
            for name, t in (("trellis", trellis), ("suh", suh.contiguous()), ("svh", svh.contiguous())):
                layer.register_parameter(f"exl3_{name}_{i}", torch.nn.Parameter(t, requires_grad=False))
            if _DENSE_CACHE and n <= _DENSE_CACHE_MAX_N:
                w = reference.reconstruct(trellis, _CODEBOOK_ID[codebook])
                layer.register_parameter(f"exl3_dense_{i}", torch.nn.Parameter(w, requires_grad=False))
        if sum(widths) != sum(layer.exl3_out_sizes):
            raise ValueError(f"{self.prefix}: shards cover {sum(widths)} of {sum(layer.exl3_out_sizes)} output columns")
        for suffix in _SUFFIXES:
            delattr(layer, suffix)
        layer.exl3_count, layer.exl3_codebooks, layer.exl3_widths = len(ids), codebooks, widths
        layer.exl3_hopper_count, layer.exl3_hopper_ends = 0, []
        if os.environ.get("AIKIDO_EXL3_HOPPER", "0") == "1":
            mats = [[getattr(layer, f"exl3_{name}_{i}") for i in range(len(ids))] for name in ("trellis", "suh", "svh")]
            ok, why = hopper.supports(mats[0], widths, codebooks)
            if ok:
                *tensors, layer.exl3_hopper_ends = hopper.prepare(*mats)
                for j, t in enumerate(tensors):
                    layer.register_buffer(f"exl3_hopper_{j}", t, persistent=False)
                layer.exl3_hopper_count = len(tensors)
            else:
                logger.info("%s: ExLlamaV3 kernels (%s)", self.prefix, why)
        layer.exl3_sliced, layer.exl3_sliced_min_rows = [], 1
        if len(ids) > 1 and len(set(codebooks)) == 1 and os.environ.get("AIKIDO_EXL3_SLICED", "1") == "1":
            mats = [[getattr(layer, f"exl3_{name}_{i}") for i in range(len(ids))] for name in ("trellis", "suh", "svh")]
            try:
                group = reference.SlicedGroup(*mats, codebooks[0], max_rows=max(ops.DENSE_ROWS, ops.DENSE_CACHED_ROWS))
            except ValueError as e:      # mixed bitrates or no common 128-aligned slice width: separate launches
                logger.info("%s: no fused launch (%s)", self.prefix, e)
            else:
                layer.exl3_sliced_keepalive = group
                for j, t in enumerate(group.tables()):
                    layer.register_buffer(f"exl3_sliced_{j}", t, persistent=False)
                layer.exl3_sliced_count = len(group.tables())
                layer.exl3_sliced_min_rows = 3 if max(group.widths) >= 16384 else 1     # measured on H200
        layer.exl3_has_dense = all(hasattr(layer, f"exl3_dense_{i}") for i in range(len(ids)))

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        n = layer.exl3_count
        y = ops.linear(x, [getattr(layer, f"exl3_trellis_{i}") for i in range(n)],
                       [getattr(layer, f"exl3_suh_{i}") for i in range(n)],
                       [getattr(layer, f"exl3_svh_{i}") for i in range(n)],
                       [getattr(layer, f"exl3_dense_{i}") for i in range(n)] if layer.exl3_has_dense else [],
                       [getattr(layer, f"exl3_sliced_{j}") for j in range(getattr(layer, "exl3_sliced_count", 0))],
                       layer.exl3_sliced_min_rows,
                       [getattr(layer, f"exl3_hopper_{j}") for j in range(layer.exl3_hopper_count)], layer.exl3_hopper_ends,
                       layer.exl3_codebooks, layer.exl3_widths)
        return y if bias is None else y + bias
