"""Original-basis weights: ExLlamaV3's own large-batch path (rows >= 1024 in ExLlamaV3, research/04 section 2.4).

An EXL3 linear is y = Had(Had(x * suh) @ W_hat) * svh. Both Hadamards and both scale vectors are linear, so they fold
into the weight:

    W = diag(suh) . H . W_hat . H . diag(svh)        (k x n, fp16)        y = x @ W

with NO transform of the activations: the transforms are paid once on the weight, O(k n), instead of O(rows (k + n)) per
call. ExLlamaV3 builds W in fp16 with `reconstruct_had_slice` (one matrix) / `reconstruct_had_batch` (all experts of a
projection through pointer tables) and multiplies with cuBLAS. This module only CALLS those builders, so W is ExLlamaV3's
W bit for bit by construction (parity/original_basis_parity.py re-checks it, and a future builder of ours must match it
bit for bit). W stays fp16: a cast to bf16 drops mantissa bits and is not lossless.

Once suh lives inside W, matrices that share an input (q/k/v, gate/up, GDN qkv/z) can be concatenated along n and run as
ONE matmul: the per-shard suh problem of the rotated basis does not exist here.

Shared by the MoE front (experts, vllm_glue/moe.py) and the dense front; imports nothing from vLLM.
Orientation: ExLlamaV3 (k, n) = (in, out), y = x @ W. vLLM's fused MoE wants [E, out, in]: see `to_vllm_experts`.
"""
from __future__ import annotations

import torch

from . import reference

CODEBOOK_MCG, CODEBOOK_MUL1 = 1, 2


def _flags(codebook: int) -> tuple[bool, bool]:
    return codebook == CODEBOOK_MCG, codebook == CODEBOOK_MUL1


def build_matrix(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: int,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """-> W fp16 [k, n] = diag(suh) H W_hat H diag(svh), by exllamav3_ext.reconstruct_had_slice (whole matrix)."""
    ext = reference._load()
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    if k % 128 or n % 128:
        raise ValueError(f"original basis needs k, n multiples of 128, got {k}, {n}")
    w = out if out is not None else torch.empty((k, n), dtype=torch.float16, device=trellis.device)
    mcg, mul1 = _flags(codebook)
    ext.reconstruct_had_slice(w, trellis.contiguous(), suh.contiguous(), svh.contiguous(), reference.bits_of(trellis), mcg, mul1, 0)
    return w


def build_group(trellis: list[torch.Tensor], suh: list[torch.Tensor], svh: list[torch.Tensor], codebooks: list[int]) -> torch.Tensor:
    """Matrices sharing the input -> ONE W fp16 [k, sum n] (each shard built on its own, concatenated along n)."""
    return torch.cat([build_matrix(t, su, sv, cb) for t, su, sv, cb in zip(trellis, suh, svh, codebooks)], dim=1).contiguous()


def build_experts(trellis_ptrs: torch.Tensor, suh_ptrs: torch.Tensor, svh_ptrs: torch.Tensor, k: int, n: int, bits: float,
                  codebook: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """All experts behind the pointer tables (int64 device tensors, as kernels.reference.pointer_table builds them; pass a
    subset of the rows to build only some experts) -> W fp16 [len(ptrs), k, n], by exllamav3_ext.reconstruct_had_batch."""
    ext = reference._load()
    e = trellis_ptrs.shape[0]
    w = out if out is not None else torch.empty((e, k, n), dtype=torch.float16, device=trellis_ptrs.device)
    if w.shape != (e, k, n) or w.dtype != torch.float16 or not w.is_contiguous():
        raise ValueError(f"out must be contiguous fp16 {(e, k, n)}, got {tuple(w.shape)} {w.dtype}")
    mcg, mul1 = _flags(codebook)
    ext.reconstruct_had_batch(w, trellis_ptrs, suh_ptrs, svh_ptrs, bits, mcg, mul1)
    return w


def to_vllm_experts(w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ExLlamaV3 orientation [E, in, out] -> vLLM fused-MoE layout: w13 [E, 2 * inter, hidden] (gate rows first, as
    silu_and_mul expects), w2 [E, hidden, inter]. Pure transposes / concatenation: no value changes."""
    w13 = torch.cat([w_gate, w_up], dim=2).transpose(1, 2).contiguous()
    w2 = w_down.transpose(1, 2).contiguous()
    return w13, w2


def build_experts_vllm(out: torch.Tensor, trellis_ptrs: torch.Tensor, suh_ptrs: torch.Tensor, svh_ptrs: torch.Tensor,
                       n: int, row_offset: int, codebook: int, bits: int = 4) -> torch.Tensor:
    """ExLlamaV3's W (same arithmetic, statement for statement: csrc/exl3_moe_orig_builder.cu) written TRANSPOSED into
    out fp16 [E, rows_total, k] at rows row_offset .. row_offset + n - 1, i.e. straight into vLLM's [E, out, in] layout with
    gate and up stacked on the out dimension. Equals build_experts(...).transpose(1, 2) bit for bit; K = 3 and K = 4.
    No allocation: `out` is the caller's arena, so the tier can be rebuilt per prefill chunk."""
    import importlib
    importlib.import_module("aikido_exl3_moe_kernels").moe_build_orig_vllm(out, trellis_ptrs, suh_ptrs, svh_ptrs, n, row_offset, codebook, bits)
    return out


def build_w13_w2(w13: torch.Tensor, w2: torch.Tensor, gate, up, down, codebook: int, bits: int = 4) -> None:
    """gate / up / down = (trellis_ptrs, suh_ptrs, svh_ptrs). Fills w13 [E, 2 * inter, hidden] and w2 [E, hidden, inter]."""
    inter = w2.shape[2]
    build_experts_vllm(w13, *gate, inter, 0, codebook, bits)
    build_experts_vllm(w13, *up, inter, inter, codebook, bits)
    build_experts_vllm(w2, *down, w2.shape[1], 0, codebook, bits)


def build_w13_w2_from_pack(w13: torch.Tensor, w2: torch.Tensor, pack, out_ids: torch.Tensor | None = None) -> None:
    """The transient tier's builder: fills the arena (w13 [E, 2 * inter, hidden], w2 [E, hidden, inter]) from the grouped
    kernel's RESIDENT pack (kernels.hopper_moe.ExpertPack, K = 3 or K = 4: repacked trellises + stacked suh / svh), so no
    second copy of the experts is kept. Same values as build_w13_w2 (bit for bit; parity/original_basis_parity.py). Three
    launches, no allocation, shapes fixed: safe inside a captured graph and cheap enough (about 1.3 ms per layer on H200
    at full clocks) to run per chunk. out_ids (int32 [E_pack]): the arena expert of each pack expert (mixed-K layers)."""
    import importlib
    mod = importlib.import_module("aikido_exl3_moe_kernels")
    h, i, cb = pack.hidden, pack.inter, pack.codebook
    mod.moe_build_orig_vllm_stacked(w13, pack.w13, pack.suh13, pack.svh13, i, 0, 0, 0, 0, cb, out_ids)          # gate: cols 0.., suh slab 0
    mod.moe_build_orig_vllm_stacked(w13, pack.w13, pack.suh13, pack.svh13, i, i, h, i, i, cb, out_ids)          # up: cols inter.., suh slab 1
    mod.moe_build_orig_vllm_stacked(w2, pack.w2, pack.suh2, pack.svh2, h, 0, 0, 0, 0, cb, out_ids)


def build_w13_w2_from_mixed(w13: torch.Tensor, w2: torch.Tensor, mixed) -> None:
    """Mixed-K layer (kernels.hopper_moe.MixedPack, GLM-5.3 TR3): every K class writes its experts into their GLOBAL rows
    of the full arena (w13 [E, 2 * inter, hidden], w2 [E, hidden, inter]); 3 launches per class, no allocation. Afterwards
    vLLM's fused MoE runs once over all E experts with the router's ids as they are."""
    for pk, ids in zip(mixed.packs, mixed.expert_ids):
        build_w13_w2_from_pack(w13, w2, pk, ids)


def expert_bytes(num_experts: int, hidden: int, inter: int) -> int:
    """Resident fp16 bytes of one MoE layer's routed experts in the original basis (gate + up + down)."""
    return num_experts * 3 * hidden * inter * 2


def run_experts_grouped(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, w_gate: torch.Tensor,
                        w_up: torch.Tensor, w_down: torch.Tensor, combine, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Routed experts in the original basis with `torch._grouped_mm`, weights in ExLlamaV3's own orientation
    [E, in, out] (what reconstruct_had_batch writes: no transposed copy). x [T, hidden] fp16 | bf16 (cast to fp16 at the
    boundary, as the plugin does everywhere); topk_weights float32 [T, top_k]; topk_ids int32 | int64 [T, top_k].

    Shapes are static for a fixed T (argsort / bincount / cumsum stay on the device: no host sync). Ids outside [0, E)
    (dummy / padding tokens) sort behind every real expert and are skipped by the combine. The GLU is computed in fp32
    and rounded once to fp16; `combine(yd, w, ids, E, y)` is the fp32 router-weighted sum of kernels.hopper_moe
    (`moe_combine`), so only the matmuls differ from the trellis path."""
    tokens, hidden = x.shape
    e, top_k = w_gate.shape[0], topk_ids.shape[1]
    flat = topk_ids.reshape(-1)
    valid = (flat >= 0) & (flat < e)
    key = torch.where(valid, flat, e).to(torch.int64)
    order = torch.argsort(key, stable=True)
    offs = torch.cumsum(torch.bincount(key, minlength=e + 1)[:e], 0).to(torch.int32)
    xs = x.to(torch.float16).index_select(0, order // top_k)                     # [S, hidden], grouped by expert
    g = torch._grouped_mm(xs, w_gate, offs=offs)
    u = torch._grouped_mm(xs, w_up, offs=offs)
    act = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.float16)
    d = torch._grouped_mm(act, w_down, offs=offs)                                # rows behind offs[-1] are never used
    yd = torch.empty_like(d)
    yd.index_copy_(0, order, d)                                                  # back to slot order
    y = torch.empty((tokens, hidden), dtype=out_dtype or x.dtype, device=x.device)
    combine(yd, topk_weights, topk_ids.reshape(tokens, top_k).contiguous(), e, y)
    return y


# ---------------------------------------------------------------------------------------------------------------
# Lower-noise builder: the same W, folded in fp32 and rounded to fp16 ONCE.
#
# `reconstruct_had_batch` is ExLlamaV3's builder and the bit-exact reference of this tier. Whatever it rounds in between,
# its W is 9.2e-4 (mean relative) away from a float64 fold of the same decoded W_hat, and a MoE layer on it is 2.3e-3
# from float64 against 1.7e-3 for the rotated-basis kernels (research/12 section 7.4). The builder below decodes W_hat
# exactly (`reconstruct_batch`: the decoded values are fp16 by definition), applies both blockwise Hadamards as exact
# fp32 butterflies (additions / subtractions only, so no TF32 or matmul-precision setting can touch it), scales in fp32
# and rounds once. It is NOT bit-identical to ExLlamaV3's W (it is closer to the exact fold); which one ships is a
# reported decision, and the parity test always reports both.

def _fwht128_(x: torch.Tensor, dim: int) -> torch.Tensor:
    """In-place unnormalised Walsh-Hadamard transform over consecutive 128-blocks of `dim` (length % 128 == 0)."""
    x = x.movedim(dim, -1)
    shape = x.shape
    v = x.reshape(*shape[:-1], shape[-1] // 128, 128)
    if v.data_ptr() != x.data_ptr():
        raise ValueError("_fwht128_ needs a view (contiguous along the transformed dim after movedim)")
    h = 1
    while h < 128:
        w = v.view(*v.shape[:-1], 128 // (2 * h), 2, h)
        a, b = w[..., 0, :].clone(), w[..., 1, :]
        w[..., 0, :] = a + b
        w[..., 1, :] = a - b
        h *= 2
    return x.movedim(-1, dim)


def build_experts_fp32(trellis_ptrs: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, k: int, n: int, bits: float,
                       codebook: int, transpose: bool = False, chunk: int = 32) -> torch.Tensor:
    """suh fp16 [E, k], svh fp16 [E, n] (stacked) -> W fp16 [E, k, n], or [E, n, k] with transpose=True (vLLM's
    [E, out, in]: the transposition rides on the final rounding copy, there is no separate transposing pass)."""
    ext = reference._load()
    e = trellis_ptrs.shape[0]
    mcg, mul1 = _flags(codebook)
    out = torch.empty((e, n, k) if transpose else (e, k, n), dtype=torch.float16, device=trellis_ptrs.device)
    for e0 in range(0, e, chunk):
        e1 = min(e, e0 + chunk)
        w_hat = torch.empty((e1 - e0, k, n), dtype=torch.float16, device=out.device)
        ext.reconstruct_batch(w_hat, trellis_ptrs[e0:e1].contiguous(), bits, mcg, mul1)
        w = w_hat.float()                                   # [c, k, n], contiguous
        del w_hat
        _fwht128_(w, 2)                                     # W_hat H   (columns; last dim is contiguous)
        wt = w.transpose(1, 2).contiguous()                 # [c, n, k]
        del w
        _fwht128_(wt, 2)                                    # H W_hat H (rows)
        wt *= (1.0 / 128.0)                                 # two orthonormal 128-point transforms: 1/sqrt(128) each
        wt *= suh[e0:e1].float().unsqueeze(1)               # [c, 1, k]
        wt *= svh[e0:e1].float().unsqueeze(2)               # [c, n, 1]
        out[e0:e1] = wt if transpose else wt.transpose(1, 2)    # ONE rounding to fp16
        del wt
    return out
