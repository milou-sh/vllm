"""Custom op for the skinny bf16 linear (AIKIDO_NONMOE_SKINNY=1). Opaque to Dynamo: the row-count switch between
our kernel (M <= MAX_ROWS and a planned shape) and cuBLAS lives inside, so the traced graph is shape-generic."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..kernels import skinny as K

_lib = torch.library.Library("aikido_nonmoe", "FRAGMENT")  # must stay alive at module scope
_lib.define("skinny_linear(Tensor x, Tensor w, int site) -> Tensor")


def _impl(x: torch.Tensor, w: torch.Tensor, site: int) -> torch.Tensor:
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    cfg = K.config_for(w.shape[0], k, m) if 0 < m <= K.MAX_ROWS else None
    if cfg is None or x2.stride(-1) != 1 or x2.dtype != torch.bfloat16 or (x2.stride(0) % 8) or (x2.data_ptr() % 16):
        return F.linear(x, w)
    return K.skinny(x2, w, cfg, site).view(*x.shape[:-1], w.shape[0])


def _fake(x: torch.Tensor, w: torch.Tensor, site: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w.shape[0]))


_lib.impl("skinny_linear", _impl, "CUDA")
torch.library.register_fake("aikido_nonmoe::skinny_linear")(_fake)


def skinny_linear(x: torch.Tensor, w: torch.Tensor, site: int) -> torch.Tensor:
    return torch.ops.aikido_nonmoe.skinny_linear(x, w, site)


# ---- MLA absorbed projections (vLLM DeepseekV32Attention): ql_nope = q_nope @ W_UK_T per head, out = x @ W_UV per head.
# W_UK_T [H, P, L] is N-contiguous (a permuted view of kv_b_proj.weight); the kernel needs K-contiguous rows, so a
# K-major copy [H, L, P] is built once per layer (3.1 MB per layer per rank on GLM-5.3 TP4) and cached by address.
_uk_kmajor: dict[int, torch.Tensor] = {}
_lib.define("mla_uk(Tensor q_nope, Tensor w_uk_t) -> Tensor")
_lib.define("mla_uv(Tensor x, Tensor w_uv, Tensor(a!) out) -> ()")


def _uk_impl(q_nope: torch.Tensor, w_uk_t: torch.Tensor) -> torch.Tensor:
    m = q_nope.shape[0]
    h, p, l = w_uk_t.shape
    wk = _uk_kmajor.get(w_uk_t.data_ptr())
    if wk is None and w_uk_t.dtype == torch.bfloat16:
        wk = w_uk_t.transpose(1, 2).contiguous()
        _uk_kmajor[w_uk_t.data_ptr()] = wk
    xt = q_nope.transpose(0, 1)                                   # [H, M, P]
    cfg = K.config_for(l, p, m, batch=h) if 0 < m <= K.MAX_ROWS and wk is not None else None
    if cfg is None or xt.stride(-1) != 1 or xt.stride(1) % 8 or xt.stride(0) % 8 or xt.data_ptr() % 16:
        return torch.bmm(xt, w_uk_t).transpose(0, 1)
    out = torch.empty(h, m, l, dtype=q_nope.dtype, device=q_nope.device)
    K.skinny(xt, wk, cfg, 0, out)
    return out.transpose(0, 1)


def _uk_fake(q_nope: torch.Tensor, w_uk_t: torch.Tensor) -> torch.Tensor:
    return q_nope.new_empty((w_uk_t.shape[0], q_nope.shape[0], w_uk_t.shape[2])).transpose(0, 1)


def _uv_impl(x: torch.Tensor, w_uv: torch.Tensor, out: torch.Tensor) -> None:
    h, m, l = x.shape
    v = w_uv.shape[2]
    wt = w_uv.transpose(1, 2)                                     # [H, V, L], K-contiguous when w_uv is kv_b's view
    cfg = K.config_for(v, l, m, batch=h) if 0 < m <= K.MAX_ROWS else None
    if (cfg is None or wt.stride(-1) != 1 or x.stride(-1) != 1 or out.stride(-1) != 1 or x.stride(1) % 8
            or x.stride(0) % 8 or wt.stride(1) % 8 or wt.stride(0) % 8 or x.data_ptr() % 16 or wt.data_ptr() % 16):
        torch.bmm(x, w_uv, out=out)
        return
    K.skinny(x, wt, cfg, 0, out)


def _uv_fake(x: torch.Tensor, w_uv: torch.Tensor, out: torch.Tensor) -> None:
    return None


_lib.impl("mla_uk", _uk_impl, "CUDA")
_lib.impl("mla_uv", _uv_impl, "CUDA")
torch.library.register_fake("aikido_nonmoe::mla_uk")(_uk_fake)
torch.library.register_fake("aikido_nonmoe::mla_uv")(_uv_fake)


def mla_uk(q_nope: torch.Tensor, w_uk_t: torch.Tensor) -> torch.Tensor:
    return torch.ops.aikido_nonmoe.mla_uk(q_nope, w_uk_t)


def mla_uv(x: torch.Tensor, w_uv: torch.Tensor, out: torch.Tensor) -> None:
    torch.ops.aikido_nonmoe.mla_uv(x, w_uv, out)
