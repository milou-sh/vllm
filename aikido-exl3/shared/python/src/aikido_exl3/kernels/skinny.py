"""Skinny bf16 GEMM for decode-sized row counts (kernels/hopper/nonmoe/skinny_gemm.cu, module `aikido_nonmoe_kernels`).

out[M, N] = x[M, K] @ w[N, K]^T for unquantized bf16 linears at M <= 32 rows: weight-streaming mma.sync kernel with
the split-K reduction inside the launch (deterministic last-CTA fixup in split order), optional PDL. The launch
geometry per (N, K) and row count comes from PLAN, measured on box2 (H200 at 200 W / 345 MHz) with
wt/nonmoe.kiter/bench_skinny.py. Shapes not in PLAN, other dtypes and M > 32 are not handled here (caller falls back).
"""
from __future__ import annotations

import importlib
import os

import torch

_mod = None
PDL = os.environ.get("AIKIDO_NONMOE_PDL", "1") == "1"
MAX_ROWS = int(os.environ.get("AIKIDO_NONMOE_SKINNY_MAX_ROWS", "32"))

# (N, K) -> ((max rows, (rt, u, wk, sc, l2 prefetch) or None = cuBLAS), ...), ascending max rows. Filled from the box2 sweep (see module doc).
PLAN: dict = {}
# batched (per-head) GEMMs, MLA's absorbed W_UK / W_UV: (batch, N, K) -> same entry format
PLAN_BMM: dict = {}

_WS_FLOATS = 4 * 1024 * 1024      # 16 MB fp32 per site: >= groups * sc * rt*nt*128 for every PLAN entry (checked)
_sites: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _load():
    global _mod
    if _mod is None:
        _mod = importlib.import_module("aikido_nonmoe_kernels")
    return _mod


def available() -> bool:
    try:
        _load()
        return True
    except ImportError:
        return False


def config_for(n: int, k: int, m: int, batch: int = 0):
    for max_m, cfg in (PLAN_BMM.get((batch, n, k), ()) if batch else PLAN.get((n, k), ())):
        if m <= max_m:
            return cfg
    return None


def _site(device: torch.device, site: int):
    key = (device.index if device.index is not None else torch.cuda.current_device(), site)
    buf = _sites.get(key)
    if buf is None:
        buf = (torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
               torch.zeros(8192, dtype=torch.int32, device=device))
        _sites[key] = buf
    return buf


def skinny(x: torch.Tensor, w: torch.Tensor, cfg, site: int = 0, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] bf16, w [N, K] bf16 -> out [M, N] bf16, or batched [B, M, K] x [B, N, K] -> [B, M, N]; unit last
    strides, other strides free. Sites that can run at the same time on different streams (vLLM's shared-expert aux
    stream) must use different `site` ids: a site owns the split-K workspace and the arrival counters."""
    rt, u, wk, sc, pf = cfg
    if out is None:
        out = torch.empty(*x.shape[:-1], w.shape[-2], dtype=x.dtype, device=x.device)
    ws, cnt = _site(x.device, site)
    _load().skinny_gemm(x, w, out, ws, cnt, rt, u, wk, sc, PDL, pf)
    return out


def check_plan() -> None:
    items = [((1, n, k), e) for (n, k), e in PLAN.items()] + list(PLAN_BMM.items())
    for (b, n, k), entries in items:
        for max_m, cfg in entries:
            if cfg is None:          # measured: cuBLAS wins at this row count
                continue
            rt, u, wk, sc, pf = cfg
            nt = (min(max_m, 32) + 7) // 8
            need = b * (n // (rt * 16)) * sc * rt * nt * 128
            assert n % (rt * 16) == 0 and k % 32 == 0 and need <= _WS_FLOATS and b * n // (rt * 16) <= 8192, (b, n, k)




# ---- measured plan (box2, 200 W / 345 MHz, wt/nonmoe.kiter/r2a.json,r3a.json)
PLAN = {
    (160, 6144): ((1, (1, 2, 8, 16, 0)), (2, (1, 2, 8, 16, 0)), (4, (1, 2, 8, 16, 0)), (8, (1, 2, 8, 16, 0)), (16, (1, 2, 8, 16, 0)), (20, (1, 1, 8, 8, 0)), (32, (1, 1, 8, 16, 0))),
    (1024, 6144): ((1, (1, 2, 4, 8, 0)), (2, (1, 2, 8, 4, 0)), (4, (1, 2, 8, 4, 0)), (8, (2, 1, 8, 8, 0)), (16, None), (20, (1, 1, 8, 4, 0)), (32, None)),
    (4096, 2048): ((1, (1, 4, 8, 1, 0)), (2, (1, 4, 8, 1, 0)), (4, (1, 2, 8, 1, 0)), (8, (1, 2, 8, 1, 0)), (16, None), (20, None), (32, None)),
    (6144, 512): ((1, (1, 1, 8, 1, 0)), (2, (1, 1, 8, 1, 0)), (4, (1, 1, 8, 1, 0)), (8, (1, 4, 4, 1, 0)), (16, None), (20, None), (32, None)),
    (6144, 3072): ((1, (1, 2, 8, 1, 0)), (2, None), (4, None), (8, None), (16, None), (20, None), (32, None)),
    (6144, 4096): ((1, (1, 2, 8, 1, 0)), (2, (1, 2, 8, 1, 0)), (4, (1, 2, 8, 1, 0)), (8, None), (16, None), (20, None), (32, None)),
    (6144, 6144): ((1, (1, 2, 8, 1, 0)), (2, (1, 2, 8, 1, 0)), (4, (1, 2, 8, 1, 0)), (8, None), (16, None), (20, None), (32, None)),
}
PLAN_BMM = {
    (16, 256, 512): ((1, (1, 1, 8, 1, 0)), (2, (1, 1, 8, 1, 0)), (4, (1, 1, 8, 1, 0)), (8, (1, 2, 8, 1, 0)), (16, (1, 2, 8, 1, 0)), (20, None), (32, None)),
    (16, 512, 192): ((1, (1, 2, 4, 1, 0)), (2, (1, 1, 4, 1, 0)), (4, (1, 2, 4, 1, 0)), (8, (1, 2, 4, 1, 0)), (16, None), (20, None), (32, None)),
}
