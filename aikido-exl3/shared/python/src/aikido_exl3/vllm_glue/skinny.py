"""Opt-in skinny bf16 GEMM for vLLM's unquantized linears (AIKIDO_NONMOE_SKINNY=1, default off).

Every bf16 linear whose (N, K) is in kernels.skinny.PLAN (GLM-5.3 TP4: fused qkv_a, q_b, o_proj, indexer wq_b and
wk_weights_proj, shared-expert gate_up / down, dense MLP) goes through aikido_nonmoe::skinny_linear; that op runs
our kernel at M <= 32 rows and cuBLAS above. Installed by patching UnquantizedLinearMethod.apply, because some of
these layers are built with quant_config=None (indexer wk_weights_proj) and never reach our Exl3Config. Layers with
a bias, batch-invariant mode and non-bf16 weights keep vLLM's path. Shared-expert layers get their own workspace site
(they run on vLLM's aux stream next to the main stream).
"""
from __future__ import annotations

import os

import torch

from . import compat

ENABLED = os.environ.get("AIKIDO_NONMOE_SKINNY", "0") == "1"
MLA = os.environ.get("AIKIDO_NONMOE_MLA", os.environ.get("AIKIDO_NONMOE_SKINNY", "0")) == "1"   # W_UK / W_UV bmms
# qkv_a + indexer wk_weights_proj as ONE GEMM on the 21 indexer layers (both read hidden_states). The two weights
# become row views of one [2624 + 160, 6144] tensor (no extra memory); the outputs are column slices of one result.
MERGE = os.environ.get("AIKIDO_NONMOE_MERGE", "0") == "1"
_installed = False
logger = compat.init_logger(__name__)


def _site_of(layer) -> int:
    return 1 if "shared_expert" in getattr(layer, "prefix", "") else 0


def install() -> None:
    global _installed
    if MERGE and not _installed:
        _patch_merge()
    if _installed or not ENABLED:
        _installed = _installed or MERGE
        return
    if os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1":
        logger.info("AIKIDO_NONMOE_SKINNY ignored under VLLM_BATCH_INVARIANT")
        return
    from ..kernels import skinny as K
    from ..runtime import skinny_ops
    if not K.available():
        logger.warning("AIKIDO_NONMOE_SKINNY=1 but aikido_nonmoe_kernels is not importable; keeping cuBLAS")
        return
    K.check_plan()
    orig = compat.UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        w = getattr(layer, "weight", None)
        if (bias is None and w is not None and w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_contiguous()
                and (w.shape[0], w.shape[1]) in K.PLAN):
            return skinny_ops.skinny_linear(x, w, _site_of(layer))
        return orig(self, layer, x, bias)

    compat.UnquantizedLinearMethod.apply = apply
    if MLA:
        _patch_mla(skinny_ops)
    _installed = True
    logger.info("aikido skinny bf16 GEMM installed for %d shapes (pdl=%s, max rows %d)", len(K.PLAN), K.PDL,
                K.MAX_ROWS)


# vLLM 0.29.0 models/deepseek_v32/attention.py: the two absorbed-projection bmms, replaced in the method source by our
# custom ops (fails loudly if vLLM's source moved; nothing is patched in that case).
_MLA_EDITS = (
    ("forward", "ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)",
     "ql_nope = _aikido_mla_uk(q_nope, self.W_UK_T)"),
    ("_sparse_indexer_and_attn", "torch.bmm(x, self.W_UV, out=out)", "_aikido_mla_uv(x, self.W_UV, out)"),
)


def _patch_mla(skinny_ops) -> None:
    try:
        cls = compat.deepseek_v32_attention_cls()
    except ImportError as e:
        logger.info("AIKIDO_NONMOE_MLA: DeepseekV32Attention not importable here (%s)", e)
        return
    extra = dict(_aikido_mla_uk=skinny_ops.mla_uk, _aikido_mla_uv=skinny_ops.mla_uv)
    ok = all(_apply_edits(cls, meth, [(old, rep)], extra) for meth, old, rep in _MLA_EDITS)
    if ok:
        logger.info("aikido skinny GEMM: MLA W_UK / W_UV bmms patched in %s", cls.__name__)


_MERGE_OLD = """        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if self.indexer is not None and not self.skip_topk:
            kw = self.indexer.wk_weights_proj(hidden_states)[0]"""
_MERGE_NEW = """        _aikido_w = getattr(self, "_aikido_qkvw", None)
        if _aikido_w is not None and self.indexer is not None and not self.skip_topk:
            _aikido_y = torch.nn.functional.linear(hidden_states, _aikido_w)
            _aikido_n = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            qkv_lora, _aikido_kw = _aikido_y[:, :_aikido_n], _aikido_y[:, _aikido_n:]
        else:
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            _aikido_kw = None
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if self.indexer is not None and not self.skip_topk:
            kw = _aikido_kw if _aikido_kw is not None else self.indexer.wk_weights_proj(hidden_states)[0]"""


def _merge_weights(attn) -> None:
    """After loading: [qkv_a; wk_weights_proj] as one tensor, both parameters re-pointed at row views of it."""
    ind = getattr(attn, "indexer", None)
    if ind is None or getattr(attn, "skip_topk", False):
        return
    a, b = attn.fused_qkv_a_proj.weight, ind.wk_weights_proj.weight
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or a.shape[1] != b.shape[1] or a.device != b.device:
        return
    w = torch.cat([a.data, b.data], dim=0)
    a.data = w[: a.shape[0]]
    b.data = w[a.shape[0]:]
    attn.register_buffer("_aikido_qkvw", w, persistent=False)


def _patch_merge() -> None:
    try:
        cls = compat.deepseek_v32_attention_cls()
    except ImportError as e:
        logger.info("AIKIDO_NONMOE_MERGE: DeepseekV32Attention not importable here (%s)", e)
        return
    def undent(t):   # the edits are written at class-body indentation; _apply_edits works on dedented source
        return "\n".join(line[4:] for line in t.split("\n"))
    if not _apply_edits(cls, "forward", [(undent(_MERGE_OLD), undent(_MERGE_NEW))], {}):
        return
    orig_pwal = cls.process_weights_after_loading

    def process_weights_after_loading(self, *args, **kwargs):
        out = orig_pwal(self, *args, **kwargs)
        _merge_weights(self)
        return out

    cls.process_weights_after_loading = process_weights_after_loading
    logger.info("aikido: qkv_a + indexer wk_weights_proj merged into one GEMM on indexer layers")


_SRC: dict = {}   # method -> cumulative edited source (MLA and MERGE edits compose; getsource of an exec'd function
                  # would return the file's original text)


def _apply_edits(cls, meth: str, edits, extra: dict) -> bool:
    import inspect
    import textwrap
    fn = cls.__dict__[meth]
    base = inspect.unwrap(fn) if hasattr(fn, "__wrapped__") else fn
    key = (cls, meth)
    if key not in _SRC:
        _SRC[key] = (textwrap.dedent(inspect.getsource(fn)), dict(base.__globals__), inspect.getsourcefile(fn))
    src, g, fname = _SRC[key]
    for old, new in edits:
        if src.count(old) != 1:
            logger.warning("aikido: %s.%s source changed (edit not applied)", cls.__name__, meth)
            return False
    for old, new in edits:
        src = src.replace(old, new)
    g.update(extra)
    ns: dict = {}
    exec(compile(src, fname or "<aikido>", "exec"), g, ns)
    setattr(cls, meth, ns[meth])
    _SRC[key] = (src, g, fname)
    return True
