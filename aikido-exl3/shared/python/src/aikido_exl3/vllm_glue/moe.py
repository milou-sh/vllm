"""Exl3MoEMethod: routed experts stored as ExLlamaV3 stores them, one EXL3 linear per expert per projection.

Status (M2 first cut): TP=1, no expert parallelism, SiLU-gated experts, one K and one codebook per
projection group, executed by kernels.reference.moe_mgemm (correct, graph-capturable, slow for long
prefills). It uses vLLM's direct `apply` path rather than the modular-kernel framework; moving to a
FusedMoEExpertsModular subclass is tracked in docs/STATUS.md.
"""
from __future__ import annotations

import os
import re

import torch

from ..kernels import original_basis, reference
from . import compat

logger = compat.init_logger(__name__)
_CODEBOOK_ID = {"3inst": 0, "mcg": 1, "mul1": 2}
_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
_RANK_RE = re.compile(r"\.rank(\d+)$")     # TR3 rank-sliced experts (format/tr3.py): <expert>.<proj>.rank<r>
_DEBUG_CALLS = int(os.environ.get("AIKIDO_EXL3_HOPPER_MOE_DEBUG", "0"))     # MoE front: log the first n hopper_moe inputs
_ALIGN_DECODE = os.environ.get("AIKIDO_EXL3_ALIGN_DECODE", "1") == "1"       # mixed-K decode routing in one launch (0 = vLLM's align)
# MoE front, OPT-IN large-batch tier in the ORIGINAL basis (kernels/original_basis.py = ExLlamaV3's own >= 1024-row
# arithmetic): fp16 W = diag(suh) H W_hat H diag(svh) per expert through vLLM's unquantized fused MoE: no activation
# Hadamards. Opt-in because with ExLlamaV3's builder it is further from float64 than the trellis path (2.3e-3 vs 1.7e-3,
# research/12 section 9); decode and small batches always use the trellis kernel.
#   AIKIDO_EXL3_MOE_ORIG=1          tier on (default off). A layer is either
#     RESIDENT  W kept on the GPU (1.5 GiB per layer on the 35B-A3B), used from _MIN_TOKENS tokens per step, or
#     TRANSIENT W rebuilt per step from the grouped kernel's resident pack into ONE arena shared by all layers (allocated
#               at load, so vLLM's memory profiling sees it; csrc/exl3_moe_orig_builder.cu, ~1.3 ms per layer), used from
#               _TRANSIENT_MIN_TOKENS tokens per step. Extra memory of the whole tier = the arena.
#   AIKIDO_EXL3_MOE_ORIG_CACHE_GB   GiB of RESIDENT layers (default 0 = all transient); layers become resident in load order
#                                   until the budget is used up. A positive value alone also turns the tier on.
#   AIKIDO_EXL3_MOE_ORIG_BUILDER    exl3 (default) = ExLlamaV3's W bit for bit (resident: exllamav3_ext.reconstruct_had_batch;
#                                   transient: our builder, bit-identical to it); fp32 = the same fold in fp32 with ONE
#                                   rounding (closer to the exact W, not bit-identical to ExLlamaV3's; RESIDENT ONLY: it is a
#                                   122 ms per layer torch builder, so layers outside the budget stay on the trellis kernel).
#   AIKIDO_EXL3_MOE_ORIG_MIN_TOKENS / _TRANSIENT_MIN_TOKENS   measured crossovers with the trellis kernel (512 / 4096).
_ORIG_CAP = float(os.environ.get("AIKIDO_EXL3_MOE_ORIG_CACHE_GB", "0"))
_ORIG_ON = os.environ.get("AIKIDO_EXL3_MOE_ORIG", "0") == "1" or _ORIG_CAP > 0
_ORIG_BUDGET = _ORIG_CAP * 2 ** 30
_ORIG_BUILDER = os.environ.get("AIKIDO_EXL3_MOE_ORIG_BUILDER", "exl3")
_ORIG_MIN_TOKENS = int(os.environ.get("AIKIDO_EXL3_MOE_ORIG_MIN_TOKENS", "512"))
_ORIG_TRANSIENT_MIN_TOKENS = int(os.environ.get("AIKIDO_EXL3_MOE_ORIG_TRANSIENT_MIN_TOKENS", "12288"))   # break-even ~7k tokens/step on GLM-5.3 (build 97 ms/layer vs 17 us/token trellis), research/12 §12.4
_orig_used = 0.0
_orig_layers = [0, 0]      # resident, not resident
_orig_arena: dict = {}     # (device index, E, hidden, inter) -> (w13, w2): the transient tier's only buffers


def _validate_expert_tensors(prefix: str, proj: str, expert: int, tensors: dict, expected: tuple, k: int, n: int) -> None:
    expected_k, expected_n, expected_twice, expected_codebook = expected[:4]
    if (expected_k, expected_n) != (k, n):
        raise ValueError(f"{prefix}: {proj}_proj expert {expert} header shape {(expected_k, expected_n)} "
                         f"does not match the vLLM layer shape {(k, n)}")
    trellis = tensors["trellis"][expert]
    if trellis.dtype != torch.int16 or tuple(trellis.shape[:2]) != (k // 16, n // 16):
        raise ValueError(f"{prefix}: {proj}_proj expert {expert} expected I16 trellis {(k // 16, n // 16)}, "
                         f"got {trellis.dtype} {tuple(trellis.shape)}")
    if trellis.ndim != 3 or trellis.shape[2] != expected_twice * 8:
        raise ValueError(f"{prefix}: {proj}_proj expert {expert} trellis does not match the header bitrate")
    for suffix, size in (("suh", k), ("svh", n)):
        scale = tensors[suffix][expert]
        if scale.dtype != torch.float16 or tuple(scale.shape) != (size,):
            raise ValueError(f"{prefix}: {proj}_proj expert {expert} expected F16 {suffix} {(size,)}, "
                             f"got {scale.dtype} {tuple(scale.shape)}")
    markers = [suffix for suffix in ("mcg", "mul1") if expert in tensors[suffix]]
    if len(markers) > 1:
        raise ValueError(f"{prefix}: {proj}_proj expert {expert} has both MCG and MUL1 markers")
    codebook = markers[0] if markers else "3inst"
    if codebook != expected_codebook:
        raise ValueError(f"{prefix}: {proj}_proj expert {expert} codebook {codebook} does not match "
                         f"the header codebook {expected_codebook}")
    if markers:
        marker = tensors[markers[0]][expert]
        if marker.dtype != torch.int32 or tuple(marker.shape) not in ((), (1,)):
            raise ValueError(f"{prefix}: {proj}_proj expert {expert} has malformed {markers[0]} marker")


def _transient_arena(device, num_experts: int, hidden: int, inter: int):
    """The transient tier's ONE arena per (device, shape), allocated at load so vLLM's memory profiling sees it."""
    key = (device.index, num_experts, hidden, inter)
    if key not in _orig_arena:
        f16 = dict(dtype=torch.float16, device=device)
        _orig_arena[key] = (torch.empty((num_experts, 2 * inter, hidden), **f16), torch.empty((num_experts, hidden, inter), **f16))
    return _orig_arena[key]


class _Tr3RankSlot(torch.nn.Module):
    """TR3 (GLM-5.3, rank-sliced experts) loader hook for models with a hand-rolled `load_weights` (DeepSeek-V2 family):
    such models route `…experts.E.<proj>.rank<r>.<suffix>` through their expert_params_mapping to the parameter named
    `…experts.routed_experts.w13_rank<r>.<suffix>` (w2_ for down). One child module per rank with zero-sized parameters
    named like the checkpoint suffixes makes that lookup succeed; the slot of a rank that is not ours answers
    "not on this rank" (return False), which the model loader treats as a skip - the same convention it uses for EP.
    GLM/TR3-only: the stock (Qwen-style) path is untouched."""

    def __init__(self, mine: bool, store: dict, role: dict, device):
        super().__init__()
        self.mine, self.store, self.role, self.device = mine, store, role, device
        for suffix in _SUFFIXES:
            param = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
            param.exl3_suffix = suffix
            param.weight_loader = self._load
            setattr(self, suffix, param)

    def _load(self, param, loaded_weight, weight_name=None, shard_id=None, expert_id=None, return_success=False):
        if not self.mine:
            return False if return_success else None
        self.store.setdefault((self.role[shard_id], param.exl3_suffix), {})[int(expert_id)] = \
            loaded_weight.to(self.device, copy=True)
        return True if return_success else None


class Exl3MoEMethod(compat.FusedMoEMethodBase):
    def __init__(self, prefix: str, modules: dict[str, tuple], moe_config):
        super().__init__(moe_config)
        self.prefix = prefix            # "...mlp.experts"
        self.modules = modules          # header scan in vLLM names: key -> (k, n, 2K, codebook, bias, legacy)
        # TR3 rank-sliced checkpoint: every expert slice carries its rank in the name; the checkpoint's TP degree is
        # the number of distinct ranks and must equal the runtime TP (each rank loads only its own slices, no slicing).
        self.tr3_ranks = sorted({int(m.group(1)) for k in modules if (m := _RANK_RE.search(k))})

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        cfg = self.moe
        pc = cfg.moe_parallel_config
        tp_size, tp_rank = getattr(pc, "tp_size", 1), getattr(pc, "tp_rank", 0)
        if getattr(pc, "use_ep", False) or num_experts != cfg.num_experts:
            raise NotImplementedError(f"{self.prefix}: expert-parallel EXL3 MoE is not implemented yet (M4)")
        if self.tr3_ranks:
            # TR3: the checkpoint is pre-sliced; vLLM's per-partition intermediate size must equal the slice width
            if self.tr3_ranks != list(range(tp_size)):
                raise ValueError(f"{self.prefix}: TR3 checkpoint carries ranks {self.tr3_ranks} but runtime TP is {tp_size}; "
                                 f"run with --tensor-parallel-size {len(self.tr3_ranks)}")
            if intermediate_size_per_partition != cfg.intermediate_size_per_partition:
                raise NotImplementedError(f"{self.prefix}: unexpected partition {intermediate_size_per_partition}")
        elif tp_size != 1 or intermediate_size_per_partition != cfg.intermediate_size_per_partition:
            raise NotImplementedError(f"{self.prefix}: tensor-parallel EXL3 MoE needs a rank-sliced checkpoint (M4)")
        names = (layer.ckpt_gate_proj_name, layer.ckpt_up_proj_name, layer.ckpt_down_proj_name)
        pattern = re.compile(r"(?:^|\.)(\d+)\.(%s)(?:\.rank(\d+))?\.(%s)$"
                              % ("|".join(map(re.escape, names)), "|".join(_SUFFIXES)))
        module_pattern = re.compile(r"(?:^|\.)(\d+)\.(%s)(?:\.rank(\d+))?$"
                                    % "|".join(map(re.escape, names)))
        store: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        device = torch.device("cuda", torch.cuda.current_device())
        role = dict(zip(names, ("gate", "up", "down")))
        expected = {}
        for key, info in self.modules.items():
            match = module_pattern.search(key)
            if match is None or (match.group(3) is not None and int(match.group(3)) != tp_rank):
                continue
            expected[(role[match.group(2)], int(match.group(1)))] = info

        def load_weights(weights):
            # Replaces RoutedExperts.load_weights on this instance only: the stock one treats every rank-3
            # tensor as "all experts fused" (an EXL3 trellis is rank-3) and drops suffixes it does not know.
            # TR3: slices of other ranks are skipped by name; ours lose the .rank<r> in the key.
            for name, tensor in weights:
                m = pattern.search(name)
                if m is None:
                    raise ValueError(f"{self.prefix}: unexpected expert tensor {name!r}")
                if m.group(3) is not None and int(m.group(3)) != tp_rank:
                    continue
                store.setdefault((role[m.group(2)], m.group(4)), {})[int(m.group(1))] = tensor.to(device, copy=True)
                yield name

        layer.load_weights = load_weights
        if self.tr3_ranks:
            # Models with a hand-rolled load_weights (DeepSeek-V2 / GLM-5.3) never call layer.load_weights; they look
            # the destination parameter up by name. Give them one per rank (see _Tr3RankSlot).
            shard_role = {"w1": "gate", "w3": "up", "w2": "down"}
            for r in self.tr3_ranks:
                layer.add_module(f"w13_rank{r}", _Tr3RankSlot(r == tp_rank, store, shard_role, device))
                layer.add_module(f"w2_rank{r}", _Tr3RankSlot(r == tp_rank, store, shard_role, device))
        layer.exl3_store, layer.exl3_num_experts = store, num_experts
        layer.exl3_intermediate = intermediate_size_per_partition
        layer.exl3_hidden = hidden_size
        layer.exl3_expected = expected

    def get_fused_moe_quant_config(self, layer):
        return None     # only used by the modular kernel stack, which this first cut bypasses

    def process_weights_after_loading(self, layer) -> None:
        store, n = layer.exl3_store, layer.exl3_num_experts
        books, bits, per_expert_bits = set(), {}, {}
        for proj in ("gate", "up", "down"):
            tensors = {s: store.get((proj, s), {}) for s in _SUFFIXES}
            missing = [e for e in range(n) if not all(e in tensors[s] for s in ("trellis", "suh", "svh"))]
            if missing:
                raise ValueError(f"{self.prefix}: {proj}_proj tensors missing for {len(missing)} experts, e.g. "
                                 f"{missing[0]} (vLLM does not report missing tensors for quantized models)")
            k, out = ((layer.exl3_hidden, layer.exl3_intermediate) if proj != "down"
                      else (layer.exl3_intermediate, layer.exl3_hidden))
            for expert in range(n):
                header_spec = layer.exl3_expected.get((proj, expert))
                if header_spec is None:
                    raise ValueError(f"{self.prefix}: {proj}_proj expert {expert} was not present in the header scan")
                _validate_expert_tensors(self.prefix, proj, expert, tensors, header_spec, k, out)
            dims = {tuple(tensors["trellis"][e].shape[:2]) for e in range(n)}
            if len(dims) != 1:
                raise NotImplementedError(f"{self.prefix}: experts of {proj}_proj differ in shape {dims}")
            per_expert_bits[proj] = [tensors["trellis"][e].shape[2] / 16 for e in range(n)]
            bits[proj] = per_expert_bits[proj][0]
            books |= {"mcg" if e in tensors["mcg"] else "mul1" if e in tensors["mul1"] else "3inst" for e in range(n)}
            for s in ("trellis", "suh", "svh"):
                keep = [tensors[s][e].contiguous() for e in range(n)]
                setattr(layer, f"exl3_{proj}_{s}", keep)                    # keeps the storage alive and in place
                layer.register_buffer(f"exl3_{proj}_{s}_ptrs", reference.pointer_table(keep), persistent=False)
        if len(books) != 1 or per_expert_bits["gate"] != per_expert_bits["up"]:
            raise NotImplementedError(f"{self.prefix}: mixed codebooks {books} or gate/up bitrates differ per expert")
        layer.exl3_codebook, layer.exl3_bits = _CODEBOOK_ID[books.pop()], bits
        store.clear()
        for r in self.tr3_ranks:                                            # the loader hooks have done their job
            for w in ("w13", "w2"):
                if hasattr(layer, f"{w}_rank{r}"):
                    delattr(layer, f"{w}_rank{r}")
        # Mixed K per expert (TR3 3.25 bpw: 192 experts K=3 + 64 experts K=4 per layer): exl3_mgemm takes one K per
        # launch, so the experts are grouped by (K gate/up, K down); apply() runs one group after another with the
        # other groups' slots routed to a dummy expert at weight 0. Uniform-K checkpoints keep the single-launch path.
        classes = sorted({(per_expert_bits["gate"][e], per_expert_bits["down"][e]) for e in range(n)})
        layer.exl3_groups = None
        if len(classes) > 1:
            layer.exl3_groups = []
            for gi, (bgu, bd) in enumerate(classes):
                ids = [e for e in range(n) if (per_expert_bits["gate"][e], per_expert_bits["down"][e]) == (bgu, bd)]
                remap = torch.zeros(n, dtype=torch.long)
                remap[ids] = torch.arange(len(ids))
                member = torch.zeros(n, dtype=torch.float32)
                member[ids] = 1.0
                dev = layer.exl3_gate_trellis[0].device
                layer.register_buffer(f"exl3_g{gi}_remap", remap.to(dev), persistent=False)
                layer.register_buffer(f"exl3_g{gi}_member", member.to(dev), persistent=False)
                tables = {}
                for proj in ("gate", "up", "down"):
                    for s in ("trellis", "suh", "svh"):
                        subset = [getattr(layer, f"exl3_{proj}_{s}")[e] for e in ids]
                        layer.register_buffer(f"exl3_g{gi}_{proj}_{s}_ptrs", reference.pointer_table(subset), persistent=False)
                    tables[proj] = tuple(f"exl3_g{gi}_{proj}_{s}_ptrs" for s in ("trellis", "suh", "svh"))
                layer.exl3_groups.append({"bits_gate_up": bgu, "bits_down": bd, "count": len(ids), "tables": tables,
                                          "remap": f"exl3_g{gi}_remap", "member": f"exl3_g{gi}_member"})
            logger.info("%s: %d expert K classes %s -> %d grouped launches per step (reference path)",
                        self.prefix, len(classes), classes, len(classes))
        # ---- MoE front (begin): grouped Hopper expert kernels, AIKIDO_EXL3_HOPPER_MOE=1 (default off). The experts
        # are re-laid out once into stacked [E, ...] tensors (pure permutation); the per-expert tensors and pointer
        # tables of the fallback path are released so the experts are resident once.
        layer.exl3_hopper_moe = None
        layer.exl3_hopper_mixed = None
        if os.environ.get("AIKIDO_EXL3_HOPPER_MOE", "0") == "1" and layer.exl3_groups is not None:
            # Mixed K per expert (TR3): one grouped pack per K class routed through vLLM's expert_map (kernels/hopper_moe
            # MixedPack): one input transform, one GLU, one combine, one GEMM launch per class and projection, no dummy
            # experts. The per-expert tensors stay resident for the reference fallback.
            from ..kernels import hopper_moe
            status = hopper_moe.probe()
            per = lambda p: tuple(getattr(layer, f"exl3_{p}_{s}") for s in ("trellis", "suh", "svh"))
            ok, why = (True, "") if status.available else (False, status.detail)
            if ok:
                for g in layer.exl3_groups:
                    ids = (getattr(layer, g["member"]) > 0).nonzero().flatten().tolist()
                    sub = lambda p: tuple([getattr(layer, f"exl3_{p}_{s}")[e] for e in ids] for s in ("trellis", "suh", "svh"))
                    ok, why = hopper_moe.supports(sub("gate")[0], sub("up")[0], sub("down")[0], layer.exl3_codebook)
                    if not ok:
                        break
            if ok:
                layer.exl3_hopper_mixed = hopper_moe.prepare_mixed(per("gate"), per("up"), per("down"), layer.exl3_codebook)
                logger.info("%s: grouped Hopper kernel, K classes %s (%s experts)", self.prefix,
                            layer.exl3_hopper_mixed.bits, [p.num_experts for p in layer.exl3_hopper_mixed.packs])
                # Large-batch tier for mixed-K layers: TRANSIENT only (every K class builds its experts into their global
                # rows of one full arena; csrc/exl3_moe_orig_builder.cu K = 3 and K = 4), from _ORIG_TRANSIENT_MIN_TOKENS.
                layer.exl3_orig, layer.exl3_orig_transient = None, False
                if _ORIG_ON and _ORIG_BUILDER != "fp32":
                    mx = layer.exl3_hopper_mixed
                    layer.exl3_orig = _transient_arena(mx.suh13.device, mx.num_experts, mx.hidden, mx.inter)
                    layer.exl3_orig_transient = True
                    _orig_layers[1] += 1
                    if _orig_layers[1] == 1:
                        logger.warning("original-basis expert tier (mixed K): TRANSIENT from %d tokens per step, shared arena %.2f GiB",
                                       _ORIG_TRANSIENT_MIN_TOKENS, original_basis.expert_bytes(mx.num_experts, mx.hidden, mx.inter) / 2 ** 30)
                # The packs are the experts now: release the per-expert tensors, pointer tables and reference groups
                # (kept resident, they double the expert memory: 2 x 70 GiB per GPU on GLM-5.3 TR3 at TP4).
                for proj in ("gate", "up", "down"):
                    for s in ("trellis", "suh", "svh"):
                        setattr(layer, f"exl3_{proj}_{s}", None)
                        setattr(layer, f"exl3_{proj}_{s}_ptrs", None)
                for gi in range(len(layer.exl3_groups)):
                    for proj in ("gate", "up", "down"):
                        for s in ("trellis", "suh", "svh"):
                            setattr(layer, f"exl3_g{gi}_{proj}_{s}_ptrs", None)
                layer.exl3_groups = None
                torch.cuda.empty_cache()
            else:
                logger.warning("%s: grouped kernel cannot serve this mixed-K layer (%s); reference path", self.prefix, why)
        if os.environ.get("AIKIDO_EXL3_HOPPER_MOE", "0") == "1" and layer.exl3_groups is None and layer.exl3_hopper_mixed is None:
            from ..kernels import hopper_moe
            per = lambda p: tuple(getattr(layer, f"exl3_{p}_{s}") for s in ("trellis", "suh", "svh"))
            status = hopper_moe.probe()
            ok, why = (hopper_moe.supports(per("gate")[0], per("up")[0], per("down")[0], layer.exl3_codebook)
                       if status.available else (False, status.detail))
            if ok:
                layer.exl3_hopper_moe = hopper_moe.prepare(per("gate"), per("up"), per("down"), layer.exl3_codebook)
                layer.exl3_orig = None
                layer.exl3_orig_transient = False
                if _ORIG_ON:
                    global _orig_used
                    pk = layer.exl3_hopper_moe
                    need = original_basis.expert_bytes(pk.num_experts, pk.hidden, pk.inter)
                    if _orig_used + need <= _ORIG_BUDGET:
                        ptrs = lambda p: tuple(getattr(layer, f"exl3_{p}_{s}_ptrs") for s in ("trellis", "suh", "svh"))
                        if _ORIG_BUILDER == "fp32":
                            b32 = lambda p, su, sv, k, n: original_basis.build_experts_fp32(
                                ptrs(p)[0], su.contiguous(), sv.contiguous(), k, n, layer.exl3_bits[p], layer.exl3_codebook, transpose=True)
                            layer.exl3_orig = (
                                torch.cat([b32("gate", pk.suh13[:, 0], pk.svh13[:, :pk.inter], pk.hidden, pk.inter),
                                           b32("up", pk.suh13[:, 1], pk.svh13[:, pk.inter:], pk.hidden, pk.inter)], dim=1).contiguous(),
                                b32("down", pk.suh2, pk.svh2, pk.inter, pk.hidden))
                        else:
                            build = lambda p, k, n: original_basis.build_experts(*ptrs(p), k, n, layer.exl3_bits[p], layer.exl3_codebook)
                            layer.exl3_orig = original_basis.to_vllm_experts(build("gate", pk.hidden, pk.inter), build("up", pk.hidden, pk.inter),
                                                                             build("down", pk.inter, pk.hidden))
                        _orig_used += need
                        _orig_layers[0] += 1
                        if _orig_layers[0] == 1:
                            logger.warning("original-basis expert tier ON: builder=%s, RESIDENT from %d tokens per step, %.2f GiB per layer",
                                           _ORIG_BUILDER, _ORIG_MIN_TOKENS, need / 2 ** 30)
                    else:
                        _orig_layers[1] += 1
                        transient = _ORIG_BUILDER != "fp32"
                        if transient:
                            layer.exl3_orig = _transient_arena(pk.w13.device, pk.num_experts, pk.hidden, pk.inter)
                            layer.exl3_orig_transient = True
                        if _orig_layers[1] == 1:
                            logger.warning("original-basis expert tier: %d resident layers (budget %.1f GiB); further layers %s",
                                           _orig_layers[0], _ORIG_BUDGET / 2 ** 30,
                                           f"TRANSIENT from {_ORIG_TRANSIENT_MIN_TOKENS} tokens per step (shared arena {need / 2 ** 30:.2f} GiB)"
                                           if transient else "stay on the trellis kernel (fp32 builder is resident-only)")
                for proj in ("gate", "up", "down"):
                    for s in ("trellis", "suh", "svh"):
                        setattr(layer, f"exl3_{proj}_{s}", None)
                        setattr(layer, f"exl3_{proj}_{s}_ptrs", None)
            else:
                logger.warning("%s: AIKIDO_EXL3_HOPPER_MOE=1 but the grouped kernel cannot serve this layer (%s); "
                               "using exl3_mgemm", self.prefix, why)
        # ---- MoE front (end)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        del shared_experts, shared_experts_input     # vLLM's runner runs shared experts itself for this method
        if getattr(layer, "apply_router_weight_on_input", False) or layer.expert_map is not None:
            raise NotImplementedError(f"{self.prefix}: router-weight-on-input / expert maps are not supported yet")
        flat = x.reshape(-1, x.shape[-1])
        # ---- MoE front (begin): grouped Hopper expert kernels (5 launches + moe_align_block_size per layer)
        pack = getattr(layer, "exl3_hopper_moe", None)
        if pack is not None:
            from ..kernels import hopper_moe
            flat = flat.contiguous()
            ids = topk_ids.reshape(flat.shape[0], -1).contiguous()
            global _DEBUG_CALLS
            if _DEBUG_CALLS > 0:      # AIKIDO_EXL3_HOPPER_MOE_DEBUG=<n>: what the engine really passes (first n calls,
                _DEBUG_CALLS -= 1     # i.e. the eager profile run; it synchronises, so never leave it on when measuring)
                logger.warning("hopper_moe input: x %s %s contiguous=%s | ids %s %s min=%d max=%d outside[0,%d)=%d | w %s",
                               tuple(x.shape), x.dtype, x.is_contiguous(), tuple(topk_ids.shape), topk_ids.dtype,
                               int(ids.min()), int(ids.max()), pack.num_experts,
                               int(((ids < 0) | (ids >= pack.num_experts)).sum()), topk_weights.dtype)
            orig = getattr(layer, "exl3_orig", None)
            transient = getattr(layer, "exl3_orig_transient", False)
            if orig is not None and flat.shape[0] >= (_ORIG_TRANSIENT_MIN_TOKENS if transient else _ORIG_MIN_TOKENS):  # shapes only
                if transient:       # rebuild this layer's W into the shared arena: 3 launches, no allocation
                    original_basis.build_w13_w2_from_pack(orig[0], orig[1], pack)
                y = compat.fused_experts(flat.to(torch.float16), orig[0], orig[1], topk_weights.reshape(ids.shape).to(torch.float32),
                                         ids, global_num_experts=pack.num_experts)
                return y.to(x.dtype).view_as(x)
            block = hopper_moe.moe_block_size(flat.shape[0], ids.shape[1], pack.num_experts)   # shapes only: graph-safe
            routing = compat.moe_align_block_size(ids, block, pack.num_experts, None, ignore_invalid_experts=True)
            y = hopper_moe.run(flat, topk_weights.reshape(ids.shape).to(torch.float32).contiguous(), ids, *routing,
                               block, pack)
            return y.view_as(x)
        mixed = getattr(layer, "exl3_hopper_mixed", None)
        if mixed is not None:   # mixed K per expert (TR3) on the grouped Hopper kernel: one pack per K class
            from ..kernels import hopper_moe
            flat = flat.contiguous()
            ids = topk_ids.reshape(flat.shape[0], -1).contiguous()
            w = topk_weights.reshape(ids.shape).to(torch.float32).contiguous()
            orig = getattr(layer, "exl3_orig", None)
            if orig is not None and flat.shape[0] >= _ORIG_TRANSIENT_MIN_TOKENS:      # shapes only: graph-safe
                original_basis.build_w13_w2_from_mixed(orig[0], orig[1], mixed)       # 6 launches into the shared arena
                y = compat.fused_experts(flat.to(torch.float16), orig[0], orig[1], w, ids, global_num_experts=mixed.num_experts)
                return y.to(x.dtype).view_as(x)
            if ids.numel() <= hopper_moe.ALIGN_DECODE_MAX_SLOTS and _ALIGN_DECODE:
                routings = hopper_moe.align_decode(ids, mixed, flat.shape[0], ids.shape[1])     # one launch, both classes
            else:
                routings = []
                for pk, emap, block in zip(mixed.packs, mixed.expert_maps, mixed.block_sizes(flat.shape[0], ids.shape[1])):
                    routings.append((*compat.moe_align_block_size(ids, block, mixed.num_experts, emap, ignore_invalid_experts=True), block))
            return hopper_moe.run_mixed(flat, w, ids, routings, mixed).view_as(x)
        # ---- MoE front (end)
        groups = getattr(layer, "exl3_groups", None)
        if groups:      # mixed K per expert (TR3): one exl3_mgemm group per K class, dummy slots at weight 0
            ids = topk_ids.reshape(flat.shape[0], -1)
            w = topk_weights.reshape(ids.shape).to(torch.float32)
            y = None
            for g in groups:
                gid = getattr(layer, g["remap"])[ids]
                gw = w * getattr(layer, g["member"])[ids]
                tab = lambda p: tuple(getattr(layer, name) for name in g["tables"][p])
                yg = reference.moe_mgemm(flat, gid, gw, tab("gate"), tab("up"), tab("down"), g["bits_gate_up"],
                                         g["bits_down"], layer.exl3_intermediate, g["count"], layer.exl3_codebook)
                y = yg if y is None else y + yg
            return y.view_as(x)
        table = lambda p: tuple(getattr(layer, f"exl3_{p}_{s}_ptrs") for s in ("trellis", "suh", "svh"))
        y = reference.moe_mgemm(flat, topk_ids, topk_weights, table("gate"), table("up"), table("down"),
                                layer.exl3_bits["gate"], layer.exl3_bits["down"], layer.exl3_intermediate,
                                layer.exl3_num_experts, layer.exl3_codebook)
        return y.view_as(x)
