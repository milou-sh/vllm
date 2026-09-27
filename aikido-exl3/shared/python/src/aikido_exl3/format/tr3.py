"""TR3 rank-sliced EXL3 checkpoints (GLM-5.2 / GLM-5.3 `hybrid_tr3_tail`), header level only.

In a TR3 checkpoint every routed expert is stored as one EXL3 matrix PER TENSOR-PARALLEL RANK:

    model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.rank{r}.{trellis|suh|svh|mcg}

gate/up are N-sliced (rank r owns output rows [I/tp * r, I/tp * (r+1)) of [I, H]), down is K-sliced (rank r owns input
columns of [H, I]), and each slice has its own suh/svh, so a rank's slices are ordinary EXL3 linears of shape
gate/up (k=H, n=I/tp) and down (k=I/tp, n=H). K may differ per expert (mixed K3/K4 in the 3.25 bpw GLM-5.3 quant);
`tier_bitmap.json` declares it, the trellis shapes are the truth. Everything that is not an expert slice (attention,
dense MLPs, shared experts, router, norms, embeddings, lm_head) is stored as plain tensors, BF16 in the checkpoints seen
so far, and is left to the model's own loader.

`config.json.quantization_config` in these checkpoints is a `modelopt`/`NVFP4` dispatch shim for a vLLM fork, so this
module never consults it: it reads the safetensors headers, `hybrid_tr3_tail` and `tier_bitmap.json` only.

Pure Python (L0): no torch, no vllm. Reference: research/05 section 2.2, davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .manifest import Manifest, build_manifest
from .safetensors_header import read_headers
from .spec import FormatError, MatrixSpec

PROJS = ("gate_proj", "up_proj", "down_proj")
RANK_KEY = re.compile(r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.experts)\.(?P<expert>\d+)\."
                      r"(?P<proj>gate_proj|up_proj|down_proj)\.rank(?P<rank>\d+)$")
TR3_FORMAT = "exl3-trellis"


@dataclass(frozen=True)
class ExpertSlices:
    """One expert of one layer: proj -> rank -> MatrixSpec."""
    layer: int
    expert: int
    slices: dict[str, dict[int, MatrixSpec]]

    def k_bits(self) -> float:
        bits = {s.bits.value for per_rank in self.slices.values() for s in per_rank.values()}
        return bits.pop() if len(bits) == 1 else float("nan")

    def nbytes_rank(self, rank: int) -> int:
        return sum(per_rank[rank].nbytes for per_rank in self.slices.values() if rank in per_rank)


@dataclass
class Tr3Plan:
    tp: int
    experts: dict[tuple[int, int], ExpertSlices] = field(default_factory=dict)   # (layer, expert) -> slices
    other_matrices: dict[str, MatrixSpec] = field(default_factory=dict)            # stock EXL3 linears, if any
    declared: dict = field(default_factory=dict)                                   # hybrid_tr3_tail
    tier_bitmap: dict[int, list[int]] = field(default_factory=dict)                # layer -> K per expert
    expected_layers: list[int] = field(default_factory=list)
    experts_per_layer: int | None = None
    files_missing: list[str] = field(default_factory=list)                         # index entries with no shard on disk
    warnings: list[str] = field(default_factory=list)

    # ---- derived views -------------------------------------------------------------------------------------
    @property
    def layers(self) -> list[int]:
        return sorted({layer for layer, _ in self.experts})

    @property
    def complete(self) -> bool:
        return not self.files_missing and (not self.expected_layers or set(self.expected_layers) <= set(self.layers))

    def layer_experts(self, layer: int) -> list[ExpertSlices]:
        return [e for (l, _), e in sorted(self.experts.items()) if l == layer]

    def k_histogram(self, layer: int) -> Counter:
        return Counter(e.k_bits() for e in self.layer_experts(layer))

    def kernel_classes(self) -> Counter:
        """(proj, k, n, K, codebook) -> number of slices. What the grouped MoE kernel must serve."""
        c = Counter()
        for e in self.experts.values():
            for proj, per_rank in e.slices.items():
                for s in per_rank.values():
                    c[(proj, s.k, s.n, str(s.bits), s.codebook.value)] += 1
        return c

    def expert_bytes_per_rank(self) -> list[int]:
        return [sum(e.nbytes_rank(r) for e in self.experts.values()) for r in range(self.tp)]

    def summary(self) -> dict:
        layers = self.layers
        per_layer_k = {}
        for layer in layers:
            per_layer_k[layer] = {f"K={k:g}": n for k, n in sorted(self.k_histogram(layer).items())}
        k_pattern = Counter(json.dumps(v, sort_keys=True) for v in per_layer_k.values())
        return {
            "format": TR3_FORMAT, "tp": self.tp, "complete": self.complete,
            "layers_present": len(layers), "layers_expected": len(self.expected_layers) or None,
            "layers_missing": sorted(set(self.expected_layers) - set(layers)),
            "experts_per_layer": self.experts_per_layer,
            "experts_present": len(self.experts),
            "k_per_layer_patterns": {pat: n for pat, n in k_pattern.most_common()},
            "kernel_classes": [{"proj": p, "k": k, "n": n, "K": K, "codebook": cb, "slices": c}
                               for (p, k, n, K, cb), c in sorted(self.kernel_classes().items())],
            "expert_bytes_per_rank": self.expert_bytes_per_rank(),
            "other_exl3_linears": len(self.other_matrices),
            "files_missing": len(self.files_missing),
            "declared": {k: self.declared.get(k) for k in ("producer", "bits", "bits_scheme", "codebook", "tp", "moe_layers",
                                                            "experts_per_layer", "k_values", "exllamav3_version")
                         if k in self.declared},
            "warnings": list(self.warnings),
        }


def build_tr3_plan(man: Manifest, declared: dict | None = None, tier_bitmap: dict | None = None,
                   files_missing: list[str] | None = None) -> Tr3Plan:
    """Group the manifest's rank-sliced EXL3 linears into experts and validate the TR3 structure.

    Raises FormatError on structural violations (ranks missing inside an expert, projections missing, shapes that
    do not mirror, a rank set that differs between experts). Layers that are absent altogether (partial download)
    are reported through `expected_layers` / `complete`, not raised, so a preflight can run on a growing directory.
    """
    declared = dict(declared or {})
    plan = Tr3Plan(tp=0, declared=declared, files_missing=list(files_missing or []))
    grouped: dict[tuple[int, int], dict[str, dict[int, MatrixSpec]]] = defaultdict(lambda: defaultdict(dict))
    for key, spec in man.matrices.items():
        m = RANK_KEY.match(key)
        if m is None:
            plan.other_matrices[key] = spec
            continue
        grouped[(int(m["layer"]), int(m["expert"]))][m["proj"]][int(m["rank"])] = spec
    if not grouped:
        raise FormatError("no rank-sliced expert tensors (…experts.E.<proj>.rankR.trellis): not a TR3 checkpoint")

    rank_sets = {frozenset(r for per_rank in projs.values() for r in per_rank) for projs in grouped.values()}
    if len(rank_sets) != 1:
        raise FormatError(f"experts disagree on the rank set: {sorted(map(sorted, rank_sets))[:4]}")
    ranks = sorted(rank_sets.pop())
    tp = len(ranks)
    if ranks != list(range(tp)):
        raise FormatError(f"ranks are not 0..{tp - 1}: {ranks}")
    if declared.get("tp") not in (None, tp):
        raise FormatError(f"hybrid_tr3_tail.tp={declared.get('tp')} but the tensors carry {tp} ranks")
    plan.tp = tp

    for (layer, expert), projs in sorted(grouped.items()):
        missing = [f"{p}.rank{r}" for p in PROJS for r in ranks if r not in projs.get(p, {})]
        if missing:
            raise FormatError(f"layer {layer} expert {expert}: missing slices {missing[:6]}"
                              f"{' …' if len(missing) > 6 else ''}")
        for r in ranks:
            g, u, d = projs["gate_proj"][r], projs["up_proj"][r], projs["down_proj"][r]
            if (g.k, g.n) != (u.k, u.n):
                raise FormatError(f"layer {layer} expert {expert} rank {r}: gate {g.k}x{g.n} != up {u.k}x{u.n}")
            if (d.k, d.n) != (g.n, g.k):
                raise FormatError(f"layer {layer} expert {expert} rank {r}: down {d.k}x{d.n} does not mirror gate {g.k}x{g.n}")
        plan.experts[(layer, expert)] = ExpertSlices(layer, expert, {p: dict(projs[p]) for p in PROJS})

    # per-layer consistency: one shape class and one codebook per layer (grouped launches), K per expert uniform
    for layer in plan.layers:
        exps = plan.layer_experts(layer)
        shapes = {(e.slices["gate_proj"][0].k, e.slices["gate_proj"][0].n) for e in exps}
        if len(shapes) != 1:
            raise FormatError(f"layer {layer}: experts have different slice shapes {sorted(shapes)}")
        codebooks = {s.codebook for e in exps for pr in e.slices.values() for s in pr.values()}
        if len(codebooks) != 1:
            plan.warnings.append(f"layer {layer}: mixed codebooks {sorted(c.value for c in codebooks)}")
        mixed = [e.expert for e in exps if e.k_bits() != e.k_bits()]   # NaN: K differs inside the expert
        if mixed:
            plan.warnings.append(f"layer {layer}: {len(mixed)} experts mix K across projections/ranks, e.g. {mixed[0]}")
        counts = Counter(e.expert for e in exps)
        if any(c != 1 for c in counts.values()):
            raise FormatError(f"layer {layer}: duplicate expert ids")

    # declared structure -> expectations (partial directories are reported, not rejected)
    moe = declared.get("moe_layers")
    if isinstance(moe, (list, tuple)) and len(moe) == 2:
        plan.expected_layers = list(range(int(moe[0]), int(moe[1]) + 1))
    epl = declared.get("experts_per_layer")
    plan.experts_per_layer = int(epl) if epl is not None else None
    if plan.experts_per_layer:
        short = {layer: len(plan.layer_experts(layer)) for layer in plan.layers
                 if len(plan.layer_experts(layer)) != plan.experts_per_layer}
        if short:
            raise FormatError(f"layers with an expert count != {plan.experts_per_layer}: "
                              f"{dict(list(short.items())[:4])}")

    # tier_bitmap.json (declared K per expert) against the tensors
    if tier_bitmap:
        for layer, entry in tier_bitmap.items():
            ks = entry.get("k") if isinstance(entry, dict) else entry
            try:
                layer_i, ks = int(layer), [float(k) for k in ks]
            except (TypeError, ValueError):
                plan.warnings.append(f"tier_bitmap: unreadable entry for layer {layer!r}")
                continue
            plan.tier_bitmap[layer_i] = ks
            exps = plan.layer_experts(layer_i)
            if not exps:
                continue
            if len(ks) != len(exps):
                plan.warnings.append(f"tier_bitmap layer {layer_i}: {len(ks)} entries, {len(exps)} experts in the tensors")
                continue
            bad = [e.expert for e in exps if ks[e.expert] != e.k_bits()]
            if bad:
                plan.warnings.append(f"tier_bitmap layer {layer_i}: K disagrees with the trellis shapes for "
                                     f"{len(bad)} experts, e.g. {bad[0]} (tensors win)")
    return plan


def load_tr3(model_dir: str | os.PathLike) -> tuple[Manifest, Tr3Plan]:
    """Manifest + TR3 plan of a local checkpoint directory (headers only; tolerates a partial download)."""
    model_dir = os.fspath(model_dir)
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FormatError(f"{model_dir}: no .safetensors files")
    headers = read_headers(files)
    weight_map, files_missing = {}, []
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f).get("weight_map", {})
        present = {os.path.basename(f) for f in files}
        files_missing = sorted(set(weight_map.values()) - present)
    man = build_manifest(headers, quant_config={}, weight_map=weight_map)   # the modelopt shim is deliberately ignored
    declared, tier = {}, None
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        declared = cfg.get("hybrid_tr3_tail") or {}
        if declared and declared.get("format") not in (None, TR3_FORMAT):
            raise FormatError(f"hybrid_tr3_tail.format={declared.get('format')!r}, expected {TR3_FORMAT!r}")
    tier_name = str(declared.get("tier_bitmap") or "tier_bitmap.json")
    if os.path.isabs(tier_name):
        raise FormatError("hybrid_tr3_tail.tier_bitmap must be checkpoint-relative")
    model_root = os.path.realpath(model_dir)
    tier_path = os.path.realpath(os.path.join(model_root, tier_name))
    if os.path.commonpath((model_root, tier_path)) != model_root:
        raise FormatError("hybrid_tr3_tail.tier_bitmap escapes the checkpoint directory")
    if os.path.exists(tier_path):
        with open(tier_path) as f:
            tier = json.load(f)
    plan = build_tr3_plan(man, declared, tier, files_missing)
    if files_missing:
        plan.warnings.append(f"{len(files_missing)} shards listed in the index are not on disk yet, e.g. {files_missing[0]}")
    return man, plan
