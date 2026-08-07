"""Validate the NVFP4->FP8 conversion against real GLM-5.2 checkpoint weights.

Synthetic uniform-random nibbles are a worst case for block quantization. This
measures the error the model will actually see, on real tensors.
"""

import glob
import sys

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    FP8_COMPUTE_BLOCK_SIZE,
    NVFP4_GROUP_SIZE,
    convert_nvfp4_weight_to_fp8_block,
    ensure_e2m1_table_on_device,
)

device = torch.device("cuda")
ensure_e2m1_table_on_device(device)
blk = FP8_COMPUTE_BLOCK_SIZE[0]

shards = sorted(glob.glob("/models/model-*.safetensors"))
print(f"shards: {len(shards)}")

# Find weight tensors that have both a block scale and a global scale.
examined = 0
results = []

for shard in shards[:6]:
    with safe_open(shard, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        weight_keys = [
            k
            for k in keys
            if k.endswith(".weight")
            and f"{k}_scale" in keys
            and f"{k}_scale_2" in keys
        ]
        # Prefer MoE expert weights, since those dominate a 504B MoE.
        expert_keys = [k for k in weight_keys if "experts" in k]
        chosen = (expert_keys or weight_keys)[:4]
        for k in chosen:
            w = f.get_tensor(k)
            if w.dtype != torch.uint8:
                continue
            s = f.get_tensor(f"{k}_scale")
            gs = f.get_tensor(f"{k}_scale_2")
            if w.ndim != 2:
                continue

            w = w.to(device)
            s = s.to(device)
            gs = gs.to(device).to(torch.float32).reshape(())

            N, Kh = w.shape
            K = Kh * 2
            if N % blk or K % blk:
                continue

            ref = dequantize_to_dtype(
                w, s, gs, torch.bfloat16, block_size=NVFP4_GROUP_SIZE, swizzle=False
            )
            w8, w8s = convert_nvfp4_weight_to_fp8_block(w, s, gs)
            recon = (
                w8.to(torch.float32).view(N // blk, blk, K // blk, blk)
                * w8s.to(torch.float32).view(N // blk, 1, K // blk, 1)
            ).reshape(N, K)

            cos = torch.nn.functional.cosine_similarity(
                recon.flatten(), ref.flatten().float(), dim=0
            ).item()
            rel = (
                (recon - ref.float()).norm() / ref.float().norm()
            ).item()
            results.append((k, tuple(w.shape), cos, rel))
            examined += 1
            del w, s, ref, w8, w8s, recon
            torch.cuda.empty_cache()
    if examined >= 8:
        break

if not results:
    print("NO SUITABLE TENSORS FOUND")
    sys.exit(1)

print(f"\n{'tensor':<62} {'shape':<18} {'cosine':>9} {'rel_err':>9}")
for k, shape, cos, rel in results:
    print(f"{k[:60]:<62} {str(shape):<18} {cos:>9.6f} {rel:>9.5f}")

worst_cos = min(r[2] for r in results)
worst_rel = max(r[3] for r in results)
print(f"\nworst cosine : {worst_cos:.6f}")
print(f"worst rel_err: {worst_rel:.5f}")

# FP8 e4m3 carries 3 mantissa bits, so a ~6.25% max / ~3% RMS relative step is
# inherent to the format, not a conversion defect. Cosine is the metric that
# tracks model quality; upstream's own bar is 0.99.
ok = worst_cos > 0.999 and worst_rel < 0.05
print("\n" + ("REAL-WEIGHT FIDELITY OK" if ok else "REAL-WEIGHT FIDELITY BELOW THRESHOLD"))
sys.exit(0 if ok else 1)
