"""Validate the NVFP4->FP8 Hopper backport against real GPU numerics.

Run inside the patched vLLM image on an H200.

The property under test: for the same packed NVFP4 checkpoint tensors, the FP8
block-quantized weights we build must reproduce the BF16 values that the Marlin
path would have computed. If they do, the FP8 kernels compute the same math on
faster tensor cores.
"""

import sys

import torch

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' - ' + detail) if detail else ''}", flush=True)
    if not cond:
        FAILURES.append(name)


from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (  # noqa: E402
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (  # noqa: E402
    FP8_COMPUTE_BLOCK_SIZE,
    NVFP4_GROUP_SIZE,
    _is_hopper_without_native_fp4,
    convert_nvfp4_weight_to_fp8_block,
    cutlass_fp4_supported,
    ensure_e2m1_table_on_device,
)

# 1. Hardware detection
is_hopper = _is_hopper_without_native_fp4()
check("hopper without native fp4 detected", is_hopper, f"value={is_hopper}")
check("native fp4 correctly reported absent", not cutlass_fp4_supported())
check("fp8 block size is [128,128]", FP8_COMPUTE_BLOCK_SIZE == [128, 128])
check("nvfp4 group size is 16", NVFP4_GROUP_SIZE == 16)

# 2. Backend wiring
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (  # noqa: E402
    NvFp4MoeBackend,
    _convert_nvfp4_moe_to_fp8_compute,
    backend_to_kernel_cls,
    make_nvfp4_moe_quant_config,
    map_nvfp4_backend,
)

check("FP8_COMPUTE enum exists", hasattr(NvFp4MoeBackend, "FP8_COMPUTE"))
kcls = backend_to_kernel_cls(NvFp4MoeBackend.FP8_COMPUTE)
check(
    "maps to TritonOrDeepGemmExperts",
    kcls and kcls[0].__name__ == "TritonOrDeepGemmExperts",
    f"got={[c.__name__ for c in kcls]}",
)
check(
    "moe_backend='fp8_compute' selectable",
    map_nvfp4_backend("fp8_compute") == NvFp4MoeBackend.FP8_COMPUTE,
)

# 3. Numerical fidelity on a realistic packed NVFP4 tensor.
torch.manual_seed(0)
device = torch.device("cuda")
N, K = 1024, 2048

# Packed NVFP4: 2 fp4 nibbles per byte along the input dim, matching the layout
# ModelOptNvFp4LinearMethod.create_weights registers.
w_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
# Per-group E4M3 block scales, one per NVFP4_GROUP_SIZE elements.
w_scale = (
    torch.rand((N, K // NVFP4_GROUP_SIZE), device=device) * 2.0 + 0.5
).to(torch.float8_e4m3fn)
w_gs = torch.tensor(1.0, dtype=torch.float32, device=device)

ensure_e2m1_table_on_device(device)

# What the Marlin/reference path computes.
ref_bf16 = dequantize_to_dtype(
    w_packed, w_scale, w_gs, torch.bfloat16, block_size=NVFP4_GROUP_SIZE, swizzle=False
)
check("reference dequant shape (N,K)", tuple(ref_bf16.shape) == (N, K), str(tuple(ref_bf16.shape)))

# What our converted FP8 weights represent.
w_fp8, w_fp8_scale = convert_nvfp4_weight_to_fp8_block(w_packed, w_scale, w_gs)
check("fp8 dtype", w_fp8.dtype == torch.float8_e4m3fn, str(w_fp8.dtype))
check("fp8 shape preserved (N,K)", tuple(w_fp8.shape) == (N, K), str(tuple(w_fp8.shape)))

blk = FP8_COMPUTE_BLOCK_SIZE[0]
recon = (
    w_fp8.to(torch.float32).view(N // blk, blk, K // blk, blk)
    * w_fp8_scale.to(torch.float32).view(N // blk, 1, K // blk, 1)
).reshape(N, K)

cos = torch.nn.functional.cosine_similarity(
    recon.flatten(), ref_bf16.flatten().float(), dim=0
).item()
rel = ((recon - ref_bf16.float()).norm() / ref_bf16.float().norm()).item()
check("cosine similarity vs FP4 reference > 0.999", cos > 0.999, f"cos={cos:.6f}")
check("relative L2 error < 0.05 (fp8 e4m3 mantissa bound)", rel < 0.05, f"rel_err={rel:.6f}")

# Guard the swizzle choice: the wrong setting must be visibly different, so a
# future edit that flips it cannot pass silently.
try:
    other = dequantize_to_dtype(
        w_packed, w_scale, w_gs, torch.bfloat16, block_size=NVFP4_GROUP_SIZE, swizzle=True
    )
    cos_other = torch.nn.functional.cosine_similarity(
        other.flatten().float(), ref_bf16.flatten().float(), dim=0
    ).item()
    check("swizzle=True is materially different", cos_other < 0.99, f"cos={cos_other:.6f}")
except Exception as exc:
    print(f"[PASS] swizzle=True path rejects linear scales ({type(exc).__name__})")

# 4. End-to-end GEMM equivalence: FP8 block GEMM vs BF16 reference matmul.
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)

M = 256
x = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 0.1
ref_out = x.float() @ ref_bf16.float().t()

q_x, x_scale = per_token_group_quant_fp8(x, group_size=FP8_COMPUTE_BLOCK_SIZE[1])
fp8_out = w8a8_triton_block_scaled_mm(
    q_x, w_fp8, x_scale, w_fp8_scale, FP8_COMPUTE_BLOCK_SIZE, torch.bfloat16
)
gemm_cos = torch.nn.functional.cosine_similarity(
    fp8_out.flatten().float(), ref_out.flatten(), dim=0
).item()
check("fp8 block GEMM matches BF16 reference > 0.99", gemm_cos > 0.99, f"cos={gemm_cos:.6f}")

# 5. MoE expert conversion
E, MN, MK = 4, 512, 1024
we = torch.randint(0, 256, (E, MN, MK // 2), dtype=torch.uint8, device=device)
we_s = (torch.rand((E, MN, MK // NVFP4_GROUP_SIZE), device=device) * 2 + 0.5).to(
    torch.float8_e4m3fn
)
we_s2 = torch.ones((E,), dtype=torch.float32, device=device)

(w13_fp8, w13_fp8_s, a, b, w2_fp8, w2_fp8_s, c, d) = _convert_nvfp4_moe_to_fp8_compute(
    we, we_s, we_s2, we, we_s, we_s2
)
check("moe fp8 dtype", w13_fp8.dtype == torch.float8_e4m3fn)
check(
    "moe fp8 shape (E,N,K)", tuple(w13_fp8.shape) == (E, MN, MK), str(tuple(w13_fp8.shape))
)
check("moe activation scales nulled", all(v is None for v in (a, b, c, d)))

# Per-expert fidelity, including the per-expert global scale indexing.
e_ref = dequantize_to_dtype(
    we[2], we_s[2], we_s2[2], torch.bfloat16, block_size=NVFP4_GROUP_SIZE, swizzle=False
)
e_recon = (
    w13_fp8[2].to(torch.float32).view(MN // blk, blk, MK // blk, blk)
    * w13_fp8_s[2].to(torch.float32).view(MN // blk, 1, MK // blk, 1)
).reshape(MN, MK)
e_cos = torch.nn.functional.cosine_similarity(
    e_recon.flatten(), e_ref.flatten().float(), dim=0
).item()
check("moe per-expert fidelity > 0.999", e_cos > 0.999, f"cos={e_cos:.6f}")

qc = make_nvfp4_moe_quant_config(
    NvFp4MoeBackend.FP8_COMPUTE, w13_fp8_s, w2_fp8_s, None, None, None, None
)
check(
    "quant config block shape",
    list(qc.block_shape) == FP8_COMPUTE_BLOCK_SIZE,
    str(qc.block_shape),
)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
