// aikido-exl3 Hopper MoE PREFILL expert GEMM, WGMMA family (extension module `aikido_exl3_moe_wgmma`, sm_90a only).
//
// K = 3 MCG gate+up of the GLM-5.3 TR3 experts, transposed product so the exact decoded weights are the REGISTER A
// operand and the transformed activations the SHARED-MEMORY B operand of wgmma:
//     Y^T [n, rows] = W_hat^T [n, k] x Xh^T [k, rows]
// One CTA = 2 warpgroups = one 128-channel output-Hadamard block x one tile of up to 128 routed rows of one expert.
// Warpgroup g owns channels 64g .. 64g+63 of the block (64 x N fp32 accumulator, N/2 registers per thread); warp w
// of the CTA owns channels 16w .. 16w+15, i.e. exactly ONE 16x16 EXL3 tile per k16 step: lane l decodes the 8
// weights ExLlamaV3's dq8_regs_3bits gives it (k rows 2(l%4)+{0,1,8,9}, tile columns l/4 and l/4+8), which is
// verbatim the wgmma A fragment of W^T (rows l/4, l/4+8; cols 2(l%4)+{0,1}, +8): a pure register renaming,
// no shuffles. Decoded values are bit-identical to exllamav3_ext.reconstruct (same instruction sequence,
// exl3_decode.cuh); only the fp32 accumulation order differs from the Marlin-template family.
//
// Routing: the SAME (sorted_ids, expert_ids, num_post_padded) as the 64-row Marlin family (moe block 64); a tile is
// two consecutive 64-row blocks of one expert (the expert's blocks 0+1, 2+3, ...); an odd last block runs alone with
// N = 64. So an expert with 262 rows = 5 blocks = tiles 128, 128, 64: 3 weight decodes instead of 5.
// Epilogue = the Marlin family's: fp32 -> fp16 (rn), then Had128 * svh (aikido_had_out_reg, ExLlamaV3's arithmetic)
// on the fp16 row, fp16 store. No split-k, no cross-block reduction: deterministic.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "exl3_decode.cuh"
#include "exl3_had.cuh"

namespace aikido_wgmma {

struct Frag2 {   // what dq8_regs_3bits fills: 2 x half2 (the m16n8k16 B fragment = the wgmma A fragment of W^T)
  half2 v[2];
  __device__ __forceinline__ half2& operator[](int i) { return v[i]; }
};

__device__ __forceinline__ uint32_t h2u(half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int R>
__device__ __forceinline__ void fence_acc(float (&d)[R]) {
#pragma unroll
  for (int i = 0; i < R; i++) asm volatile("" : "+f"(d[i])::"memory");
}
__device__ __forceinline__ void fence_a(uint32_t (&a)[4]) {
#pragma unroll
  for (int i = 0; i < 4; i++) asm volatile("" : "+r"(a[i])::"memory");
}

// K-major, 128-byte swizzle operand descriptor (sm90 GmmaDescriptor: start >> 4 [0,14), LBO >> 4 [16,30) (unused for
// swizzled K-major), SBO >> 4 [32,46) = 1024 B between 8-row groups, layout 1 = SWIZZLE_128B [62,64)).
__device__ __forceinline__ uint64_t desc_sw128(const void* p) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// D[64 x 128] (+)= A[64 x 16] (registers, fp16) * B[16 x 128] (shared memory, K-major, 128B swizzle); fp32 accumulate.
// scale_d = 0: D = A * B (first k step), 1: D += A * B.
__device__ __forceinline__ void wgmma_m64n128k16_rs(float (&d)[64], const uint32_t (&a)[4], uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
      "}, "
      "{%64, %65, %66, %67}, %68, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

// D[64 x 64] (+)= A[64 x 16] (registers, fp16) * B[16 x 64] (shared memory, K-major, 128B swizzle); fp32 accumulate.
// scale_d = 0: D = A * B (first k step), 1: D += A * B.
__device__ __forceinline__ void wgmma_m64n64k16_rs(float (&d)[32], const uint32_t (&a)[4], uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
      "}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

__device__ __forceinline__ void wgmma_m64n256k16_rs(float (&d)[128], const uint32_t (&a)[4], uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %133, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,"
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,"
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"
      "}, "
      "{%128, %129, %130, %131}, %132, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

__device__ __forceinline__ void wgmma_m64n192k16_rs(float (&d)[96], const uint32_t (&a)[4], uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %101, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95"
      "}, "
      "{%96, %97, %98, %99}, %100, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  // "memory": ordered against __syncthreads / cp.async waits like an ordinary shared load (no hoisting across them)
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr) : "memory");
  return v;
}

// A fragment from this lane's two tile words (high word a, low word b).
__device__ __forceinline__ void decode_a_k3w(uint32_t wa, uint32_t wb, int s2, uint32_t (&a)[4]) {
  Frag2 f0, f1;
  aikido_exl3::dq8_regs_3bits<Frag2, 1>(wa, wb, s2, f0, f1);
  a[0] = h2u(f0[0]); a[1] = h2u(f1[0]); a[2] = h2u(f0[1]); a[3] = h2u(f1[1]);
}

// Decode this lane's A fragment (one 16x16 K = 3 tile per warp) from the staged tile words.
__device__ __forceinline__ void decode_a_k3(const uint32_t* __restrict__ tile, int src_a, int src_b, int s2, uint32_t (&a)[4]) {
  Frag2 f0, f1;
  aikido_exl3::dq8_regs_3bits<Frag2, 1>(tile[src_a], tile[src_b], s2, f0, f1);
  a[0] = h2u(f0[0]);   // (row l/4,   k 2(l%4)+{0,1})
  a[1] = h2u(f1[0]);   // (row l/4+8, k 2(l%4)+{0,1})
  a[2] = h2u(f0[1]);   // (row l/4,   k 2(l%4)+{8,9})
  a[3] = h2u(f1[1]);   // (row l/4+8, k 2(l%4)+{8,9})
}

// Word w of K = 3 tile t of a k16 row (t counted from a 4-tile-aligned start). Stack layout as the Marlin-template
// module stores it (stack_repacked): [4 tiles][24 words] per 64-column unit, or with AIKIDO_K3_TILE_INTERLEAVE=1
// (slope #143) [24 words][4 tiles]. Both are pure permutations of the stored words, so the decode is unchanged.
__device__ __forceinline__ int k3_word(int t, int w) {
#if AIKIDO_K3_TILE_INTERLEAVE
  return (t >> 2) * 96 + w * 4 + (t & 3);
#else
  return t * 24 + w;
#endif
}

// CN = output channels per CTA: 128 (2 warpgroups) or 256 (4 warpgroups sharing one activation tile: half the
// activation traffic per decoded weight, which is L2-bound at 345 MHz); threads = 2 * CN.
constexpr int kTK = 64;                       // k per pipeline stage (one 128-byte swizzle row per token row)
constexpr int kStages = 4;                    // prefetch distance kStages - 2 (see main loop)
template <int TMAX> constexpr int x_stage() { return TMAX * kTK * 2; }   // TMAX token rows x 64 k fp16 (16 / 32 KB)
// weights per stage: 4 k16 rows x CN/16 tiles, a row padded to the K = 4 size (K = 3: 96 B / tile, K = 4: 128 B / tile)
template <int CN> constexpr int w_row() { return CN / 16 * 128; }
template <int CN> constexpr int w_stage() { return 4 * w_row<CN>(); }
template <int CN> constexpr int c_stride() { return CN + 8; }   // epilogue fp16 tile [rows][CN + 8]
template <int TMAX, int CN> constexpr int smem_bytes() { return kStages * (x_stage<TMAX>() + w_stage<CN>()) + 1024; }

template <int N>
__device__ __forceinline__ void wgmma_rs(float (&d)[N / 2], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  if constexpr (N == 256) wgmma_m64n256k16_rs(d, a, desc, scale_d);
  else if constexpr (N == 192) wgmma_m64n192k16_rs(d, a, desc, scale_d);
  else if constexpr (N == 128) wgmma_m64n128k16_rs(d, a, desc, scale_d);
  else wgmma_m64n64k16_rs(d, a, desc, scale_d);
}

// Variant bits (knob moe_wgmma_set_variant; 0 = served): 1 = TIMING PROBE no weight decode (raw tile words as A),
// 2 = TIMING PROBE no wgmma (decode only), 4 = 4 A buffers + wgmma.wait_group 2 (two wgmmas in flight).
constexpr int kVarNoDecode = 1, kVarNoMma = 2, kVarDeep = 4;

// One tile: N routed rows (128, or 64 for an expert's odd last block) x 128 channels, full k.
template <int BITS, int TMAX, int CN, int N, int V>
__device__ __forceinline__ void tile_gemm(uint8_t* smem, const int* sh_ids, const half* __restrict__ A,
                                          const uint32_t* __restrict__ B, half* __restrict__ C,
                                          const half* __restrict__ svh, int e, int nb, int n, int k) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint8_t* sx = smem;
  constexpr int kXStage = x_stage<TMAX>();
  constexpr int kThreads = 2 * CN, kWStage = w_stage<CN>(), kWRow = w_row<CN>(), kCStride = c_stride<CN>();
  constexpr int kWRowWords = kWRow / 4;
  uint8_t* sw = smem + kStages * kXStage;
  const int KT = k / kTK;
  const size_t tiles_n = (size_t)(n / 16);
  // K = 3: [E, k/16, n/16, 24] tiles as stored; K = 4: [E, k/16, n/64, 32 lanes, 4] lane-major repack (word l of tile
  // (i, 4g + j) at [i, g, l, j]). Either way the CTA's 8 tiles of one k16 row are contiguous.
  constexpr int row_words = (CN / 16) * (BITS == 3 ? 24 : 32);   // this CTA's tiles of one k16 row
  const size_t krow_words = BITS == 3 ? tiles_n * 24 : tiles_n * 32;
  const uint32_t* bexp = B + (size_t)e * (k / 16) * krow_words + (size_t)nb * row_words;

  // W copy: 4 k16 rows x row_words * 4 contiguous bytes (8 tiles), one int4 per thread < row_words
  constexpr int q_per_row = row_words / 4;
  const int ws_ = tid / q_per_row, wq_ = tid - ws_ * q_per_row;
  const uint32_t* wsrc = bexp + (size_t)ws_ * krow_words + wq_ * 4;
  const uint32_t wdst = ws_ * kWRow + wq_ * 16;
  auto load_stage = [&](int kt, int slot) {
    // Gather of N routed rows x 8 chunks of 16 B, addresses computed per k-tile. (A version with the row offsets
    // hoisted out of the k loop corrupted whole 128 x 128 output tiles nondeterministically although its SASS
    // address arithmetic reads correct; not understood, not shipped: wf/wgmma LOG r9-r10.)
#pragma unroll
    for (int i = 0; i < N * 8 / kThreads; i++) {
      const int c = tid + kThreads * i;
      const int row = c >> 3, ch = c & 7;
      const int sid = sh_ids[row];
      const half* src = A + (size_t)(sid < 0 ? 0 : sid) * k + kt * kTK + ch * 8;
      cp_async16(sx + slot * kXStage + row * 128 + ((ch ^ (row & 7)) << 4), src, sid < 0 ? 0 : 16);
    }
    if (tid < row_words)                      // 4 k16 rows x row_words * 4 contiguous bytes (8 tiles)
      cp_async16(sw + slot * kWStage + wdst, wsrc + (size_t)kt * 4 * krow_words, 16);
  };

  int src_a, src_b, s2;
  aikido_exl3::k3_lane_words(lane, src_a, src_b, s2);
  // this warp's 16x16 tile inside a k16 row of 8 tiles (K = 4: own word b and the previous lane's word a)
  const int tile_off = BITS == 3 ? warp * 24 : (warp >> 2) * 128 + lane * 4 + (warp & 3);
  const int tile_off_a = BITS == 3 ? 0 : (warp >> 2) * 128 + ((lane + 31) & 31) * 4 + (warp & 3);
  // word offsets (in a k16 row) of this lane's high / low source words
  const int wofs_a = BITS == 3 ? k3_word(warp, src_a) : tile_off_a;
  const int wofs_b = BITS == 3 ? k3_word(warp, src_b) : tile_off;

  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; i++) acc[i] = 0.0f;
  constexpr int NB = (V & kVarDeep) ? 4 : 2;
  uint32_t afr[NB][4];

#pragma unroll
  for (int s = 0; s < kStages - 2; s++) {
    if (s < KT) load_stage(s, s);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; kt++) {
    // stage kt landed (this thread's copies), and after the barrier everybody's; every thread is past its
    // wgmma.wait of tile kt-1 step 0, so all wgmmas of tile kt-2 are complete and its stage may be refilled.
    cp_async_wait<kStages - 3>();
    fence_async_smem();                       // cp.async (generic proxy) writes -> wgmma (async proxy) reads
    __syncthreads();
    if (kt + kStages - 2 < KT) load_stage(kt + kStages - 2, (kt + kStages - 2) % kStages);
    cp_async_commit();
    const int slot = kt % kStages;
    const uint32_t* wbase = reinterpret_cast<const uint32_t*>(sw + slot * kWStage);
    const uint32_t* wt = wbase + tile_off;
    const uint8_t* xs = sx + slot * kXStage;
    const uint64_t dx = desc_sw128(xs);       // + 2 per k16 step (32 bytes >> 4): the stage is 1024-aligned
    const uint32_t wsh = static_cast<uint32_t>(__cvta_generic_to_shared(wbase));
    uint32_t wa = 0, wb = 0;
    if constexpr (!(V & kVarNoDecode)) {
      wa = lds32(wsh + 4 * wofs_a);
      wb = lds32(wsh + 4 * wofs_b);
    }
#pragma unroll
    for (int ss = 0; ss < 4; ss++) {
      uint32_t (&af)[4] = afr[ss % NB];
      if constexpr (V & kVarNoDecode) {
        const uint32_t* tw = wt + ss * kWRowWords;
        af[0] = tw[src_a]; af[1] = tw[src_b]; af[2] = tw[src_a] ^ tw[src_b]; af[3] = tw[src_b] + s2;
      } else {
        if constexpr (BITS == 3) {
          decode_a_k3w(wa, wb, s2, af);
        } else {
          Frag2 f0, f1;
          aikido_exl3::dq8_regs_4bits<Frag2, 1>(wa, wb, f0, f1);
          af[0] = h2u(f0[0]); af[1] = h2u(f1[0]); af[2] = h2u(f0[1]); af[3] = h2u(f1[1]);
        }
        if (ss < 3) {            // next step's words now: their latency hides behind this wgmma
          wa = lds32(wsh + 4 * ((ss + 1) * kWRowWords + wofs_a));
          wb = lds32(wsh + 4 * ((ss + 1) * kWRowWords + wofs_b));
        }
      }
      if constexpr (V & kVarNoMma) {
        fence_a(af);
        acc[ss] += __uint_as_float(af[0] ^ af[1] ^ af[2] ^ af[3]) * 0.0f;
      } else {
        wg_fence();
        fence_acc<N / 2>(acc);
        wgmma_rs<N>(acc, af, dx + 2 * ss, 1);
        wg_commit();
        fence_acc<N / 2>(acc);
        if constexpr (V & kVarDeep) wg_wait<2>(); else wg_wait<1>();   // oldest in-flight A buffer is free again
        fence_a(afr[(ss + 1) % NB]);
      }
    }
  }
  wg_wait<0>();
  fence_acc<N / 2>(acc);
  cp_async_wait<0>();
  __syncthreads();

  // Epilogue: fp32 -> fp16 (round to nearest) into a [row][channel] tile, then per row Had128 * svh, fp16 store.
  half* sc = reinterpret_cast<half*>(smem);
  const int ch0 = warp * 16 + (lane >> 2);
#pragma unroll
  for (int i = 0; i < N / 2; i++) {
    const int ch = ch0 + ((i & 2) ? 8 : 0);
    const int row = 8 * (i >> 2) + 2 * (lane & 3) + (i & 1);
    sc[row * kCStride + ch] = __float2half_rn(acc[i]);
  }
  __syncthreads();
  constexpr int nblk = CN / 128, nwarps = kThreads / 32;
  half4 sv[nblk];
#pragma unroll
  for (int b = 0; b < nblk; b++)
    if (svh) sv[b] = reinterpret_cast<const half4*>(svh + (size_t)e * n + nb * CN + b * 128)[lane];
  for (int item = warp; item < N * nblk; item += nwarps) {   // one (row, 128-block) per warp and pass
    const int row = item / nblk, b = item % nblk;
    const int sid = sh_ids[row];
    if (sid < 0) continue;
    half4 v = *reinterpret_cast<const half4*>(sc + row * kCStride + b * 128 + 4 * lane);
    if (svh) v = aikido_had_out_reg<false>(v, sv[b], 0.088388347648f, lane);
    *reinterpret_cast<half4*>(C + (size_t)sid * n + nb * CN + b * 128 + 4 * lane) = v;
  }
}

// grid (n / 128, moe blocks of 64 rows); a: fp16 [shards * rows, k] (columns >= shard_end read slab 1).
template <int BITS, int V, int TMAX, int CN>
__global__ void __launch_bounds__(2 * CN, 1)
moe_wgmma_k3_kernel(const half* __restrict__ A, const uint32_t* __restrict__ B, half* __restrict__ C,
                    const half* __restrict__ svh, const int* __restrict__ sorted_ids,
                    const int* __restrict__ expert_ids, const int* __restrict__ num_post_padded, int rows, int n,
                    int k, int shard_end) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~(uintptr_t)1023);
  __shared__ int sh_ids[TMAX];
  constexpr int tb = TMAX / 64;               // 64-row blocks per tile
  const int nb = blockIdx.x, mb = blockIdx.y;
  const int npp = num_post_padded[0];
  if (mb * 64 >= npp) return;
  const int e = expert_ids[mb];
  if (e < 0) return;
  int j = 0;                                  // index of this 64-row block inside its expert's run
  while (mb - j - 1 >= 0 && expert_ids[mb - j - 1] == e) j++;
  if (j % tb) return;                         // not the first block of a tile
  int cnt = 1;                                // blocks in this tile: up to tb consecutive blocks of the expert
  while (cnt < tb && (mb + cnt) * 64 < npp && expert_ids[mb + cnt] == e) cnt++;
  for (int r = threadIdx.x; r < TMAX; r += 2 * CN) {
    const int sid = r < cnt * 64 ? sorted_ids[mb * 64 + r] : rows;
    sh_ids[r] = (sid >= 0 && sid < rows) ? sid : -1;
  }
  __syncthreads();
  const half* a = A + ((shard_end > 0 && nb * CN >= shard_end) ? (size_t)rows * k : 0);
  if constexpr (TMAX == 256) {
    if (cnt == 4) tile_gemm<BITS, TMAX, CN, 256, V>(smem, sh_ids, a, B, C, svh, e, nb, n, k);
    else if (cnt == 3) tile_gemm<BITS, TMAX, CN, 192, V>(smem, sh_ids, a, B, C, svh, e, nb, n, k);
    else if (cnt == 2) tile_gemm<BITS, TMAX, CN, 128, V>(smem, sh_ids, a, B, C, svh, e, nb, n, k);
    else tile_gemm<BITS, TMAX, CN, 64, V>(smem, sh_ids, a, B, C, svh, e, nb, n, k);
  } else {
    if (cnt == 2) tile_gemm<BITS, TMAX, CN, 128, V>(smem, sh_ids, a, B, C, svh, e, nb, n, k);
    else tile_gemm<BITS, TMAX, CN, 64, V>(smem, sh_ids, a, B, C, svh, e, nb, n, k);
  }
}

__device__ __forceinline__ void wgmma_m64n256k16_ss(float (&d)[128], uint64_t desc_a, uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95,"
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111,"
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127"
      "}, "
      "%128, %129, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

__device__ __forceinline__ void wgmma_m64n192k16_ss(float (&d)[96], uint64_t desc_a, uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %98, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63,"
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79,"
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95"
      "}, "
      "%96, %97, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

__device__ __forceinline__ void wgmma_m64n128k16_ss(float (&d)[64], uint64_t desc_a, uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31,"
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47,"
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63"
      "}, "
      "%64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

__device__ __forceinline__ void wgmma_m64n64k16_ss(float (&d)[32], uint64_t desc_a, uint64_t desc_b, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15,"
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31"
      "}, "
      "%32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d));
#else
  __trap();
#endif
}

// ---------------------------------------------------------------------------------------------------------------
// Warp-specialized variant (geometry 3): CTA = 3 warpgroups. WG0 = producer: cp.async of the activation tile (up to
// 256 routed rows) and of the raw trellis words, then the SAME exact decode (decode_a_k3 / dq8_regs_4bits) into a
// fp16 A tile in shared memory (K-major, 128-byte swizzle). WG1 / WG2 = consumers: wgmma SS (A = decoded weights of
// channels 64(g-1) .. +63, B = activations), N = 64 x (64-row blocks in the tile) <= 256, fp32 accumulators. Full /
// empty mbarriers per stage. Decoded values are stored and read unchanged (fp16 -> fp16), so the weights the tensor
// core multiplies with are the same bits as in the register-A kernel.
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@!P1 bra WAIT_%=;\n"
      "}\n" ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void bar_named(int id, int count) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory"); }

constexpr int kWsStages = 4;
constexpr int kWsX = 256 * 128;              // 32 KB: 256 rows x 64 k fp16
constexpr int kWsA = 128 * 128;              // 16 KB: 128 channels x 64 k fp16 (decoded)
constexpr int kWsW = 4 * 1024;               // 4 KB: raw trellis words, 4 k16 rows x 8 tiles
constexpr int kWsStage = kWsX + kWsA + kWsW;
constexpr int kWsSmem = kWsStages * kWsStage + 1024;
template <int PW> constexpr int ws_threads() { return 128 * (PW + 2); }   // PW producer warpgroups + 2 consumers

template <int N>
__device__ __forceinline__ void wgmma_ss(float (&d)[N / 2], uint64_t da, uint64_t db, int scale_d) {
  if constexpr (N == 256) wgmma_m64n256k16_ss(d, da, db, scale_d);
  else if constexpr (N == 192) wgmma_m64n192k16_ss(d, da, db, scale_d);
  else if constexpr (N == 128) wgmma_m64n128k16_ss(d, da, db, scale_d);
  else wgmma_m64n64k16_ss(d, da, db, scale_d);
}

template <int BITS, int N, int PW>
__device__ __forceinline__ void tile_ws(uint8_t* smem, uint64_t* full, uint64_t* empty, const int* sh_ids,
                                        const half* __restrict__ A, const uint32_t* __restrict__ B, half* __restrict__ C,
                                        const half* __restrict__ svh, int e, int nb, int n, int k) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wg = warp >> 2;
  const int KT = k / kTK;
  const size_t tiles_n = (size_t)(n / 16);
  constexpr int row_words = BITS == 3 ? 192 : 256;
  const size_t krow_words = BITS == 3 ? tiles_n * 24 : tiles_n * 32;
  const uint32_t* bexp = B + (size_t)e * (k / 16) * krow_words + (size_t)nb * row_words;
  auto sx = [&](int s) { return smem + s * kWsStage; };
  auto sa = [&](int s) { return smem + s * kWsStage + kWsX; };
  auto sw = [&](int s) { return smem + s * kWsStage + kWsX + kWsA; };

  constexpr int PT = 128 * PW;                          // producer threads
  float acc[N / 2];
  if (wg < PW) {
    // ---------------- producer ----------------
    if constexpr (PW == 2) asm volatile("setmaxnreg.dec.sync.aligned.u32 56;\n" ::: "memory");
    const int pt = tid;                                   // 0..PT-1
    auto load = [&](int kt) {
      const int s = kt % kWsStages;
#pragma unroll 4
      for (int i = 0; i < N * 8 / PT; i++) {
        const int c = pt + PT * i;
        const int row = c >> 3, ch = c & 7;
        const int sid = sh_ids[row];
        const half* src = A + (size_t)(sid < 0 ? 0 : sid) * k + kt * kTK + ch * 8;
        cp_async16(sx(s) + row * 128 + ((ch ^ (row & 7)) << 4), src, sid < 0 ? 0 : 16);
      }
      for (int q = pt; q < row_words; q += PT) {          // 4 k16 rows x row_words / 4 int4
        const int r = q / (row_words / 4), qq = q - r * (row_words / 4);
        cp_async16(sw(s) + r * 1024 + qq * 16, bexp + (size_t)(kt * 4 + r) * krow_words + qq * 4, 16);
      }
    };
    int src_a = 0, src_b = 0, s2 = 0;
    if constexpr (BITS == 3) aikido_exl3::k3_lane_words(lane, src_a, src_b, s2);
    load(0);
    cp_async_commit();
    for (int kt = 0; kt < KT; kt++) {
      const int s = kt % kWsStages;
      if (kt + 1 < KT) {
        const int s1 = (kt + 1) % kWsStages;
        if (kt + 1 >= kWsStages) mbar_wait(&empty[s1], ((kt + 1) / kWsStages - 1) & 1);
        load(kt + 1);
      }
      cp_async_commit();
      cp_async_wait<1>();                                 // this thread's copies of k-tile kt landed
      bar_named(1, PT);                                   // ... and every producer thread's (raw words are shared)
      // decode: producer warp w = n-tiles w * 8 / (4 PW) .. of the CTA, 4 k16 rows each
      const uint32_t* wrow = reinterpret_cast<const uint32_t*>(sw(s));
      uint8_t* at = sa(s);
#pragma unroll
      for (int ss = 0; ss < 4; ss++) {
#pragma unroll
        for (int h = 0; h < 2 / PW; h++) {
          const int t = (2 / PW) * warp + h;              // n-tile 0..7 (channels 16t .. 16t+15)
          uint32_t a4[4];
          if constexpr (BITS == 3) {
            const uint32_t* wr = wrow + ss * 256;
            decode_a_k3w(wr[k3_word(t, src_a)], wr[k3_word(t, src_b)], s2, a4);
          } else {
            const uint32_t* w0 = wrow + ss * 256 + (t >> 2) * 128 + (t & 3);
            Frag2 f0, f1;
            aikido_exl3::dq8_regs_4bits<Frag2, 1>(w0[((lane + 31) & 31) * 4], w0[lane * 4], f0, f1);
            a4[0] = h2u(f0[0]); a4[1] = h2u(f1[0]); a4[2] = h2u(f0[1]); a4[3] = h2u(f1[1]);
          }
          // a4: (ch r, k 2q..), (ch r+8, k 2q..), (ch r, k 2q+8..), (ch r+8, k 2q+8..) of the 16x16 tile
          const int r = lane >> 2, q = lane & 3;
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const int ch = 16 * t + r + ((v & 1) ? 8 : 0);
            const int chunk = 2 * ss + ((v & 2) ? 1 : 0);
            *reinterpret_cast<uint32_t*>(at + ch * 128 + ((chunk ^ (ch & 7)) << 4) + 4 * q) = a4[v];
          }
        }
      }
      fence_async_smem();                                 // generic-proxy writes (cp.async, st.shared) -> wgmma
      mbar_arrive(&full[s]);
    }
    cp_async_wait<0>();
  } else {
    // ---------------- consumers ----------------
    if constexpr (PW == 2) asm volatile("setmaxnreg.inc.sync.aligned.u32 200;\n" ::: "memory");
    const int g = wg - PW;                                // channels 64g .. 64g+63
#pragma unroll
    for (int i = 0; i < N / 2; i++) acc[i] = 0.0f;
    for (int kt = 0; kt < KT; kt++) {
      const int s = kt % kWsStages;
      mbar_wait(&full[s], (kt / kWsStages) & 1);
      const uint8_t* a_base = sa(s) + g * 64 * 128;
      const uint8_t* x_base = sx(s);
      wg_fence();
      fence_acc<N / 2>(acc);
#pragma unroll
      for (int ss = 0; ss < 4; ss++) wgmma_ss<N>(acc, desc_sw128(a_base + ss * 32), desc_sw128(x_base + ss * 32), 1);
      wg_commit();
      fence_acc<N / 2>(acc);
      wg_wait<1>();                                       // k-tile kt-1's wgmmas are complete: release its stage
      if (kt > 0) mbar_arrive(&empty[(kt - 1) % kWsStages]);
    }
    wg_wait<0>();
    fence_acc<N / 2>(acc);
  }
  __syncthreads();
  // Epilogue (as the register-A kernel): fp32 -> fp16 rn into [row][channel], then Had128 * svh per row, fp16 store.
  constexpr int cs = 136;
  half* sc = reinterpret_cast<half*>(smem);
  if (wg >= PW) {
    const int ch0 = (warp - 4 * PW) * 16 + (lane >> 2);
#pragma unroll
    for (int i = 0; i < N / 2; i++) {
      const int ch = ch0 + ((i & 2) ? 8 : 0);
      const int row = 8 * (i >> 2) + 2 * (lane & 3) + (i & 1);
      sc[row * cs + ch] = __float2half_rn(acc[i]);
    }
  }
  __syncthreads();
  half4 sv;
  if (svh) sv = reinterpret_cast<const half4*>(svh + (size_t)e * n + nb * 128)[lane];
  for (int row = warp; row < N; row += ws_threads<PW>() / 32) {
    const int sid = sh_ids[row];
    if (sid < 0) continue;
    half4 v = *reinterpret_cast<const half4*>(sc + row * cs + 4 * lane);
    if (svh) v = aikido_had_out_reg<false>(v, sv, 0.088388347648f, lane);
    *reinterpret_cast<half4*>(C + (size_t)sid * n + nb * 128 + 4 * lane) = v;
  }
}

template <int BITS, int PW>
__global__ void __launch_bounds__(ws_threads<PW>(), 1)
moe_wgmma_ws_kernel(const half* __restrict__ A, const uint32_t* __restrict__ B, half* __restrict__ C,
                    const half* __restrict__ svh, const int* __restrict__ sorted_ids,
                    const int* __restrict__ expert_ids, const int* __restrict__ num_post_padded, int rows, int n,
                    int k, int shard_end) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~(uintptr_t)1023);
  __shared__ int sh_ids[256];
  __shared__ __align__(8) uint64_t full[kWsStages], empty[kWsStages];
  const int nb = blockIdx.x, mb = blockIdx.y;
  const int npp = num_post_padded[0];
  if (mb * 64 >= npp) return;
  const int e = expert_ids[mb];
  if (e < 0) return;
  int j = 0;
  while (mb - j - 1 >= 0 && expert_ids[mb - j - 1] == e) j++;
  if (j % 4) return;
  int cnt = 1;
  while (cnt < 4 && (mb + cnt) * 64 < npp && expert_ids[mb + cnt] == e) cnt++;
  for (int r = threadIdx.x; r < 256; r += ws_threads<PW>()) {
    const int sid = r < cnt * 64 ? sorted_ids[mb * 64 + r] : rows;
    sh_ids[r] = (sid >= 0 && sid < rows) ? sid : -1;
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < kWsStages; s++) { mbar_init(&full[s], 128 * PW); mbar_init(&empty[s], 256); }
  }
  __syncthreads();
  const half* a = A + ((shard_end > 0 && nb * 128 >= shard_end) ? (size_t)rows * k : 0);
  if (cnt == 4) tile_ws<BITS, 256, PW>(smem, full, empty, sh_ids, a, B, C, svh, e, nb, n, k);
  else if (cnt == 3) tile_ws<BITS, 192, PW>(smem, full, empty, sh_ids, a, B, C, svh, e, nb, n, k);
  else if (cnt == 2) tile_ws<BITS, 128, PW>(smem, full, empty, sh_ids, a, B, C, svh, e, nb, n, k);
  else tile_ws<BITS, 64, PW>(smem, full, empty, sh_ids, a, B, C, svh, e, nb, n, k);
}

// Fragment dump (unit check): out fp16 [k, n] of expert e = the A fragments this kernel's decode produces, written
// back through the wgmma A-fragment layout (row = channel l/4 (+8), col = k 2(l%4) + {0,1} (+8)). grid (n/16, k/16), 32.
__global__ void moe_wgmma_k3_dump_kernel(const uint32_t* __restrict__ B, half* __restrict__ out, int e, int n, int k) {
  const int nt = blockIdx.x, kt = blockIdx.y, lane = threadIdx.x;
  int src_a, src_b, s2;
  aikido_exl3::k3_lane_words(lane, src_a, src_b, s2);
  const uint32_t* krow = B + ((size_t)e * (k / 16) + kt) * (n / 16) * 24;   // this k16 row (n/16 tiles, 4-aligned)
  uint32_t a[4];
  decode_a_k3w(krow[k3_word(nt, src_a)], krow[k3_word(nt, src_b)], s2, a);
  const int r = lane >> 2, c = 2 * (lane & 3);
  auto put = [&](int row, int col, uint32_t v) {
    const half2 h = *reinterpret_cast<half2*>(&v);
    out[(size_t)(kt * 16 + col) * n + nt * 16 + row] = __low2half(h);
    out[(size_t)(kt * 16 + col + 1) * n + nt * 16 + row] = __high2half(h);
  };
  put(r, c, a[0]);
  put(r + 8, c, a[1]);
  put(r, c + 8, a[2]);
  put(r + 8, c + 8, a[3]);
}

}  // namespace aikido_wgmma

// ---------------------------------------------------------------------------------------------------------------
// Torch entry points

static int g_variant = 0;
static int g_geom = 0;   // knob moe_wgmma_set_geometry: 0 = 128 rows x 128 ch, 1 = 256 rows x 128 ch, 2 = 128 rows x 256 ch
void moe_wgmma_set_geometry(int64_t g) {
  // 2 (128 rows x 256 channels, 4 warpgroups) is a measured dead end with an open race (wf/wgmma LOG r5): refused.
  TORCH_CHECK(g == 0 || g == 1 || g == 3, "geometry must be 0, 1 or 3");
  g_geom = (int)g;
}
void moe_wgmma_set_variant(int64_t v) {
  TORCH_CHECK(v == 0 || v == 1 || v == 2 || v == 4, "variant must be 0, 1, 2 or 4");
  g_variant = (int)v;
}

// Same arguments as aikido_exl3_moe_kernels.moe_gemm (moe_block_size must be 64, K = 3 stack, cb = 1 (MCG)).
void moe_gemm_wgmma(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, const c10::optional<at::Tensor>& svh,
                    const at::Tensor& sorted_ids, const at::Tensor& expert_ids, const at::Tensor& num_post_padded,
                    int64_t moe_block_size, int64_t shard_end, int64_t cb) {
  const at::cuda::OptionalCUDAGuard device_guard(a.device());
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2 && a.dtype() == at::kHalf && c.dtype() == at::kHalf);
  TORCH_CHECK(a.is_contiguous() && c.is_contiguous());
  const int64_t rows = c.size(0), k = a.size(1), n = c.size(1), shards = shard_end > 0 ? 2 : 1;
  TORCH_CHECK(a.size(0) == shards * rows, "a must be [shards * rows, k]");
  TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.dtype() == at::kInt && b.dim() == 5 &&
              ((!AIKIDO_K3_TILE_INTERLEAVE && b.size(3) == 4 && b.size(4) == 24) ||
               (AIKIDO_K3_TILE_INTERLEAVE && b.size(3) == 24 && b.size(4) == 4) || (b.size(3) == 32 && b.size(4) == 4)) &&
              b.size(1) * 16 == k && b.size(2) * 64 == n,
              "stack [E, k/16, n/64, 4, 24] (K = 3; [.., 24, 4] when built with AIKIDO_K3_TILE_INTERLEAVE) or "
              "[E, k/16, n/64, 32, 4] (K = 4) expected");
  TORCH_CHECK(moe_block_size == 64, "wgmma family: moe_block_size must be 64");
  TORCH_CHECK(cb == 1, "wgmma family: MCG codebook only");
  TORCH_CHECK(n % 128 == 0 && k % 64 == 0 && shard_end % 128 == 0 && shard_end >= 0 && shard_end < n);
  TORCH_CHECK(sorted_ids.dtype() == at::kInt && expert_ids.dtype() == at::kInt && num_post_padded.dtype() == at::kInt);
  TORCH_CHECK(sorted_ids.is_contiguous() && expert_ids.is_contiguous() && num_post_padded.numel() == 1);
  const half* svh_ptr = nullptr;
  if (svh.has_value()) {
    TORCH_CHECK(svh->dtype() == at::kHalf && svh->is_contiguous() && svh->dim() == 2 && svh->size(0) == b.size(0) &&
                svh->size(1) == n, "svh must be fp16 [E, n]");
    svh_ptr = (const half*)svh->data_ptr();
  }
  if (rows == 0) return;
  const int64_t mblocks = std::min<int64_t>(expert_ids.numel(), sorted_ids.numel() / 64);
  if (mblocks == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const bool k3 = b.size(4) == 24 || b.size(3) == 24;
  using namespace aikido_wgmma;
  // geometry: 0 = 128-row tiles x 128 channels (served), 1 = 256-row tiles x 128 channels, 2 = 128-row tiles x 256 channels
  const int geom = g_geom;
  TORCH_CHECK(geom != 2 || (n % 256 == 0 && shard_end % 256 == 0), "geometry 2 needs n, shard_end multiples of 256");
  const int cn = geom == 2 ? 256 : 128;
  const int smem = geom == 1 ? smem_bytes<256, 128>() : geom == 2 ? smem_bytes<128, 256>() : smem_bytes<128, 128>();
#define AIKIDO_WG_K(bits, v) (geom == 1 ? moe_wgmma_k3_kernel<bits, v, 256, 128> : geom == 2 ? moe_wgmma_k3_kernel<bits, v, 128, 256> : moe_wgmma_k3_kernel<bits, v, 128, 128>)
  auto kern = k3 ? AIKIDO_WG_K(3, 0) : AIKIDO_WG_K(4, 0);
  switch (g_variant) {
    case 1: kern = k3 ? AIKIDO_WG_K(3, 1) : AIKIDO_WG_K(4, 1); break;
    case 2: kern = k3 ? AIKIDO_WG_K(3, 2) : AIKIDO_WG_K(4, 2); break;
    case 4: kern = k3 ? AIKIDO_WG_K(3, 4) : AIKIDO_WG_K(4, 4); break;
    default: break;
  }
#undef AIKIDO_WG_K
  if (geom >= 3) {   // warp-specialized: producer(s) decode into shared memory, 2 consumer warpgroups, up to 256 rows
    auto kws = k3 ? moe_wgmma_ws_kernel<3, 1> : moe_wgmma_ws_kernel<4, 1>;   // PW = 2 does not compile (see LOG r8)
    const int wst = ws_threads<1>();
    cudaFuncSetAttribute(kws, cudaFuncAttributeMaxDynamicSharedMemorySize, kWsSmem);
    kws<<<dim3((unsigned)(n / 128), (unsigned)mblocks), wst, kWsSmem, stream>>>(
        (const half*)a.data_ptr(), (const uint32_t*)b.data_ptr(), (half*)c.data_ptr(), svh_ptr, sorted_ids.data_ptr<int>(),
        expert_ids.data_ptr<int>(), num_post_padded.data_ptr<int>(), (int)rows, (int)n, (int)k, (int)shard_end);
    return;
  }
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid((unsigned)(n / cn), (unsigned)mblocks);
  kern<<<grid, 2 * cn, smem, stream>>>(
      (const half*)a.data_ptr(), (const uint32_t*)b.data_ptr(), (half*)c.data_ptr(), svh_ptr, sorted_ids.data_ptr<int>(),
      expert_ids.data_ptr<int>(), num_post_padded.data_ptr<int>(), (int)rows, (int)n, (int)k, (int)shard_end);
}

// out fp16 [k, n] <- decoded A fragments of expert e (unit check of the fragment mapping)
void moe_wgmma_dump_a(const at::Tensor& b, int64_t e, at::Tensor& out) {
  const at::cuda::OptionalCUDAGuard device_guard(b.device());
  TORCH_CHECK(b.dim() == 5 && (AIKIDO_K3_TILE_INTERLEAVE ? (b.size(3) == 24 && b.size(4) == 4) : (b.size(3) == 4 && b.size(4) == 24)) &&
              b.dtype() == at::kInt && b.is_contiguous(), "K = 3 stack expected (layout per AIKIDO_K3_TILE_INTERLEAVE)");
  const int64_t k = b.size(1) * 16, n = b.size(2) * 64;
  TORCH_CHECK(out.dtype() == at::kHalf && out.is_contiguous() && out.size(0) == k && out.size(1) == n);
  dim3 grid((unsigned)(n / 16), (unsigned)(k / 16));
  aikido_wgmma::moe_wgmma_k3_dump_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      (const uint32_t*)b.data_ptr(), (half*)out.data_ptr(), (int)e, (int)n, (int)k);
}

int64_t moe_wgmma_k3_tile_interleave() { return AIKIDO_K3_TILE_INTERLEAVE; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_wgmma_k3_tile_interleave", &moe_wgmma_k3_tile_interleave, "1: reads K = 3 stacks as [.., 24 words, 4 tiles] (must match aikido_exl3_moe_kernels)");
  m.def("moe_gemm_wgmma", &moe_gemm_wgmma, "K=3 MCG grouped GEMM, WGMMA prefill family (128-row tiles over moe blocks of 64)",
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("svh"), py::arg("sorted_ids"), py::arg("expert_ids"),
        py::arg("num_post_padded"), py::arg("moe_block_size"), py::arg("shard_end"), py::arg("cb"));
  m.def("moe_wgmma_set_variant", &moe_wgmma_set_variant, "knob: 0 served, 1 probe no decode, 2 probe no wgmma, 4 two wgmmas in flight");
  m.def("moe_wgmma_set_geometry", &moe_wgmma_set_geometry, "knob: 0 = 128 rows x 128 ch (default), 1 = 256 rows x 128 ch, 3 = warp-specialized smem-A, up to 256 rows");
  m.def("moe_wgmma_dump_a", &moe_wgmma_dump_a, "decoded wgmma A fragments of one expert as fp16 [k, n] (unit check)");
}
