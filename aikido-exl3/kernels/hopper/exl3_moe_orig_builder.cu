// aikido-exl3: original-basis expert weights written STRAIGHT into vLLM's fused-MoE layout.
//
//   W = diag(suh) . H128 . W_hat . H128 . diag(svh)      (ExLlamaV3's own >= 1024-row arithmetic)
//
// `aikido_orig_batch_kernel` below is ExLlamaV3's `reconstruct_had_tile` + `reconstruct_had_batch_kernel` (exllamav3_ext/quant/reconstruct.cu, MIT, Copyright (c)
// 2025 Turboderp), K = 4, statement for statement - the same shared-memory tile, the same fp16 butterfly with the
// pre-applied 1/sqrt(128), the same fp32 first stage of the second transform, the same three fp16 multiplies (r_scale, suh,
// svh) in the same order - with these changes, all marked AIKIDO, none of which touches a computed value:
//   1. the K = 4 window decode is our `aikido_exl3::dq8_regs_4bits(prev_word, own_word)` (exl3_decode.cuh), the same
//      instructions as ExLlamaV3's `dq8_aligned_4bits(ptr, lane * 8)`: a = ptr[(lane + 31) & 31], b = ptr[lane];
//      the K = 3 decode (GLM-5.3 TR3: 192 of 256 experts per layer) is our `dq8_regs_3bits(a, b, s2)` with the lane's
//      two source words from `k3_lane_words` = ExLlamaV3's `dq8<3, cb, 4>(ptr, lane * 8)` (exl3_dq.cuh): a = ptr[i0 % 24],
//      b = ptr[i2 % 24], the same funnel shifts, the same decode; a K = 3 tile is 24 words staged byte for byte;
//   2. the final STORE: ExLlamaV3 writes lane l's four values to row (kb*128 + R), columns nb*128 + 4l .. 4l+3 of a
//      [k, n] matrix. vLLM's fused MoE wants [E, out, in] = the transpose, with gate and up stacked on the out dimension,
//      so the same four values go to rows row_offset + nb*128 + 4l + i, column kb*128 + R of a [rows_total, k] matrix.
//   3. (STACKED variant) the trellis is read from the grouped kernel's resident pack [E, k/16, n/64, 32 lanes, 4 tiles]
//      (hopper_moe.stack_repacked: a pure permutation of the stream words, word l of tile (i, 4g + j) at [i, g, l, j];
//      K = 3: [E, k/16, n/64, 4 tiles, 24 words], the tiles exactly as stored) instead of per-expert int16 tensors behind
//      pointer tables, and suh / svh from the stacked [E, ...] tensors. Only the shared-memory load and the two word reads
//      of the decode change their INDICES. So the transient tier needs no second copy of the experts: its only extra
//      memory is the arena. An optional `out_ids` table (int32, local expert -> output expert) lets one K class of a
//      mixed-K layer (GLM-5.3 TR3) write its experts into their GLOBAL rows of a full arena, so vLLM's fused MoE runs
//      once over all experts with the unmapped router ids.
//   4. (AIKIDO V, wf/builder) DATA-MOVEMENT variants of the two 128-point Hadamard passes and of the decode-phase tile
//      stores, selected by the variant bitmask V (compile-time; runtime choice through AIKIDO_ORIG_VARIANT / the
//      moe_orig_set_variant binding). Every butterfly stage is still the same fp16 HADD2 / fp32 FADD on the same pair of
//      values in the same stage order (row bits 0,1 -> r_scale -> bits 2..6; column bits 0,1 in fp32 -> r_scale -> bits
//      2..6); what changes is WHICH lane holds which value, so that stages on the low bits of a lane's own rows /
//      columns are plain register butterflies instead of SHFL rounds:
//        V & 1  pass 1 (rows): a lane holds 16 rows x 4 columns (stages on row bits 0..3 in registers, 3 SHFL rounds for
//               bits 4..6 instead of 5);
//        V & 2  pass 2 (columns): 8 lanes per row, a lane holds 4 column quads (stages on column bits 2..4 by SHFL xor
//               1, 2, 4 = 3 rounds, bits 5, 6 in registers);
//        V & 4  decode-phase stores: every lane stores 4 half2 (one SHFL.BFLY pair instead of four SHFL.DOWN, all 32
//               lanes active) instead of half the lanes storing 8;
//        V & 8  the 8-iteration decode loop unrolled;
//        V & 16 transposed store: a warp writes 4 output rows x 128 B per STG.128 (4 cache lines) instead of 32 rows x
//               16 B (32 lines); the shared-memory reads stay conflict free (rows c, c+1 read the same words).
//        V & 32 P2B (pass 2 with 2 lanes per row, 1 SHFL round), V & 64 SUHVEC (suh as one 16-byte load), V & 128 P1B
//               (pass 1 with 32 rows per lane, 2 SHFL rounds), V & 1024 SGNFMA (+-1 HFMA2 sign stage): bit-exact, no gain;
//        V & 256 / 512 min-blocks 5 / 4 (occupancy only);
//        V & 2048 STADDR, V & 4096 DADDR: the store phase / the decode-phase tile stores address the same shared-memory
//               words through per-lane bases + one LOP3 per XOR-swizzle constant (the compiler did not see that the
//               swizzle terms have disjoint bits): -300 / -130 SASS per thread;
//        V & 8192 DPRMT: the DSPLIT packing as lane-selector PRMTs (fewer SEL).
//      Default (wf/builder kiter #32): 6431 = P1 | P2 | DSPLIT | DUNROLL | STORE4 | min-blocks 5 | STADDR | DADDR.
//      Arithmetic in the register butterflies uses the .rn intrinsics: plain add.f16x2 / mul.f16x2 may be contracted to
//      HFMA2 by ptxas once a multiply feeds an add directly (the SHFL form never allows that), which broke P1 before.
//      In fp16, a - b == a + (-b) and a + b == b + a bit for bit, so "partner - self" in a register butterfly equals the
//      SHFL form's "(-self) + partner"; nothing is re-associated.
// No value is computed differently, so the result equals `exllamav3_ext.reconstruct_had_batch(...).transpose(1, 2)` BIT FOR
// BIT (parity/original_basis_parity.py), and the 5.4 ms per layer transposing copy of the first implementation disappears:
// the tier can be rebuilt per prefill chunk into a fixed arena instead of keeping 1.5 GiB per layer resident.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <optional>

#include "exl3_decode.cuh"
#include "third_party/exllamav3/hadamard_inner.cuh"

#ifndef AIKIDO_ORIG_DEFAULT_VARIANT
#define AIKIDO_ORIG_DEFAULT_VARIANT 6431
#endif
// Variants compiled in (bitmask, see AIKIDO (4) above). X(V) list.
#ifndef AIKIDO_ORIG_VARIANTS
#define AIKIDO_ORIG_VARIANTS(X) X(0) X(23) X(31) X(2079) X(6175) X(6431) X(14623)
#endif

namespace aikido_orig {

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};
using FragB = Vec<half2, 2>;

#define RH_THREADS 256

// One butterfly stage across lanes at distance d (one iteration of ExLlamaV3's shuffle_had_h2x32 for i = d).
// FMA = true (AIKIDO (4), V & 1024): (+-1) * self + partner in one HFMA2 instead of the sign-bit LOP3 + HADD2. The product
// by +-1 is exact, so the single rounding of the fma is the rounding of (-self) + partner / self + partner, bit for bit.
template <int d, bool FMA = false>
__device__ __forceinline__ half2 had_stage_shfl(half2 v, int lane_id) {
  half2 pv = __shfl_xor_sync(0xffffffff, v, d);
  if constexpr (FMA) {
    const half2 sg = (lane_id & d) ? __float2half2_rn(-1.0f) : __float2half2_rn(1.0f);
    return __hfma2(v, sg, pv);
  }
  uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
  int32_t sfm = -static_cast<int16_t>(lane_id & d) >> 31;
  *vi ^= (sfm & 0x80008000);
  return __hadd2(v, pv);
}

// The same stage on a lane's own values (index bit d): element i (bit clear) = self + partner, element i | d (bit set) =
// partner - self, i.e. (-self) + partner of the SHFL form, bit for bit.
template <int N, int d>
__device__ __forceinline__ void had_stage_reg(half2* v) {
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    if (!(i & d)) {
      // .rn forms: a plain add.f16x2 after the r_scale mul.f16x2 may be contracted to HFMA2 by ptxas (PTX: ops
      // without a rounding modifier may fuse), which the SHFL form can never do. Same rounding otherwise.
      half2 s = __hadd2_rn(v[i], v[i | d]);
      half2 t = __hsub2_rn(v[i], v[i | d]);
      v[i] = s;
      v[i | d] = t;
    }
  }
}

// STACKED = false: packed_ptrs / suh_ptrs / svh_ptrs are per-expert pointer tables (ExLlamaV3's calling convention).
// STACKED = true: packed_ptrs is really `const int32_t*` = the pack of ALL experts, expert e at e * pack_stride ints, this
// matrix's column groups start at group_offset (gate | up are concatenated along n/64); suh / svh are base pointers of
// stacked tensors, expert e at e * s?h_stride + s?h_off halfs.
// AIKIDO (4, V & 256 / V & 512): occupancy only (register cap for 5 / 4 co-resident blocks per SM); no value changes.
template <int cb, bool STACKED, int K, int V>
__global__ __launch_bounds__(RH_THREADS, (V & 256) ? 5 : ((V & 512) ? 4 : 0))
void aikido_orig_batch_kernel(half* __restrict__ g_out, const void* __restrict__ packed_ptrs,
                              const void* __restrict__ suh_ptrs, const void* __restrict__ svh_ptrs,
                              int packed_blocks_n, size_t out_stride, int row_offset, int k_len,
                              size_t pack_stride, int groups_total, int group_offset, size_t suh_stride, size_t suh_off,
                              size_t svh_stride, size_t svh_off, const int32_t* __restrict__ out_ids) {
  static_assert(K == 3 || K == 4, "K = 3 and K = 4 only");
  constexpr bool P2B = (V & 32) != 0, SUHVEC = (V & 64) != 0, P1B = (V & 128) != 0, SGNFMA = (V & 1024) != 0,
                 STADDR = (V & 2048) != 0, DADDR = (V & 4096) != 0,
                 DPRMT = (V & 8192) != 0;
  constexpr bool P1 = (V & 1) != 0 && !P1B, P2 = (V & 2) != 0 && !P2B, DSPLIT = (V & 4) != 0, DUNROLL = (V & 8) != 0, STORE4 = (V & 16) != 0;
  constexpr int packed_size = 16 * K;                 // uint16 per tile: 48 (K = 3) / 64 (K = 4)
  constexpr int words_per_group = 32 * K;             // int32 per 64-column group of the STACKED pack: 96 / 128
  constexpr float r_scale = 0.08838834764831845f;

  const int bz = blockIdx.z;
  const uint16_t* g_packed = nullptr;
  const int32_t* g_pack = nullptr;
  const half* suh;
  const half* svh;
  if constexpr (STACKED) {
    g_pack = (const int32_t*)packed_ptrs + (size_t)bz * pack_stride;
    suh = (const half*)suh_ptrs + (size_t)bz * suh_stride + suh_off;
    svh = (const half*)svh_ptrs + (size_t)bz * svh_stride + svh_off;
  } else {
    g_packed = ((const uint16_t* const*)packed_ptrs)[bz];
    suh = ((const half* const*)suh_ptrs)[bz];
    svh = ((const half* const*)svh_ptrs)[bz];
  }
  // AIKIDO (3): this expert's rows of the output (out_ids: local -> global expert of a full mixed-K arena)
  half* g_unpacked = g_out + (size_t)(out_ids ? out_ids[bz] : bz) * out_stride;

  int t = threadIdx.x;
  int lane_id = t % 32;
  int warp_id = t / 32;
  int kb = blockIdx.y;
  int nb = blockIdx.x;
  int n = nb * 8;

  __shared__ uint32_t s_packed[8][8][packed_size / 2];
  __shared__ __align__(16) half2 stile[128 * 64];

  auto tix = [&](int R, int q, int p) { return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p; };

  constexpr int j_int4 = packed_size / 8;
  for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS) {
    int j = u / (8 * j_int4);
    int r = u % (8 * j_int4);
    if constexpr (STACKED) {
      // AIKIDO (3): k-tile row (kb*8 + j) of the pack: 2 column groups of this 128-column block x 32 lanes x 4 tiles.
      // int4 r = (group r / 32, lane r % 32) holds that lane's stream word for the 4 tiles of the group; it lands at
      // s_packed[j] int index 4r, i.e. word of (group g2, lane l, tile jj) at s_packed[j] flat index (g2*32 + l)*4 + jj.
      // K = 3: the group is 4 tiles x 24 words as stored, so the two groups of this block land as s_packed[j][wn][24],
      // exactly ExLlamaV3's staging of 8 consecutive tiles.
      const int32_t* gp = g_pack + ((size_t)(kb * 8 + j) * groups_total + group_offset + nb * 2) * words_per_group;
      ((int4*)s_packed[j])[r] = ((const int4*)gp)[r];
    } else {
      const uint16_t* gp = g_packed + ((size_t)((kb * 8 + j) * packed_blocks_n + n)) * packed_size;
      ((int4*)s_packed[j])[r] = ((const int4*)gp)[r];
    }
  }
  __syncthreads();

  constexpr int DEC_UNROLL = DUNROLL ? 8 : 1;
  #pragma unroll DEC_UNROLL
  for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj) {
    int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
    int wn = warp_id % 8;
    register FragB frag[2];
    // AIKIDO (1): same decode, our entry points. ExLlamaV3: dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, ...)
    if constexpr (K == 3) {
      int src_a, src_b, s2;
      aikido_exl3::k3_lane_words(lane_id, src_a, src_b, s2);
      if constexpr (STACKED && AIKIDO_K3_TILE_INTERLEAVE) {   // stack unit = [24 words][4 tiles]
        const uint32_t* flat = (const uint32_t*)s_packed[j];
        const int g2 = wn / 4, tj = wn % 4;
        aikido_exl3::dq8_regs_3bits<FragB, cb>(flat[(g2 * 24 + src_a) * 4 + tj], flat[(g2 * 24 + src_b) * 4 + tj], s2,
                                               frag[0], frag[1]);
      } else {
        const uint32_t* ptr = s_packed[j][wn];
        aikido_exl3::dq8_regs_3bits<FragB, cb>(ptr[src_a], ptr[src_b], s2, frag[0], frag[1]);
      }
    } else if constexpr (STACKED) {
      const uint32_t* flat = (const uint32_t*)s_packed[j];     // [(g2 * 32 + lane) * 4 + jj], tile wn = g2 * 4 + jj
      const int g2 = wn / 4, tj = wn % 4;
      aikido_exl3::dq8_regs_4bits<FragB, cb>(flat[(g2 * 32 + ((lane_id + 31) & 31)) * 4 + tj], flat[(g2 * 32 + lane_id) * 4 + tj],
                                             frag[0], frag[1]);
    } else {
      const uint32_t* ptr = s_packed[j][wn];
      aikido_exl3::dq8_regs_4bits<FragB, cb>(ptr[(lane_id + 31) & 31], ptr[lane_id], frag[0], frag[1]);
    }

    int r0 = j * 16 + (lane_id % 4) * 2;
    int r1 = r0 + 1;
    int r2 = r0 + 8;
    int r3 = r0 + 9;
    int c0 = lane_id / 8;
    if constexpr (!DSPLIT) {
      half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
      half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
      half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
      half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

      if (!(lane_id & 4)) {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
        int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
        stile[tix(r0, q0, p0)] = m0;
        stile[tix(r1, q0, p0)] = m1;
        stile[tix(r2, q0, p0)] = m2;
        stile[tix(r3, q0, p0)] = m3;
        stile[tix(r0, q1, p1)] = m4;
        stile[tix(r1, q1, p1)] = m5;
        stile[tix(r2, q1, p1)] = m6;
        stile[tix(r3, q1, p1)] = m7;
      }
    } else {
      // AIKIDO (4, V & 4): lane l (bit 2 clear) pairs its frag[0] with lane l+4's frag[0] (= m0..m3 above), lane l+4
      // pairs lane l's frag[1] with its own frag[1] (= m4..m7 above): the same half2 values to the same slots, one
      // SHFL.BFLY per fragment word instead of four SHFL.DOWN, and all 32 lanes store 4 words each (branch free).
      const bool hi = (lane_id & 4) != 0;
      half2 s0 = __shfl_xor_sync(0xFFFFFFFF, hi ? frag[0][0] : frag[1][0], 4, 32);
      half2 s1 = __shfl_xor_sync(0xFFFFFFFF, hi ? frag[0][1] : frag[1][1], 4, 32);
      half2 x0 = hi ? s0 : frag[0][0], y0 = hi ? frag[1][0] : s0;
      half2 x1 = hi ? s1 : frag[0][1], y1 = hi ? frag[1][1] : s1;
      int cc = c0 + (hi ? 4 : 0);
      int q = (wn * 8 + cc) >> 1, p = cc & 1;
      if constexpr (DPRMT) {
        // AIKIDO (4, V & 8192): the same four half2 as below, as PRMT with a lane-constant selector on (t, s):
        // lo lanes (t = frag[0][.]): (low t, low s) / (high t, high s); hi lanes (t = frag[1][.]): (low s, low t) /
        // (high s, high t). One SEL per fragment word instead of two, one PRMT per stored word.
        const uint32_t selL = hi ? 0x1054u : 0x5410u, selH = hi ? 0x3276u : 0x7632u;
        const half2 t0 = hi ? frag[1][0] : frag[0][0], t1 = hi ? frag[1][1] : frag[0][1];
        const uint32_t ut0 = *reinterpret_cast<const uint32_t*>(&t0), ut1 = *reinterpret_cast<const uint32_t*>(&t1);
        const uint32_t us0 = *reinterpret_cast<const uint32_t*>(&s0), us1 = *reinterpret_cast<const uint32_t*>(&s1);
        uint32_t o0 = __byte_perm(ut0, us0, selL), o1 = __byte_perm(ut0, us0, selH);
        uint32_t o2 = __byte_perm(ut1, us1, selL), o3 = __byte_perm(ut1, us1, selH);
        const int qa = q ^ ((lane_id & 3) >> 1);
        uint32_t* b0 = reinterpret_cast<uint32_t*>(stile + (j * 16 + (lane_id & 3) * 2) * 64 + p);
        uint32_t* w0 = b0 + (qa ^ (4 * j)) * 2;
        uint32_t* w2 = b0 + 8 * 64 + (qa ^ (4 * j) ^ 2) * 2;
        w0[0] = o0;
        w0[64] = o1;
        w2[0] = o2;
        w2[64] = o3;
      } else if constexpr (DADDR) {
        // AIKIDO (4, V & 4096): the same four slots through a per-lane base. r0 = 16j + 2a' (a' = lane & 3), so
        // (r0 >> 2) = 4j + (a' >> 1) and (r2 >> 2) = that + 2, disjoint bits: q ^ (r >> 2) = (q ^ (a' >> 1)) ^ 4j [^ 2].
        const int qa = q ^ ((lane_id & 3) >> 1);
        half2* b0 = stile + (j * 16 + (lane_id & 3) * 2) * 64 + p;
        half2* w0 = b0 + (qa ^ (4 * j)) * 2;
        half2* w2 = b0 + 8 * 64 + (qa ^ (4 * j) ^ 2) * 2;
        w0[0] = __halves2half2(__low2half(x0), __low2half(y0));
        w0[64] = __halves2half2(__high2half(x0), __high2half(y0));
        w2[0] = __halves2half2(__low2half(x1), __low2half(y1));
        w2[64] = __halves2half2(__high2half(x1), __high2half(y1));
      } else {
      stile[tix(r0, q, p)] = __halves2half2(__low2half(x0), __low2half(y0));
      stile[tix(r1, q, p)] = __halves2half2(__high2half(x0), __high2half(y0));
      stile[tix(r2, q, p)] = __halves2half2(__low2half(x1), __low2half(y1));
      stile[tix(r3, q, p)] = __halves2half2(__high2half(x1), __high2half(y1));
      }
    }
  }
  __syncthreads();

  const half2 rs2 = __float2half2_rn(r_scale);
  if constexpr (P1B) {
    // AIKIDO (4, V & 128): lane (p = lane & 1, qh = (lane >> 1) & 3, m = lane >> 3) of warp w holds half2 column p of
    // logical quad q = (w & 3) | 4qh | 16(w >> 2) for the 32 rows R = (i & 3) | 4m | 16(i >> 2), i = 0..31. Row bits 0, 1
    // are the register 4-point (+ r_scale) as above, bits 2, 3 = lane bits 3, 4 are SHFL xor 8, 16, bits 4, 5, 6 = i bits
    // 2, 3, 4 are register butterflies (same stage order). One LDS.32 per value: (q ^ sw) mod 16 has bits 0, 1 = (w & 3) ^ m
    // and bits 2, 3 = qh ^ (i >> 2): 32 distinct banks.
    const int pp = lane_id & 1, qh = (lane_id >> 1) & 3, m = lane_id >> 3;
    const int q = (warp_id & 3) | (qh << 2) | ((warp_id >> 2) << 4);
    half2 v[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int R = (i & 3) | (m << 2) | ((i >> 2) << 4);
      v[i] = stile[R * 64 + (q ^ ((R >> 2) & 31)) * 2 + pp];
    }
    #pragma unroll
    for (int i4 = 0; i4 < 32; i4 += 4) {
      half2 s0 = __hadd2(v[i4], v[i4 + 1]), d0 = __hsub2(v[i4], v[i4 + 1]);
      half2 s1 = __hadd2(v[i4 + 2], v[i4 + 3]), d1 = __hsub2(v[i4 + 2], v[i4 + 3]);
      v[i4] = __hmul2_rn(__hadd2(s0, s1), rs2);
      v[i4 + 1] = __hmul2_rn(__hadd2(d0, d1), rs2);
      v[i4 + 2] = __hmul2_rn(__hsub2(s0, s1), rs2);
      v[i4 + 3] = __hmul2_rn(__hsub2(d0, d1), rs2);
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = had_stage_shfl<8, SGNFMA>(v[i], lane_id);
    #pragma unroll
    for (int i = 0; i < 32; ++i) v[i] = had_stage_shfl<16, SGNFMA>(v[i], lane_id);
    had_stage_reg<32, 4>(v);
    had_stage_reg<32, 8>(v);
    had_stage_reg<32, 16>(v);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      const int R = (i & 3) | (m << 2) | ((i >> 2) << 4);
      stile[R * 64 + (q ^ ((R >> 2) & 31)) * 2 + pp] = v[i];
    }
  } else if constexpr (!P1) {
    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq) {
      int q = warp_id * CHUNKS_PW + qq;
      int qs = q ^ lane_id;
      half2 a[4], b[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        half4 v = *((const half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2));
        a[i] = v.x;
        b[i] = v.y;
      }
      #pragma unroll
      for (int x = 0; x < 2; ++x) {
        half2* v = x == 0 ? a : b;
        half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
        half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
        v[0] = __hmul2(__hadd2(s0, s1), rs2);
        v[1] = __hmul2(__hadd2(d0, d1), rs2);
        v[2] = __hmul2(__hsub2(s0, s1), rs2);
        v[3] = __hmul2(__hsub2(d0, d1), rs2);
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = shuffle_had_h2x32(v[i], lane_id);
      }
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        half4 v;
        v.x = a[i];
        v.y = b[i];
        *((half4*)(stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
      }
    }
  } else {
    // AIKIDO (4, V & 1): lane (g = lane & 3, m = lane >> 2) holds rows 16m .. 16m+15 of logical column quad
    // warp*4 + g (physical quad (q ^ 4m) ^ (i >> 2) in row 16m + i: a half-warp touches 16 distinct 8-byte bank pairs).
    // Row bits 0, 1 (+ r_scale), 2, 3 are register butterflies; bits 4, 5, 6 = lane bits 2, 3, 4 are SHFL xor 4, 8, 16.
    const int g = lane_id & 3, m = lane_id >> 2;
    const int q = warp_id * 4 + g;
    const int qm = q ^ (4 * m);
    half2 a[16], b[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      half4 v = *((const half4*)(stile + (16 * m + i) * 64 + (qm ^ (i >> 2)) * 2));
      a[i] = v.x;
      b[i] = v.y;
    }
    #pragma unroll
    for (int x = 0; x < 2; ++x) {
      half2* v = x == 0 ? a : b;
      #pragma unroll
      for (int i4 = 0; i4 < 16; i4 += 4) {
        half2 s0 = __hadd2(v[i4], v[i4 + 1]), d0 = __hsub2(v[i4], v[i4 + 1]);
        half2 s1 = __hadd2(v[i4 + 2], v[i4 + 3]), d1 = __hsub2(v[i4 + 2], v[i4 + 3]);
        v[i4] = __hmul2_rn(__hadd2(s0, s1), rs2);
        v[i4 + 1] = __hmul2_rn(__hadd2(d0, d1), rs2);
        v[i4 + 2] = __hmul2_rn(__hsub2(s0, s1), rs2);
        v[i4 + 3] = __hmul2_rn(__hsub2(d0, d1), rs2);
      }
      had_stage_reg<16, 4>(v);
      had_stage_reg<16, 8>(v);
      #pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = had_stage_shfl<4, SGNFMA>(v[i], lane_id);
      #pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = had_stage_shfl<8, SGNFMA>(v[i], lane_id);
      #pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = had_stage_shfl<16, SGNFMA>(v[i], lane_id);
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      half4 v;
      v.x = a[i];
      v.y = b[i];
      *((half4*)(stile + (16 * m + i) * 64 + (qm ^ (i >> 2)) * 2)) = v;
    }
  }
  __syncthreads();

  if constexpr (P2B) {
    // AIKIDO (4, V & 32): 2 lanes per row; lane (x = lane & 1, r = (lane >> 1) & 7, h = lane >> 4) of warp w holds row
    // R = w + 8r + 64h (row bits 0..2 = w, 3..5 = r, 6 = h) and logical column quads 2j + x, j = 0..15. Column bits 0, 1
    // are the fp32 4-point (+ r_scale) as above; column bit 2 = quad bit 0 = lane bit 0 is SHFL xor 1; column bits 3..6 =
    // quad bits 1..4 = j bits 0..3 are register butterflies at c[] distance 2, 4, 8, 16 (same stage order 1, 2, 4, 8, 16).
    // Bank pairs of a half-warp: (q ^ sw) mod 16 has bit 0 = x ^ (w >> 2), bits 1..3 = j ^ r: 16 distinct.
    const int x = lane_id & 1, rr8 = (lane_id >> 1) & 7, h = lane_id >> 4;
    const int R = warp_id + 8 * rr8 + 64 * h;
    const int sw = (R >> 2) & 31;
    half2 c[32];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int base = R * 64 + ((2 * j + x) ^ sw) * 2;
      half4 v = *((const half4*)(stile + base));
      half2 v01 = v.x;
      half2 v23 = v.y;
      float v0 = __low2float(v01), v1 = __high2float(v01);
      float v2 = __low2float(v23), v3 = __high2float(v23);
      float s0 = v0 + v1, d0 = v0 - v1;
      float s1 = v2 + v3, d1 = v2 - v3;
      c[2 * j] = __hmul2_rn(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
      c[2 * j + 1] = __hmul2_rn(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) c[i] = had_stage_shfl<1, SGNFMA>(c[i], lane_id);
    had_stage_reg<32, 2>(c);
    had_stage_reg<32, 4>(c);
    had_stage_reg<32, 8>(c);
    had_stage_reg<32, 16>(c);
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int base = R * 64 + ((2 * j + x) ^ sw) * 2;
      half4 v;
      v.x = c[2 * j];
      v.y = c[2 * j + 1];
      *((half4*)(stile + base)) = v;
    }
  } else if constexpr (!P2) {
    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr) {
      int R = warp_id * ROWS_PW + rr;
      int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
      half2 v01 = stile[base];
      half2 v23 = stile[base + 1];
      float v0 = __low2float(v01), v1 = __high2float(v01);
      float v2 = __low2float(v23), v3 = __high2float(v23);
      float s0 = v0 + v1, d0 = v0 - v1;
      float s1 = v2 + v3, d1 = v2 - v3;
      half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
      half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
      h01 = shuffle_had_h2x32(h01, lane_id);
      h23 = shuffle_had_h2x32(h23, lane_id);
      // AIKIDO (2): ExLlamaV3 scales here (o = (h * suh[kb*128 + R]) * svh[nb*128 + 4*lane + j]) and stores one half4 at
      // [kb*128 + R][nb*128 + 4*lane .. +3] of a [k, n] matrix. We keep the transformed tile in shared memory (the slots
      // this lane just read) and scale + store it transposed below, 16 bytes per store, instead of four 2-byte stores per
      // lane 4 rows apart (k_len * 2 bytes each): same two fp16 multiplies per element in the same order.
      stile[base] = h01;
      stile[base + 1] = h23;
    }
  } else {
    // AIKIDO (4, V & 2): 8 lanes per row; lane (m = lane & 7, h = lane >> 3) holds logical column quads 8j + m
    // (j = 0..3) of row R = warp*4 + pp + 32h (physical quad 8(j ^ h) + (m ^ warp): a half-warp touches 16 distinct
    // 8-byte bank pairs). Column bits 0, 1 are the fp32 4-point as above (+ r_scale), bits 2, 3, 4 = lane bits 0, 1, 2
    // are SHFL xor 1, 2, 4, bits 5, 6 = j are register butterflies (index distance 2, 4 in c[]).
    const int m = lane_id & 7, h = lane_id >> 3;
    const int pm = m ^ warp_id;
    #pragma unroll
    for (int pp = 0; pp < 4; ++pp) {
      const int R = warp_id * 4 + pp + 32 * h;
      half2 c[8];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int base = R * 64 + (8 * (j ^ h) + pm) * 2;
        half4 v = *((const half4*)(stile + base));
        half2 v01 = v.x;
        half2 v23 = v.y;
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        c[2 * j] = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        c[2 * j + 1] = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
      }
      #pragma unroll
      for (int i = 0; i < 8; ++i) c[i] = had_stage_shfl<1, SGNFMA>(c[i], lane_id);
      #pragma unroll
      for (int i = 0; i < 8; ++i) c[i] = had_stage_shfl<2, SGNFMA>(c[i], lane_id);
      #pragma unroll
      for (int i = 0; i < 8; ++i) c[i] = had_stage_shfl<4, SGNFMA>(c[i], lane_id);
      had_stage_reg<8, 2>(c);
      had_stage_reg<8, 4>(c);
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int base = R * 64 + (8 * (j ^ h) + pm) * 2;
        half4 v;
        v.x = c[2 * j];
        v.y = c[2 * j + 1];
        *((half4*)(stile + base)) = v;
      }
    }
  }
  __syncthreads();

  // Transposed, vectorised store: thread t owns output row (row_offset + nb*128 + c), c = t % 128, and the 64 columns
  // kb*128 + k0 .. + 63, k0 = (t / 128) * 64 (a warp reads 32 consecutive c of one row R: 16 distinct words, no bank
  // conflicts); element (R, c) of the tile sits at stile[tix(R, c / 4, (c / 2) & 1)] half (c & 1). Pairs of rows
  // (R, R + 1) form one half2 so the scaling is the same HMUL2 arithmetic elementwise: (h * su[R]) * sv[c].
  if constexpr (!STORE4) {
    const int c = t & 127, k0 = (t >> 7) * 64;
    const int q = c >> 2, p = (c >> 1) & 1, e = c & 1;
    const half2 sv2 = __half2half2(svh[nb * 128 + c]);
    half* dst = g_unpacked + (size_t)(row_offset + nb * 128 + c) * k_len + (kb * 128 + k0);
    #pragma unroll 2
    for (int r8 = 0; r8 < 64; r8 += 8) {
      half2 o[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int R = k0 + r8 + 2 * i;
        const half2 a = stile[tix(R, q, p)], b = stile[tix(R + 1, q, p)];
        const half2 h2 = e ? __halves2half2(__high2half(a), __high2half(b)) : __halves2half2(__low2half(a), __low2half(b));
        const half2 su2 = __halves2half2(suh[kb * 128 + R], suh[kb * 128 + R + 1]);
        o[i] = __hmul2(__hmul2(h2, su2), sv2);
      }
      *((int4*)(dst + r8)) = *((const int4*)o);
    }
  } else {
    // AIKIDO (4, V & 16): per pass a warp owns 4 consecutive output rows c0 .. c0+3 (c0 = pass*32 + warp*4); lane
    // (rl = lane >> 3, ch = lane & 7) writes row c0 + rl, columns kb*128 + 8ch .. +7 and + 64: one STG.128 per warp
    // covers 4 rows x 128 contiguous bytes. Rows c0, c0+1 (and c0+2, c0+3) read the same tile words (they differ in
    // the half e only), so the 16 distinct words of a read hit 16 distinct bank pairs. Same HMUL2 per element.
    const int rl = lane_id >> 3, ch = lane_id & 7;
    // AIKIDO (4, V & 2048): the same tile words through precomputed addresses. For R = 8ch + 64hh + 4ih + {0, 2} (+ 1),
    // tix(R, q, p) = R * 64 + (q ^ (R >> 2)) * 2 + p with q = 8pass | warp and R >> 2 = 2ch | 16hh | ih (disjoint bits), so
    // q ^ (R >> 2) = X ^ (8pass ^ 16hh ^ ih), X = warp ^ 2ch: one LOP3 per (pass, hh, ih), immediate row offsets.
    const int sa_p = (rl >> 1) & 1;
    const int sa_x = warp_id ^ (2 * ch);
    const half2* sa_base = stile + (8 * ch) * 64 + sa_p;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
      const int c = pass * 32 + warp_id * 4 + rl;
      const int q = c >> 2, p = (c >> 1) & 1, e = c & 1;
      const half2 sv2 = __half2half2(svh[nb * 128 + c]);
      half* dst = g_unpacked + (size_t)(row_offset + nb * 128 + c) * k_len + kb * 128;
      #pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int k0 = (ch + 8 * hh) * 8;
        half2 o[4];
        // AIKIDO (4, V & 64): suh[kb*128 + k0 .. +7] as one 16-byte load (the host checks the alignment), the same halves.
        int4 su8;
        if constexpr (SUHVEC) su8 = *((const int4*)(suh + kb * 128 + k0));
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int R = k0 + 2 * i;
          half2 a, b;
          if constexpr (STADDR) {
            const half2* rowp = sa_base + (64 * hh + 4 * (i >> 1) + 2 * (i & 1)) * 64 + (sa_x ^ (8 * pass ^ 16 * hh ^ (i >> 1))) * 2;
            a = rowp[0];
            b = rowp[64];
          } else {
            a = stile[tix(R, q, p)];
            b = stile[tix(R + 1, q, p)];
          }
          const half2 h2 = e ? __halves2half2(__high2half(a), __high2half(b)) : __halves2half2(__low2half(a), __low2half(b));
          half2 su2;
          if constexpr (SUHVEC) su2 = ((const half2*)&su8)[i];
          else su2 = __halves2half2(suh[kb * 128 + R], suh[kb * 128 + R + 1]);
          o[i] = __hmul2(__hmul2(h2, su2), sv2);
        }
        *((int4*)(dst + k0)) = *((const int4*)o);
      }
    }
  }
}

// Runtime variant (AIKIDO (4)): AIKIDO_ORIG_VARIANT at first use, or moe_orig_set_variant().
static int g_variant = -1;
int variant() {
  if (g_variant < 0) {
    const char* s = std::getenv("AIKIDO_ORIG_VARIANT");
    g_variant = s ? std::atoi(s) : AIKIDO_ORIG_DEFAULT_VARIANT;
  }
  return g_variant;
}
void set_variant(int v) { g_variant = v; }

template <int cb, bool STACKED, int K, typename... Args>
void launch_variant(int v, dim3 grid, cudaStream_t stream, Args... args) {
  if constexpr (cb != 1) {      // MUL1: the reference variant only (neither model uses it)
    (void)v;                    // variants are built for cb 1 (MCG) only; MUL1 always runs the reference variant
    aikido_orig_batch_kernel<cb, STACKED, K, 0><<<grid, RH_THREADS, 0, stream>>>(args...);
    return;
  } else {
    switch (v) {
#define AIKIDO_ORIG_CASE(VV) case VV: aikido_orig_batch_kernel<cb, STACKED, K, VV><<<grid, RH_THREADS, 0, stream>>>(args...); return;
      AIKIDO_ORIG_VARIANTS(AIKIDO_ORIG_CASE)
#undef AIKIDO_ORIG_CASE
      default: TORCH_CHECK(false, "original-basis builder: variant ", v, " not built");
    }
  }
}

template <bool STACKED, typename... Args>
void dispatch(int64_t cb, int64_t bits, int v, dim3 grid, cudaStream_t stream, Args... args) {
  if (cb == 1 && bits == 4) launch_variant<1, STACKED, 4>(v, grid, stream, args...);
  else if (cb == 2 && bits == 4) launch_variant<2, STACKED, 4>(v, grid, stream, args...);
  else if (cb == 1 && bits == 3) launch_variant<1, STACKED, 3>(v, grid, stream, args...);
  else if (cb == 2 && bits == 3) launch_variant<2, STACKED, 3>(v, grid, stream, args...);
  else TORCH_CHECK(false, "original-basis builder: codebook ", cb, " / K ", bits, " not built (cb 1 = MCG, 2 = MUL1; K 3, 4)");
}

}  // namespace aikido_orig

void moe_orig_set_variant(int64_t v) { aikido_orig::set_variant((int)v); }
int64_t moe_orig_get_variant() { return aikido_orig::variant(); }

// out: fp16 [E, rows_total, k] contiguous (vLLM [E, out, in]); this call fills rows row_offset .. row_offset + n - 1 of every
// expert with the n x k transpose of W. Pointer tables: int64 device tensors (trellis int16 [k/16, n/16, 64], suh fp16 [k],
// svh fp16 [n] per expert), as exllamav3_ext.reconstruct_had_batch takes them. bits = K (3 or 4); cb 1 = MCG, 2 = MUL1.
void moe_build_orig_vllm(at::Tensor& out, const at::Tensor& trellis_ptrs, const at::Tensor& suh_ptrs,
                         const at::Tensor& svh_ptrs, int64_t n, int64_t row_offset, int64_t cb, int64_t bits) {
  const at::cuda::OptionalCUDAGuard device_guard(out.device());
  TORCH_CHECK(out.dim() == 3 && out.dtype() == at::kHalf && out.is_contiguous(), "out must be contiguous fp16 [E, rows, k]");
  const int64_t e = out.size(0), rows_total = out.size(1), k = out.size(2);
  TORCH_CHECK(k % 128 == 0 && n % 128 == 0 && row_offset % 128 == 0 && row_offset >= 0 && row_offset + n <= rows_total,
              "k, n, row_offset must be multiples of 128 and fit the output");
  for (const at::Tensor* p : {&trellis_ptrs, &suh_ptrs, &svh_ptrs})
    TORCH_CHECK(p->dtype() == at::kLong && p->is_contiguous() && p->device() == out.device() && p->numel() >= e,
                "pointer tables must be contiguous int64 tensors on the output device with one entry per expert");
  if (e == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)(n / 128), (unsigned)(k / 128), (unsigned)e);
  aikido_orig::dispatch<false>(cb, bits, aikido_orig::variant() & ~64, grid, stream, (half*)out.data_ptr(), (const void*)trellis_ptrs.data_ptr(),
                               (const void*)suh_ptrs.data_ptr(), (const void*)svh_ptrs.data_ptr(),
                               (int)(n / 16), (size_t)rows_total * k, (int)row_offset, (int)k,
                               (size_t)0, 0, 0, (size_t)0, (size_t)0, (size_t)0, (size_t)0, (const int32_t*)nullptr);
}

// STACKED variant: reads the grouped kernel's resident pack. pack: int32 [E, k/16, groups_total, 32, 4] for K = 4 or
// [E, k/16, groups_total, 4, 24] for K = 3 (n_total = 64 * groups_total columns; this matrix = columns col_offset ..
// col_offset + n - 1 of it); suh: fp16 [E, ..., k] with this matrix's vector at flat offset suh_off inside an expert's
// slab; svh: fp16 [E, n_total'] with offset svh_off. out_ids (optional int32 [E_pack]): output expert of each pack expert;
// without it expert e of the pack writes expert e of out, and out must hold exactly the pack's experts.
void moe_build_orig_vllm_stacked(at::Tensor& out, const at::Tensor& pack, const at::Tensor& suh, const at::Tensor& svh,
                                 int64_t n, int64_t col_offset, int64_t suh_off, int64_t svh_off, int64_t row_offset, int64_t cb,
                                 const std::optional<at::Tensor>& out_ids) {
  const at::cuda::OptionalCUDAGuard device_guard(out.device());
  TORCH_CHECK(out.dim() == 3 && out.dtype() == at::kHalf && out.is_contiguous(), "out must be contiguous fp16 [E, rows, k]");
  const int64_t rows_total = out.size(1), k = out.size(2);
  TORCH_CHECK(pack.dim() == 5 && pack.dtype() == at::kInt && pack.is_contiguous() && pack.size(1) * 16 == k &&
              ((pack.size(3) == 32 && pack.size(4) == 4) || (!AIKIDO_K3_TILE_INTERLEAVE && pack.size(3) == 4 && pack.size(4) == 24) || (AIKIDO_K3_TILE_INTERLEAVE && pack.size(3) == 24 && pack.size(4) == 4)),
              "pack must be contiguous int32 [E, k/16, n_total/64, 32, 4] (K = 4) or [E, k/16, n_total/64, 4, 24] (K = 3)");
  const int64_t e = pack.size(0), bits = (pack.size(4) == 24 || pack.size(3) == 24) ? 3 : 4;
  const int32_t* ids = nullptr;
  if (out_ids.has_value() && out_ids->defined()) {
    TORCH_CHECK(out_ids->dtype() == at::kInt && out_ids->is_contiguous() && out_ids->device() == out.device() && out_ids->numel() == e,
                "out_ids must be a contiguous int32 tensor on the output device with one entry per pack expert");
    ids = out_ids->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(out.size(0) == e, "out must hold exactly the pack's experts when no out_ids table is given");
  }
  const int64_t groups_total = pack.size(2);
  TORCH_CHECK(k % 128 == 0 && n % 128 == 0 && col_offset % 128 == 0 && col_offset + n <= groups_total * 64 && row_offset % 128 == 0 &&
              row_offset >= 0 && row_offset + n <= rows_total, "k, n, offsets must be multiples of 128 and fit");
  TORCH_CHECK(suh.dtype() == at::kHalf && svh.dtype() == at::kHalf && suh.is_contiguous() && svh.is_contiguous() &&
              suh.size(0) == e && svh.size(0) == e, "suh / svh must be contiguous fp16 stacked per expert");
  const int64_t suh_stride = suh.numel() / e, svh_stride = svh.numel() / e;
  TORCH_CHECK(suh_off >= 0 && suh_off + k <= suh_stride && svh_off >= 0 && svh_off + n <= svh_stride, "suh / svh offsets out of range");
  if (e == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)(n / 128), (unsigned)(k / 128), (unsigned)e);
  int v = aikido_orig::variant();
  if (((uintptr_t)suh.data_ptr() % 16) || (suh_stride % 8) || (suh_off % 8)) v &= ~64;   // V & 64 needs 16-byte suh rows
  aikido_orig::dispatch<true>(cb, bits, v, grid, stream, (half*)out.data_ptr(), (const void*)pack.data_ptr(),
                              (const void*)suh.data_ptr(), (const void*)svh.data_ptr(), 0, (size_t)rows_total * k, (int)row_offset, (int)k,
                              (size_t)(pack.numel() / e), (int)groups_total, (int)(col_offset / 64),
                              (size_t)suh_stride, (size_t)suh_off, (size_t)svh_stride, (size_t)svh_off, ids);
}
