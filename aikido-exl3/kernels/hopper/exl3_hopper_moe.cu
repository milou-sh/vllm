// aikido-exl3 Hopper MoE kernels: host side + torch bindings (extension module `aikido_exl3_moe_kernels`).
//
// Routed experts of an EXL3 MoE layer, all experts of a projection stacked [E, ...] in the lane-major layout of
// `aikido_exl3_kernels.repack_trellis`, routed with vLLM's moe_align_block_size (sorted_token_ids / expert_ids /
// num_tokens_post_padded: the inputs vLLM's Marlin MoE takes). One layer =
//   1. moe_had_in      xh[s][slot] = Had128(fp16(x[slot / top_k]) * suh[expert(slot)][s])      (s = gate, up)
//   2. moe_gemm        gate+up of every moe block in ONE launch (shard map: column slices < I read slab 0, the
//                      others slab 1), output transform Had128 -> * svh[expert] inside the launch
//   3. moe_glu_had_in  xd[slot] = Had128(fp16(silu(g) * u) * suh_down[expert(slot)])
//   4. moe_gemm        down, output transform inside the launch
//   5. moe_combine     y[t] = sum_j w[t, j] * yd[t * top_k + j]   (fp32, ordered; fp16 or bf16 out)
//
// - GEMM: exl3_hopper_moe_template.h = vLLM's Marlin MoE kernel template (Apache-2.0) with the int4 dequant replaced
//   by EXL3's trellis decode (ExLlamaV3, MIT). Host launch logic follows vLLM v0.29.0
//   csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu (Apache-2.0, Neural Magic / vLLM project).
// - Hadamards: ExLlamaV3's had_hf_r_128_inner arithmetic (exl3_had.cuh); SiLU: ExLlamaV3's fp32 act_silu
//   (exl3_moe_coop_kernel.cuh, MIT).
//
// Decoded expert weights are bit-identical to exllamav3_ext.reconstruct; only the accumulation order differs.
// Every launch has a static shape for a fixed (tokens, top_k, moe_block_size): CUDA-graph safe. The number of valid
// moe blocks is read on the device (num_tokens_post_padded), as in Marlin MoE.

#define MARLIN_NAMESPACE_NAME aikido_exl3_marlin_moe

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <climits>
#include <map>
#include <vector>

#include "exl3_hopper_moe_kernels.h"
#include "exl3_had.cuh"
#include "exl3_pdl.cuh"      // AIKIDO: programmatic dependent launch
#include "exl3_decode.cuh"   // AIKIDO_CB_MCG_LUT, decode_3inst (the table generator uses the codebook arithmetic itself)

// AIKIDO PDL: which launches carry cudaLaunchAttributeProgrammaticStreamSerialization (exl3_pdl.cuh). Bit mask:
// 1 had_in, 2 gemm gate+up, 4 glu_had_in, 8 gemm down, 16 combine, 32 align_decode (a kernel launched with the
// attribute also executes griddepcontrol.wait; nothing is executed in kernels launched without it);
// 64: had_in runs concurrently with align (it does not depend on align: triggers at its top, waits at its end) and
// the gate+up GEMM waits at ITS top instead of after its routing prologue;
// 128: the per-slot kernels (had_in, glu, combine, align) execute griddepcontrol.launch_dependents explicitly
// (measured: ~1 ns per block, serialized, so 12k-block grids pay 10+ us; off = the implicit trigger at block exit);
// 256: the GEMMs execute griddepcontrol.launch_dependents explicitly (264 blocks: cheap).
// Compile-time default AIKIDO_MOE_PDL_DEFAULT (setup.py: env AIKIDO_MOE_PDL), overridden at load by env
// AIKIDO_EXL3_MOE_PDL and at run time by moe_set_pdl (tools/moe_decode_profile.py sweeps it).
#ifndef AIKIDO_MOE_PDL_DEFAULT
#define AIKIDO_MOE_PDL_DEFAULT 0
#endif
static int pdl_mask_init() {
  const char* e = getenv("AIKIDO_EXL3_MOE_PDL");
  return e ? atoi(e) : AIKIDO_MOE_PDL_DEFAULT;
}
static int g_pdl = pdl_mask_init();
enum : int { kPdlHadIn = 1, kPdlGemm13 = 2, kPdlGlu = 4, kPdlGemm2 = 8, kPdlCombine = 16, kPdlAlign = 32,
             kPdlHadInEarly = 64, kPdlSlotTrigger = 128, kPdlGemmTrigger = 256,
             kPdlBPrefetch = 512, kPdlSlotTailTrigger = 1024, kPdlAlignTrigTop = 2048,
             kPdlHadInResident = 4096 };
// per-slot kernel argument: bit 0 wait, bit 1 trigger after the wait, bit 2 (had_in only) trigger at the top and
// wait at the end
static int slot_pdl(int attr_bit) {
  // triggers (128 / 1024) apply whether or not this launch carries the attribute: they only release the next launch
  const int trig = ((g_pdl & kPdlSlotTrigger) ? 2 : 0) | ((g_pdl & kPdlSlotTailTrigger) ? 8 : 0);
  if (!(g_pdl & attr_bit)) return trig | ((attr_bit == kPdlAlign && (g_pdl & kPdlAlignTrigTop)) ? 16 : 0);
  return 1 | trig | ((attr_bit == kPdlHadIn && (g_pdl & kPdlHadInEarly)) ? 4 : 0)
           | ((attr_bit == kPdlAlign && (g_pdl & kPdlAlignTrigTop)) ? 16 : 0);
}


// cudaLaunchKernelEx with the programmatic-stream-serialization attribute when `pdl` (else a plain launch); the
// attribute survives stream capture as a programmatic graph edge (CUDA >= 12.3).
template <typename... KArgs, typename... Args>
static void aikido_launch(bool pdl, void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                          Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

namespace MARLIN_NAMESPACE_NAME {

// Variants compiled into this build: AIKIDO_MOE_VARIANTS(X) = X(mb, cb) ... (header generated by setup.py)
#include "aikido_moe_families.h"

#define AIKIDO_MOE_SEL(threads_, tn_, tk_, mb_, cb_, bits_)                                         \
  if (threads == threads_ && tn == tn_ && tk == tk_ && mb == mb_ && cb == cb_ && bits == bits_)     \
    return AIKIDO_MOE_KERNEL(threads_, tn_, tk_, mb_, cb_, bits_);
#define AIKIDO_MOE_SEL_VARIANT(mb, cb, bits) AIKIDO_MOE_THREAD_CFGS(AIKIDO_MOE_SEL, mb, cb, bits)

static MoeFuncPtr select_kernel(int threads, int tn, int tk, int mb, int cb, int bits) {
  AIKIDO_MOE_VARIANTS(AIKIDO_MOE_SEL_VARIANT)
  return nullptr;
}

struct ThreadCfg {
  int thread_k, thread_n, num_threads;
};

// Same tables and priority as Marlin MoE.
static const ThreadCfg small_cfgs[] = {{128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
static const ThreadCfg large_cfgs[] = {{64, 256, 256}, {64, 128, 128}, {128, 64, 128}};

// Marlin MoE's get_kernel_cache_size for 4-bit weights, fp16 activations, no groups / zero points / act-order.
static int kernel_cache_size(const ThreadCfg& c, int thread_m_blocks) {
  int tb_k = c.thread_k, tb_n = c.thread_n, tb_m = thread_m_blocks * 16;
  int sh_block_meta_size = tb_m * 16;
  int sh_a_size = kMoeStages * (tb_m * tb_k) * 2;
#if AIKIDO_WARP_STAGE
  if (thread_m_blocks == 1) sh_a_size = std::max(sh_a_size, kMoeStages * 8 * tb_k * 2 * (tb_n / 64));  // A per n-warp
#endif
  int sh_b_size = kMoeStages * (tb_k * tb_n / 8) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);
  int sh_s_size = tb_n * 2 * kMoeStages;
  return tmp_size + sh_a_size + sh_s_size + sh_block_meta_size;
}

static bool valid_cfg(const ThreadCfg& c, int thread_m_blocks, int n, int k, int max_shared_mem, int shard_end,
                      bool need_had, int extra_smem = 0) {
  if (k % c.thread_k != 0 || n % c.thread_n != 0) return false;
  if (shard_end > 0 && shard_end % c.thread_n != 0) return false;  // a column slice must not straddle gate / up
  if (need_had && c.thread_n % 128 != 0) return false;             // in-launch output transform: whole 128-blocks
  return kernel_cache_size(c, thread_m_blocks) + extra_smem <= max_shared_mem;
}

static constexpr int kMaxBlocksPerSm = 4;
static int g_blocks_per_sm = -1;  // knob (moe_set_blocks_per_sm): -1 = automatic
// Automatic = Marlin MoE's determine_exec_config ("the config that allows the most co-resident blocks per SM wins"),
// EXCEPT for moe blocks of more than 16 rows, where co-resident blocks are capped at g_large_block_bps (default 1):
// measured on H200 at 64-row blocks (research/12 section 7.2) gate+up 1.57 -> 1.43 ms and down 1.27 -> 1.14 ms at 8k
// tokens with 1 block per SM and the 64 x 256 thread config; at 8-row blocks Marlin's choice is already the best.
static int g_large_block_bps = 1;  // knob (moe_set_large_block_bps): 0 = no cap
#ifndef AIKIDO_MOE_OCC_MIN_ROWS
#define AIKIDO_MOE_OCC_MIN_ROWS 64   // 8-row launches with at least this many rows (slots) use moe family 5 (128-register
#endif                               // 4-blocks-per-SM build) for the 128-thread configs; fewer rows: family 0
static int g_occ_min_rows = AIKIDO_MOE_OCC_MIN_ROWS;  // knob (moe_set_occ_min_rows); 0 = always family 5, huge = never
#ifndef AIKIDO_MOE_WIDE_DOWN_ROWS
#define AIKIDO_MOE_WIDE_DOWN_ROWS 32   // 8-row DOWN launches (no shard map) with >= this many rows use the 256-thread
#endif                                 // 64 x 256 config (2 blocks per SM): half the column slices, so half the per-slice
#ifndef AIKIDO_MOE_WIDE_GU_ROWS        // epilogues / pipeline refills of the 8-stage k = 512 slices. 0 = off
#define AIKIDO_MOE_WIDE_GU_ROWS 32      // same for gate+up (0 = off)
#endif
static int g_wide_down_rows = AIKIDO_MOE_WIDE_DOWN_ROWS, g_wide_gu_rows = AIKIDO_MOE_WIDE_GU_ROWS;
static int g_grid_limit = 0;       // knob (moe_set_grid_limit): > 0 caps the grid (blocks) below sms * bps: longer stripes, less per-slice overhead

struct DevState {
  int sms = 0;
  int max_shared_mem = 0;
  // Two scratch sets (locks + fp32 reduce buffer) so that two grouped GEMMs (the K=3 and K=4 classes of a mixed pack)
  // can run concurrently on two streams: `scratch` picks the set.
  at::Tensor locks[2];  // int32 [sms * 4], zero
  at::Tensor c_tmp[2];  // float: fp32 reduce scratch, Marlin MoE's upper bound (sms * 4 blocks x 64 rows x 256 cols, x2)
  at::Tensor lut_mcg;  // half [65536]: decode_3inst<1>(w) for every window (cb 13, shared-memory table decode)
  at::Tensor glu_counters[2];  // int32 [kGluCounters], zero between launches: fused GLU arrival counters (glue step 3)
};
static constexpr int64_t kGluCounters = 1 << 18;   // moe blocks x (inter / 128) the fused GLU epilogue can track

// lut[w] = the MCG codebook value of window w, by the codebook arithmetic itself (bit-exact by construction)
__global__ void aikido_build_mcg_lut(half* __restrict__ out) {
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < 65536u) out[w] = decode_3inst<1>(w);
}
static constexpr int kLutBytes = 65536 * 2;

static DevState& dev_state(int device) {
  static std::map<int, DevState> states;
  auto it = states.find(device);
  if (it != states.end()) return it->second;
  DevState st;
  cudaDeviceGetAttribute(&st.sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&st.max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  TORCH_CHECK(st.sms > 0 && st.max_shared_mem > 0, "aikido_exl3 moe: device query failed");
  auto dev = at::Device(at::kCUDA, device);
  for (int i = 0; i < 2; i++) {
    st.locks[i] = at::zeros({st.sms * kMaxBlocksPerSm}, at::TensorOptions().dtype(at::kInt).device(dev));
    st.c_tmp[i] = at::empty({(int64_t)st.sms * kMaxBlocksPerSm * 64 * 256 * 2}, at::TensorOptions().dtype(at::kFloat).device(dev));
    st.glu_counters[i] = at::zeros({kGluCounters}, at::TensorOptions().dtype(at::kInt).device(dev));
  }
  st.lut_mcg = at::empty({65536}, at::TensorOptions().dtype(at::kHalf).device(dev));
  aikido_build_mcg_lut<<<64, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>((half*)st.lut_mcg.data_ptr());
  states[device] = st;
  return states[device];
}

// One grouped launch. a: fp16 [shards * rows, k] (slab s = rows s*rows ..); b: [E, k/16, n/64, 32, 4]; c: fp16
// [rows, n]; sorted ids index ROWS (slots), i.e. Marlin MoE with top_k = 1. svh: fp16 [E, n] or nullptr.
static void gemm_launch(const half* a_ptr, const int* b_ptr, half* c_ptr, const half* svh, const int* sorted_ids,
                        const int* expert_ids, const int* num_post_padded, int moe_block_size, int rows, int n, int k,
                        int shard_end, int cb, int bits, int device, cudaStream_t stream, int force_thread_k,
                        int force_thread_n, int scratch, const half* glu_suh = nullptr, half* glu_xd = nullptr) {
  DevState& st = dev_state(device);
  const int thread_m_blocks = (moe_block_size + 15) / 16;
  const bool m8 = moe_block_size == 8;
  const bool occ5 = rows >= g_occ_min_rows;
  const int sms = st.sms;
  int max_shared_mem = st.max_shared_mem;

  ThreadCfg cfg{-1, -1, -1};
  int bps = 1;
  const int extra_smem = cb == AIKIDO_CB_MCG_LUT ? kLutBytes : 0;   // the table lives in dynamic shared memory
  if (force_thread_k > 0 && force_thread_n > 0) {
    cfg = ThreadCfg{force_thread_k, force_thread_n, force_thread_k * force_thread_n / 64};
    TORCH_CHECK(valid_cfg(cfg, thread_m_blocks, n, k, max_shared_mem - 512, shard_end, svh != nullptr, extra_smem),
                "aikido_exl3 moe: bad forced thread config");
    if (g_blocks_per_sm > 0) bps = g_blocks_per_sm;
  } else {
    // Marlin MoE's determine_exec_config: the valid config that allows the most co-resident blocks per SM wins.
    const ThreadCfg* cfgs = thread_m_blocks > 1 ? large_cfgs : small_cfgs;
#ifndef AIKIDO_MOE_REGFILE_MIN_ROWS
#define AIKIDO_MOE_REGFILE_MIN_ROWS 0     // 64 (a 4th block only from 8 tokens) measured a mixed-path pathology at 16 tokens (2395 us): keep 0
#endif
#ifndef AIKIDO_MOE_REGFILE_BYTES
#define AIKIDO_MOE_REGFILE_BYTES (255 * 1024)   // Marlin's bound; 262144 = the real 64K x 4 B register file (lets a
#endif                                          // 128-register 128-thread kernel run 4 blocks per SM)
    // AIKIDO_MOE_REGFILE_BYTES = 262144 lets the 128-register 128-thread kernel run 4 blocks per SM
    const int device_max_reg_size = rows >= AIKIDO_MOE_REGFILE_MIN_ROWS ? AIKIDO_MOE_REGFILE_BYTES : 255 * 1024;
    int count = 0;
    for (int i = 0; i < 3; i++) {
      if (!valid_cfg(cfgs[i], thread_m_blocks, n, k, max_shared_mem - 512, shard_end, svh != nullptr, extra_smem)) continue;
      MoeFuncPtr kern = select_kernel(cfgs[i].num_threads, cfgs[i].thread_n, cfgs[i].thread_k, m8 ? ((cfgs[i].num_threads == 128 && occ5) ? 5 : 0) : thread_m_blocks, cb, bits);
      if (kern == nullptr) continue;
      int cache_size = kernel_cache_size(cfgs[i], thread_m_blocks) + extra_smem;
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, kern);
      int reg_size = std::max((int)attr.numRegs, 1) * cfgs[i].num_threads * 4;
      int allow = std::min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1536));
      allow = std::max(std::min(allow, thread_m_blocks == 1 ? 4 : 2), 1);
      if (n / cfgs[i].thread_n * rows * 4 < sms * allow) allow = std::max(n / cfgs[i].thread_n * rows * 4 / sms, 1);
      if (g_blocks_per_sm > 0) allow = std::min(allow, g_blocks_per_sm);
      else if (thread_m_blocks > 1 && g_large_block_bps > 0) allow = std::min(allow, g_large_block_bps);
      if (allow > count) {
        count = allow;
        cfg = cfgs[i];
        bps = allow;
      }
    }
  }
  {  // AIKIDO wide config for the 8-row family (auto mode only)
    const int wide_rows = shard_end > 0 ? g_wide_gu_rows : g_wide_down_rows;
    const ThreadCfg wide{64, 256, 256};
    if (m8 && force_thread_k <= 0 && g_blocks_per_sm <= 0 && wide_rows > 0 && rows >= wide_rows &&
        valid_cfg(wide, thread_m_blocks, n, k, max_shared_mem - 512, shard_end, svh != nullptr, extra_smem)) {
      MoeFuncPtr kern = select_kernel(256, 256, 64, 0, cb, bits);
      if (kern != nullptr) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, kern);
        const int reg_size = std::max((int)attr.numRegs, 1) * 256 * 4;
        const int cache_size = kernel_cache_size(wide, thread_m_blocks) + extra_smem;
        const int allow = std::max(std::min({AIKIDO_MOE_REGFILE_BYTES / reg_size, max_shared_mem / (cache_size + 1536), 2}), 1);
        cfg = wide; bps = allow;
      }
    }
  }
  TORCH_CHECK(cfg.thread_k != -1, "aikido_exl3 moe: no thread config for block=", moe_block_size, " k=", k, " n=", n);
  MoeFuncPtr kernel = select_kernel(cfg.num_threads, cfg.thread_n, cfg.thread_k, m8 ? ((cfg.num_threads == 128 && occ5) ? 5 : 0) : thread_m_blocks, cb, bits);
  TORCH_CHECK(kernel != nullptr, "aikido_exl3 moe: kernel not compiled: threads=", cfg.num_threads, " thread_n=",
              cfg.thread_n, " thread_k=", cfg.thread_k, " moe_block=", moe_block_size, " cb=", cb, " K=", bits);
  TORCH_CHECK(bps >= 1 && bps <= kMaxBlocksPerSm);
  if (bps > 1) max_shared_mem = max_shared_mem / bps - 1024;
  TORCH_CHECK(kernel_cache_size(cfg, thread_m_blocks) + extra_smem <= max_shared_mem, "aikido_exl3 moe: shared memory too small");
  // AIKIDO: launch with what the kernel needs (Marlin passes the whole opt-in maximum, which pins one block per SM even
  // when registers would allow two): another launch's blocks (the other K class on a second stream, or a second
  // block of this one under AIKIDO_MOE_MINBLOCKS_M8=2) can then co-reside.
  const int dyn_smem = std::min(max_shared_mem, ((kernel_cache_size(cfg, thread_m_blocks) + extra_smem + 4096 + 1023) / 1024) * 1024);
  const int4* lut_ptr = cb == AIKIDO_CB_MCG_LUT ? (const int4*)st.lut_mcg.data_ptr() : nullptr;

  const int a_shard_stride = shard_end > 0 ? rows * (k / 8) : 0;  // int4 units (16 bytes) per input slab
#ifndef AIKIDO_PDL_GEMM
#define AIKIDO_PDL_GEMM 0
#endif
#if AIKIDO_PDL_GEMM
#error "GEMM-side PDL is not in this build: the MoE template has no griddepcontrol.wait, so its launch must never carry the attribute"
#endif
  // AIKIDO_PDL_GEMM 0: the GEMM has no griddepcontrol code, so its launch never carries the attribute
  const bool pdl = AIKIDO_PDL_GEMM && (g_pdl & (shard_end > 0 ? kPdlGemm13 : kPdlGemm2)) != 0;
  // out_flags: bit 0 output transform (svh), bit 2 fused GLU + down-input transform (glue step 3); the PDL bits of
  // wf/pdl (4, 8, 16, 32) are not used: GEMM-side PDL is not compiled in (see #error above)
  const int out_flags = (svh ? 1 : 0) | (glu_xd ? 4 : 0);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
  // clang-format off
  const int grid = g_grid_limit > 0 ? std::min(sms * bps, g_grid_limit) : sms * bps;
  aikido_launch(pdl, kernel, dim3(grid), dim3(cfg.num_threads), (size_t)dyn_smem, stream,
      (const int4*)a_ptr, (const int4*)b_ptr, (int4*)c_ptr, (int4*)st.c_tmp[scratch].data_ptr(), (const int4*)svh,
      nullptr, lut_ptr, nullptr, nullptr, nullptr,
      sorted_ids, expert_ids, num_post_padded, nullptr, /*top_k=*/1, /*mul_topk_weights=*/false,
      /*num_groups=*/-1, rows, n, k, (int*)st.locks[scratch].data_ptr(), /*has_bias=*/false, /*use_atomic_add=*/false,
      /*use_fp32_reduce=*/true, a_shard_stride, shard_end > 0 ? shard_end : INT_MAX, INT_MAX, INT_MAX, out_flags,
      glu_suh, glu_xd, (int*)st.glu_counters[scratch].data_ptr());
  // clang-format on
}

}  // namespace MARLIN_NAMESPACE_NAME

// ---------------------------------------------------------------------------------------------------------------
// Per-slot launches. Block (x, y) handles 128-block y of row x; 32 threads = the 32 lanes of the warp.

static constexpr float kHadScale = 0.088388347648f;  // 1/sqrt(128)
#define AIKIDO_COMBINE_MAX_K 8      // moe_combine: top_k up to this many slots are loaded before the (slot-ordered) sum
#define AIKIDO_ROUTE_MAX_E 256      // fused had_in + routing: experts per layer the one-warp builder handles
#define AIKIDO_ROUTE_MAX_SLOTS 4096

// xh[s * slots + slot] = Had128(fp16(x[slot / top_k]) * suh[ids[slot]][s]); grid (shards * slots, k / 128).
template <bool in_bf16, bool ids64>
__global__ __launch_bounds__(32)
void aikido_moe_had_in_kernel(const half* __restrict__ x, half* __restrict__ xh, const half* __restrict__ suh,
                              const void* __restrict__ ids, const int slots, const int top_k, const int shards,
                              const int num_experts, const int pdl) {
  // AIKIDO PDL (pdl: bit 0 wait for align, bit 1 trigger, bit 2 = nothing here depends on align: trigger now, work
  // concurrently with align, wait at the end; the gate+up GEMM then waits at its top)
  if (pdl & 4) aikido_pdl_trig(pdl);   // early mode: optional (all / tail) trigger, wait at the end
  else aikido_pdl_slot(pdl);
  const size_t width = (size_t)gridDim.y * 128;
  const int slot = blockIdx.x % slots, shard = blockIdx.x / slots;
  int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
  // An id outside [0, E) (padding / dummy tokens, expert-parallel sentinels) is dropped by moe_align_block_size, so the
  // GEMM never reads this slot and moe_combine skips it; clamp so the scale lookup stays inside suh.
  if (e < 0 || e >= num_experts) e = 0;
  const half* in = x + width * (slot / top_k) + blockIdx.y * 128;
  half* out = xh + width * blockIdx.x + blockIdx.y * 128;
  aikido_had_r_128_inner<in_bf16, false, true, false>(in, out, suh + width * (e * shards + shard), blockIdx.y * 32,
                                                      kHadScale);
  if (pdl & 4) aikido_pdl_wait();   // grid completion must imply align's completion
}

// AIKIDO glue step 2: decode routing built by ONE warp, as extra blocks of the moe_had_in launch (no separate align
// launch). Block c (c < num_classes) produces class c's (sorted_token_ids, expert_ids, num_tokens_post_padded) exactly
// as aikido_moe_align_decode_kernel / vLLM's moe_align_block_size do (padding slot = numel, padding block = -1, local
// ids, per-expert counts padded to the class block size, experts ascending); the slots of an expert are placed in
// ascending slot order (deterministic; the GEMM output does not depend on the order inside an expert's range).
struct RouteClass {
  const int* map; int block; int* sorted; int cap; int* eids; int mblk; int* post;
};

template <bool ids64>
__device__ __forceinline__ void aikido_route_warp(const void* __restrict__ ids, const int slots, const int num_experts,
                                                  const RouteClass rc, int* count, int* start, int* fill) {
  const int lane = threadIdx.x & 31;
  const unsigned full = 0xffffffffu;
  auto id_of = [&](int s) -> int {
    return ids64 ? (int)((const int64_t*)ids)[s] : ((const int32_t*)ids)[s];
  };
  for (int e = lane; e < num_experts; e += 32) count[e] = 0;
  __syncwarp();
  for (int s = lane; s < slots; s += 32) {
    const int e = id_of(s);
    if (e >= 0 && e < num_experts) atomicAdd(&count[e], 1);
  }
  __syncwarp();
  // lane owns experts lane * EPL .. +EPL-1 (ascending global order across lanes); exclusive scan of the padded counts
  constexpr int EPL = AIKIDO_ROUTE_MAX_E / 32;
  int padded[EPL];
  int local = 0;
#pragma unroll
  for (int i = 0; i < EPL; i++) {
    const int e = lane * EPL + i;
    padded[i] = (e < num_experts && rc.map[e] >= 0) ? ((count[e] + rc.block - 1) / rc.block) * rc.block : 0;
    local += padded[i];
  }
  int incl = local;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int n = __shfl_up_sync(full, incl, off);
    if (lane >= off) incl += n;
  }
  const int total = __shfl_sync(full, incl, 31);
  int excl = incl - local;
#pragma unroll
  for (int i = 0; i < EPL; i++) {
    const int e = lane * EPL + i;
    if (e < num_experts) { start[e] = excl; fill[e] = 0; }
    excl += padded[i];
  }
  // expert ids per block (local ids), padding blocks -1
#pragma unroll
  for (int i = 0; i < EPL; i++) {
    const int e = lane * EPL + i;
    if (padded[i] > 0) {
      const int s0 = start[e], m = rc.map[e];
      for (int p = s0; p < s0 + padded[i]; p += rc.block) rc.eids[p / rc.block] = m;
    }
  }
  for (int i = (total + rc.block - 1) / rc.block + lane; i < rc.mblk; i += 32) rc.eids[i] = -1;
  // every sorted entry = the padding sentinel first (cap is a multiple of 4 and the tensor is 16-byte aligned)
  {
    int4 pad4; pad4.x = pad4.y = pad4.z = pad4.w = slots;
    int4* s4 = (int4*)rc.sorted;
    for (int i = lane; i < rc.cap / 4; i += 32) s4[i] = pad4;
  }
  __syncwarp();
  // then this class's slots into their expert's range, ascending slot order inside an expert
  for (int base = 0; base < slots; base += 32) {
    const int s = base + lane;
    int e = s < slots ? id_of(s) : -1;
    const bool ok = e >= 0 && e < num_experts && rc.map[e] >= 0;
    const unsigned peers = __match_any_sync(full, ok ? e : -1 - lane);
    const int rank = __popc(peers & ((1u << lane) - 1u));
    if (ok) rc.sorted[start[e] + fill[e] + rank] = s;
    __syncwarp();
    if (ok && rank == 0) fill[e] += __popc(peers);
    __syncwarp();
  }
  if (lane == 0) *rc.post = total;
}

// moe_had_in (above) + the routing tables of every K class in the same launch: grid (shards * slots + num_classes,
// k / 128); the last num_classes x-blocks build the tables (blockIdx.y == 0 only), the others are the input transform.
template <bool in_bf16, bool ids64>
__global__ __launch_bounds__(32)
void aikido_moe_had_in_route_kernel(const half* __restrict__ x, half* __restrict__ xh, const half* __restrict__ suh,
                                    const void* __restrict__ ids, const int slots, const int top_k, const int shards,
                                    const int num_experts, const RouteClass rc0, const RouteClass rc1) {
  __shared__ int count[AIKIDO_ROUTE_MAX_E], start[AIKIDO_ROUTE_MAX_E], fill[AIKIDO_ROUTE_MAX_E];
  const int width = (int)gridDim.y * 128;
  const int items = shards * slots;
  if ((int)blockIdx.x >= items) {
    if (blockIdx.y != 0) return;
    const int c = (int)blockIdx.x - items;
    aikido_route_warp<ids64>(ids, slots, num_experts, c == 0 ? rc0 : rc1, count, start, fill);
    return;
  }
  const int slot = blockIdx.x % slots, shard = blockIdx.x / slots;
  int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
  if (e < 0 || e >= num_experts) e = 0;
  const half* in = x + (size_t)width * (slot / top_k) + blockIdx.y * 128;
  half* out = xh + (size_t)width * blockIdx.x + blockIdx.y * 128;
  aikido_had_r_128_inner<in_bf16, false, true, false>(in, out, suh + (size_t)width * (e * shards + shard), blockIdx.y * 32,
                                                      kHadScale);
}

// ExLlamaV3's fp32 SiLU (exl3_moe_coop_kernel.cuh act_silu)
__device__ __forceinline__ float aikido_act_silu(float x) { return aikido_glu_silu(x); }  // exl3_had.cuh

// xd[slot] = Had128(fp16_rn(silu(g) * u) * suh_down[ids[slot]]); gu = [slots, 2 * inter] (gate | up), fp16.
// silu and the product are fp32, rounded to fp16 once (round to nearest even); from there it is exactly the input
// transform of a dense EXL3 linear, so moe_glu_had_in(gu) == had_in(fp16_rn(silu32(g) * u32)) bit for bit.
// grid (slots, inter / 128).
template <bool ids64>
__global__ __launch_bounds__(32)
void aikido_moe_glu_had_in_kernel(const half* __restrict__ gu, half* __restrict__ xd, half* __restrict__ act_tmp,
                                  const half* __restrict__ suh, const void* __restrict__ ids, const int num_experts,
                                  const int pdl) {
  const size_t inter = (size_t)gridDim.y * 128;
  const int slot = blockIdx.x;
  const int t = threadIdx.x & 31;
  int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
  if (e < 0 || e >= num_experts) e = 0;  // dropped slot (see moe_had_in): its gu row was never written, result unused
  aikido_pdl_slot(pdl);   // AIKIDO PDL: gu is the gate+up GEMM's output
  const half4 g = ((const half4*)(gu + 2 * inter * slot + blockIdx.y * 128))[t];
  const half4 u = ((const half4*)(gu + 2 * inter * slot + inter + blockIdx.y * 128))[t];
  float2 g0 = __half22float2(g.x), g1 = __half22float2(g.y), u0 = __half22float2(u.x), u1 = __half22float2(u.y);
  half4 a;
  a.x = __floats2half2_rn(aikido_act_silu(g0.x) * u0.x, aikido_act_silu(g0.y) * u0.y);
  a.y = __floats2half2_rn(aikido_act_silu(g1.x) * u1.x, aikido_act_silu(g1.y) * u1.y);
  // stage the activation in the caller's scratch (its own lane slot: no conflict), then the standard transform
  half* act = act_tmp + inter * slot + blockIdx.y * 128;
  ((half4*)act)[t] = a;
  __syncthreads();
  aikido_had_r_128_inner<false, false, true, false>(act, xd + inter * slot + blockIdx.y * 128, suh + inter * e,
                                                    blockIdx.y * 32, kHadScale);
}

// y[t] = sum_j w[t, j] * yd[t * top_k + j], fp32 accumulation in slot order, one rounding to fp16 / bf16.
// grid (tokens, hidden / 128).
template <bool out_bf16, bool ids64>
__global__ __launch_bounds__(32)
void aikido_moe_combine_kernel(const half* __restrict__ yd, const float* __restrict__ w, half* __restrict__ y,
                               const void* __restrict__ ids, const int top_k, const int num_experts, const int pdl) {
  const size_t hidden = (size_t)gridDim.y * 128;
  const int tok = blockIdx.x;
  const int t = threadIdx.x & 31;
  aikido_pdl_slot(pdl);   // AIKIDO PDL: yd is the down GEMM's output
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  constexpr int kMaxK = AIKIDO_COMBINE_MAX_K;
  if (top_k <= kMaxK) {
    // AIKIDO glue step 1: issue every id load, then every weight + row load (predicated on the id), then the SAME
    // fp32 sum in slot order as the loop below: two dependent memory round trips instead of top_k.
    int64_t e[kMaxK];
    float wj[kMaxK];
    half4 v[kMaxK];
    bool ok[kMaxK];
#pragma unroll
    for (int j = 0; j < kMaxK; j++) {
      const size_t slot = (size_t)tok * top_k + j;
      e[j] = j < top_k ? (ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot]) : (int64_t)-1;
    }
#pragma unroll
    for (int j = 0; j < kMaxK; j++) {
      const size_t slot = (size_t)tok * top_k + j;
      ok[j] = e[j] >= 0 && e[j] < num_experts;   // dropped slot: yd row never written, never read
      wj[j] = ok[j] ? w[slot] : 0.0f;
      half4 z; z.x = __float2half2_rn(0.0f); z.y = z.x;
      v[j] = ok[j] ? ((const half4*)(yd + hidden * slot + blockIdx.y * 128))[t] : z;
    }
#pragma unroll
    for (int j = 0; j < kMaxK; j++) {
      if (!ok[j]) continue;
      float2 lo = __half22float2(v[j].x), hi = __half22float2(v[j].y);
      s0 += wj[j] * lo.x; s1 += wj[j] * lo.y; s2 += wj[j] * hi.x; s3 += wj[j] * hi.y;
    }
  } else {
    for (int j = 0; j < top_k; j++) {
      const size_t slot = (size_t)tok * top_k + j;
      const int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
      if (e < 0 || e >= num_experts) continue;  // dropped slot: yd row never written
      const half4 v = ((const half4*)(yd + hidden * slot + blockIdx.y * 128))[t];
      const float wj = w[(size_t)tok * top_k + j];
      float2 lo = __half22float2(v.x), hi = __half22float2(v.y);
      s0 += wj * lo.x; s1 += wj * lo.y; s2 += wj * hi.x; s3 += wj * hi.y;
    }
  }
  half* out = y + hidden * tok + blockIdx.y * 128;
  if constexpr (out_bf16) {
    float2 f0, f1;
    f0.x = s0; f0.y = s1; f1.x = s2; f1.y = s3;
    __nv_bfloat162* wr = ((__nv_bfloat162*)out) + 2 * t;
    wr[0] = __float22bfloat162_rn(f0);
    wr[1] = __float22bfloat162_rn(f1);
  } else {
    half4 o;
    o.x = __floats2half2_rn(s0, s1);
    o.y = __floats2half2_rn(s2, s3);
    ((half4*)out)[t] = o;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Grid-strided variants (prefill): instead of one 32-thread block per (row, 128-block) item - 4.2 M blocks per layer at
// 8k tokens - a FIXED grid of 256-thread blocks (8 warps each) strides over the items, as the dense template's
// cooperative prologue does. Warp g = blockIdx.x * 8 + threadIdx.x / 32 handles items g, g + G, g + 2G, ... (G = warps
// in the grid); all 32 lanes of a warp run the same trip count, so the warp shuffles inside the Hadamard stay in
// lockstep. Same statements per item as the kernels above => bit-identical results (checked by the parity tools).
// g_gs_blocks (knob moe_set_gridstride_blocks): -1 = automatic, 0 = always per-item launches (above), n > 0 = always n
// grid blocks. Automatic (measured on H200, research/12 section 7.2: had_in 1.27 -> 0.60 ms at 8k tokens with 8,448
// blocks, same at 16 tokens, worse with too few blocks): grid-strided with up to 8,448 blocks once there are at least
// 16,384 items (64 tokens here), per-item launches below that (decode sizes: no measurable difference).
static int g_gs_blocks = -1;
static constexpr int64_t kGsAutoMinItems = 16384;  // 64 tokens x top-8 x 2 slabs x 16 blocks: 0.046 -> 0.027 ms measured
static constexpr int kGsAutoBlocks = 8448;

static int gs_blocks_for(int64_t items) {
  if (g_gs_blocks >= 0) return g_gs_blocks;
  return items >= kGsAutoMinItems ? kGsAutoBlocks : 0;
}

template <bool in_bf16, bool ids64>
__global__ __launch_bounds__(256)
void aikido_moe_had_in_gs_kernel(const half* __restrict__ x, half* __restrict__ xh, const half* __restrict__ suh,
                                 const void* __restrict__ ids, const int slots, const int top_k, const int shards,
                                 const int num_experts, const int kb, const int pdl) {
  if (pdl & 4) aikido_pdl_trig(pdl);   // early mode: optional (all / tail) trigger, wait at the end
  else aikido_pdl_slot(pdl);
  const size_t width = (size_t)kb * 128;
  const int64_t total = (int64_t)shards * slots * kb, stride = (int64_t)gridDim.x * 8;
  for (int64_t w = (int64_t)blockIdx.x * 8 + threadIdx.x / 32; w < total; w += stride) {
    const int blk = (int)(w % kb);
    const int64_t rs = w / kb;
    const int slot = (int)(rs % slots), shard = (int)(rs / slots);
    int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
    if (e < 0 || e >= num_experts) e = 0;
    aikido_had_r_128_inner<in_bf16, false, true, false>(x + width * (slot / top_k) + blk * 128, xh + width * rs + blk * 128,
                                                        suh + width * (e * shards + shard), blk * 32, kHadScale);
  }
  if (pdl & 4) aikido_pdl_wait();
}

template <bool ids64>
__global__ __launch_bounds__(256)
void aikido_moe_glu_had_in_gs_kernel(const half* __restrict__ gu, half* __restrict__ xd, half* __restrict__ act_tmp,
                                     const half* __restrict__ suh, const void* __restrict__ ids, const int num_experts,
                                     const int slots, const int kb, const int pdl) {
  aikido_pdl_slot(pdl);
  const size_t inter = (size_t)kb * 128;
  const int64_t total = (int64_t)slots * kb, stride = (int64_t)gridDim.x * 8;
  const int t = threadIdx.x & 31;
  for (int64_t w = (int64_t)blockIdx.x * 8 + threadIdx.x / 32; w < total; w += stride) {
    const int blk = (int)(w % kb);
    const int64_t slot = w / kb;
    int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
    if (e < 0 || e >= num_experts) e = 0;
    const half4 g = ((const half4*)(gu + 2 * inter * slot + blk * 128))[t];
    const half4 u = ((const half4*)(gu + 2 * inter * slot + inter + blk * 128))[t];
    float2 g0 = __half22float2(g.x), g1 = __half22float2(g.y), u0 = __half22float2(u.x), u1 = __half22float2(u.y);
    half4 a;
    a.x = __floats2half2_rn(aikido_act_silu(g0.x) * u0.x, aikido_act_silu(g0.y) * u0.y);
    a.y = __floats2half2_rn(aikido_act_silu(g1.x) * u1.x, aikido_act_silu(g1.y) * u1.y);
    half* act = act_tmp + inter * slot + blk * 128;
    ((half4*)act)[t] = a;
    // No __syncthreads() here: it would synchronise the whole 256-thread block, and the 8 warps of a block have
    // different trip counts at the tail (hang). Each lane reads back only the vector it has just written itself.
    aikido_had_r_128_inner<false, false, true, false>(act, xd + inter * slot + blk * 128, suh + inter * e, blk * 32, kHadScale);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Row-per-warp variants (prefill, wf/prefill). One warp per slot (moe_had_in) / per token (moe_combine), 32-bit index
// math once per row instead of 64-bit divisions per item. moe_had_in: L lanes per 128-block, EPL = 128 / L elements per
// lane, G = 32 / L blocks per warp step, both shards from one x load. The butterfly is the SAME sequence of fp32
// operations as had_hf_r_128_inner - stage on element bit 0, then bit 1, ..., bit 6; lo' = lo + hi, hi' = (-hi) + lo
// via the same sign-bit xor - only the first log2(EPL) stages run inside the lane and the last log2(L) across lanes (fewer
// shuffles). Pre-scale (__hmul2), bf16 -> fp16, * 1/sqrt(128) and the fp16 rounding are the same statements, so every
// output element is bit-identical (parity: hopper_moe_parity check_transforms, vs the per-item kernels).
static int g_had_lanes = 8;  // knob (moe_set_had_lanes): 0 = grid-strided per-item kernel above; 2 / 4 / 8 lanes per block

template <int EPL>
__device__ __forceinline__ void aikido_had_lane_stages(float (&v)[EPL]) {
#pragma unroll
  for (int b = 1; b < EPL; b <<= 1) {
#pragma unroll
    for (int i = 0; i < EPL; i++) {
      if (!(i & b)) {
        const float lo = v[i], hi = v[i + b];
        v[i] = lo + hi;
        v[i + b] = lo - hi;
      }
    }
  }
}

template <int EPL, int L>
__device__ __forceinline__ void aikido_had_shfl_stages(float (&v)[EPL], const int q) {
#pragma unroll
  for (int m = 1; m < L; m <<= 1) {
    const uint32_t sfm = (q & m) ? 0x80000000u : 0u;
#pragma unroll
    for (int i = 0; i < EPL; i++) {
      const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
      v[i] = __uint_as_float(__float_as_uint(v[i]) ^ sfm) + p;
    }
  }
}

// U = warp steps whose loads are issued together (more bytes in flight per warp: at 345 MHz the kernel is latency-bound).
static int g_had_unroll = 2;  // knob (moe_set_had_unroll): 1..3

template <bool in_bf16, bool ids64, int L, int U>
__global__ __launch_bounds__(256)
void aikido_moe_had_in_rw_kernel(const half* __restrict__ x, half* __restrict__ xh, const half* __restrict__ suh,
                                 const void* __restrict__ ids, const int slots, const int top_k, const int shards,
                                 const int num_experts, const int kb) {
  constexpr int EPL = 128 / L, G = 32 / L, V = EPL / 8;
  const int lane = threadIdx.x & 31, sub = lane / L, q = lane % L;
  const size_t width = (size_t)kb * 128;
  for (int slot = blockIdx.x * 8 + threadIdx.x / 32; slot < slots; slot += gridDim.x * 8) {
    int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
    if (e < 0 || e >= num_experts) e = 0;
    const half* xrow = x + width * (slot / top_k);
    const half* srow = suh + width * (e * shards);
    for (int b0 = 0; b0 < kb; b0 += G * U) {
      uint4 xr[U][V], sr[U][2][V];
      bool ok[U];
      int col[U];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int blk = b0 + u * G + sub;
        ok[u] = blk < kb;
        col[u] = (ok[u] ? blk : kb - 1) * 128 + q * EPL;
#pragma unroll
        for (int i = 0; i < V; i++) xr[u][i] = ((const uint4*)(xrow + col[u]))[i];
#pragma unroll
        for (int s = 0; s < 2; s++)
          if (s < shards)
#pragma unroll
            for (int i = 0; i < V; i++) sr[u][s][i] = ((const uint4*)(srow + width * s + col[u]))[i];
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        half2 xv[EPL / 2];
#pragma unroll
        for (int i = 0; i < V; i++) {
          if constexpr (in_bf16) {
            const __nv_bfloat162* bb = (const __nv_bfloat162*)&xr[u][i];
#pragma unroll
            for (int j = 0; j < 4; j++) {
              const float2 f = __bfloat1622float2(bb[j]);
              xv[4 * i + j] = __floats2half2_rn(f.x, f.y);
            }
          } else {
            const half2* h = (const half2*)&xr[u][i];
#pragma unroll
            for (int j = 0; j < 4; j++) xv[4 * i + j] = h[j];
          }
        }
#pragma unroll
        for (int s = 0; s < 2; s++) {
          if (s >= shards) break;
          float v[EPL];
#pragma unroll
          for (int i = 0; i < V; i++) {
            const half2* h = (const half2*)&sr[u][s][i];
#pragma unroll
            for (int j = 0; j < 4; j++) {
              const half2 pv = __hmul2(xv[4 * i + j], h[j]);
              v[8 * i + 2 * j] = __half2float(__low2half(pv));
              v[8 * i + 2 * j + 1] = __half2float(__high2half(pv));
            }
          }
          aikido_had_lane_stages<EPL>(v);
          aikido_had_shfl_stages<EPL, L>(v, q);
          if (ok[u]) {
            uint4* out = (uint4*)(xh + width * ((size_t)s * slots + slot) + col[u]);
#pragma unroll
            for (int i = 0; i < V; i++) {
              uint4 r;
              half2* h = (half2*)&r;
#pragma unroll
              for (int j = 0; j < 4; j++) h[j] = __floats2half2_rn(v[8 * i + 2 * j] * kHadScale, v[8 * i + 2 * j + 1] * kHadScale);
              out[i] = r;
            }
          }
        }
      }
    }
  }
}

template <bool in_bf16, bool ids64, int L>
static void launch_had_in_rw_u(int gb, cudaStream_t stream, const half* x, half* xh, const half* suh, const void* ids,
                               int slots, int top_k, int shards, int ne, int kb) {
  if (g_had_unroll >= 3)      aikido_moe_had_in_rw_kernel<in_bf16, ids64, L, 3><<<gb, 256, 0, stream>>>(x, xh, suh, ids, slots, top_k, shards, ne, kb);
  else if (g_had_unroll == 2) aikido_moe_had_in_rw_kernel<in_bf16, ids64, L, 2><<<gb, 256, 0, stream>>>(x, xh, suh, ids, slots, top_k, shards, ne, kb);
  else                        aikido_moe_had_in_rw_kernel<in_bf16, ids64, L, 1><<<gb, 256, 0, stream>>>(x, xh, suh, ids, slots, top_k, shards, ne, kb);
}

// xd[slot] = Had128(fp16_rn(silu(g) * u) * suh_down[e]) row-per-warp (L lanes per 128-block, as moe_had_in above): the
// activation is the same fp32 statements rounded once to fp16 (also stored to act_tmp, as the per-item kernel does), then
// the same pre-scale + butterfly sequence => bit-identical to aikido_moe_glu_had_in_kernel.
static int g_glu_lanes = 8;  // knob (moe_set_glu_lanes): 0 = grid-strided per-item kernel; 2 / 4 / 8 lanes per block

template <bool ids64, int L>
__global__ __launch_bounds__(256)
void aikido_moe_glu_had_in_rw_kernel(const half* __restrict__ gu, half* __restrict__ xd, half* __restrict__ act_tmp,
                                     const half* __restrict__ suh, const void* __restrict__ ids, const int num_experts,
                                     const int slots, const int kb) {
  constexpr int EPL = 128 / L, G = 32 / L, V = EPL / 8;
  const int lane = threadIdx.x & 31, sub = lane / L, q = lane % L;
  const size_t inter = (size_t)kb * 128;
  for (int slot = blockIdx.x * 8 + threadIdx.x / 32; slot < slots; slot += gridDim.x * 8) {
    int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
    if (e < 0 || e >= num_experts) e = 0;
    for (int b0 = 0; b0 < kb; b0 += G) {
      const int blk = b0 + sub;
      const bool ok = blk < kb;
      const int col = (ok ? blk : kb - 1) * 128 + q * EPL;
      const uint4* gp = (const uint4*)(gu + 2 * inter * slot + col);
      const uint4* up = (const uint4*)(gu + 2 * inter * slot + inter + col);
      const uint4* sp = (const uint4*)(suh + inter * e + col);
      uint4* ap = (uint4*)(act_tmp + inter * slot + col);
      float v[EPL];
#pragma unroll
      for (int i = 0; i < V; i++) {
        const uint4 gr = gp[i], ur = up[i], sr = sp[i];
        const half2* g2 = (const half2*)&gr;
        const half2* u2 = (const half2*)&ur;
        const half2* s2 = (const half2*)&sr;
        uint4 ar;
        half2* a2 = (half2*)&ar;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const float2 gf = __half22float2(g2[j]), uf = __half22float2(u2[j]);
          a2[j] = __floats2half2_rn(aikido_act_silu(gf.x) * uf.x, aikido_act_silu(gf.y) * uf.y);
          const half2 pv = __hmul2(a2[j], s2[j]);
          v[8 * i + 2 * j] = __half2float(__low2half(pv));
          v[8 * i + 2 * j + 1] = __half2float(__high2half(pv));
        }
        if (ok) ap[i] = ar;
      }
      aikido_had_lane_stages<EPL>(v);
      aikido_had_shfl_stages<EPL, L>(v, q);
      if (ok) {
        uint4* out = (uint4*)(xd + inter * slot + col);
#pragma unroll
        for (int i = 0; i < V; i++) {
          uint4 r;
          half2* h = (half2*)&r;
#pragma unroll
          for (int j = 0; j < 4; j++) h[j] = __floats2half2_rn(v[8 * i + 2 * j] * kHadScale, v[8 * i + 2 * j + 1] * kHadScale);
          out[i] = r;
        }
      }
    }
  }
}

template <bool ids64>
static void launch_glu_rw(int lanes, int gb, cudaStream_t stream, const half* gu, half* xd, half* act, const half* suh,
                          const void* ids, int ne, int slots, int kb) {
  if (lanes == 2)      aikido_moe_glu_had_in_rw_kernel<ids64, 2><<<gb, 256, 0, stream>>>(gu, xd, act, suh, ids, ne, slots, kb);
  else if (lanes == 4) aikido_moe_glu_had_in_rw_kernel<ids64, 4><<<gb, 256, 0, stream>>>(gu, xd, act, suh, ids, ne, slots, kb);
  else                 aikido_moe_glu_had_in_rw_kernel<ids64, 8><<<gb, 256, 0, stream>>>(gu, xd, act, suh, ids, ne, slots, kb);
}

// Column-stationary variant (default for prefill): at prefill sizes the input transform is bound by L2 / memory traffic,
// and the row-per-warp kernel re-reads 2 x 12 KB of suh per slot (as much as it writes). Here a CTA owns ONE (shard,
// 128-column block) and a range of slots: it stages suh[:, shard, block] of every expert (E x 256 B, 64 KB at E = 256)
// in shared memory once, then each warp step transforms 8 consecutive slots (4 lanes per slot, EPL = 32: the L = 4
// arithmetic of the row-per-warp kernel above, statement for statement). Traffic ~ the xh write + x once per shard.
static int g_had_cs = 0;  // knob (moe_set_had_cs): 1 = column-stationary kernel when E * 256 B fits, 0 = off (integ: off,
                          // the row-per-warp L = 8 kernel measured faster at 8k tokens, wf/prefill r4 / r5)
static constexpr int kHadCsMaxExperts = 384;

template <bool in_bf16, bool ids64, int L, int THREADS>
__global__ __launch_bounds__(THREADS)
void aikido_moe_had_in_cs_kernel(const half* __restrict__ x, half* __restrict__ xh, const half* __restrict__ suh,
                                 const void* __restrict__ ids, const int slots, const int top_k, const int shards,
                                 const int num_experts, const int kb, const int chunk_slots) {
  constexpr int EPL = 128 / L, G = 32 / L, V = EPL / 8, WARPS = THREADS / 32;
  extern __shared__ uint4 sh_suh[];  // [E][128] halves of this (shard, block)
  const int lane = threadIdx.x & 31, sub = lane / L, q = lane % L;
  const size_t width = (size_t)kb * 128;
  const int cols = shards * kb;
  const int sb = blockIdx.x % cols, chunk = blockIdx.x / cols;
  const int shard = sb / kb, blk = sb % kb;
  for (int i = threadIdx.x; i < num_experts * 16; i += THREADS) {
    const int ee = i >> 4, c = i & 15;
    sh_suh[i] = ((const uint4*)(suh + width * ((size_t)ee * shards + shard) + blk * 128))[c];
  }
  __syncthreads();
  const int s0 = chunk * chunk_slots, s1 = min(slots, s0 + chunk_slots);
  // consecutive slots of a warp step share their token (slot / top_k): the x loads of its lane groups coalesce
  for (int base = s0 + (threadIdx.x / 32) * G; base < s1; base += WARPS * G) {
    const int slot = base + sub;
    const bool ok = slot < s1;
    const int sl = ok ? slot : s1 - 1;
    int e = (int)(ids64 ? ((const int64_t*)ids)[sl] : (int64_t)((const int32_t*)ids)[sl]);
    if (e < 0 || e >= num_experts) e = 0;
    const uint4* xp = (const uint4*)(x + width * (sl / top_k) + blk * 128 + q * EPL);
    const uint4* sp = sh_suh + e * 16 + q * V;
    float v[EPL];
#pragma unroll
    for (int i = 0; i < V; i++) {
      const uint4 r = xp[i];
      half2 xv[4];
      if constexpr (in_bf16) {
        const __nv_bfloat162* bb = (const __nv_bfloat162*)&r;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const float2 f = __bfloat1622float2(bb[j]);
          xv[j] = __floats2half2_rn(f.x, f.y);
        }
      } else {
        const half2* h = (const half2*)&r;
#pragma unroll
        for (int j = 0; j < 4; j++) xv[j] = h[j];
      }
      const uint4 sr = sp[i];
      const half2* h = (const half2*)&sr;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const half2 pv = __hmul2(xv[j], h[j]);
        v[8 * i + 2 * j] = __half2float(__low2half(pv));
        v[8 * i + 2 * j + 1] = __half2float(__high2half(pv));
      }
    }
    aikido_had_lane_stages<EPL>(v);
    aikido_had_shfl_stages<EPL, L>(v, q);
    if (ok) {
      uint4* out = (uint4*)(xh + width * ((size_t)shard * slots + slot) + blk * 128 + q * EPL);
#pragma unroll
      for (int i = 0; i < V; i++) {
        uint4 r;
        half2* h = (half2*)&r;
#pragma unroll
        for (int j = 0; j < 4; j++) h[j] = __floats2half2_rn(v[8 * i + 2 * j] * kHadScale, v[8 * i + 2 * j + 1] * kHadScale);
        out[i] = r;
      }
    }
  }
}

static int g_had_cs_lanes = 8;     // knob (moe_set_had_cs_cfg): lanes per 128-block in the column-stationary kernel (4 / 8 / 16)
static int g_had_cs_threads = 512; // 256 / 512 / 1024
static int g_had_cs_waves = 2;     // CTAs ~ waves x co-resident count

template <bool in_bf16, bool ids64, int L, int T>
static void launch_had_in_cs_t(int sms, cudaStream_t stream, const half* x, half* xh, const half* suh, const void* ids,
                               int slots, int top_k, int shards, int ne, int kb) {
  const int smem = ne * 256;
  const int per_sm = std::max(1, std::min(2048 / T, (227 * 1024) / (smem + 1024)));
  const int cols = shards * kb;
  const int step = (T / 32) * (32 / L);
  int chunks = std::max(1, (g_had_cs_waves * sms * per_sm + cols - 1) / cols);
  int chunk_slots = (slots + chunks - 1) / chunks;
  chunk_slots = std::max(step, (chunk_slots + step - 1) / step * step);
  chunks = (slots + chunk_slots - 1) / chunk_slots;
  auto kern = aikido_moe_had_in_cs_kernel<in_bf16, ids64, L, T>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kern<<<cols * chunks, T, smem, stream>>>(x, xh, suh, ids, slots, top_k, shards, ne, kb, chunk_slots);
}

template <bool in_bf16, bool ids64>
static void launch_had_in_cs(int device, cudaStream_t stream, const half* x, half* xh, const half* suh, const void* ids,
                             int slots, int top_k, int shards, int ne, int kb) {
  static int sms = 0;
  if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
#define AIKIDO_CS(L_, T_) if (g_had_cs_lanes == L_ && g_had_cs_threads == T_) { launch_had_in_cs_t<in_bf16, ids64, L_, T_>(sms, stream, x, xh, suh, ids, slots, top_k, shards, ne, kb); return; }
  AIKIDO_CS(4, 256) AIKIDO_CS(4, 512) AIKIDO_CS(8, 256) AIKIDO_CS(8, 512) AIKIDO_CS(8, 1024) AIKIDO_CS(16, 512) AIKIDO_CS(16, 1024)
#undef AIKIDO_CS
  launch_had_in_cs_t<in_bf16, ids64, 8, 512>(sms, stream, x, xh, suh, ids, slots, top_k, shards, ne, kb);
}

template <bool in_bf16, bool ids64>
static void launch_had_in_rw(int lanes, int gb, cudaStream_t stream, const half* x, half* xh, const half* suh, const void* ids,
                             int slots, int top_k, int shards, int ne, int kb) {
  if (lanes == 2)       launch_had_in_rw_u<in_bf16, ids64, 2>(gb, stream, x, xh, suh, ids, slots, top_k, shards, ne, kb);
  else if (lanes == 4)  launch_had_in_rw_u<in_bf16, ids64, 4>(gb, stream, x, xh, suh, ids, slots, top_k, shards, ne, kb);
  else if (lanes == 16) launch_had_in_rw_u<in_bf16, ids64, 16>(gb, stream, x, xh, suh, ids, slots, top_k, shards, ne, kb);
  else                  launch_had_in_rw_u<in_bf16, ids64, 8>(gb, stream, x, xh, suh, ids, slots, top_k, shards, ne, kb);
}

// y[t] = sum_j w[t, j] * yd[t * top_k + j]: one warp per token, lane t of 128-block blk holds elements 4t..4t+3 exactly as
// aikido_moe_combine_kernel (same statements per element, same slot order), 32-bit indexing, 256-thread blocks.
static int g_combine_rw = 1;  // knob (moe_set_combine_rw): 0 = one 32-thread block per (token, 128-block)
static constexpr int64_t kCombineRwMinTokens = 64;

template <bool out_bf16, bool ids64>
__global__ __launch_bounds__(256)
void aikido_moe_combine_rw_kernel(const half* __restrict__ yd, const float* __restrict__ w, half* __restrict__ y,
                                  const void* __restrict__ ids, const int tokens, const int top_k, const int num_experts,
                                  const int kb) {
  const size_t hidden = (size_t)kb * 128;
  const int t = threadIdx.x & 31;
  for (int tok = blockIdx.x * 8 + threadIdx.x / 32; tok < tokens; tok += gridDim.x * 8) {
    for (int blk = 0; blk < kb; blk++) {
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      for (int j = 0; j < top_k; j++) {
        const size_t slot = (size_t)tok * top_k + j;
        const int64_t e = ids64 ? ((const int64_t*)ids)[slot] : (int64_t)((const int32_t*)ids)[slot];
        if (e < 0 || e >= num_experts) continue;
        const half4 v = ((const half4*)(yd + hidden * slot + blk * 128))[t];
        const float wj = w[(size_t)tok * top_k + j];
        float2 lo = __half22float2(v.x), hi = __half22float2(v.y);
        s0 += wj * lo.x; s1 += wj * lo.y; s2 += wj * hi.x; s3 += wj * hi.y;
      }
      half* out = y + hidden * tok + blk * 128;
      if constexpr (out_bf16) {
        float2 f0, f1;
        f0.x = s0; f0.y = s1; f1.x = s2; f1.y = s3;
        __nv_bfloat162* wr = ((__nv_bfloat162*)out) + 2 * t;
        wr[0] = __float22bfloat162_rn(f0);
        wr[1] = __float22bfloat162_rn(f1);
      } else {
        half4 o;
        o.x = __floats2half2_rn(s0, s1);
        o.y = __floats2half2_rn(s2, s3);
        ((half4*)out)[t] = o;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Torch entry points

static bool is_16bit_float(const at::Tensor& t) { return t.dtype() == at::kHalf || t.dtype() == at::kBFloat16; }
static bool is_index(const at::Tensor& t) { return t.dtype() == at::kInt || t.dtype() == at::kLong; }

// -> EXL3 K of the stack: [E, k/16, n/64, 32, 4] = K=4 (lane-major repack), [E, k/16, n/64, 4, 24] = K=3 (tiles as stored)
static int check_stack(const at::Tensor& b, int64_t k, int64_t n) {
  TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.dtype() == at::kInt, "stacked trellis must be contiguous int32 CUDA");
  TORCH_CHECK(b.dim() == 5 && ((b.size(3) == 32 && b.size(4) == 4) ||
                               (!AIKIDO_K3_TILE_INTERLEAVE && b.size(3) == 4 && b.size(4) == 24) ||
                               (AIKIDO_K3_TILE_INTERLEAVE && b.size(3) == 24 && b.size(4) == 4)),
              "stacked trellis must be [E, k/16, n/64, 32, 4] (K=4) or [E, k/16, n/64, 4, 24] (K=3; [.., 24, 4] when built "
              "with AIKIDO_K3_TILE_INTERLEAVE)");
  TORCH_CHECK(b.size(1) * 16 == k && b.size(2) * 64 == n, "stacked trellis shape does not match k, n");
  return (b.size(4) == 24 || b.size(3) == 24) ? 3 : 4;
}

// Grouped rotated-basis GEMM (+ optional in-launch output transform). a fp16 [shards * rows, k]; c fp16 [rows, n];
// sorted_ids int32 [padded] index rows, padding entries >= rows; expert_ids int32 [padded / block];
// num_post_padded int32 [1]; svh fp16 [E, n] or undefined; shard_end = column where shard 1 starts (0 = one shard).
void moe_gemm(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, const c10::optional<at::Tensor>& svh,
              const at::Tensor& sorted_ids, const at::Tensor& expert_ids, const at::Tensor& num_post_padded,
              int64_t moe_block_size, int64_t shard_end, int64_t cb, int64_t thread_k, int64_t thread_n, int64_t scratch,
              const c10::optional<at::Tensor>& glu_suh, const c10::optional<at::Tensor>& glu_xd) {
  const at::cuda::OptionalCUDAGuard device_guard(a.device());
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2 && a.dtype() == at::kHalf && c.dtype() == at::kHalf);
  TORCH_CHECK(a.is_contiguous() && c.is_contiguous());
  const int64_t rows = c.size(0), k = a.size(1), n = c.size(1), shards = shard_end > 0 ? 2 : 1;
  TORCH_CHECK(a.size(0) == shards * rows, "a must be [shards * rows, k]");
  const int bits = check_stack(b, k, n);
  TORCH_CHECK(moe_block_size == 8 || (moe_block_size % 16 == 0 && moe_block_size >= 16 && moe_block_size <= 64),
              "unsupported moe_block_size ", moe_block_size);
  TORCH_CHECK(sorted_ids.dtype() == at::kInt && expert_ids.dtype() == at::kInt && num_post_padded.dtype() == at::kInt);
  // moe_align_block_size sizes sorted_ids as T * top_k + E * (block - 1), which need not be a multiple of the block
  // size (e.g. 255 tokens, block 16); the kernel only visits num_post_padded / block full blocks, the tail is slack.
  TORCH_CHECK(sorted_ids.is_contiguous() && expert_ids.is_contiguous() && num_post_padded.numel() == 1);
  TORCH_CHECK(expert_ids.numel() >= sorted_ids.numel() / moe_block_size, "expert_ids too short");
  TORCH_CHECK(shard_end >= 0 && shard_end < n && shard_end % 128 == 0);
  TORCH_CHECK((int64_t)shards * rows * (k / 8) < INT_MAX, "input slabs exceed the kernel's 32-bit offsets");
  const half* svh_ptr = nullptr;
  if (svh.has_value()) {
    TORCH_CHECK(svh->dtype() == at::kHalf && svh->is_contiguous() && svh->dim() == 2 && svh->size(0) == b.size(0) &&
                svh->size(1) == n, "svh must be fp16 [E, n]");
    svh_ptr = (const half*)svh->data_ptr();
  }
  TORCH_CHECK(scratch == 0 || scratch == 1, "scratch must be 0 or 1");
  // glue step 3: fused GLU + down-input transform in the epilogue (gate+up launches only)
  const half* glu_suh_ptr = nullptr; half* glu_xd_ptr = nullptr;
  if (glu_suh.has_value() || glu_xd.has_value()) {
    TORCH_CHECK(glu_suh.has_value() && glu_xd.has_value() && shard_end > 0 && svh.has_value(), "fused GLU needs the gate+up launch with svh");
    TORCH_CHECK(glu_suh->dtype() == at::kHalf && glu_suh->is_contiguous() && glu_suh->dim() == 2 &&
                glu_suh->size(0) == b.size(0) && glu_suh->size(1) == shard_end, "glu_suh must be fp16 [E, inter]");
    TORCH_CHECK(glu_xd->dtype() == at::kHalf && glu_xd->is_contiguous() && glu_xd->numel() >= rows * shard_end, "glu_xd must be fp16 [rows, inter]");
    TORCH_CHECK(glu_xd->data_ptr() != c.data_ptr() && glu_xd->data_ptr() != a.data_ptr());
    TORCH_CHECK(expert_ids.numel() * (shard_end / 128) <= aikido_exl3_marlin_moe::kGluCounters, "fused GLU: too many moe blocks (decode sizes only)");
    glu_suh_ptr = (const half*)glu_suh->data_ptr(); glu_xd_ptr = (half*)glu_xd->data_ptr();
  }
  if (rows == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  aikido_exl3_marlin_moe::gemm_launch((const half*)a.data_ptr(), (const int*)b.data_ptr(), (half*)c.data_ptr(), svh_ptr,
                                      (const int*)sorted_ids.data_ptr(), (const int*)expert_ids.data_ptr(),
                                      (const int*)num_post_padded.data_ptr(), (int)moe_block_size, (int)rows, (int)n,
                                      (int)k, (int)shard_end, (int)cb, bits, a.get_device(), stream, (int)thread_k,
                                      (int)thread_n, (int)scratch, glu_suh_ptr, glu_xd_ptr);
}

// xh [shards * slots, k] fp16 <- x [tokens, k] fp16 | bf16, suh [E, shards, k] fp16, ids [tokens, top_k] int32 | int64
void moe_had_in(const at::Tensor& x, const at::Tensor& suh, const at::Tensor& ids, at::Tensor& xh) {
  const at::cuda::OptionalCUDAGuard device_guard(x.device());
  TORCH_CHECK(x.dim() == 2 && is_16bit_float(x) && x.is_contiguous() && x.size(1) % 128 == 0);
  TORCH_CHECK(ids.dim() == 2 && is_index(ids) && ids.is_contiguous() && ids.size(0) == x.size(0));
  TORCH_CHECK(suh.dim() == 3 && suh.dtype() == at::kHalf && suh.is_contiguous() && suh.size(2) == x.size(1));
  const int64_t tokens = x.size(0), k = x.size(1), top_k = ids.size(1), shards = suh.size(1), slots = tokens * top_k;
  TORCH_CHECK(xh.dtype() == at::kHalf && xh.is_contiguous() && xh.numel() >= shards * slots * k);
  if (slots == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)(shards * slots), (unsigned)(k / 128));
  const bool bf = x.dtype() == at::kBFloat16, i64 = ids.dtype() == at::kLong;
  const half* xp = (const half*)x.data_ptr(); half* op = (half*)xh.data_ptr();
  const half* sp = (const half*)suh.data_ptr(); const void* ip = ids.data_ptr();
  // AIKIDO PDL (mask 64 + 1 + 4096): in early mode every had_in block waits for align at its end, so blocks beyond the
  // first resident wave could only start once align is done; a grid-strided launch sized to one resident wave
  // (8 blocks of 8 warps per SM) runs all of had_in beside align. Same per-item statements: bit-identical.
  int gs_early = 0;
  if ((g_pdl & (kPdlHadIn | kPdlHadInEarly | kPdlHadInResident)) == (kPdlHadIn | kPdlHadInEarly | kPdlHadInResident)) {
    const int64_t items = shards * slots * (k / 128);
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    if (items > (int64_t)sms * 32) gs_early = sms * 8;
  }
  // AIKIDO prefill (wf/prefill): the column-stationary / row-per-warp kernels take the sizes the automatic grid-stride
  // rule picks (gs_blocks_for > 0, plain launches); the PDL resident grid-stride launch (gs_early) keeps the decode
  // sizes below that.
  const int gsb_auto = gs_blocks_for(shards * slots * (k / 128));
  if (const int gsb = gsb_auto > 0 ? gsb_auto : gs_early; gsb > 0) {
    const int kb = (int)(k / 128), ne = (int)suh.size(0);
    if (gsb_auto > 0 && g_had_cs && ne <= kHadCsMaxExperts) {
      TORCH_CHECK(slots < INT_MAX);
      const int dev = x.get_device();
      // clang-format off
      if (bf && i64)  launch_had_in_cs<true, true>(dev, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      else if (bf)    launch_had_in_cs<true, false>(dev, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      else if (i64)   launch_had_in_cs<false, true>(dev, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      else            launch_had_in_cs<false, false>(dev, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      // clang-format on
      return;
    }
    if (gsb_auto > 0 && g_had_lanes > 0) {
      TORCH_CHECK(slots < INT_MAX && shards * slots * k < ((int64_t)1 << 40));
      const int gb = (int)((slots + 7) / 8);
      // clang-format off
      if (bf && i64)  launch_had_in_rw<true, true>(g_had_lanes, gb, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      else if (bf)    launch_had_in_rw<true, false>(g_had_lanes, gb, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      else if (i64)   launch_had_in_rw<false, true>(g_had_lanes, gb, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      else            launch_had_in_rw<false, false>(g_had_lanes, gb, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb);
      // clang-format on
      return;
    }
    const int64_t warps = shards * slots * kb;
    const int gb = (int)std::min<int64_t>(gsb, (warps + 7) / 8);
    // clang-format off
    const bool pdl = (g_pdl & kPdlHadIn) != 0; const int pf = slot_pdl(kPdlHadIn);
    if (bf && i64)  aikido_launch(pdl, &aikido_moe_had_in_gs_kernel<true, true>, dim3(gb), dim3(256), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb, pf);
    else if (bf)    aikido_launch(pdl, &aikido_moe_had_in_gs_kernel<true, false>, dim3(gb), dim3(256), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb, pf);
    else if (i64)   aikido_launch(pdl, &aikido_moe_had_in_gs_kernel<false, true>, dim3(gb), dim3(256), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb, pf);
    else            aikido_launch(pdl, &aikido_moe_had_in_gs_kernel<false, false>, dim3(gb), dim3(256), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, ne, kb, pf);
    // clang-format on
    return;
  }
  // clang-format off
  const bool pdl = (g_pdl & kPdlHadIn) != 0; const int pf = slot_pdl(kPdlHadIn);
  if (bf && i64)       aikido_launch(pdl, &aikido_moe_had_in_kernel<true, true>, grid, dim3(32), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, (int)suh.size(0), pf);
  else if (bf)         aikido_launch(pdl, &aikido_moe_had_in_kernel<true, false>, grid, dim3(32), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, (int)suh.size(0), pf);
  else if (i64)        aikido_launch(pdl, &aikido_moe_had_in_kernel<false, true>, grid, dim3(32), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, (int)suh.size(0), pf);
  else                 aikido_launch(pdl, &aikido_moe_had_in_kernel<false, false>, grid, dim3(32), 0, stream, xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, (int)suh.size(0), pf);
  // clang-format on
}

// moe_had_in + decode routing for every K class in ONE launch (glue step 2). maps: int32 [E] per class (global -> local
// id, -1 elsewhere); blocks: class moe block size; sorted / eids / post: preallocated per class as align_decode does.
void moe_had_in_align(const at::Tensor& x, const at::Tensor& suh, const at::Tensor& ids, at::Tensor& xh,
                      const std::vector<at::Tensor>& maps, const std::vector<int64_t>& blocks, int64_t num_experts,
                      const std::vector<at::Tensor>& sorted, const std::vector<at::Tensor>& eids,
                      const std::vector<at::Tensor>& post) {
  const at::cuda::OptionalCUDAGuard device_guard(x.device());
  TORCH_CHECK(x.dim() == 2 && is_16bit_float(x) && x.is_contiguous() && x.size(1) % 128 == 0);
  TORCH_CHECK(ids.dim() == 2 && is_index(ids) && ids.is_contiguous() && ids.size(0) == x.size(0));
  TORCH_CHECK(suh.dim() == 3 && suh.dtype() == at::kHalf && suh.is_contiguous() && suh.size(2) == x.size(1));
  TORCH_CHECK(suh.size(0) == num_experts && num_experts <= AIKIDO_ROUTE_MAX_E, "moe_had_in_align: <= 256 experts");
  const int64_t tokens = x.size(0), k = x.size(1), top_k = ids.size(1), shards = suh.size(1), slots = tokens * top_k;
  TORCH_CHECK(xh.dtype() == at::kHalf && xh.is_contiguous() && xh.numel() >= shards * slots * k);
  TORCH_CHECK(slots >= 1 && slots <= AIKIDO_ROUTE_MAX_SLOTS, "moe_had_in_align: decode sizes only");
  const int nc = (int)maps.size();
  TORCH_CHECK(nc >= 1 && nc <= 2 && (int)blocks.size() == nc && (int)sorted.size() == nc && (int)eids.size() == nc && (int)post.size() == nc);
  RouteClass rc[2] = {};
  for (int c = 0; c < nc; c++) {
    TORCH_CHECK(maps[c].dtype() == at::kInt && maps[c].numel() == num_experts && maps[c].is_contiguous());
    TORCH_CHECK(sorted[c].dtype() == at::kInt && eids[c].dtype() == at::kInt && post[c].dtype() == at::kInt);
    TORCH_CHECK(sorted[c].is_contiguous() && eids[c].is_contiguous() && sorted[c].numel() % 4 == 0 &&
                ((uintptr_t)sorted[c].data_ptr() & 15) == 0, "sorted_ids must be 16-byte aligned, length % 4 == 0");
    TORCH_CHECK(eids[c].numel() * blocks[c] >= sorted[c].numel(), "expert_ids too short");
    rc[c] = RouteClass{maps[c].data_ptr<int>(), (int)blocks[c], sorted[c].data_ptr<int>(), (int)sorted[c].numel(),
                       eids[c].data_ptr<int>(), (int)eids[c].numel(), post[c].data_ptr<int>()};
  }
  if (nc == 1) rc[1] = rc[0];
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)(shards * slots + nc), (unsigned)(k / 128));
  const bool bf = x.dtype() == at::kBFloat16, i64 = ids.dtype() == at::kLong;
  const half* xp = (const half*)x.data_ptr(); half* op = (half*)xh.data_ptr();
  const half* sp = (const half*)suh.data_ptr(); const void* ip = ids.data_ptr();
  const int E = (int)num_experts;
  // clang-format off
  if (bf && i64)       aikido_moe_had_in_route_kernel<true, true><<<grid, 32, 0, stream>>>(xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, E, rc[0], rc[1]);
  else if (bf)         aikido_moe_had_in_route_kernel<true, false><<<grid, 32, 0, stream>>>(xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, E, rc[0], rc[1]);
  else if (i64)        aikido_moe_had_in_route_kernel<false, true><<<grid, 32, 0, stream>>>(xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, E, rc[0], rc[1]);
  else                 aikido_moe_had_in_route_kernel<false, false><<<grid, 32, 0, stream>>>(xp, op, sp, ip, (int)slots, (int)top_k, (int)shards, E, rc[0], rc[1]);
  // clang-format on
}

// xd [slots, inter] fp16 <- gu [slots, 2 * inter] fp16 (gate | up), suh_down [E, inter], ids [tokens, top_k];
// act_tmp: fp16 scratch [slots, inter] (receives fp16_rn(silu(g) * u), the activation tensor itself)
void moe_glu_had_in(const at::Tensor& gu, const at::Tensor& suh, const at::Tensor& ids, at::Tensor& act_tmp,
                    at::Tensor& xd) {
  const at::cuda::OptionalCUDAGuard device_guard(gu.device());
  TORCH_CHECK(gu.dim() == 2 && gu.dtype() == at::kHalf && gu.is_contiguous() && gu.size(1) % 256 == 0);
  const int64_t slots = gu.size(0), inter = gu.size(1) / 2;
  TORCH_CHECK(is_index(ids) && ids.is_contiguous() && ids.numel() == slots);
  TORCH_CHECK(suh.dim() == 2 && suh.dtype() == at::kHalf && suh.is_contiguous() && suh.size(1) == inter);
  TORCH_CHECK(xd.dtype() == at::kHalf && xd.is_contiguous() && xd.numel() >= slots * inter);
  TORCH_CHECK(act_tmp.dtype() == at::kHalf && act_tmp.is_contiguous() && act_tmp.numel() >= slots * inter);
  TORCH_CHECK(act_tmp.data_ptr() != xd.data_ptr() && act_tmp.data_ptr() != gu.data_ptr());
  if (slots == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)slots, (unsigned)(inter / 128));
  if (const int gsb = gs_blocks_for(slots * (inter / 128)); gsb > 0) {
    const int kb = (int)(inter / 128), ne = (int)suh.size(0);
    if (g_glu_lanes > 0) {
      const int gbr = (int)((slots + 7) / 8);
      if (ids.dtype() == at::kLong)
        launch_glu_rw<true>(g_glu_lanes, gbr, stream, (const half*)gu.data_ptr(), (half*)xd.data_ptr(), (half*)act_tmp.data_ptr(), (const half*)suh.data_ptr(), ids.data_ptr(), ne, (int)slots, kb);
      else
        launch_glu_rw<false>(g_glu_lanes, gbr, stream, (const half*)gu.data_ptr(), (half*)xd.data_ptr(), (half*)act_tmp.data_ptr(), (const half*)suh.data_ptr(), ids.data_ptr(), ne, (int)slots, kb);
      return;
    }
    const int gb = (int)std::min<int64_t>(gsb, (slots * kb + 7) / 8);
    // clang-format off
    const bool pdl = (g_pdl & kPdlGlu) != 0; const int pf = slot_pdl(kPdlGlu);
    if (ids.dtype() == at::kLong)
      aikido_launch(pdl, &aikido_moe_glu_had_in_gs_kernel<true>, dim3(gb), dim3(256), 0, stream, (const half*)gu.data_ptr(), (half*)xd.data_ptr(), (half*)act_tmp.data_ptr(), (const half*)suh.data_ptr(), (const void*)ids.data_ptr(), ne, (int)slots, kb, pf);
    else
      aikido_launch(pdl, &aikido_moe_glu_had_in_gs_kernel<false>, dim3(gb), dim3(256), 0, stream, (const half*)gu.data_ptr(), (half*)xd.data_ptr(), (half*)act_tmp.data_ptr(), (const half*)suh.data_ptr(), (const void*)ids.data_ptr(), ne, (int)slots, kb, pf);
    // clang-format on
    return;
  }
  // clang-format off
  const bool pdl = (g_pdl & kPdlGlu) != 0; const int pf = slot_pdl(kPdlGlu);
  if (ids.dtype() == at::kLong)
    aikido_launch(pdl, &aikido_moe_glu_had_in_kernel<true>, grid, dim3(32), 0, stream, (const half*)gu.data_ptr(), (half*)xd.data_ptr(), (half*)act_tmp.data_ptr(), (const half*)suh.data_ptr(), (const void*)ids.data_ptr(), (int)suh.size(0), pf);
  else
    aikido_launch(pdl, &aikido_moe_glu_had_in_kernel<false>, grid, dim3(32), 0, stream, (const half*)gu.data_ptr(), (half*)xd.data_ptr(), (half*)act_tmp.data_ptr(), (const half*)suh.data_ptr(), (const void*)ids.data_ptr(), (int)suh.size(0), pf);
  // clang-format on
}

// y [tokens, hidden] fp16 | bf16 <- yd [tokens * top_k, hidden] fp16, w [tokens, top_k] float32; slots whose id is
// outside [0, num_experts) are skipped (moe_align_block_size dropped them, their yd row was never written)
void moe_combine(const at::Tensor& yd, const at::Tensor& w, const at::Tensor& ids, int64_t num_experts, at::Tensor& y) {
  const at::cuda::OptionalCUDAGuard device_guard(yd.device());
  TORCH_CHECK(yd.dim() == 2 && yd.dtype() == at::kHalf && yd.is_contiguous() && yd.size(1) % 128 == 0);
  TORCH_CHECK(w.dim() == 2 && w.dtype() == at::kFloat && w.is_contiguous());
  TORCH_CHECK(y.dim() == 2 && is_16bit_float(y) && y.is_contiguous() && y.size(1) == yd.size(1));
  const int64_t tokens = y.size(0), top_k = w.size(1);
  TORCH_CHECK(w.size(0) == tokens && yd.size(0) == tokens * top_k);
  TORCH_CHECK(is_index(ids) && ids.is_contiguous() && ids.numel() == tokens * top_k && num_experts > 0);
  if (tokens == 0) return;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid((unsigned)tokens, (unsigned)(y.size(1) / 128));
  // clang-format off
  const half* ydp = (const half*)yd.data_ptr(); const float* wp = (const float*)w.data_ptr(); half* yp = (half*)y.data_ptr();
  const bool bf = y.dtype() == at::kBFloat16, i64 = ids.dtype() == at::kLong;
  // AIKIDO prefill (wf/prefill): row-per-warp combine from 64 tokens (plain launch); below, the PDL per-token launch
  if (g_combine_rw && tokens >= kCombineRwMinTokens) {
    const int gb = (int)((tokens + 7) / 8), kb = (int)(y.size(1) / 128);
    if (bf && i64)  aikido_moe_combine_rw_kernel<true, true><<<gb, 256, 0, stream>>>(ydp, wp, yp, ids.data_ptr(), (int)tokens, (int)top_k, (int)num_experts, kb);
    else if (bf)    aikido_moe_combine_rw_kernel<true, false><<<gb, 256, 0, stream>>>(ydp, wp, yp, ids.data_ptr(), (int)tokens, (int)top_k, (int)num_experts, kb);
    else if (i64)   aikido_moe_combine_rw_kernel<false, true><<<gb, 256, 0, stream>>>(ydp, wp, yp, ids.data_ptr(), (int)tokens, (int)top_k, (int)num_experts, kb);
    else            aikido_moe_combine_rw_kernel<false, false><<<gb, 256, 0, stream>>>(ydp, wp, yp, ids.data_ptr(), (int)tokens, (int)top_k, (int)num_experts, kb);
    return;
  }
  const bool pdl = (g_pdl & kPdlCombine) != 0; const int pf = slot_pdl(kPdlCombine);
  const void* idp = ids.data_ptr();
  if (bf && i64)  aikido_launch(pdl, &aikido_moe_combine_kernel<true, true>, grid, dim3(32), 0, stream, ydp, wp, yp, idp, (int)top_k, (int)num_experts, pf);
  else if (bf)    aikido_launch(pdl, &aikido_moe_combine_kernel<true, false>, grid, dim3(32), 0, stream, ydp, wp, yp, idp, (int)top_k, (int)num_experts, pf);
  else if (i64)   aikido_launch(pdl, &aikido_moe_combine_kernel<false, true>, grid, dim3(32), 0, stream, ydp, wp, yp, idp, (int)top_k, (int)num_experts, pf);
  else            aikido_launch(pdl, &aikido_moe_combine_kernel<false, false>, grid, dim3(32), 0, stream, ydp, wp, yp, idp, (int)top_k, (int)num_experts, pf);
  // clang-format on
}

// exl3_moe_orig_builder.cu: original-basis expert weights straight into vLLM's [E, out, in] layout
void moe_build_orig_vllm(at::Tensor& out, const at::Tensor& trellis_ptrs, const at::Tensor& suh_ptrs,
                         const at::Tensor& svh_ptrs, int64_t n, int64_t row_offset, int64_t cb, int64_t bits);

void moe_build_orig_vllm_stacked(at::Tensor& out, const at::Tensor& pack, const at::Tensor& suh, const at::Tensor& svh,
                                 int64_t n, int64_t col_offset, int64_t suh_off, int64_t svh_off, int64_t row_offset, int64_t cb,
                                 const std::optional<at::Tensor>& out_ids);

void moe_orig_set_variant(int64_t v);
int64_t moe_orig_get_variant();

void moe_init_device(int64_t device) { aikido_exl3_marlin_moe::dev_state((int)device); }

void moe_set_pdl(int64_t mask) {
  TORCH_CHECK(mask >= 0 && mask < 65536);
  g_pdl = (int)mask;
}

int64_t moe_get_pdl() { return g_pdl; }

void moe_set_gridstride_blocks(int64_t n) {
  TORCH_CHECK(n >= -1 && n <= (1 << 22));
  g_gs_blocks = (int)n;
}

void moe_set_had_lanes(int64_t n) {
  TORCH_CHECK(n == 0 || n == 2 || n == 4 || n == 8 || n == 16);
  g_had_lanes = (int)n;
}

void moe_set_combine_rw(int64_t n) { g_combine_rw = n != 0; }

void moe_set_had_cs(int64_t n) { g_had_cs = n != 0; }

void moe_set_had_cs_cfg(int64_t lanes, int64_t threads, int64_t waves) {
  TORCH_CHECK((lanes == 4 || lanes == 8 || lanes == 16) && (threads == 256 || threads == 512 || threads == 1024) && waves >= 1 && waves <= 64);
  g_had_cs_lanes = (int)lanes; g_had_cs_threads = (int)threads; g_had_cs_waves = (int)waves;
}

void moe_set_had_unroll(int64_t n) {
  TORCH_CHECK(n >= 1 && n <= 3);
  g_had_unroll = (int)n;
}

void moe_set_glu_lanes(int64_t n) {
  TORCH_CHECK(n == 0 || n == 2 || n == 4 || n == 8);
  g_glu_lanes = (int)n;
}

void moe_set_large_block_bps(int64_t n) {
  TORCH_CHECK(n >= 0 && n <= aikido_exl3_marlin_moe::kMaxBlocksPerSm);
  aikido_exl3_marlin_moe::g_large_block_bps = (int)n;
}

void moe_set_blocks_per_sm(int64_t n) {
  TORCH_CHECK(n == -1 || (n >= 1 && n <= aikido_exl3_marlin_moe::kMaxBlocksPerSm));
  aikido_exl3_marlin_moe::g_blocks_per_sm = (int)n;
}

void moe_set_occ_min_rows(int64_t n) { aikido_exl3_marlin_moe::g_occ_min_rows = (int)n; }
void moe_set_wide_rows(int64_t down, int64_t gu) { aikido_exl3_marlin_moe::g_wide_down_rows = (int)down; aikido_exl3_marlin_moe::g_wide_gu_rows = (int)gu; }
int64_t moe_k3_tile_interleave() { return AIKIDO_K3_TILE_INTERLEAVE; }
void moe_set_grid_limit(int64_t n) {
  TORCH_CHECK(n >= 0);
  aikido_exl3_marlin_moe::g_grid_limit = (int)n;
}

// ---------------------------------------------------------------------------------------------------------------
// Decode-sized routing for mixed-K layers: ONE launch produces, for every K class, the (sorted_token_ids, expert_ids,
// num_tokens_post_padded) triple vLLM's moe_align_block_size would produce with that class's expert_map and
// ignore_invalid_experts=True (padding slots = numel, padding blocks = -1, local expert ids, per-expert counts padded
// to the class's block size, experts in ascending global order). vLLM's kernel is generic (grids, cub scans over 256
// experts, a separate fill launch): 21-49 us per class at 1 token on box2; this is ~5 us for both classes.
// One block of 1024 threads; slots <= AIKIDO_ALIGN_MAX_SLOTS, experts <= 1024, classes <= 2.
// Deterministic (integ): inside an expert's range the slots are placed in ASCENDING slot order, run to run, like the
// glue route warp. The old shared-memory atomicAdd fill put them in arrival order, so once an expert spans more than
// one moe block (16+ tokens) the k-split / reduction partition changed between runs (up to 1 fp16 ulp). The rank of a
// slot inside its expert is class independent: one warp computes it once (__match_any_sync rank per 32-slot batch,
// batches in order; slots / 32 iterations, 4 at 16 tokens x top-8), which also yields the per-expert counts.
#include <cub/cub.cuh>
#define AIKIDO_ALIGN_MAX_SLOTS 4096
#define AIKIDO_ALIGN_MAX_E 1024

__global__ __launch_bounds__(1024)
void aikido_moe_align_decode_kernel(const int* __restrict__ topk_ids, int slots, int num_experts, int num_classes,
                                    const int* __restrict__ map0, const int* __restrict__ map1, int block0, int block1,
                                    int* __restrict__ sorted0, int* __restrict__ sorted1, int cap0, int cap1,
                                    int* __restrict__ eids0, int* __restrict__ eids1, int mblk0, int mblk1,
                                    int* __restrict__ post0, int* __restrict__ post1, const int pdl) {
  __shared__ int count[AIKIDO_ALIGN_MAX_E];
  __shared__ int start[AIKIDO_ALIGN_MAX_E];
  __shared__ int rank_s[AIKIDO_ALIGN_MAX_SLOTS];   // topk id, then (valid slots) the slot's rank inside its expert
  __shared__ int total_c[2];
  using Scan = cub::BlockScan<int, 1024>;
  __shared__ typename Scan::TempStorage tmp;
  const int t = threadIdx.x;
  for (int e = t; e < num_experts; e += blockDim.x) count[e] = 0;
  aikido_pdl_slot(pdl);   // AIKIDO PDL: topk_ids come from the router
  for (int s = t; s < slots; s += blockDim.x) rank_s[s] = topk_ids[s];
  __syncthreads();
  if (t < 32) {
    for (int base = 0; base < slots; base += 32) {
      const int s = base + t;
      const int e = s < slots ? rank_s[s] : -1;
      const bool ok = e >= 0 && e < num_experts;
      // invalid lanes get unique negative keys (-1 - lane), so only equal valid experts group together
      const unsigned peers = __match_any_sync(0xffffffffu, ok ? e : -1 - t);
      const int r = __popc(peers & ((1u << t) - 1u));
      if (ok) rank_s[s] = count[e] + r;
      __syncwarp();
      if (ok && r == 0) count[e] += __popc(peers);   // the group's lowest lane
      __syncwarp();
    }
  }
  __syncthreads();
  for (int c = 0; c < num_classes; c++) {
    const int* map = c == 0 ? map0 : map1;
    const int block = c == 0 ? block0 : block1;
    int padded = 0;
    if (t < num_experts && map[t] >= 0) padded = ((count[t] + block - 1) / block) * block;
    int excl, total;
    Scan(tmp).ExclusiveSum(padded, excl, total);
    if (t < num_experts) start[t] = excl;
    if (t == 0) total_c[c] = total;
    __syncthreads();
    int* sorted = c == 0 ? sorted0 : sorted1;
    int* eids = c == 0 ? eids0 : eids1;
    const int cap = c == 0 ? cap0 : cap1;
    const int mblk = c == 0 ? mblk0 : mblk1;
    // expert ids per block (local ids), padding blocks -1
    if (t < num_experts && map[t] >= 0)
      for (int i = start[t]; i < start[t] + padded; i += block) eids[i / block] = map[t];
    for (int i = (total + block - 1) / block + t; i < mblk; i += blockDim.x) eids[i] = -1;
    // padding slots = numel (vLLM's sentinel); then the class's slots into their expert's range
    for (int i = total + t; i < cap; i += blockDim.x) sorted[i] = slots;
    for (int i = t; i < total; i += blockDim.x) sorted[i] = slots;   // pad the tails inside expert ranges too
    __syncthreads();
    for (int s = t; s < slots; s += blockDim.x) {
      int e = topk_ids[s];
      if (e >= 0 && e < num_experts && map[e] >= 0) sorted[start[e] + rank_s[s]] = s;
    }
    if (t == 0) *(c == 0 ? post0 : post1) = total;
    __syncthreads();
  }
}

// topk_ids int32 [tokens, top_k] (global ids); maps: int32 [E] per class; outputs preallocated per class
void moe_align_decode(const at::Tensor& topk_ids, const std::vector<at::Tensor>& maps, const std::vector<int64_t>& blocks,
                      int64_t num_experts, const std::vector<at::Tensor>& sorted, const std::vector<at::Tensor>& eids,
                      const std::vector<at::Tensor>& post) {
  const at::cuda::OptionalCUDAGuard device_guard(topk_ids.device());
  const int nc = (int)maps.size();
  TORCH_CHECK(nc >= 1 && nc <= 2 && (int)blocks.size() == nc && (int)sorted.size() == nc && (int)eids.size() == nc && (int)post.size() == nc);
  TORCH_CHECK(topk_ids.dtype() == at::kInt && topk_ids.is_contiguous());
  const int slots = (int)topk_ids.numel();
  TORCH_CHECK(slots <= AIKIDO_ALIGN_MAX_SLOTS && num_experts <= AIKIDO_ALIGN_MAX_E, "moe_align_decode: decode sizes only");
  for (int c = 0; c < nc; c++) {
    TORCH_CHECK(maps[c].dtype() == at::kInt && maps[c].numel() == num_experts && maps[c].is_contiguous());
    TORCH_CHECK(sorted[c].dtype() == at::kInt && eids[c].dtype() == at::kInt && post[c].dtype() == at::kInt);
    TORCH_CHECK(eids[c].numel() * blocks[c] >= sorted[c].numel(), "expert_ids too short");
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto P = [&](const std::vector<at::Tensor>& v, int c) { return c < nc ? v[c].data_ptr<int>() : nullptr; };
  auto M = [&](int c) { return c < nc ? maps[c].data_ptr<int>() : nullptr; };
  aikido_launch((g_pdl & kPdlAlign) != 0, &aikido_moe_align_decode_kernel, dim3(1), dim3(1024), 0, stream,
      (const int*)topk_ids.data_ptr<int>(), slots, (int)num_experts, nc, (const int*)M(0), (const int*)M(1), (int)blocks[0], nc > 1 ? (int)blocks[1] : 1,
      P(sorted, 0), P(sorted, 1), (int)sorted[0].numel(), nc > 1 ? (int)sorted[1].numel() : 0,
      P(eids, 0), P(eids, 1), (int)eids[0].numel(), nc > 1 ? (int)eids[1].numel() : 0, P(post, 0), P(post, 1), slot_pdl(kPdlAlign));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_align_decode", &moe_align_decode, "decode-sized routing for mixed-K layers: vLLM moe_align_block_size outputs for every K class in ONE launch",
        py::arg("topk_ids"), py::arg("maps"), py::arg("blocks"), py::arg("num_experts"), py::arg("sorted"), py::arg("expert_ids"), py::arg("num_post"));
  m.def("moe_gemm", &moe_gemm, "grouped rotated-basis EXL3 GEMM over moe blocks (+ in-launch output transform)",
        py::arg("a"), py::arg("b"), py::arg("c"), py::arg("svh"), py::arg("sorted_ids"), py::arg("expert_ids"),
        py::arg("num_post_padded"), py::arg("moe_block_size"), py::arg("shard_end"), py::arg("cb"),
        py::arg("thread_k") = -1, py::arg("thread_n") = -1, py::arg("scratch") = 0,
        py::arg("glu_suh") = py::none(), py::arg("glu_xd") = py::none());
  m.def("moe_had_in", &moe_had_in, "per-slot input transform with per-expert suh (gather + scale + Had128)");
  m.def("moe_had_in_align", &moe_had_in_align, "moe_had_in + the decode routing tables of every K class in the same launch (one warp per class)",
        py::arg("x"), py::arg("suh"), py::arg("ids"), py::arg("xh"), py::arg("maps"), py::arg("blocks"), py::arg("num_experts"),
        py::arg("sorted"), py::arg("expert_ids"), py::arg("num_post"));
  m.def("moe_glu_had_in", &moe_glu_had_in, "silu(gate) * up -> fp16 -> per-expert suh -> Had128 (down input)");
  m.def("moe_combine", &moe_combine, "router-weighted fp32 sum over a token's top-k slots");
  m.def("moe_build_orig_vllm", &moe_build_orig_vllm,
        "W = diag(suh) H W_hat H diag(svh) of every expert (ExLlamaV3's reconstruct_had arithmetic) written transposed into out[E, rows, k]",
        py::arg("out"), py::arg("trellis_ptrs"), py::arg("suh_ptrs"), py::arg("svh_ptrs"), py::arg("n"), py::arg("row_offset"), py::arg("cb"),
        py::arg("bits") = 4);
  m.def("moe_build_orig_vllm_stacked", &moe_build_orig_vllm_stacked,
        "the same W, read from the grouped kernel's resident pack ([E, k/16, n/64, 32, 4] K = 4 or [E, k/16, n/64, 4, 24] K = 3) and stacked "
        "suh / svh (no pointer tables); out_ids (int32 [E_pack], optional) = output expert per pack expert (mixed-K arenas)",
        py::arg("out"), py::arg("pack"), py::arg("suh"), py::arg("svh"), py::arg("n"), py::arg("col_offset"), py::arg("suh_off"), py::arg("svh_off"),
        py::arg("row_offset"), py::arg("cb"), py::arg("out_ids") = py::none());
  m.def("moe_orig_set_variant", &moe_orig_set_variant, "original-basis builder: select a compiled data-movement variant (bitmask)");
  m.def("moe_orig_get_variant", &moe_orig_get_variant, "original-basis builder: the active variant");
  m.def("moe_init_device", &moe_init_device, "allocate per-device state (locks, fp32 reduce scratch)");
  m.def("moe_set_pdl", &moe_set_pdl, "knob: programmatic-dependent-launch mask (1 had_in, 2 gemm gate+up, 4 glu, 8 gemm down, 16 combine, 32 align, 64 had_in overlaps align, 128 explicit triggers in the per-slot kernels, 256 explicit triggers in the GEMMs, 512 GEMM weight prefetch before the wait, 1024 tail-wave triggers in the per-slot kernels)");
  m.def("moe_get_pdl", &moe_get_pdl, "current programmatic-dependent-launch mask");
  m.def("moe_set_gridstride_blocks", &moe_set_gridstride_blocks,
        "knob: -1 = automatic (default), 0 = one 32-thread block per (row, 128-block) item in the per-slot transforms, n > 0 = n grid-strided 256-thread blocks");
  m.def("moe_set_had_lanes", &moe_set_had_lanes, "knob: prefill input transform, 0 = grid-strided per-item kernel, 2 / 4 / 8 = row-per-warp kernel with that many lanes per 128-block (default 4)");
  m.def("moe_set_glu_lanes", &moe_set_glu_lanes, "knob: prefill GLU transform, 0 = grid-strided per-item kernel, 2 / 4 / 8 = row-per-warp kernel (default 8)");
  m.def("moe_set_had_cs", &moe_set_had_cs, "knob: 1 = column-stationary prefill input transform (E <= 384), 0 = row-per-warp / grid-strided (default)");
  m.def("moe_set_had_unroll", &moe_set_had_unroll, "knob: warp steps with loads in flight together in the row-per-warp input transform (1..3, default 2)");
  m.def("moe_set_had_cs_cfg", &moe_set_had_cs_cfg, "knob: column-stationary input transform geometry (lanes 4/8/16, threads 256/512/1024, waves)");
  m.def("moe_set_combine_rw", &moe_set_combine_rw, "knob: 1 = row-per-warp combine from 64 tokens (default), 0 = one 32-thread block per (token, 128-block)");
  m.def("moe_set_large_block_bps", &moe_set_large_block_bps,
        "knob: cap of co-resident blocks per SM for moe blocks of more than 16 rows (default 1, 0 = Marlin MoE's choice)");
  m.def("moe_set_blocks_per_sm", &moe_set_blocks_per_sm, "knob: co-resident blocks per SM (-1 = Marlin MoE's choice)");
  m.def("moe_set_occ_min_rows", &moe_set_occ_min_rows, "knob: rows per 8-row launch from which the 4-blocks-per-SM family 5 is used");
  m.def("moe_set_wide_rows", &moe_set_wide_rows, "knob: rows from which 8-row down / gate+up launches use the 64x256 config (0 = off)");
  m.def("moe_k3_tile_interleave", &moe_k3_tile_interleave, "1: K = 3 stacks are [.., 24 words, 4 tiles] (stack_repacked)");
  m.def("moe_set_grid_limit", &moe_set_grid_limit, "knob: cap the GEMM grid at n blocks (0 = sms * blocks per SM)");
}
