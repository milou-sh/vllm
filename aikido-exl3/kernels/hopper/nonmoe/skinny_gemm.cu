// aikido_nonmoe_kernels: skinny bf16 GEMM for decode-sized row counts (M <= 32) on sm_90.
//
//   out[M, N] (bf16) = x[M, K] (bf16) @ W[N, K]^T (bf16),  fp32 accumulation.
//
// Weight-streaming design ("swap AB"): W rows are the MMA M dimension (m16), tokens are the MMA N dimension (n8),
// so one warp covers RT x 16 weight rows and NT x 8 tokens with mma.sync.m16n8k16 bf16 -> fp32.
// Each thread loads 16 contiguous bytes of a weight row per 32-wide k chunk; the k order inside a chunk is permuted
// identically for A and B (a dot product is invariant under a shared permutation of k), so the fragments come
// straight from coalesced 16-byte global loads, no shared memory, no ldmatrix.
// K is split over WK warps inside a CTA (reduced through shared memory in warp order) and over SC CTAs (fp32
// partials in a workspace; the last-arriving CTA of a row group sums the SC partials in split order 0..SC-1, so the
// result does not depend on arrival order: deterministic, no atomics on data, no separate reduce kernel).
// PDL: the first weight stage is issued before griddepcontrol.wait (weights never depend on the previous kernel);
// x, the workspace, the counters and out are touched only after it.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

struct Params {
  const __nv_bfloat16* x; long ldx; long xbs;      // batch stride (blockIdx.z), 0 for plain GEMMs
  const __nv_bfloat16* w; long ldw; long wbs; int K;
  __nv_bfloat16* out; long ldo; long obs;
  float* ws; int* counters;
  int M, N;
  int chunks_per_warp;               // k32 chunks per warp split
  int total_chunks;                  // K / 32
  int prefetch;                      // 1: bulk-prefetch this warp's weight rows into L2 before griddepcontrol.wait
};

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldg_x(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <int RT, int NT, int U>
__global__ void __launch_bounds__(256) skinny_gemm_kernel(const Params p) {
  extern __shared__ float red[];      // [WK][E] fp32, E = RT*NT*128
  constexpr int E = RT * NT * 128;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wk = blockDim.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rg0 = blockIdx.x, s = blockIdx.y, sc = gridDim.y, b = blockIdx.z;
  const int rg = rg0 + b * gridDim.x;                  // workspace / counter index
  const __nv_bfloat16* __restrict__ xb_ = p.x + b * p.xbs;
  const __nv_bfloat16* __restrict__ wb_ = p.w + b * p.wbs;
  __nv_bfloat16* __restrict__ ob_ = p.out + b * p.obs;
  const int ksplit = s * wk + warp;
  const int c0 = ksplit * p.chunks_per_warp;
  const int c1 = min(p.total_chunks, c0 + p.chunks_per_warp);
  const int row0 = rg0 * RT * 16;

  // weight row pointers (rows g and g+8 of each m16 tile), offset by this lane's 16-byte column
  const __nv_bfloat16* wp[RT][2];
#pragma unroll
  for (int r = 0; r < RT; r++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int row = min(row0 + r * 16 + g + 8 * h, p.N - 1);   // clamp (N % 16 == 0 is required by the host anyway)
      wp[r][h] = wb_ + (long)row * p.ldw + 8 * t;
    }
  }
  const __nv_bfloat16* xp[NT];
  bool xv[NT];
#pragma unroll
  for (int j = 0; j < NT; j++) {
    int tok = j * 8 + g;
    xv[j] = tok < p.M;
    xp[j] = xb_ + (long)(xv[j] ? tok : 0) * p.ldx + 8 * t;
  }

  float acc[RT][NT][4];
#pragma unroll
  for (int r = 0; r < RT; r++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) acc[r][j][e] = 0.f;

  uint4 wa[U][RT][2], xa[U][NT];
  const uint4 z = make_uint4(0, 0, 0, 0);

  // optional: stream this warp's whole weight slice into L2 now (no registers, full memory-level parallelism); with
  // PDL this overlaps the previous kernel. The main loop then reads L2 (or merges with the in-flight prefetch).
  if (p.prefetch && c1 > c0) {
    const unsigned bytes = (unsigned)(c1 - c0) * 64u;
    for (int i = lane; i < RT * 16; i += 32) {
      int row = min(row0 + i, p.N - 1);
      const void* a = wb_ + (long)row * p.ldw + (long)c0 * 32;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"(bytes) : "memory");
    }
  }
  // prologue: first weight stage before the grid dependency wait
#pragma unroll
  for (int u = 0; u < U; u++) {
    int c = c0 + u;
#pragma unroll
    for (int r = 0; r < RT; r++)
#pragma unroll
      for (int h = 0; h < 2; h++) wa[u][r][h] = (c < c1) ? ldg_stream(wp[r][h] + c * 32) : z;
  }
  pdl_wait();
  pdl_trigger();
#pragma unroll
  for (int u = 0; u < U; u++) {
    int c = c0 + u;
#pragma unroll
    for (int j = 0; j < NT; j++) xa[u][j] = (c < c1 && xv[j]) ? ldg_x(xp[j] + c * 32) : z;
  }

#pragma unroll 1
  for (int c = c0; c < c1; c += U) {
    uint4 wb[U][RT][2], xb[U][NT];
    const bool more = c + U < c1;
    if (more) {
#pragma unroll
      for (int u = 0; u < U; u++) {
        int cn = c + U + u;
#pragma unroll
        for (int r = 0; r < RT; r++)
#pragma unroll
          for (int h = 0; h < 2; h++) wb[u][r][h] = (cn < c1) ? ldg_stream(wp[r][h] + cn * 32) : z;
#pragma unroll
        for (int j = 0; j < NT; j++) xb[u][j] = (cn < c1 && xv[j]) ? ldg_x(xp[j] + cn * 32) : z;
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
#pragma unroll
      for (int r = 0; r < RT; r++)
#pragma unroll
        for (int j = 0; j < NT; j++) {
          mma_bf16(acc[r][j], wa[u][r][0].x, wa[u][r][1].x, wa[u][r][0].y, wa[u][r][1].y, xa[u][j].x, xa[u][j].y);
          mma_bf16(acc[r][j], wa[u][r][0].z, wa[u][r][1].z, wa[u][r][0].w, wa[u][r][1].w, xa[u][j].z, xa[u][j].w);
        }
    }
    if (more) {
#pragma unroll
      for (int u = 0; u < U; u++) {
#pragma unroll
        for (int r = 0; r < RT; r++) { wa[u][r][0] = wb[u][r][0]; wa[u][r][1] = wb[u][r][1]; }
#pragma unroll
        for (int j = 0; j < NT; j++) xa[u][j] = xb[u][j];
      }
    }
  }

  // ---- in-CTA reduction over the WK warps, in warp order
  float* mine = red + warp * E;
#pragma unroll
  for (int r = 0; r < RT; r++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) mine[((r * NT + j) * 4 + e) * 32 + lane] = acc[r][j][e];
  __syncthreads();

  const int nthr = blockDim.x;
  __shared__ int is_last;
  if (sc == 1) {
    for (int i = threadIdx.x; i < E; i += nthr) {
      float v = 0.f;
      for (int q = 0; q < wk; q++) v += red[q * E + i];
      int ln = i & 31, e = (i >> 5) & 3, rj = i >> 7, j = rj % NT, r = rj / NT;
      int row = row0 + r * 16 + (ln >> 2) + ((e & 2) ? 8 : 0);
      int tok = j * 8 + (ln & 3) * 2 + (e & 1);
      if (tok < p.M && row < p.N) ob_[(long)tok * p.ldo + row] = __float2bfloat16_rn(v);
    }
    return;
  }
  float* part = p.ws + ((long)rg * sc + s) * E;
  for (int i = threadIdx.x; i < E; i += nthr) {
    float v = 0.f;
    for (int q = 0; q < wk; q++) v += red[q * E + i];
    __stcg(part + i, v);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) is_last = (atomicAdd(p.counters + rg, 1) == sc - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const float* base = p.ws + (long)rg * sc * E;
  for (int i = threadIdx.x; i < E; i += nthr) {
    float v = 0.f;
    for (int q = 0; q < sc; q++) v += __ldcg(base + (long)q * E + i);
    int ln = i & 31, e = (i >> 5) & 3, rj = i >> 7, j = rj % NT, r = rj / NT;
    int row = row0 + r * 16 + (ln >> 2) + ((e & 2) ? 8 : 0);
    int tok = j * 8 + (ln & 3) * 2 + (e & 1);
    if (tok < p.M && row < p.N) ob_[(long)tok * p.ldo + row] = __float2bfloat16_rn(v);
  }
  if (threadIdx.x == 0) p.counters[rg] = 0;   // ready for the next launch (graph replay)
}

template <int RT, int NT, int U>
void launch(const Params& p, int wk, int sc, int batch, bool pdl, cudaStream_t stream) {
  constexpr int E = RT * NT * 128;
  int smem = wk * E * (int)sizeof(float);
  auto kern = skinny_gemm_kernel<RT, NT, U>;
  static int configured_smem = 0;   // per instantiation
  if (smem > 32 * 1024 && smem > configured_smem) {   // dynamic + static must fit the 48 KB default
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "skinny_gemm: smem attribute");
    configured_smem = smem;
  }
  dim3 grid(p.N / (RT * 16), sc, batch);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(32 * wk, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, kern, p) == cudaSuccess, "skinny_gemm: launch failed: ", cudaGetErrorString(cudaGetLastError()));
}

template <int RT, int U>
void dispatch_nt(const Params& p, int nt, int wk, int sc, int batch, bool pdl, cudaStream_t st) {
  switch (nt) {
    case 1: launch<RT, 1, U>(p, wk, sc, batch, pdl, st); break;
    case 2: launch<RT, 2, U>(p, wk, sc, batch, pdl, st); break;
    case 3: launch<RT, 3, U>(p, wk, sc, batch, pdl, st); break;
    case 4: launch<RT, 4, U>(p, wk, sc, batch, pdl, st); break;
    default: TORCH_CHECK(false, "skinny_gemm: nt must be 1..4");
  }
}

// bench helper: a predecessor that (optionally) triggers its dependents at its top like a PDL-aware kernel,
// spins for `ns` nanoseconds (an all-reduce waiting on peers, attention, ...), then writes dst = src.
__global__ void spin_copy_kernel(uint4* dst, const uint4* src, long n16, long ns, int trigger_early) {
  if (trigger_early) pdl_trigger();
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); } while ((long)(t1 - t0) < ns);
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n16; i += (long)gridDim.x * blockDim.x) dst[i] = src[i];
}

}  // namespace

void spin_copy(torch::Tensor dst, torch::Tensor src, int64_t ns, bool trigger_early, int64_t blocks) {
  TORCH_CHECK(dst.is_contiguous() && src.is_contiguous() && dst.nbytes() == src.nbytes() && dst.nbytes() % 16 == 0);
  const at::cuda::CUDAGuard guard(dst.device());
  spin_copy_kernel<<<(int)blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint4*>(dst.data_ptr()), reinterpret_cast<const uint4*>(src.data_ptr()), dst.nbytes() / 16, ns,
      trigger_early ? 1 : 0);
}

// x [M, K] bf16, w [N, K] bf16, out [M, N] bf16, or batched x [B, M, K], w [B, N, K], out [B, M, N] (one GEMM per
// batch entry, e.g. MLA's per-head W_UV). Last dims must have unit stride; rows / batches may be strided.
// ws fp32 >= B * (N / (rt*16)) * sc * rt*nt*128 floats when sc > 1; counters int32 >= B * N / (rt*16), zeroed once.
void skinny_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                 int64_t rt, int64_t u, int64_t wk, int64_t sc, bool pdl, int64_t prefetch) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16);
  const int d = x.dim();
  TORCH_CHECK((d == 2 || d == 3) && w.dim() == d && out.dim() == d, "skinny_gemm: 2-D or batched 3-D operands");
  TORCH_CHECK(x.stride(d - 1) == 1 && w.stride(d - 1) == 1 && out.stride(d - 1) == 1, "skinny_gemm: unit last stride");
  const int B = d == 3 ? x.size(0) : 1;
  const int M = x.size(d - 2), K = x.size(d - 1), N = w.size(d - 2);
  TORCH_CHECK(w.size(d - 1) == K && out.size(d - 2) == M && out.size(d - 1) == N);
  if (d == 3) TORCH_CHECK(w.size(0) == B && out.size(0) == B);
  TORCH_CHECK(M >= 1 && M <= 32, "skinny_gemm: 1 <= M <= 32");
  TORCH_CHECK(K % 32 == 0 && N % (rt * 16) == 0, "skinny_gemm: K % 32, N % (rt*16)");
  const long ldx = x.stride(d - 2), ldw = w.stride(d - 2);
  const long xbs = d == 3 ? x.stride(0) : 0, wbs = d == 3 ? w.stride(0) : 0;
  TORCH_CHECK(ldx % 8 == 0 && ldw % 8 == 0 && xbs % 8 == 0 && wbs % 8 == 0 &&
              (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 &&
              (reinterpret_cast<uintptr_t>(w.data_ptr()) & 15) == 0, "skinny_gemm: 16-byte alignment");
  TORCH_CHECK(wk >= 1 && wk <= 8 && sc >= 1 && B <= 65535);
  const int nt = (M + 7) / 8;
  const int E = rt * nt * 128;
  const int groups = N / (rt * 16);
  if (sc > 1) {
    TORCH_CHECK(ws.dtype() == torch::kFloat32 && ws.numel() >= (long)B * groups * sc * E, "skinny_gemm: workspace");
    TORCH_CHECK(counters.dtype() == torch::kInt32 && counters.numel() >= (long)B * groups, "skinny_gemm: counters");
  }
  Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ldx = ldx;
  p.xbs = xbs;
  p.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  p.ldw = ldw;
  p.wbs = wbs;
  p.K = K;
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldo = out.stride(d - 2);
  p.obs = d == 3 ? out.stride(0) : 0;
  p.ws = sc > 1 ? ws.data_ptr<float>() : nullptr;
  p.counters = sc > 1 ? counters.data_ptr<int>() : nullptr;
  p.M = M;
  p.N = N;
  p.total_chunks = K / 32;
  const int splits = wk * sc;
  p.chunks_per_warp = (p.total_chunks + splits - 1) / splits;
  const int batch = B;
  p.prefetch = (int)prefetch;
  const at::cuda::CUDAGuard guard(x.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (rt == 1) {
    if (u == 1) dispatch_nt<1, 1>(p, nt, wk, sc, batch, pdl, st);
    else if (u == 2) dispatch_nt<1, 2>(p, nt, wk, sc, batch, pdl, st);
    else if (u == 4) dispatch_nt<1, 4>(p, nt, wk, sc, batch, pdl, st);
    else TORCH_CHECK(false, "u");
  } else if (rt == 2) {
    if (u == 1) dispatch_nt<2, 1>(p, nt, wk, sc, batch, pdl, st);
    else if (u == 2) dispatch_nt<2, 2>(p, nt, wk, sc, batch, pdl, st);
    else TORCH_CHECK(false, "u");
  } else if (rt == 4) {
    if (u == 1) dispatch_nt<4, 1>(p, nt, wk, sc, batch, pdl, st);
    else if (u == 2) dispatch_nt<4, 2>(p, nt, wk, sc, batch, pdl, st);
    else TORCH_CHECK(false, "u");
  } else {
    TORCH_CHECK(false, "skinny_gemm: rt must be 1, 2 or 4");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("skinny_gemm", &skinny_gemm, "bf16 skinny GEMM (M <= 32), in-kernel split-K reduce, optional PDL + L2 prefetch");
  m.def("spin_copy", &spin_copy, "bench helper: spin ns (optionally triggering PDL dependents first), then copy");
}
