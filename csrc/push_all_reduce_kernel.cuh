// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Push-based 2-buffer allreduce kernel, ported from SGLang's
// all_reduce_one_shot_push_kernel. Uses epoch-based double-buffered
// protocol with positive-zero sentinel for data arrival detection.
//
// Original source:
//   sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh
// Protocol reference: SGLang commit edb1b3f
//
// Changes from SGLang:
//   - All code placed in namespace vllm::push_ar
//   - SGL_DEVICE macros replaced with __device__ __forceinline__
//   - SGL_CUDA_ARCH replaced with __CUDA_ARCH__
//   - std::integral (C++20) replaced with explicit overloads (C++17)
//   - kMaxVecBytes hardcoded to 16 (push kernel uses 16-byte vectors)
//   - TVM/FFI dependencies removed; file is self-contained

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vllm {
namespace push_ar {

// ============================================================
// Section A: Type aliases (from SGLang utils.cuh lines 50-75)
// ============================================================
using fp32_t = float;
using fp16_t = __half;
using bf16_t = __nv_bfloat16;
using fp32x2_t = float2;
using fp16x2_t = __half2;
using bf16x2_t = __nv_bfloat162;

static constexpr uint32_t kWarpThreads = 32u;

// ============================================================
// Section B: kMaxVecBytes (from SGLang utils.cuh line 112)
// ============================================================
// Hardcoded to 16 since the push kernel uses 16-byte vectors.
// The kernel's kVecSize = 16 / (sizeof(DType) * 2) yields
// AlignedVector<packed_t<DType>, kVecSize> = 16 bytes always.
inline constexpr std::size_t kMaxVecBytes = 16;

// ============================================================
// Section C: PDL helpers (from SGLang utils.cuh lines 119-148)
// ============================================================
// CHANGED: SGL_ARCH_HOPPER_OR_GREATER -> __CUDA_ARCH__ >= 900
template <bool kUsePDL>
__device__ __forceinline__ void PDLWaitPrimary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif
}

template <bool kUsePDL>
__device__ __forceinline__ void PDLTriggerSecondary() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
#endif
}

// ============================================================
// Section D: Pointer offset helpers (from SGLang utils.cuh 181-195)
// ============================================================
// CHANGED: Removed std::integral (C++20) constraint.
// Use explicit overloads for 1 and 2 offsets (C++17 compatible).

// Byte-level offset (replaces pointer::offset<char>)
__device__ __forceinline__ void* ptr_byte_offset(void* ptr, int64_t off1) {
  return static_cast<char*>(ptr) + off1;
}

__device__ __forceinline__ void* ptr_byte_offset(void* ptr, int64_t off1,
                                                 int64_t off2) {
  return static_cast<char*>(ptr) + off1 + off2;
}

// Typed offset for AlignedVector load/store addressing
// (replaces pointer::offset<T>)
template <typename T>
__device__ __forceinline__ void* ptr_typed_offset(void* ptr, int64_t offset) {
  return static_cast<T*>(ptr) + offset;
}

template <typename T>
__device__ __forceinline__ const void* ptr_typed_offset(const void* ptr,
                                                        int64_t offset) {
  return static_cast<const T*>(ptr) + offset;
}

// Host-side pointer offset (for storage layout calculations)
inline void* host_ptr_offset(void* ptr, int64_t off) {
  return static_cast<char*>(ptr) + off;
}

// ============================================================
// Section E: dtype_trait system (from SGLang type.cuh)
// ============================================================
// COPY AS IS with namespace adjustment.

template <typename T>
struct dtype_trait {};

template <>
struct dtype_trait<fp32_t> {
  using self_t = fp32_t;
  using packed_t = fp32x2_t;
  template <typename S>
  __device__ __forceinline__ static self_t from(const S& value) {
    return static_cast<fp32_t>(value);
  }
  __device__ __forceinline__ static self_t from(const fp16_t& x) {
    return __half2float(x);
  }
  __device__ __forceinline__ static self_t from(const bf16_t& x) {
    return __bfloat162float(x);
  }
};

template <>
struct dtype_trait<fp16_t> {
  using self_t = fp16_t;
  using packed_t = fp16x2_t;
  template <typename S>
  __device__ __forceinline__ static self_t from(const S& value) {
    return static_cast<fp16_t>(value);
  }
};

template <>
struct dtype_trait<bf16_t> {
  using self_t = bf16_t;
  using packed_t = bf16x2_t;
  template <typename S>
  __device__ __forceinline__ static self_t from(const S& value) {
    return static_cast<bf16_t>(value);
  }
};

template <>
struct dtype_trait<fp32x2_t> {
  using self_t = fp32x2_t;
  template <typename S>
  __device__ __forceinline__ static self_t from(const S& value) {
    return static_cast<fp32x2_t>(value);
  }
  __device__ __forceinline__ static self_t from(const fp16x2_t& x) {
    return __half22float2(x);
  }
  __device__ __forceinline__ static self_t from(const bf16x2_t& x) {
    return __bfloat1622float2(x);
  }
};

template <>
struct dtype_trait<fp16x2_t> {
  using self_t = fp16x2_t;
  template <typename S>
  __device__ __forceinline__ static self_t from(const S& value) {
    return static_cast<fp16x2_t>(value);
  }
  __device__ __forceinline__ static self_t from(const fp32x2_t& x) {
    return __float22half2_rn(x);
  }
};

template <>
struct dtype_trait<bf16x2_t> {
  using self_t = bf16x2_t;
  template <typename S>
  __device__ __forceinline__ static self_t from(const S& value) {
    return static_cast<bf16x2_t>(value);
  }
  __device__ __forceinline__ static self_t from(const fp32x2_t& x) {
    return __float22bfloat162_rn(x);
  }
};

template <typename T>
using packed_t = typename dtype_trait<T>::packed_t;

template <typename To, typename From>
__device__ __forceinline__ To cast(const From& value) {
  return dtype_trait<To>::from(value);
}

// ============================================================
// Section F: AlignedVector (from SGLang vec.cuh lines 73-116)
// ============================================================
// COPY AS IS with SGL_DEVICE -> __device__ __forceinline__
// and kMaxVecBytes = 16.

namespace detail {

template <std::size_t N>
struct uint_trait {};
template <>
struct uint_trait<1> {
  using type = uint8_t;
};
template <>
struct uint_trait<2> {
  using type = uint16_t;
};
template <>
struct uint_trait<4> {
  using type = uint32_t;
};
template <>
struct uint_trait<8> {
  using type = uint64_t;
};

template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}  // namespace detail

template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

template <typename T, std::size_t N>
struct AlignedVector {
 private:
  static_assert((N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= kMaxVecBytes,
                "CUDA vector size exceeds arch limit (max 16 bytes)");
  using element_t = typename detail::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  __device__ __forceinline__ void load(const void* ptr, int64_t offset = 0) {
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  __device__ __forceinline__ void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
  __device__ __forceinline__ void fill(T value) {
    const auto store_value = *reinterpret_cast<element_t*>(&value);
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      m_storage.data[i] = store_value;
    }
  }
  __device__ __forceinline__ auto operator[](std::size_t idx) -> T& {
    return reinterpret_cast<T*>(&m_storage)[idx];
  }
  __device__ __forceinline__ auto operator[](std::size_t idx) const -> T {
    return reinterpret_cast<const T*>(&m_storage)[idx];
  }

 private:
  storage_t m_storage;
};

// ============================================================
// Section G: PushController (from SGLang common.cuh lines 93-118)
// ============================================================
// COPY AS IS with SGL_DEVICE -> __device__ __forceinline__

static constexpr uint32_t kMaxNumGPU = 8;

struct PushController {
  using SignalType = uint32_t;
  static constexpr int64_t kNumStages = 2;  // double-buffered epochs

  PushController() : m_local_signal(nullptr) {}

  PushController(void* ptr) : m_local_signal(static_cast<SignalType*>(ptr)) {}

  __device__ __forceinline__ SignalType epoch() const {
    return m_local_signal[blockIdx.x];
  }

  __device__ __forceinline__ void exit() const {
    __syncthreads();
    if (threadIdx.x == 0) {
      exit_unsafe(blockIdx.x);
    }
  }

  __device__ __forceinline__ void exit_unsafe(uint32_t which) const {
    auto& signal = m_local_signal[which];
    signal = (signal + 1) % kNumStages;
  }

  SignalType* m_local_signal;
};

// ============================================================
// Section H: AllReducePushData (from SGLang
//   custom_all_reduce_push.cuh 23-31)
// ============================================================
// COPY AS IS.

struct AllReducePushData {
  void* __restrict__ buffer[kMaxNumGPU];
  const void* input;
  void* output;
  uint32_t rank;
  uint32_t num_items;
  uint32_t buffer_bytes;
  uint32_t epoch_bytes;
};

// ============================================================
// Section I: fp_trait sentinel types (from SGLang push.cuh 35-64)
// ============================================================
// COPY AS IS.

template <typename T>
struct fp_trait {};

template <>
struct fp_trait<bf16_t> {
  using type = uint16_t;
  [[maybe_unused]] static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]] static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<fp16_t> {
  using type = uint16_t;
  [[maybe_unused]] static constexpr uint16_t pos_zero = 0x0000u;
  [[maybe_unused]] static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  [[maybe_unused]] static constexpr uint32_t pos_zero = 0x00000000u;
  [[maybe_unused]] static constexpr uint32_t neg_zero = 0x80000000u;
};

// ============================================================
// Section J: Sentinel helpers (from SGLang push.cuh 66-84)
// ============================================================
// COPY AS IS.

template <typename DType>
__device__ __forceinline__ void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) *ptr = Trait::neg_zero;
}

template <typename DType>
__device__ __forceinline__ bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
__device__ __forceinline__ DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}

// ============================================================
// Section K: Volatile 16-byte load/store (from SGLang push.cuh 87-105)
// ============================================================
// CHANGED: pointer::offset<T> -> ptr_typed_offset<T>

template <typename T>
__device__ __forceinline__ void ld_global_volatile_16B(T& x, const void* addr,
                                                       int64_t offset) {
  static_assert(alignof(T) == 16 && sizeof(T) == 16);
  addr = ptr_typed_offset<T>(addr, offset);
  uint4 val;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  x = *reinterpret_cast<const T*>(&val);
}

template <typename T>
__device__ __forceinline__ void st_global_volatile_16B(const T& x, void* addr,
                                                       int64_t offset) {
  static_assert(alignof(T) == 16 && sizeof(T) == 16);
  const uint4 val = *reinterpret_cast<const uint4*>(&x);
  addr = ptr_typed_offset<T>(addr, offset);
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x),
               "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

// ============================================================
// Section L: reduce_impl (from SGLang custom_all_reduce.cuh 331-354)
// ============================================================
// COPY AS IS.

template <typename DType2, size_t N, uint32_t M>
__device__ __forceinline__ auto reduce_impl(
    AlignedVector<DType2, N> (&storage)[M]) -> AlignedVector<DType2, N> {
  fp32x2_t acc[N] = {};
#pragma unroll
  for (uint32_t i = 0; i < M; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < N; ++j) {
      const auto [x, y] = cast<fp32x2_t>(storage[i][j]);
      auto& [x_acc, y_acc] = acc[j];
      x_acc += x;
      y_acc += y;
    }
  }
  AlignedVector<DType2, N> result;
#pragma unroll
  for (uint32_t j = 0; j < N; ++j) {
    result[j] = cast<DType2>(acc[j]);
  }
  return result;
}

// ============================================================
// Section M: push_impl (from SGLang push.cuh 107-128)
// ============================================================
// CHANGED: pointer::offset -> ptr_byte_offset

template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void push_impl(DType* (&push_buf)[kNumGPU],
                                          const void* data,
                                          uint32_t num_items) {
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage vec;
    vec.load(data, offset);
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      clear_pos_zero(vec[j].x);
      clear_pos_zero(vec[j].y);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      st_global_volatile_16B(vec, push_buf[i], offset);
    }
  }
}

// ============================================================
// Section N: poll_impl (from SGLang push.cuh 130-165)
// ============================================================
// COPY AS IS.

template <typename DType, uint32_t kNumGPU>
__device__ __forceinline__ void poll_impl(DType* (&poll_buf)[kNumGPU],
                                          void* data, uint32_t num_items) {
  constexpr uint32_t kVecSize = 16 / (sizeof(DType) * 2);
  using Storage = AlignedVector<packed_t<DType>, kVecSize>;

  for (auto i = blockIdx.x;; i += gridDim.x) {
    const auto offset = i * blockDim.x + threadIdx.x;
    if (offset * kVecSize * 2 >= num_items) break;
    Storage storage[kNumGPU];

    while (true) {
      bool has_pos_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kNumGPU; ++i) {
        ld_global_volatile_16B(storage[i], poll_buf[i], offset);
#pragma unroll
        for (auto j = 0; j < kVecSize; ++j) {
          has_pos_zero |= is_pos_zero(storage[i][j].x);
          has_pos_zero |= is_pos_zero(storage[i][j].y);
        }
      }
      if (!has_pos_zero) break;
    }

    const Storage result = reduce_impl(storage);
    result.store(data, offset);

    Storage pos_zeros;
    pos_zeros.fill({get_pos_zero<DType>(), get_pos_zero<DType>()});
#pragma unroll
    for (uint32_t i = 0; i < kNumGPU; ++i) {
      pos_zeros.store(poll_buf[i], offset);
    }
  }
}

// ============================================================
// Section O: THE KERNEL (from SGLang push.cuh 167-196)
// ============================================================
// COPY AS IS. CHANGED: CUSTOM_AR_KERNEL macro expanded.
// The kernel uses __grid_constant__ for params passed by value.
// cudaLaunchKernelEx copies params to constant memory before launch.

template <typename DType, uint32_t kNumGPU, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void all_reduce_one_shot_push_kernel(
    const AllReducePushData __grid_constant__ params,
    const PushController __grid_constant__ ctrl) {
  const auto [buffer, input, output, rank, num_items, buffer_bytes,
              epoch_bytes] = params;

  PDLWaitPrimary<kUsePDL>();

  // Phase 1: Push data from input to all ranks' push buffers
  const auto epoch_offset = ctrl.epoch() * epoch_bytes;
  DType* push_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    push_buf[i] = static_cast<DType*>(
        ptr_byte_offset(buffer[i], rank * buffer_bytes, epoch_offset));
  }
  push_impl(push_buf, input, num_items);

  PDLTriggerSecondary<kUsePDL>();

  // Phase 2: Poll local buffer, reduce, write output, reset
  DType* poll_buf[kNumGPU];
#pragma unroll
  for (uint32_t i = 0; i < kNumGPU; ++i) {
    poll_buf[i] = static_cast<DType*>(
        ptr_byte_offset(buffer[rank], i * buffer_bytes, epoch_offset));
  }
  poll_impl(poll_buf, output, num_items);
  ctrl.exit();
}

// GLM/Hopper specialization: fuse the residual add and RMSNorm into the
// push-allreduce poll phase. One CTA owns one complete token row, which makes
// the RMS reduction local to a block and avoids a second kernel launch. The
// partial all-reduce result is first rounded to BF16, then the residual add is
// rounded to BF16 before variance accumulation, matching vLLM's existing
// fused_add_rms_norm semantics.
template <uint32_t kNumGPU, bool kUsePDL>
__global__ __launch_bounds__(1024, 1)
    void all_reduce_residual_rms_norm_push_kernel(
        const AllReducePushData __grid_constant__ params,
        const PushController __grid_constant__ ctrl,
        const bf16_t* __restrict__ residual,
        const bf16_t* __restrict__ weight, bf16_t* __restrict__ norm_out,
        const uint32_t hidden_size, const float epsilon) {
  constexpr uint32_t kVecSize = 16 / (sizeof(bf16_t) * 2);
  using Storage = AlignedVector<packed_t<bf16_t>, kVecSize>;

  const auto [buffer, input, residual_out, rank, num_items, buffer_bytes,
              epoch_bytes] = params;
  const uint32_t row = blockIdx.x;
  const uint32_t vectors_per_row = hidden_size / (kVecSize * 2);
  const uint32_t row_vector_base = row * vectors_per_row;

  PDLWaitPrimary<kUsePDL>();

  const auto epoch_offset = ctrl.epoch() * epoch_bytes;
  bf16_t* push_buf[kNumGPU];
#pragma unroll
  for (uint32_t gpu = 0; gpu < kNumGPU; ++gpu) {
    push_buf[gpu] = static_cast<bf16_t*>(
        ptr_byte_offset(buffer[gpu], rank * buffer_bytes, epoch_offset));
  }

  for (uint32_t vec = threadIdx.x; vec < vectors_per_row;
       vec += blockDim.x) {
    const uint32_t offset = row_vector_base + vec;
    Storage value;
    value.load(input, offset);
#pragma unroll
    for (uint32_t item = 0; item < kVecSize; ++item) {
      clear_pos_zero(value[item].x);
      clear_pos_zero(value[item].y);
    }
#pragma unroll
    for (uint32_t gpu = 0; gpu < kNumGPU; ++gpu) {
      st_global_volatile_16B(value, push_buf[gpu], offset);
    }
  }
  __syncthreads();
  PDLTriggerSecondary<kUsePDL>();

  bf16_t* poll_buf[kNumGPU];
#pragma unroll
  for (uint32_t gpu = 0; gpu < kNumGPU; ++gpu) {
    poll_buf[gpu] = static_cast<bf16_t*>(
        ptr_byte_offset(buffer[rank], gpu * buffer_bytes, epoch_offset));
  }

  float variance = 0.0f;
  for (uint32_t vec = threadIdx.x; vec < vectors_per_row;
       vec += blockDim.x) {
    const uint32_t offset = row_vector_base + vec;
    Storage rank_values[kNumGPU];
    while (true) {
      bool waiting = false;
#pragma unroll
      for (uint32_t gpu = 0; gpu < kNumGPU; ++gpu) {
        ld_global_volatile_16B(rank_values[gpu], poll_buf[gpu], offset);
#pragma unroll
        for (uint32_t item = 0; item < kVecSize; ++item) {
          waiting |= is_pos_zero(rank_values[gpu][item].x);
          waiting |= is_pos_zero(rank_values[gpu][item].y);
        }
      }
      if (!waiting) break;
    }

    const Storage reduced = reduce_impl(rank_values);
    Storage residual_value;
    residual_value.load(residual, offset);
    Storage updated;
#pragma unroll
    for (uint32_t item = 0; item < kVecSize; ++item) {
      const fp32x2_t reduced_f32 = cast<fp32x2_t>(reduced[item]);
      const fp32x2_t residual_f32 = cast<fp32x2_t>(residual_value[item]);
      const fp32x2_t sum = {reduced_f32.x + residual_f32.x,
                            reduced_f32.y + residual_f32.y};
      updated[item] = cast<bf16x2_t>(sum);
      const fp32x2_t rounded = cast<fp32x2_t>(updated[item]);
      variance += rounded.x * rounded.x + rounded.y * rounded.y;
    }
    updated.store(residual_out, offset);

    Storage positive_zeros;
    positive_zeros.fill({get_pos_zero<bf16_t>(), get_pos_zero<bf16_t>()});
#pragma unroll
    for (uint32_t gpu = 0; gpu < kNumGPU; ++gpu) {
      positive_zeros.store(poll_buf[gpu], offset);
    }
  }

  // Block-wide FP32 sum. The maximum supported block size is 1024 threads.
  const uint32_t lane = threadIdx.x & (kWarpThreads - 1);
  const uint32_t warp = threadIdx.x / kWarpThreads;
#pragma unroll
  for (uint32_t delta = kWarpThreads / 2; delta > 0; delta /= 2) {
    variance += __shfl_down_sync(0xffffffff, variance, delta);
  }
  __shared__ float warp_sums[32];
  __shared__ float inverse_rms;
  if (lane == 0) warp_sums[warp] = variance;
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < (blockDim.x + kWarpThreads - 1) / kWarpThreads
                          ? warp_sums[lane]
                          : 0.0f;
#pragma unroll
    for (uint32_t delta = kWarpThreads / 2; delta > 0; delta /= 2) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, delta);
    }
    if (lane == 0) {
      inverse_rms = rsqrtf(block_sum / static_cast<float>(hidden_size) +
                           epsilon);
    }
  }
  __syncthreads();

  for (uint32_t vec = threadIdx.x; vec < vectors_per_row;
       vec += blockDim.x) {
    const uint32_t offset = row_vector_base + vec;
    Storage updated;
    Storage gamma;
    updated.load(residual_out, offset);
    gamma.load(weight, vec);
    Storage normalized;
#pragma unroll
    for (uint32_t item = 0; item < kVecSize; ++item) {
      const fp32x2_t value_f32 = cast<fp32x2_t>(updated[item]);
      const fp32x2_t gamma_f32 = cast<fp32x2_t>(gamma[item]);
      const fp32x2_t norm_f32 = {
          value_f32.x * inverse_rms * gamma_f32.x,
          value_f32.y * inverse_rms * gamma_f32.y,
      };
      normalized[item] = cast<bf16x2_t>(norm_f32);
    }
    normalized.store(norm_out, offset);
  }
  ctrl.exit();
}

}  // namespace push_ar
}  // namespace vllm
