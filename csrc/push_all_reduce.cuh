// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side manager for push-based allreduce.
// Manages IPC storage, PushController initialization, and kernel launch.
// Replaces SGLang's CustomAllReduceBase + CustomAllReducePush.

#pragma once
#include "push_all_reduce_kernel.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm {
namespace push_ar {

#define PUSH_AR_CUDACHECK(cmd)                                                 \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("push_all_reduce CUDA error at ") + \
                               __FILE__ + ":" + std::to_string(__LINE__) +     \
                               " '" + cudaGetErrorString(e) + "'");            \
    }                                                                          \
  } while (0)

class PushAllReduceManager {
 public:
  PushAllReduceManager(int rank, int world_size, int64_t push_buffer_bytes,
                       int max_num_cta)
      : rank_(rank),
        world_size_(world_size),
        push_buffer_bytes_(push_buffer_bytes),
        max_num_cta_(max_num_cta),
        storage_(nullptr) {
    assert(world_size_ >= 2 && world_size_ <= 8);
    assert(max_num_cta_ > 0 && max_num_cta_ <= 512);
    assert(push_buffer_bytes_ > 0);

    // Determine PDL support from device capability
    int device_id;
    PUSH_AR_CUDACHECK(cudaGetDevice(&device_id));
    int major;
    PUSH_AR_CUDACHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device_id));
    use_pdl_ = (major >= 9);  // Hopper (sm90) or newer

    // Allocate storage
    storage_bytes_ = push_signal_bytes() + push_buffer_total_bytes();
    PUSH_AR_CUDACHECK(cudaMalloc(&storage_, storage_bytes_));
    // Zeros signals (epoch=0 for all CTAs) AND push buffer
    // (0x0000 = IEEE 754 positive-zero = "empty" sentinel)
    PUSH_AR_CUDACHECK(cudaMemset(storage_, 0, storage_bytes_));

    peer_storage_.resize(world_size_, nullptr);
  }

  ~PushAllReduceManager() {
    for (int i = 0; i < world_size_; i++) {
      if (i != rank_ && peer_storage_[i] != nullptr) {
        cudaIpcCloseMemHandle(peer_storage_[i]);
      }
    }
    if (storage_) {
      cudaFree(storage_);
    }
  }

  // Phase 1: Return IPC handle for local storage
  cudaIpcMemHandle_t get_ipc_handle() {
    cudaIpcMemHandle_t handle;
    PUSH_AR_CUDACHECK(cudaIpcGetMemHandle(&handle, storage_));
    return handle;
  }

  // Phase 2: Open peer IPC handles and init PushController
  void post_init(const std::vector<cudaIpcMemHandle_t>& peer_handles) {
    assert(peer_handles.size() == (size_t)world_size_);
    for (int i = 0; i < world_size_; i++) {
      if (i == rank_) {
        peer_storage_[i] = storage_;
      } else {
        PUSH_AR_CUDACHECK(cudaIpcOpenMemHandle(&peer_storage_[i],
                                               peer_handles[i],
                                               cudaIpcMemLazyEnablePeerAccess));
      }
    }
    // Create PushController pointing to local signal region
    push_ctrl_ = PushController(get_push_signal(storage_));
    ctrl_initialized_ = true;
  }

  // Main allreduce dispatch
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int num_elements) {
    assert(ctrl_initialized_);
    assert(num_elements > 0);

    const uint32_t num_items = static_cast<uint32_t>(num_elements);
    const int num_threads = select_num_threads<T>(num_items);

    // Verify input fits in push buffer (runtime check, not compiled out)
    const int64_t input_bytes = static_cast<int64_t>(sizeof(T)) * num_elements;
    if (input_bytes > push_buffer_bytes_) {
      throw std::runtime_error("push_all_reduce: input (" +
                               std::to_string(input_bytes) +
                               " bytes) exceeds push buffer capacity (" +
                               std::to_string(push_buffer_bytes_) + " bytes)");
    }

    // Build kernel params
    AllReducePushData params;
    for (int i = 0; i < world_size_; i++) {
      params.buffer[i] = get_push_buffer(peer_storage_[i]);
    }
    // Fill remaining buffer slots with nullptr (safety for kMaxNumGPU=8)
    for (int i = world_size_; i < (int)kMaxNumGPU; i++) {
      params.buffer[i] = nullptr;
    }
    params.input = input;
    params.output = output;
    params.rank = rank_;
    params.num_items = num_items;
    params.buffer_bytes = static_cast<uint32_t>(push_buffer_bytes_);
    params.epoch_bytes = world_size_ * params.buffer_bytes;

    // Build launch config for cudaLaunchKernelEx
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(max_num_cta_);
    config.blockDim = dim3(num_threads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    config.numAttrs = 0;
    config.attrs = attrs;

    if (use_pdl_) {
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = 1;
      config.numAttrs = 1;
    }

    // Template dispatch: world_size x pdl
    launch_kernel<T>(config, params);
  }

  void allreduce_residual_rms_norm(
      cudaStream_t stream, nv_bfloat16* input, const nv_bfloat16* residual,
      const nv_bfloat16* weight, nv_bfloat16* norm_out, int num_elements,
      int hidden_size, float epsilon, int block_threads) {
    assert(ctrl_initialized_);
    assert(num_elements > 0);
    if (hidden_size <= 0 || num_elements % hidden_size != 0 ||
        hidden_size % 8 != 0) {
      throw std::runtime_error(
          "push_allreduce_rms_norm: invalid hidden size or shape");
    }
    const int num_rows = num_elements / hidden_size;
    if (num_rows > max_num_cta_) {
      throw std::runtime_error(
          "push_allreduce_rms_norm: rows exceed allocated CTA signals");
    }
    if (block_threads != 128 && block_threads != 256 && block_threads != 512 &&
        block_threads != 1024) {
      throw std::runtime_error(
          "push_allreduce_rms_norm: threads must be 128, 256, 512 or 1024");
    }
    const int64_t input_bytes =
        static_cast<int64_t>(sizeof(nv_bfloat16)) * num_elements;
    if (input_bytes > push_buffer_bytes_) {
      throw std::runtime_error(
          "push_allreduce_rms_norm: input exceeds push buffer capacity");
    }

    AllReducePushData params;
    for (int i = 0; i < world_size_; ++i) {
      params.buffer[i] = get_push_buffer(peer_storage_[i]);
    }
    for (int i = world_size_; i < static_cast<int>(kMaxNumGPU); ++i) {
      params.buffer[i] = nullptr;
    }
    params.input = input;
    params.output = input;
    params.rank = rank_;
    params.num_items = static_cast<uint32_t>(num_elements);
    params.buffer_bytes = static_cast<uint32_t>(push_buffer_bytes_);
    params.epoch_bytes = world_size_ * params.buffer_bytes;

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_rows);
    config.blockDim = dim3(block_threads);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    config.numAttrs = 0;
    config.attrs = attrs;
    if (use_pdl_) {
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = 1;
      config.numAttrs = 1;
    }

    launch_rms_norm_kernel(config, params, residual, weight, norm_out,
                           hidden_size, epsilon);
  }

 private:
  // Template dispatch helper
  template <typename T>
  void launch_kernel(const cudaLaunchConfig_t& config,
                     const AllReducePushData& params) {
    // Dispatch on world_size and use_pdl
    if (world_size_ == 8) {
      if (use_pdl_) {
        auto kernel = all_reduce_one_shot_push_kernel<T, 8, true>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      } else {
        auto kernel = all_reduce_one_shot_push_kernel<T, 8, false>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      }
    } else if (world_size_ == 4) {
      if (use_pdl_) {
        auto kernel = all_reduce_one_shot_push_kernel<T, 4, true>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      } else {
        auto kernel = all_reduce_one_shot_push_kernel<T, 4, false>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      }
    } else if (world_size_ == 2) {
      if (use_pdl_) {
        auto kernel = all_reduce_one_shot_push_kernel<T, 2, true>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      } else {
        auto kernel = all_reduce_one_shot_push_kernel<T, 2, false>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      }
    } else if (world_size_ == 6) {
      if (use_pdl_) {
        auto kernel = all_reduce_one_shot_push_kernel<T, 6, true>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      } else {
        auto kernel = all_reduce_one_shot_push_kernel<T, 6, false>;
        cudaLaunchKernelEx(&config, kernel, params, push_ctrl_);
      }
    }
  }

  void launch_rms_norm_kernel(const cudaLaunchConfig_t& config,
                              const AllReducePushData& params,
                              const nv_bfloat16* residual,
                              const nv_bfloat16* weight,
                              nv_bfloat16* norm_out, int hidden_size,
                              float epsilon) {
#define LAUNCH_PUSH_RMS(NUM_GPU, USE_PDL)                                     \
  cudaLaunchKernelEx(                                                         \
      &config,                                                                \
      all_reduce_residual_rms_norm_push_kernel<NUM_GPU, USE_PDL>, params,     \
      push_ctrl_, residual, weight, norm_out,                                 \
      static_cast<uint32_t>(hidden_size), epsilon)
    if (world_size_ == 4) {
      if (use_pdl_) {
        LAUNCH_PUSH_RMS(4, true);
      } else {
        LAUNCH_PUSH_RMS(4, false);
      }
    } else if (world_size_ == 2) {
      if (use_pdl_) {
        LAUNCH_PUSH_RMS(2, true);
      } else {
        LAUNCH_PUSH_RMS(2, false);
      }
    } else if (world_size_ == 6) {
      if (use_pdl_) {
        LAUNCH_PUSH_RMS(6, true);
      } else {
        LAUNCH_PUSH_RMS(6, false);
      }
    } else if (world_size_ == 8) {
      if (use_pdl_) {
        LAUNCH_PUSH_RMS(8, true);
      } else {
        LAUNCH_PUSH_RMS(8, false);
      }
    }
#undef LAUNCH_PUSH_RMS
  }

  // Thread count selection (from SGLang CustomAllReducePush::all_reduce)
  template <typename T>
  int select_num_threads(uint32_t num_items) const {
    constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
    for (const auto t : {128u, 256u, 512u}) {
      if (t * max_num_cta_ * 2 * kVecSize >= num_items) {
        return static_cast<int>(t);
      }
    }
    return 1024;
  }

  // Storage layout helpers
  int64_t push_signal_bytes() const {
    return align128(sizeof(uint32_t) * max_num_cta_);
  }

  int64_t push_buffer_total_bytes() const {
    return align128(PushController::kNumStages * world_size_ *
                    push_buffer_bytes_);
  }

  void* get_push_signal(void* base) const {
    return base;  // signals start at offset 0
  }

  void* get_push_buffer(void* base) const {
    return static_cast<char*>(base) + push_signal_bytes();
  }

  static int64_t align128(int64_t size) { return ((size + 127) / 128) * 128; }

  // Members
  int rank_;
  int world_size_;
  int64_t push_buffer_bytes_;
  int max_num_cta_;
  bool use_pdl_;

  void* storage_;
  int64_t storage_bytes_;
  std::vector<void*> peer_storage_;

  PushController push_ctrl_;
  bool ctrl_initialized_ = false;
};

}  // namespace push_ar
}  // namespace vllm
