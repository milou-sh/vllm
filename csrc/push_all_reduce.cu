// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// C++ bridge functions for push-based allreduce.
// Exposes PushAllReduceManager to Python via torch custom ops.

#include "push_all_reduce.cuh"
#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstddef>
#include <cstring>

using fptr_t = int64_t;
using namespace vllm::push_ar;

fptr_t init_push_ar(int64_t rank, int64_t world_size, int64_t push_buffer_bytes,
                    int64_t max_num_cta) {
  auto* mgr = new PushAllReduceManager(
      static_cast<int>(rank), static_cast<int>(world_size), push_buffer_bytes,
      static_cast<int>(max_num_cta));
  return reinterpret_cast<fptr_t>(mgr);
}

torch::stable::Tensor get_push_ar_ipc_handle(fptr_t _mgr) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  cudaIpcMemHandle_t handle = mgr->get_ipc_handle();
  auto t = torch::stable::empty(
      {static_cast<int64_t>(sizeof(handle))},
      torch::headeronly::ScalarType::Byte, std::nullopt,
      torch::stable::Device(torch::stable::DeviceType::CPU));
  std::memcpy(t.mutable_data_ptr(), &handle, sizeof(handle));
  return t;
}

void post_init_push_ar(fptr_t _mgr,
                       const torch::stable::Tensor& all_handles) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  STD_TORCH_CHECK(all_handles.dim() == 2);
  STD_TORCH_CHECK(all_handles.scalar_type() ==
                  torch::headeronly::ScalarType::Byte);
  STD_TORCH_CHECK(all_handles.size(1) == sizeof(cudaIpcMemHandle_t));
  int world_size = all_handles.size(0);
  std::vector<cudaIpcMemHandle_t> handles(world_size);
  const auto* bytes =
      static_cast<const std::byte*>(all_handles.const_data_ptr());
  for (int i = 0; i < world_size; i++) {
    std::memcpy(&handles[i], bytes + i * sizeof(cudaIpcMemHandle_t),
                sizeof(cudaIpcMemHandle_t));
  }
  mgr->post_init(handles);
}

static bool is_weak_contiguous(torch::stable::Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return true;
  }
  int64_t storage_nbytes = 0;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_storage_size(tensor.get(), &storage_nbytes));
  return storage_nbytes - tensor.storage_offset() * tensor.element_size() ==
         tensor.numel() * tensor.element_size();
}

void push_ar_all_reduce(fptr_t _mgr, torch::stable::Tensor& inp,
                        torch::stable::Tensor& out) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK(inp.scalar_type() == out.scalar_type());
  STD_TORCH_CHECK(inp.numel() == out.numel());
  STD_TORCH_CHECK(is_weak_contiguous(inp), "Input must be contiguous");
  STD_TORCH_CHECK(is_weak_contiguous(out), "Output must be contiguous");

  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::BFloat16:
      mgr->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.mutable_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), out.numel());
      break;
    case torch::headeronly::ScalarType::Half:
      mgr->allreduce<half>(stream,
                           reinterpret_cast<half*>(inp.mutable_data_ptr()),
                           reinterpret_cast<half*>(out.mutable_data_ptr()),
                           out.numel());
      break;
    case torch::headeronly::ScalarType::Float:
      mgr->allreduce<float>(
          stream, reinterpret_cast<float*>(inp.mutable_data_ptr()),
          reinterpret_cast<float*>(out.mutable_data_ptr()),
          out.numel());
      break;
    default:
      throw std::runtime_error(
          "push allreduce only supports float32, float16 and bfloat16");
  }
}

void push_ar_residual_rms_norm(fptr_t _mgr, torch::stable::Tensor& input,
                               const torch::stable::Tensor& residual,
                               const torch::stable::Tensor& weight,
                               double epsilon,
                               torch::stable::Tensor& norm_out,
                               int64_t block_threads) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  STD_TORCH_CHECK(input.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);
  STD_TORCH_CHECK(residual.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(norm_out.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(input.dim() == 2);
  STD_TORCH_CHECK(residual.dim() == 2);
  STD_TORCH_CHECK(norm_out.dim() == 2);
  STD_TORCH_CHECK(weight.dim() == 1);
  STD_TORCH_CHECK(input.numel() == residual.numel());
  STD_TORCH_CHECK(input.numel() == norm_out.numel());
  STD_TORCH_CHECK(input.size(1) == weight.size(0));
  STD_TORCH_CHECK(is_weak_contiguous(input));
  STD_TORCH_CHECK(residual.is_contiguous());
  STD_TORCH_CHECK(weight.is_contiguous());
  STD_TORCH_CHECK(norm_out.is_contiguous());

  mgr->allreduce_residual_rms_norm(
      stream, reinterpret_cast<nv_bfloat16*>(input.mutable_data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(residual.const_data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(weight.const_data_ptr()),
      reinterpret_cast<nv_bfloat16*>(norm_out.mutable_data_ptr()),
      input.numel(), input.size(1), static_cast<float>(epsilon),
      static_cast<int>(block_threads));
}

void dispose_push_ar(fptr_t _mgr) {
  auto* mgr = reinterpret_cast<PushAllReduceManager*>(_mgr);
  delete mgr;
}
