// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/cuda_linear_argmax.h"

#include <stdexcept>

#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_quant.h"

namespace mini_llama {

int CudaLinearArgMaxDeviceInput(
    const CudaTensor& x, QuantType weight_type, const void* weight_data,
    size_t weight_block_count, const std::vector<int>& weight_shape,
    CudaDeviceBuffer& candidate_workspace, CudaDeviceBuffer& result_workspace,
    int device_id) {
  if (weight_data == nullptr) {
    throw std::runtime_error("CudaLinearArgMaxDeviceInput: weight is null");
  }
  switch (weight_type) {
    case QuantType::kF32:
      return CudaF32LinearArgMaxDeviceInput(
          x, weight_data, weight_shape, candidate_workspace, result_workspace,
          device_id);
    case QuantType::kQ80: {
      return CudaQ80LinearArgMaxDeviceInput(
          x, weight_data, weight_block_count, weight_shape,
          candidate_workspace, result_workspace, device_id);
    }
    case QuantType::kQ40: {
      return CudaQ40LinearArgMaxDeviceInput(
          x, weight_data, weight_block_count, weight_shape,
          candidate_workspace, result_workspace, device_id);
    }
    case QuantType::kQ41: {
      return CudaQ41LinearArgMaxDeviceInput(
          x, weight_data, weight_block_count, weight_shape,
          candidate_workspace, result_workspace, device_id);
    }
  }
  throw std::runtime_error(
      "CudaLinearArgMaxDeviceInput: unsupported weight type");
}

}  // namespace mini_llama
