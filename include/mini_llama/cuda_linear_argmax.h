// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_LINEAR_ARGMAX_H_
#define INCLUDE_MINI_LLAMA_CUDA_LINEAR_ARGMAX_H_

#include <cstddef>
#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/quantized_tensor.h"

namespace mini_llama {

// Common greedy Decode operator for a CUDA-resident Linear weight. Dispatches
// by weight type. F32/Q8_0/Q4_0/Q4_1 all use type-specific fused candidate
// reduction kernels. Only a token id is copied to the host and tie-breaking
// always selects the lower token id.
int CudaLinearArgMaxDeviceInput(
    const CudaTensor& x, QuantType weight_type, const void* weight_data,
    size_t weight_block_count, const std::vector<int>& weight_shape,
    CudaDeviceBuffer& candidate_workspace, CudaDeviceBuffer& result_workspace,
    int device_id = 0);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_LINEAR_ARGMAX_H_
