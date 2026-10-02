// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
#define INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool CudaAttentionBuilt();

Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache,
                           int layer, int pos, int n_heads, int n_kv_heads,
                           int head_dim, int device_id = 0);

CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache,
                                          int layer, int pos, int n_heads,
                                          int n_kv_heads, int head_dim,
                                          int device_id = 0);

// Uses caller-owned device storage for attention scores. Reusing it across
// Decode steps avoids a cudaMalloc/cudaFree pair for every layer and token.
CudaTensor CudaAttentionDecodeDeviceInput(
    const CudaTensor& q, const CudaKvCache& kv_cache,
    CudaDeviceBuffer& scores_workspace, int layer, int pos, int n_heads,
    int n_kv_heads, int head_dim, int device_id = 0);

// Causal attention for all prompt tokens in one launch grid. q has shape
// [n_tokens, n_heads * head_dim]; the KV cache already contains the matching
// contiguous range [start_pos, start_pos + n_tokens).
CudaTensor CudaAttentionPrefillDeviceInput(
    const CudaTensor& q, const CudaKvCache& kv_cache, int layer,
    int start_pos, int n_heads, int n_kv_heads, int head_dim,
    int device_id = 0);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
