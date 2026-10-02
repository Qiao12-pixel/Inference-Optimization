// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_CONTEXT_H_

#include <vector>

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Inference context: holds KV cache, current position, token history, and
// stats.
struct MiniLlamaContext {
  const MiniLlamaModel* model = nullptr;
  KvCache kv_cache;
  CudaKvCache cuda_kv_cache;
  // Reused by all CUDA attention layers during Decode. It is allocated lazily
  // at max-sequence capacity when the CUDA forward path is first entered.
  CudaDeviceBuffer cuda_attention_scores_workspace;
  // One scalar is sufficient because RMSNorm calls are sequential.
  CudaDeviceBuffer cuda_rms_norm_workspace;
  // Reused output buffer for GPU-side greedy sampling.
  CudaDeviceBuffer cuda_argmax_workspace;
  // Block-local candidates for the CUDA Linear + ArgMax Decode dispatcher.
  CudaDeviceBuffer cuda_linear_argmax_workspace;
  int pos = 0;

  // All tokens that have been fed through this context (including prefill).
  std::vector<int> token_history;

  // Stats
  int n_prefill_tokens = 0;
  int n_decode_tokens = 0;

  MiniLlamaContext() = default;
  MiniLlamaContext(const MiniLlamaContext&) = delete;
  MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
  MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
  MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
  explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CONTEXT_H_
