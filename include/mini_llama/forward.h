// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_FORWARD_H_
#define INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Forward pass for a single token
// Returns logits: [vocab_size]
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token, CudaTensor* device_logits = nullptr);

// CUDA greedy Decode fast path for F32 lm_head models. It returns false when
// the model/path is unsupported, allowing callers to retain the regular
// device-logits + sampler path for non-greedy sampling and other weight types.
bool ForwardTokenGreedyCuda(MiniLlamaContext& ctx,
                            const MiniLlamaModel& model, int token,
                            int& next_token);

// Forward pass for a batch of tokens.
// Internally processes tokens sequentially and returns logits for the last
// token. This unifies prefill (multi-token) and Decode (single-token) paths.
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch,
                    CudaTensor* device_logits = nullptr);

// CUDA-only sequence Prefill. batch positions must be contiguous and the
// result is the last prompt token's logits kept on device.
CudaTensor ForwardPrefillDevice(MiniLlamaContext& ctx,
                                const MiniLlamaModel& model,
                                const MiniBatch& batch);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_FORWARD_H_
