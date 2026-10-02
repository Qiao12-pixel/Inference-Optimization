// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/batch.h"

#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/model.h"

namespace mini_llama {

MiniBatch MiniBatch::Single(int token, int pos) {
  MiniBatch batch;
  batch.tokens.push_back(token);
  batch.positions.push_back(pos);
  return batch;
}

MiniBatch MiniBatch::FromTokens(const std::vector<int>& toks, int start_pos) {
  MiniBatch batch;
  batch.tokens = toks;
  batch.positions.reserve(toks.size());
  for (size_t i = 0; i < toks.size(); ++i) {
    batch.positions.push_back(start_pos + static_cast<int>(i));
  }
  return batch;
}

// ---------------------------------------------------------------------------
// ForwardBatch
// ---------------------------------------------------------------------------
// Processes all tokens in the batch sequentially.
// Returns logits for the *last* token in the batch.
// This unifies prefill (multi-token) and Decode (single-token) paths.
// ---------------------------------------------------------------------------
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch, CudaTensor* device_logits) {
  if (batch.num_tokens() == 0) {
    throw std::runtime_error("ForwardBatch: empty batch");
  }
  if (batch.tokens.size() != batch.positions.size()) {
    throw std::runtime_error(
        "ForwardBatch: tokens and positions size mismatch");
  }

  if (model.cuda_weights) {
    bool contiguous = true;
    for (int i = 1; i < batch.num_tokens(); ++i) {
      if (batch.positions[i] != batch.positions[0] + i) {
        contiguous = false;
        break;
      }
    }
    if (batch.num_tokens() > 1 && contiguous) {
      CudaTensor last_logits = ForwardPrefillDevice(ctx, model, batch);
      ctx.pos = batch.positions.back();
      ctx.token_history.insert(ctx.token_history.end(), batch.tokens.begin(),
                               batch.tokens.end());
      if (device_logits != nullptr) {
        *device_logits = std::move(last_logits);
        return Tensor();
      }
      Tensor logits = last_logits.Download();
      RecordModelCudaDeviceToHostCopy(model, logits.size() * sizeof(float));
      return logits;
    }
    CudaTensor last_logits;
    for (int i = 0; i < batch.num_tokens(); ++i) {
      ctx.pos = batch.positions[i];
      // The logits for preceding prefill tokens are unused. Keeping them on
      // device avoids downloading a full vocabulary vector per input token.
      (void)ForwardToken(ctx, model, batch.tokens[i], &last_logits);
      ctx.token_history.push_back(batch.tokens[i]);
    }
    if (device_logits != nullptr) {
      *device_logits = std::move(last_logits);
      return Tensor();
    }
    Tensor logits = last_logits.Download();
    RecordModelCudaDeviceToHostCopy(model, logits.size() * sizeof(float));
    return logits;
  }

  if (device_logits != nullptr) {
    throw std::runtime_error(
        "ForwardBatch: device logits requested for a CPU model");
  }

  Tensor logits;
  for (int i = 0; i < batch.num_tokens(); ++i) {
    ctx.pos = batch.positions[i];
    logits = ForwardToken(ctx, model, batch.tokens[i]);
    ctx.token_history.push_back(batch.tokens[i]);
  }
  return logits;
}

}  // namespace mini_llama
