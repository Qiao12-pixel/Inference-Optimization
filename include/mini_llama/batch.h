// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_BATCH_H_
#define INCLUDE_MINI_LLAMA_BATCH_H_

#include <vector>

namespace mini_llama {

// MiniBatch mirrors llama_batch: a collection of tokens with positions.
// For this educational implementation, each token maps to one position.
struct MiniBatch {
  std::vector<int> tokens;     // token ids
  std::vector<int> positions;  // position for each token

  int num_tokens() const { return static_cast<int>(tokens.size()); }

  // Build a single-token batch (Decode step).
  static MiniBatch Single(int token, int pos);

  // Build a multi-token batch from a sequence (prefill step).
  // positions start at start_pos and increment by 1.
  static MiniBatch FromTokens(const std::vector<int>& toks, int start_pos = 0);
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_BATCH_H_
