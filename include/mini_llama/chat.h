// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CHAT_H_
#define INCLUDE_MINI_LLAMA_CHAT_H_

#include <string>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/radix_tree.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

struct ChatMessage {
  std::string role;  // "system", "user", "assistant"
  std::string content;
};

// Manages the state of an interactive chat session.
class ChatSession {
 public:
  std::vector<ChatMessage> messages;
  std::vector<int> token_history;
  RadixTree prefix_cache;
  SamplingParams sampling_params;

  // Stats
  int total_prompt_tokens = 0;
  int total_generated_tokens = 0;
  double total_time_ms = 0.0;

  ChatSession() = default;

  // Reset messages and stats.
  void Clear();

  // Add a message to the history.
  void AddMessage(const std::string& role, const std::string& content);

  // Increment stats after a turn.
  void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);

  // Replace the current token history for the latest evaluated context.
  void SetTokenHistory(const std::vector<int>& tokens);

  // Append one generated token to the tracked context.
  void AppendToken(int token);

  // Record a token sequence whose prefix has been evaluated.
  void RecordPrefix(const std::vector<int>& tokens);

  // Return the longest cached prefix for a token sequence.
  size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CHAT_H_
