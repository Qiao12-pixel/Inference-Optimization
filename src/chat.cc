// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/chat.h"

#include <string>
#include <vector>

namespace mini_llama {

void ChatSession::Clear() {
  messages.clear();
  token_history.clear();
  prefix_cache.Clear();
  total_prompt_tokens = 0;
  total_generated_tokens = 0;
  total_time_ms = 0.0;
}

void ChatSession::AddMessage(const std::string& role,
                             const std::string& content) {
  messages.push_back({role, content});
}

void ChatSession::RecordTurn(int prompt_tokens, int generated_tokens,
                             double time_ms) {
  total_prompt_tokens += prompt_tokens;
  total_generated_tokens += generated_tokens;
  total_time_ms += time_ms;
}

void ChatSession::SetTokenHistory(const std::vector<int>& tokens) {
  token_history = tokens;
}

void ChatSession::AppendToken(int token) { token_history.push_back(token); }

void ChatSession::RecordPrefix(const std::vector<int>& tokens) {
  prefix_cache.Insert(tokens);
}

size_t ChatSession::LongestCachedPrefix(const std::vector<int>& tokens) const {
  return prefix_cache.LongestPrefix(tokens);
}

}  // namespace mini_llama
