// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_TERMINAL_H_
#define INCLUDE_MINI_LLAMA_TERMINAL_H_

#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"

namespace mini_llama {

// Handles interactive terminal I/O for the chat mode.
class Terminal {
 public:
  Terminal() = default;

  // Print the user prompt (e.g., "mini-llama > ").
  void PrintUserPrompt() const;

  // Read a line from stdin. Returns empty string on EOF.
  std::string ReadLine() const;

  // Print assistant prefix (e.g., "assistant > ").
  void PrintAssistantPrefix() const;

  // Print a single decoded token (no newline, no flush).
  void PrintTokenText(const std::string& text) const;

  // Flush stdout.
  void Flush() const;

  // Print a newline.
  void NewLine() const;

  // Print the /help text.
  void PrintHelp() const;

  // Print session stats.
  void PrintStats(const ChatSession& session) const;

  // Print current sampling params.
  void PrintParams(const SamplingParams& params) const;

  // Print a generic message.
  void PrintMessage(const std::string& msg) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_TERMINAL_H_
