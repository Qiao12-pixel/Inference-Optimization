// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/tokenizer.h"

#include <string>
#include <vector>

namespace mini_llama {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.push_back(bos_id());
  for (unsigned char c : text) {
    int id = static_cast<int>(c);
    if (id >= kVocabSize) {
      tokens.push_back(unk_id());
    } else {
      tokens.push_back(id);
    }
  }
  return tokens;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
  if (token == bos_id()) {
    return "<bos>";
  }
  if (token == eos_id()) {
    return "<eos>";
  }
  if (token == unk_id()) {
    return "<unk>";
  }
  if (token >= kVocabSize) {
    return "<unk>";
  }
  if (token < 32 || token == 127) {
    // Control characters: represent as empty
    return "";
  }
  return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

}  // namespace mini_llama
