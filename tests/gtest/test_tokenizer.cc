// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mini_llama/tokenizer.h"
#include "tests/test_names.h"

namespace {

struct TempBpeFiles {
  std::filesystem::path dir;
  std::filesystem::path vocab;
  std::filesystem::path merges;
  std::filesystem::path special;
};

static TempBpeFiles WriteTempBpeFiles() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  TempBpeFiles files;
  files.dir = std::filesystem::temp_directory_path() /
              ("mini_llama_bpe_tokenizer_gtest_" + std::to_string(stamp));
  std::filesystem::create_directories(files.dir);
  files.vocab = files.dir / "vocab.json";
  files.merges = files.dir / "merges.txt";
  files.special = files.dir / "special_tokens.json";

  {
    std::ofstream out(files.vocab);
    out << "{\n"
        << "  \"h\": 0,\n"
        << "  \"e\": 1,\n"
        << "  \"l\": 2,\n"
        << "  \"o\": 3,\n"
        << "  \"he\": 4,\n"
        << "  \"hel\": 5,\n"
        << "  \"hell\": 6,\n"
        << "  \"hello\": 7,\n"
        << "  \"<|im_start|>\": 8,\n"
        << "  \"<unk>\": 9,\n"
        << "  \"<bos>\": 10,\n"
        << "  \"<eos>\": 11\n"
        << "}\n";
  }
  {
    std::ofstream out(files.merges);
    out << "h e\n"
        << "he l\n"
        << "hel l\n"
        << "hell o\n";
  }
  {
    std::ofstream out(files.special);
    out << "{ \"bos_id\": 10, \"eos_id\": 11, \"unk_id\": 9 }\n";
  }
  return files;
}

static void RemoveTempBpeFiles(const TempBpeFiles& files) {
  std::error_code ec;
  std::filesystem::remove_all(files.dir, ec);
}

}  // namespace

// ---------------------------------------------------------------------------
// AsciiTokenizer
// ---------------------------------------------------------------------------
TEST(TokenizerTest, TestAsciiEncodeBasic) {
  AsciiTokenizer tok;
  auto tokens = tok.Encode("hi");
  EXPECT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0], tok.bos_id());  // 1
  EXPECT_EQ(tokens[1], static_cast<int>('h'));
  EXPECT_EQ(tokens[2], static_cast<int>('i'));
}

TEST(TokenizerTest, TestAsciiEncodeEmpty) {
  AsciiTokenizer tok;
  auto tokens = tok.Encode("");
  EXPECT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0], tok.bos_id());
}

TEST(TokenizerTest, TestAsciiDecodeToken) {
  AsciiTokenizer tok;
  EXPECT_EQ(tok.DecodeToken(tok.bos_id()), "<bos>");
  EXPECT_EQ(tok.DecodeToken(tok.eos_id()), "<eos>");
  EXPECT_EQ(tok.DecodeToken(tok.unk_id()), "<unk>");
  EXPECT_EQ(tok.DecodeToken(static_cast<int>('a')), "a");
  EXPECT_EQ(tok.DecodeToken(32), " ");
  EXPECT_EQ(tok.DecodeToken(10), "");        // control char
  EXPECT_EQ(tok.DecodeToken(127), "");       // DEL
  EXPECT_EQ(tok.DecodeToken(200), "<unk>");  // out of range
}

TEST(TokenizerTest, TestAsciiDecodeSequence) {
  AsciiTokenizer tok;
  std::vector<int> tokens = {tok.bos_id(), static_cast<int>('h'),
                             static_cast<int>('i')};
  std::string text = tok.Decode(tokens);
  EXPECT_EQ(text, "<bos>hi");
}

TEST(TokenizerTest, TestAsciiVocabSize) {
  AsciiTokenizer tok;
  EXPECT_EQ(tok.vocab_size(), 128);
  EXPECT_EQ(tok.bos_id(), 1);
  EXPECT_EQ(tok.eos_id(), 2);
  EXPECT_EQ(tok.unk_id(), 0);
}

// ---------------------------------------------------------------------------
// JsonVocabTokenizer
// ---------------------------------------------------------------------------
TEST(TokenizerTest, TestJsonVocabLoads) {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    EXPECT_EQ(tok.vocab_size(), 128);
    EXPECT_EQ(tok.bos_id(), 1);
    EXPECT_EQ(tok.eos_id(), 2);
    EXPECT_EQ(tok.unk_id(), 0);
  } catch (const std::runtime_error& e) {
    FAIL() << "failed to load vocab.json: " << e.what();
  }
}

TEST(TokenizerTest, TestJsonVocabEncodeBasic) {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    auto tokens = tok.Encode("hi");
    EXPECT_EQ(tokens.size(), 3);
    EXPECT_EQ(tokens[0], tok.bos_id());
    EXPECT_EQ(tokens[1], static_cast<int>('h'));
    EXPECT_EQ(tokens[2], static_cast<int>('i'));
  } catch (const std::runtime_error& e) {
    FAIL() << "failed to load vocab.json: " << e.what();
  }
}

TEST(TokenizerTest, TestJsonVocabDecodeToken) {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    EXPECT_EQ(tok.DecodeToken(0), "<unk>");
    EXPECT_EQ(tok.DecodeToken(1), "<bos>");
    EXPECT_EQ(tok.DecodeToken(2), "<eos>");
    EXPECT_EQ(tok.DecodeToken(static_cast<int>('a')), "a");
    EXPECT_EQ(tok.DecodeToken(32), " ");
    EXPECT_EQ(tok.DecodeToken(10), "");        // control char maps to empty
    EXPECT_EQ(tok.DecodeToken(127), "");       // DEL
    EXPECT_EQ(tok.DecodeToken(200), "<unk>");  // out of range
  } catch (const std::runtime_error& e) {
    FAIL() << "failed to load vocab.json: " << e.what();
  }
}

TEST(TokenizerTest, TestJsonVocabDecodeSequence) {
  try {
    JsonVocabTokenizer tok("models/tiny/vocab.json");
    std::vector<int> tokens = {1, static_cast<int>('h'), static_cast<int>('i')};
    std::string text = tok.Decode(tokens);
    EXPECT_EQ(text, "<bos>hi");
  } catch (const std::runtime_error& e) {
    FAIL() << "failed to load vocab.json: " << e.what();
  }
}

TEST(TokenizerTest, TestJsonVocabMatchesAscii) {
  try {
    AsciiTokenizer ascii_tok;
    JsonVocabTokenizer json_tok("models/tiny/vocab.json");

    std::string prompt = "hello world!";
    auto ascii_tokens = ascii_tok.Encode(prompt);
    auto json_tokens = json_tok.Encode(prompt);

    EXPECT_EQ(ascii_tokens.size(), json_tokens.size());
    for (size_t i = 0; i < ascii_tokens.size(); ++i) {
      EXPECT_EQ(ascii_tokens[i], json_tokens[i]);
    }

    std::string ascii_decoded = ascii_tok.Decode(ascii_tokens);
    std::string json_decoded = json_tok.Decode(json_tokens);
    EXPECT_EQ(ascii_decoded, json_decoded);
  } catch (const std::runtime_error& e) {
    FAIL() << "failed to load vocab.json: " << e.what();
  }
}

TEST(TokenizerTest, TestJsonVocabMissingFile) {
  try {
    JsonVocabTokenizer tok("models/tiny/nonexistent_vocab.json");
    FAIL() << "expected exception for missing vocab file";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
TEST(TokenizerTest, TestFactoryUsesJsonWhenExists) {
  auto tok = CreateTokenizer("models/tiny/vocab.json");
  EXPECT_TRUE(tok != nullptr);
  EXPECT_EQ(tok->vocab_size(), 128);
}

TEST(TokenizerTest, TestFactoryFallsBackToAscii) {
  auto tok = CreateTokenizer("");
  EXPECT_TRUE(tok != nullptr);
  EXPECT_EQ(tok->vocab_size(), 128);  // AsciiTokenizer
}

TEST(TokenizerTest, TestFactoryFallsBackWhenMissing) {
  auto tok = CreateTokenizer("models/tiny/does_not_exist.json");
  EXPECT_TRUE(tok != nullptr);
  EXPECT_EQ(tok->vocab_size(), 128);  // AsciiTokenizer fallback
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------
TEST(TokenizerTest, TestBpeEncodeDecodeAndSpecialToken) {
  TempBpeFiles files = WriteTempBpeFiles();
  auto tok = CreateBpeTokenizer(files.vocab.string(), files.merges.string(),
                                files.special.string());
  ASSERT_TRUE(tok != nullptr);
  EXPECT_EQ(tok->bos_id(), 10);
  EXPECT_EQ(tok->eos_id(), 11);
  EXPECT_EQ(tok->unk_id(), 9);

  std::vector<int> hello = tok->Encode("hello");
  ASSERT_EQ(hello.size(), 1u);
  EXPECT_EQ(hello[0], 7);
  EXPECT_EQ(tok->Decode(hello), "hello");

  std::vector<int> special = tok->Encode("<|im_start|>");
  ASSERT_EQ(special.size(), 1u);
  EXPECT_EQ(special[0], 8);
  EXPECT_EQ(tok->Decode(special), "<|im_start|>");

  RemoveTempBpeFiles(files);
}

// ---------------------------------------------------------------------------
