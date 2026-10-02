// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <vector>

#include "mini_llama/debug.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------
TEST(DebugTest, TestBenchmarkResultZero) {
  BenchmarkResult r;
  EXPECT_EQ(r.n_prompt_tokens, 0);
  EXPECT_EQ(r.n_generated_tokens, 0);
  EXPECT_EQ(r.n_decode_tokens, 0);
  EXPECT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
  EXPECT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
}

TEST(DebugTest, TestBenchmarkResultTokensPerSec) {
  BenchmarkResult r;
  r.n_prompt_tokens = 10;
  r.n_generated_tokens = 100;
  r.n_decode_tokens = 100;
  r.prefill_ms = 50.0;
  r.decode_ms = 450.0;
  EXPECT_NEAR(r.tokens_per_sec(), 200.0, 1e-6);  // 100 / 0.5
  EXPECT_NEAR(r.decode_tokens_per_sec(), 1000.0 / 4.5, 1e-6);
}

TEST(DebugTest, TestBenchmarkResultNoDecode) {
  BenchmarkResult r;
  r.n_generated_tokens = 0;
  r.n_decode_tokens = 0;
  r.prefill_ms = 100.0;
  r.decode_ms = 0.0;
  EXPECT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
  EXPECT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// DumpTensorShape (smoke test: just verify no crash)
// ---------------------------------------------------------------------------
TEST(DebugTest, TestDumpTensorShape) {
  Tensor t({2, 3, 4}, 1.0f);
  DumpTensorShape(t, "TestTensor");
}

// ---------------------------------------------------------------------------
// DumpLogitsTopK (smoke test)
// ---------------------------------------------------------------------------
TEST(DebugTest, TestDumpLogitsTopK) {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits.data[i] = static_cast<float>(i);
  }
  DumpLogitsTopK(logits, 3);
}

TEST(DebugTest, TestDumpLogitsTopKLargerThanVocab) {
  Tensor logits({5}, 0.0f);
  for (int i = 0; i < 5; ++i) {
    logits.data[i] = static_cast<float>(i);
  }
  DumpLogitsTopK(logits, 10);  // k > vocab_size
}

// ---------------------------------------------------------------------------
// DumpKvCacheInfo (smoke test)
// ---------------------------------------------------------------------------
TEST(DebugTest, TestDumpKvCacheInfo) {
  KvCache cache(2, 16, 4, 8);
  DumpKvCacheInfo(cache);
}

// ---------------------------------------------------------------------------
// RunBenchmark (integration smoke test)
// ---------------------------------------------------------------------------
TEST(DebugTest, TestRunBenchmarkSmoke) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hi");
  BenchmarkResult result = RunBenchmark(model, tokens, 4, 42, false);

  EXPECT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
  EXPECT_EQ(result.n_generated_tokens, 4);
  EXPECT_EQ(result.n_decode_tokens, 3);
  EXPECT_TRUE(result.prefill_ms >= 0.0);
  EXPECT_TRUE(result.decode_ms >= 0.0);
  EXPECT_TRUE(result.tokens_per_sec() >= 0.0);
}

TEST(DebugTest, TestRunBenchmarkZeroPredict) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hi");
  BenchmarkResult result = RunBenchmark(model, tokens, 0, 42, false);

  EXPECT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
  EXPECT_EQ(result.n_generated_tokens, 0);
  EXPECT_EQ(result.n_decode_tokens, 0);
  EXPECT_TRUE(result.prefill_ms >= 0.0);
  EXPECT_TRUE(result.decode_ms == 0.0);
}

// ---------------------------------------------------------------------------
