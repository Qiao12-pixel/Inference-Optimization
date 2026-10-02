// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "mini_llama/sampler.h"
#include "mini_llama/tensor.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// Greedy sampling
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerGreedy) {
  Tensor logits({5}, 0.0f);
  logits[0] = 1.0f;
  logits[1] = 5.0f;
  logits[2] = 3.0f;
  logits[3] = 5.0f;  // same max as FlatIndex 1
  logits[4] = 2.0f;

  int idx = MiniSampler::SampleGreedy(logits);
  EXPECT_EQ(idx, 1);  // first occurrence of max
}

// ---------------------------------------------------------------------------
// temperature = 0 is equivalent to greedy
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerTemperatureZeroIsGreedy) {
  Tensor logits({5}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 2.0f;
  logits[2] = 1.0f;
  logits[3] = 0.5f;
  logits[4] = 0.2f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 0.0f;
  params.top_k = 0;

  // Run many times; should always pick the same token (greedy).
  int expected = MiniSampler::SampleGreedy(logits);
  for (int i = 0; i < 20; ++i) {
    int idx = sampler.Sample(logits, params);
    EXPECT_EQ(idx, expected);
  }
}

// ---------------------------------------------------------------------------
// top_k = 1 is equivalent to greedy
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerTopKOneIsGreedy) {
  Tensor logits({5}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 2.0f;
  logits[2] = 1.0f;
  logits[3] = 0.5f;
  logits[4] = 0.2f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 1;

  int expected = MiniSampler::SampleGreedy(logits);
  for (int i = 0; i < 20; ++i) {
    int idx = sampler.Sample(logits, params);
    EXPECT_EQ(idx, expected);
  }
}

// ---------------------------------------------------------------------------
// Fixed seed gives reproducible temperature sampling
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerTemperatureReproducible) {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i) * 0.5f;
  }

  MiniSampler sampler1(123);
  MiniSampler sampler2(123);
  SamplingParams params;
  params.temperature = 0.8f;
  params.top_k = 0;

  // Same seed should produce identical sequences.
  for (int i = 0; i < 10; ++i) {
    int a = sampler1.Sample(logits, params);
    int b = sampler2.Sample(logits, params);
    EXPECT_EQ(a, b);
  }
}

// ---------------------------------------------------------------------------
// Fixed seed gives reproducible top-k sampling
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerTopKReproducible) {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i) * 0.5f;
  }

  MiniSampler sampler1(456);
  MiniSampler sampler2(456);
  SamplingParams params;
  params.temperature = 0.8f;
  params.top_k = 3;

  for (int i = 0; i < 10; ++i) {
    int a = sampler1.Sample(logits, params);
    int b = sampler2.Sample(logits, params);
    EXPECT_EQ(a, b);
  }
}

// ---------------------------------------------------------------------------
// Different seeds give different sequences (with high probability)
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerDifferentSeedsDiffer) {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i) * 0.3f;
  }

  MiniSampler sampler1(1);
  MiniSampler sampler2(2);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 0;

  int same_count = 0;
  for (int i = 0; i < 20; ++i) {
    int a = sampler1.Sample(logits, params);
    int b = sampler2.Sample(logits, params);
    if (a == b) {
      ++same_count;
    }
  }

  // With different seeds and temperature=1.0, it's extremely unlikely
  // that all 20 samples match. Allow up to 18 matches as a loose bound.
  EXPECT_LT(same_count, 18);
}

// ---------------------------------------------------------------------------
// Top-k only samples from the top-k set
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerTopKLimitsCandidates) {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits[i] = static_cast<float>(i);
  }

  MiniSampler sampler(789);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 2;

  // With top_k=2 on logits [0..9], only tokens 8 and 9 should ever be sampled.
  for (int i = 0; i < 50; ++i) {
    int idx = sampler.Sample(logits, params);
    EXPECT_TRUE(idx == 8 || idx == 9);
  }
}

// ---------------------------------------------------------------------------
// Legacy free function still works
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerLegacyGreedy) {
  Tensor logits({5}, 0.0f);
  logits[0] = 1.0f;
  logits[1] = 5.0f;
  logits[2] = 3.0f;
  logits[3] = 4.0f;
  logits[4] = 2.0f;

  int idx = SampleGreedy(logits);
  EXPECT_EQ(idx, 1);
}

// ---------------------------------------------------------------------------
// Default params preserve greedy behavior
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerDefaultParamsAreGreedy) {
  Tensor logits({5}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 2.0f;
  logits[2] = 1.0f;
  logits[3] = 0.5f;
  logits[4] = 0.2f;

  MiniSampler sampler(42);
  SamplingParams params;

  int idx = sampler.Sample(logits, params);
  EXPECT_EQ(idx, MiniSampler::SampleGreedy(logits));
}

// ---------------------------------------------------------------------------
// Invalid parameters fail clearly
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerRejectsNegativeTopK) {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 0.2f;
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = -1;

  EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
}

TEST(SamplerTest, TestSamplerRejectsNanTemperature) {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 0.2f;
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = std::numeric_limits<float>::quiet_NaN();
  params.top_k = 0;

  EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
}

TEST(SamplerTest, TestSamplerRejectsEmptyLogits) {
  Tensor logits;
  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 0.0f;

  EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
}

TEST(SamplerTest, TestSamplerRejectsNonfiniteLogits) {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = std::numeric_limits<float>::infinity();
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  SamplingParams params;
  params.temperature = 1.0f;

  EXPECT_THROW(sampler.Sample(logits, params), std::runtime_error);
}

TEST(SamplerTest, TestSamplerTopKLargerThanVocabIsClamped) {
  Tensor logits({4}, 0.0f);
  logits[0] = 0.0f;
  logits[1] = 1.0f;
  logits[2] = 2.0f;
  logits[3] = 3.0f;

  MiniSampler sampler(42);
  for (int i = 0; i < 20; ++i) {
    int idx = sampler.SampleTopK(logits, 1.0f, 100);
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, 4);
  }
}

TEST(SamplerTest, TestSamplerDirectTopKZeroRejected) {
  Tensor logits({3}, 0.0f);
  logits[0] = 0.1f;
  logits[1] = 0.2f;
  logits[2] = 0.3f;

  MiniSampler sampler(42);
  EXPECT_THROW(sampler.SampleTopK(logits, 1.0f, 0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Top-k keeps exactly k candidates even when logits tie
// ---------------------------------------------------------------------------
TEST(SamplerTest, TestSamplerTopKTiesKeepExactK) {
  Tensor logits({5}, 0.0f);
  logits[0] = 5.0f;
  logits[1] = 5.0f;
  logits[2] = 5.0f;
  logits[3] = 1.0f;
  logits[4] = 0.0f;

  MiniSampler sampler(321);
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 2;

  for (int i = 0; i < 40; ++i) {
    int idx = sampler.Sample(logits, params);
    EXPECT_TRUE(idx == 0 || idx == 1);
  }
}

// ---------------------------------------------------------------------------
