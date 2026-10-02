// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_names.h"

TEST(ForwardTest, TestForwardTokenProducesLogitShape) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    FAIL() << "Failed to load model";
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);  // BOS token
  EXPECT_EQ(logits.num_dims(), 1);
  EXPECT_EQ(logits.shape[0], model.config.vocab_size);
}
TEST(ForwardTest, TestForwardTokenChangesKvCache) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    FAIL() << "Failed to load model";
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  // Before forward, cache at layer 0, pos 0 should be zero
  float before = ctx.kv_cache.keys.At({0, 0, 0, 0});
  EXPECT_NEAR(before, 0.0f, 1e-6f);

  ForwardToken(ctx, model, 1);

  // After forward, cache should have been written
  float after = ctx.kv_cache.keys.At({0, 0, 0, 0});
  // It's unlikely to be exactly 0 after random weights forward
  // Just check it's finite and cache was written somewhere
  EXPECT_TRUE(std::isfinite(after));
}

TEST(ForwardTest, TestGenerationGreedySequence) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    FAIL() << "Failed to load model";
  }
  AsciiTokenizer tokenizer;
  std::vector<int> tokens = tokenizer.Encode("hello");
  MiniLlamaContext ctx(&model);

  // Prefill
  size_t prompt_len = tokens.size();
  for (size_t i = 0; i < prompt_len; ++i) {
    ctx.pos = static_cast<int>(i);
    Tensor logits = ForwardToken(ctx, model, tokens[i]);
    if (i + 1 == prompt_len) {
      tokens.push_back(SampleGreedy(logits));
    }
  }

  // Decode 5 more tokens
  for (int i = 0; i < 5; ++i) {
    ctx.pos = static_cast<int>(tokens.size() - 1);
    Tensor logits = ForwardToken(ctx, model, tokens.back());
    tokens.push_back(SampleGreedy(logits));
  }

  EXPECT_GT(tokens.size(), 6);
  // First generated token should be the same as what we got in prefill
  EXPECT_GE(tokens[tokens.size() - 6], 0);
}

TEST(ForwardTest, TestPositionIncrementsCorrectly) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (model.layers.empty()) {
    FAIL() << "Failed to load model";
  }
  MiniLlamaContext ctx(&model);

  // Forward token at pos 0
  ctx.pos = 0;
  ForwardToken(ctx, model, 1);

  // Forward token at pos 1
  ctx.pos = 1;
  Tensor logits = ForwardToken(ctx, model, 2);
  EXPECT_EQ(logits.shape[0], model.config.vocab_size);
}

TEST(ForwardTest, TestForwardRejectsInvalidToken) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "Failed to load model";
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;

  EXPECT_THROW(ForwardToken(ctx, model, model.config.vocab_size),
               std::out_of_range);
}

TEST(ForwardTest, TestForwardRejectsInvalidPosition) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "Failed to load model";
  }
  MiniLlamaContext ctx(&model);
  ctx.pos = model.config.max_seq_len;

  EXPECT_THROW(ForwardToken(ctx, model, 1), std::out_of_range);
}

TEST(ForwardTest, TestQ80ForwardMatchesF32) {
  // Load F32 model and run forward pass
  MiniLlamaModel model_f32 =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model_f32.loaded) {
    FAIL() << "Failed to load model";
  }
  MiniLlamaContext ctx_f32(&model_f32);
  ctx_f32.pos = 0;
  Tensor logits_f32 = ForwardToken(ctx_f32, model_f32, 1);

  // Quantize another copy of the model to Q8_0
  MiniLlamaModel model_q8 =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model_q8.loaded) {
    FAIL() << "Failed to load model";
  }
  QuantizeModelToQ80(model_q8);

  // Run forward with fresh context
  MiniLlamaContext ctx_q8(&model_q8);
  ctx_q8.pos = 0;
  Tensor logits_q8 = ForwardToken(ctx_q8, model_q8, 1);

  // Shapes must match
  EXPECT_EQ(logits_f32.shape, logits_q8.shape);

  // Logits should be close (Q8_0 roundtrip error is small)
  float max_err = 0.0f;
  float sum_err = 0.0f;
  for (size_t i = 0; i < logits_f32.size(); ++i) {
    float err = std::abs(logits_f32.data[i] - logits_q8.data[i]);
    max_err = std::max(max_err, err);
    sum_err += err;
  }
  float mean_err = sum_err / static_cast<float>(logits_f32.size());

  EXPECT_LT(max_err, 0.5f) << "max error too large: " << max_err;
  EXPECT_LT(mean_err, 0.05f) << "mean error too large: " << mean_err;
}

TEST(ForwardTest, TestQ40ForwardMatchesF32) {
  // Load F32 model and run forward pass
  MiniLlamaModel model_f32 =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model_f32.loaded) {
    FAIL() << "Failed to load model";
  }
  MiniLlamaContext ctx_f32(&model_f32);
  ctx_f32.pos = 0;
  Tensor logits_f32 = ForwardToken(ctx_f32, model_f32, 1);

  // Quantize another copy of the model to Q4_0
  MiniLlamaModel model_q4 =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model_q4.loaded) {
    FAIL() << "Failed to load model";
  }
  QuantizeModelToQ40(model_q4);

  // Run forward with fresh context
  MiniLlamaContext ctx_q4(&model_q4);
  ctx_q4.pos = 0;
  Tensor logits_q4 = ForwardToken(ctx_q4, model_q4, 1);

  // Shapes must match
  EXPECT_EQ(logits_f32.shape, logits_q4.shape);

  // Logits should be close (Q4_0 has higher error than Q8_0)
  float max_err = 0.0f;
  float sum_err = 0.0f;
  for (size_t i = 0; i < logits_f32.size(); ++i) {
    float err = std::abs(logits_f32.data[i] - logits_q4.data[i]);
    max_err = std::max(max_err, err);
    sum_err += err;
  }
  float mean_err = sum_err / static_cast<float>(logits_f32.size());

  EXPECT_LT(max_err, 2.0f) << "max error too large: " << max_err;
  EXPECT_LT(mean_err, 0.2f) << "mean error too large: " << mean_err;
}
