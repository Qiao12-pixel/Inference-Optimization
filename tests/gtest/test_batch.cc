// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// MiniBatch construction
// ---------------------------------------------------------------------------
TEST(BatchTest, TestBatchSingle) {
  MiniBatch b = MiniBatch::Single(42, 5);
  EXPECT_EQ(b.num_tokens(), 1);
  EXPECT_EQ(b.tokens[0], 42);
  EXPECT_EQ(b.positions[0], 5);
}

TEST(BatchTest, TestBatchFromTokens) {
  std::vector<int> toks = {10, 20, 30};
  MiniBatch b = MiniBatch::FromTokens(toks, 0);
  EXPECT_EQ(b.num_tokens(), 3);
  EXPECT_EQ(b.tokens[0], 10);
  EXPECT_EQ(b.tokens[1], 20);
  EXPECT_EQ(b.tokens[2], 30);
  EXPECT_EQ(b.positions[0], 0);
  EXPECT_EQ(b.positions[1], 1);
  EXPECT_EQ(b.positions[2], 2);
}

TEST(BatchTest, TestBatchFromTokensWithOffset) {
  std::vector<int> toks = {5, 6};
  MiniBatch b = MiniBatch::FromTokens(toks, 10);
  EXPECT_EQ(b.positions[0], 10);
  EXPECT_EQ(b.positions[1], 11);
}

TEST(BatchTest, TestBatchEmpty) {
  MiniBatch b;
  EXPECT_EQ(b.num_tokens(), 0);
}

// ---------------------------------------------------------------------------
// ForwardBatch
// ---------------------------------------------------------------------------
TEST(BatchTest, TestForwardBatchPrefillMatchesIndividual) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  std::vector<int> tokens = {1, 104, 101, 108, 108, 111};

  // Individual forward
  MiniLlamaContext ctx1(&model);
  Tensor logits1;
  for (size_t i = 0; i < tokens.size(); ++i) {
    ctx1.pos = static_cast<int>(i);
    logits1 = ForwardToken(ctx1, model, tokens[i]);
  }

  // Batch forward
  MiniLlamaContext ctx2(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  Tensor logits2 = ForwardBatch(ctx2, model, batch);

  EXPECT_EQ(logits1.num_dims(), 1);
  EXPECT_EQ(logits2.num_dims(), 1);
  EXPECT_EQ(logits1.shape[0], logits2.shape[0]);
  for (size_t i = 0; i < logits1.size(); ++i) {
    EXPECT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
  }
}

TEST(BatchTest, TestForwardBatchSingleMatchesIndividual) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  // Individual forward
  MiniLlamaContext ctx1(&model);
  ctx1.pos = 3;
  Tensor logits1 = ForwardToken(ctx1, model, 42);

  // Batch forward with single token
  MiniLlamaContext ctx2(&model);
  MiniBatch batch = MiniBatch::Single(42, 3);
  Tensor logits2 = ForwardBatch(ctx2, model, batch);

  EXPECT_EQ(logits1.num_dims(), 1);
  EXPECT_EQ(logits2.num_dims(), 1);
  EXPECT_EQ(logits1.shape[0], logits2.shape[0]);
  for (size_t i = 0; i < logits1.size(); ++i) {
    EXPECT_NEAR(logits1.data[i], logits2.data[i], 1e-6f);
  }
}

TEST(BatchTest, TestForwardBatchUpdatesContext) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  std::vector<int> tokens = {1, 104, 101};
  MiniLlamaContext ctx(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  ForwardBatch(ctx, model, batch);

  EXPECT_EQ(ctx.token_history.size(), tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    EXPECT_EQ(ctx.token_history[i], tokens[i]);
  }
  EXPECT_EQ(ctx.pos, static_cast<int>(tokens.size()) - 1);
}

TEST(BatchTest, TestForwardBatchAppendsDecodeHistory) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  MiniLlamaContext ctx(&model);
  MiniBatch prefill = MiniBatch::FromTokens({1, 104}, 0);
  ForwardBatch(ctx, model, prefill);

  MiniBatch decode = MiniBatch::Single(101, 2);
  ForwardBatch(ctx, model, decode);

  EXPECT_EQ(ctx.token_history.size(), 3);
  EXPECT_EQ(ctx.token_history[0], 1);
  EXPECT_EQ(ctx.token_history[1], 104);
  EXPECT_EQ(ctx.token_history[2], 101);
  EXPECT_EQ(ctx.pos, 2);
}

TEST(BatchTest, TestForwardBatchEmptyRejected) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  try {
    ForwardBatch(ctx, model, batch);
    FAIL() << "expected exception for empty batch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(BatchTest, TestForwardBatchMismatchedSizesRejected) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  if (!model.loaded) {
    FAIL() << "failed to load model";
  }

  MiniLlamaContext ctx(&model);
  MiniBatch batch;
  batch.tokens = {1, 2};
  batch.positions = {0};
  try {
    ForwardBatch(ctx, model, batch);
    FAIL() << "expected exception for size mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ---------------------------------------------------------------------------
