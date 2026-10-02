// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf_loader.h"
#include "tests/gguf_loader_fixture.h"
#include "tests/test_names.h"

TEST(GgufLoaderTest, LoadsTinyFixture) {
  std::filesystem::path path = mini_llama::test::WriteTinyLlamaGgufFixture(
      "tiny_loader_fixture_gtest.gguf");
  MiniLlamaModel model = LoadGgufModel(path.string());
  std::filesystem::remove(path);

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());
  EXPECT_EQ(model.config.vocab_size, 128);
  EXPECT_EQ(model.config.dim, 32);
  EXPECT_EQ(model.config.n_layers, 2);
  EXPECT_EQ(model.config.n_heads, 4);
  EXPECT_EQ(model.config.n_kv_heads, 4);
  EXPECT_EQ(model.config.head_dim, 8);
  EXPECT_EQ(model.config.hidden_dim, 86);
  EXPECT_EQ(model.config.max_seq_len, 128);
  EXPECT_NEAR(model.config.rms_norm_eps, 1e-5f, 1e-7f);

  EXPECT_EQ(model.token_embedding.shape, std::vector<int>({128, 32}));
  ASSERT_EQ(model.layers.size(), 2);
  for (const LayerWeights& lw : model.layers) {
    EXPECT_EQ(lw.attention_norm.shape, std::vector<int>({32}));
    EXPECT_EQ(lw.wq.shape, std::vector<int>({32, 32}));
    EXPECT_EQ(lw.wk.shape, std::vector<int>({32, 32}));
    EXPECT_EQ(lw.wv.shape, std::vector<int>({32, 32}));
    EXPECT_EQ(lw.wo.shape, std::vector<int>({32, 32}));
    EXPECT_EQ(lw.ffn_norm.shape, std::vector<int>({32}));
    EXPECT_EQ(lw.w_gate.shape, std::vector<int>({86, 32}));
    EXPECT_EQ(lw.w_up.shape, std::vector<int>({86, 32}));
    EXPECT_EQ(lw.w_down.shape, std::vector<int>({32, 86}));
  }
  EXPECT_EQ(model.final_norm.shape, std::vector<int>({32}));
  EXPECT_EQ(model.lm_head.shape, std::vector<int>({128, 32}));
}

TEST(GgufLoaderTest, FixtureEntersForwardPath) {
  std::filesystem::path path = mini_llama::test::WriteTinyLlamaGgufFixture(
      "tiny_loader_forward_gtest.gguf");
  MiniLlamaModel model = LoadGgufModel(path.string());
  std::filesystem::remove(path);

  ASSERT_TRUE(model.loaded) << model.load_error;
  MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  Tensor logits = ForwardToken(ctx, model, 1);
  EXPECT_EQ(logits.shape, std::vector<int>({128}));
}

TEST(GgufLoaderTest, LoadsRealModelWhenAvailable) {
  const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "local demo GGUF is not available";
  }

  MiniLlamaModel model = LoadGgufModel(path);
  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());
  EXPECT_EQ(model.config.vocab_size, 151936);
  EXPECT_EQ(model.config.dim, 896);
  EXPECT_EQ(model.config.n_layers, 24);
  EXPECT_EQ(model.config.n_heads, 14);
  EXPECT_EQ(model.config.n_kv_heads, 2);
  EXPECT_EQ(model.config.head_dim, 64);
  EXPECT_EQ(model.config.hidden_dim, 4864);
  EXPECT_EQ(model.config.max_seq_len, 32768);
  EXPECT_NEAR(model.config.rms_norm_eps, 1e-6f, 1e-7f);
}

TEST(GgufLoaderTest, MissingFile) {
  MiniLlamaModel model = LoadGgufModel("models/chat/nonexistent.gguf");
  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}

TEST(GgufLoaderTest, InvalidFile) {
  MiniLlamaModel model = LoadGgufModel("models/tiny/model.json");
  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}

TEST(GgufLoaderTest, RejectsIncompleteConfig) {
  MiniLlamaModel model = LoadGgufModel("models/tiny/test.gguf");
  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}
