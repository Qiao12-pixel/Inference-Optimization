// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "mini_llama/chat.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/prompt_builder.h"
#include "tests/gguf_tokenizer_fixture.h"
#include "tests/test_names.h"

TEST(GgufTokenizerTest, LoadsFromTinyFixture) {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_load_gtest.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_TRUE(tok != nullptr);
  EXPECT_EQ(tok->vocab_size(), 14);
  EXPECT_EQ(tok->bos_id(), 3);
  EXPECT_EQ(tok->eos_id(), 4);
}

TEST(GgufTokenizerTest, EncodeDecodeRoundtrip) {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_roundtrip_gtest.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_TRUE(tok != nullptr);
  std::vector<int> ids = tok->Encode("ab");
  ASSERT_EQ(ids.size(), 1);
  EXPECT_EQ(ids[0], 2);
  EXPECT_EQ(tok->Decode(ids), "ab");
}

TEST(GgufTokenizerTest, SpecialTokens) {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_tokenizer_special_gtest.gguf");
  auto tok = CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_TRUE(tok != nullptr);
  EXPECT_EQ(tok->Encode("<|im_start|>"), std::vector<int>({5}));
  EXPECT_EQ(tok->Encode("<|im_end|>"), std::vector<int>({6}));
  EXPECT_EQ(tok->eos_id(), 4);
  EXPECT_EQ(tok->DecodeToken(4), "</s>");
}

TEST(GgufTokenizerTest, LoadsRealModelWhenAvailable) {
  const std::string path = "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "local demo GGUF is not available";
  }

  auto tok = CreateGgufTokenizer(path);
  ASSERT_TRUE(tok != nullptr);
  EXPECT_EQ(tok->vocab_size(), 151936);
  EXPECT_EQ(tok->bos_id(), 151643);
  EXPECT_EQ(tok->eos_id(), 151645);
  EXPECT_EQ(tok->Encode("<|im_start|>"), std::vector<int>({151644}));
}

TEST(GgufTokenizerTest, RejectsMissingFile) {
  auto tok = CreateGgufTokenizer("models/chat/nonexistent.gguf");
  EXPECT_TRUE(tok == nullptr);
}

TEST(ChatTemplateTest, LoadsTemplateFromGguf) {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_template_qwen2_gtest.gguf");
  std::string tmpl = LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  ASSERT_FALSE(tmpl.empty());
  EXPECT_TRUE(tmpl.find("{% for message in messages %}") != std::string::npos);

  PromptBuilder builder;
  builder.SetChatTemplate(tmpl);
  std::string prompt = builder.Build({{"user", "Hello"}});

  EXPECT_TRUE(prompt.find("<|im_start|>system") != std::string::npos);
  EXPECT_TRUE(prompt.find("<|im_start|>user") != std::string::npos);
  EXPECT_TRUE(prompt.find("Hello") != std::string::npos);
  EXPECT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);
}

TEST(ChatTemplateTest, KeepsExistingSystemMessage) {
  std::filesystem::path path = mini_llama::test::WriteTinyGgufTokenizerFixture(
      "tiny_template_system_gtest.gguf");
  std::string tmpl = LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  PromptBuilder builder;
  builder.SetChatTemplate(tmpl);
  std::string prompt = builder.Build({
      {"system", "You are a coding assistant."},
      {"user", "Write a hello world program."},
  });

  EXPECT_TRUE(prompt.find("coding assistant") != std::string::npos);
  EXPECT_TRUE(prompt.find("You are a helpful assistant") == std::string::npos);
  EXPECT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);
}

TEST(ChatTemplateTest, MissingTemplateUsesArchFallbackOnly) {
  std::filesystem::path qwen_path =
      mini_llama::test::WriteTinyGgufTokenizerFixture(
          "tiny_template_qwen2_fallback_gtest.gguf", "qwen2", false);
  std::string qwen_tmpl = LoadChatTemplateFromGguf(qwen_path.string());
  std::filesystem::remove(qwen_path);
  EXPECT_EQ(qwen_tmpl, "qwen2");

  std::filesystem::path llama_path =
      mini_llama::test::WriteTinyGgufTokenizerFixture(
          "tiny_template_llama_empty_gtest.gguf", "llama", false);
  std::string llama_tmpl = LoadChatTemplateFromGguf(llama_path.string());
  std::filesystem::remove(llama_path);
  EXPECT_TRUE(llama_tmpl.empty());
}

TEST(ChatTemplateTest, MalformedTemplateIsSafe) {
  PromptBuilder builder;
  builder.SetChatTemplate("{% endif %}{{ 'ok' }}");
  EXPECT_EQ(builder.Build({}), "ok");

  builder.SetChatTemplate("{% if missing %}bad");
  EXPECT_TRUE(builder.Build({}).empty());

  builder.SetChatTemplate("{% for message in messages %}bad");
  EXPECT_TRUE(builder.Build({}).empty());
}

TEST(ChatTemplateTest, SupportsJinjaTrimMarkers) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{%- for message in messages -%}{{- '<|im_start|>' + message['role'] + "
      "'\n' + message['content'] -}}{%- endfor -%}{{- "
      "'<|im_start|>assistant\n' -}}");
  std::string prompt = builder.Build({{"user", "Hello"}});

  EXPECT_TRUE(prompt.find("<|im_start|>user") != std::string::npos);
  EXPECT_TRUE(prompt.find("Hello") != std::string::npos);
  EXPECT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);
}

TEST(ChatTemplateTest, BuiltinQwen2Fallback) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");

  std::string prompt = builder.Build({{"user", "Hi"}});
  EXPECT_TRUE(prompt.find("<|im_start|>system") != std::string::npos);
  EXPECT_TRUE(prompt.find("<|im_start|>user") != std::string::npos);
  EXPECT_TRUE(prompt.find("<|im_start|>assistant") != std::string::npos);
}

TEST(ChatTemplateTest, PlainText) {
  PromptBuilder builder;
  std::string prompt = builder.Build({{"user", "Hello"}});
  EXPECT_TRUE(prompt.find("User: Hello") != std::string::npos);
  EXPECT_TRUE(prompt.find("Assistant:") != std::string::npos);
}
