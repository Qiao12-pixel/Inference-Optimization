// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "mini_llama/chat.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------
TEST(ChatTest, TestChatMessageStoresRoleAndContent) {
  ChatMessage msg{"user", "hello"};
  EXPECT_EQ(msg.role, "user");
  EXPECT_EQ(msg.content, "hello");
}

// ---------------------------------------------------------------------------
// ChatSession
// ---------------------------------------------------------------------------
TEST(ChatTest, TestChatSessionStartsEmpty) {
  ChatSession session;
  EXPECT_TRUE(session.messages.empty());
  EXPECT_TRUE(session.token_history.empty());
  EXPECT_EQ(session.total_prompt_tokens, 0);
  EXPECT_EQ(session.total_generated_tokens, 0);
  EXPECT_NEAR(session.total_time_ms, 0.0, 1e-6);
}

TEST(ChatTest, TestChatSessionAddMessageAppends) {
  ChatSession session;
  session.AddMessage("system", "You are helpful.");
  session.AddMessage("user", "hi");
  EXPECT_EQ(session.messages.size(), 2);
  EXPECT_EQ(session.messages[0].role, "system");
  EXPECT_EQ(session.messages[0].content, "You are helpful.");
  EXPECT_EQ(session.messages[1].role, "user");
  EXPECT_EQ(session.messages[1].content, "hi");
}

TEST(ChatTest, TestChatSessionClearResets) {
  ChatSession session;
  session.AddMessage("user", "test");
  session.SetTokenHistory({1, 2, 3});
  session.RecordTurn(10, 5, 100.0);
  session.Clear();
  EXPECT_TRUE(session.messages.empty());
  EXPECT_TRUE(session.token_history.empty());
  EXPECT_EQ(session.total_prompt_tokens, 0);
  EXPECT_EQ(session.total_generated_tokens, 0);
  EXPECT_NEAR(session.total_time_ms, 0.0, 1e-6);
}

TEST(ChatTest, TestChatSessionTracksTokenHistory) {
  ChatSession session;
  session.SetTokenHistory({1, 10, 11});
  session.AppendToken(12);
  EXPECT_EQ(session.token_history.size(), 4);
  EXPECT_EQ(session.token_history[0], 1);
  EXPECT_EQ(session.token_history[3], 12);
}

TEST(ChatTest, TestChatSessionRecordTurnAccumulates) {
  ChatSession session;
  session.RecordTurn(10, 5, 100.0);
  session.RecordTurn(20, 8, 200.0);
  EXPECT_EQ(session.total_prompt_tokens, 30);
  EXPECT_EQ(session.total_generated_tokens, 13);
  EXPECT_NEAR(session.total_time_ms, 300.0, 1e-6);
}

// ---------------------------------------------------------------------------
// PromptBuilder
// ---------------------------------------------------------------------------
TEST(ChatTest, TestPromptBuilderEmptyHistory) {
  PromptBuilder builder;
  std::vector<ChatMessage> messages;
  std::string prompt = builder.Build(messages);
  EXPECT_EQ(prompt, "Assistant:");
}

TEST(ChatTest, TestPromptBuilderMultiTurn) {
  PromptBuilder builder;
  std::vector<ChatMessage> messages = {{"system", "You are helpful."},
                                       {"user", "hi"},
                                       {"assistant", "Hello!"},
                                       {"user", "bye"}};
  std::string prompt = builder.Build(messages);
  EXPECT_EQ(prompt,
            "System: You are helpful.\n"
            "User: hi\n"
            "Assistant: Hello!\n"
            "User: bye\n"
            "Assistant:");
}

TEST(ChatTest, TestPromptBuilderIgnoresUnknownRoles) {
  PromptBuilder builder;
  std::vector<ChatMessage> messages = {{"unknown", "ignored"},
                                       {"user", "hello"}};
  std::string prompt = builder.Build(messages);
  EXPECT_EQ(prompt, "User: hello\nAssistant:");
}

TEST(ChatTest, TestPromptBuilderQwen2TemplateInjectsSystemAndPrefix) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");
  std::vector<ChatMessage> messages = {{"user", "你好"}};
  std::string prompt = builder.Build(messages);
  EXPECT_EQ(prompt,
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "你好<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST(ChatTest, TestPromptBuilderQwen2TemplateKeepsExistingSystem) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");
  std::vector<ChatMessage> messages = {{"system", "Be concise."},
                                       {"user", "hi"}};
  std::string prompt = builder.Build(messages);
  EXPECT_EQ(prompt,
            "<|im_start|>system\n"
            "Be concise.<|im_end|>\n"
            "<|im_start|>user\n"
            "hi<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// ---------------------------------------------------------------------------
// Terminal (smoke tests — just verify no crash)
// ---------------------------------------------------------------------------
TEST(ChatTest, TestTerminalPrintUserPrompt) {
  Terminal term;
  term.PrintUserPrompt();
}

TEST(ChatTest, TestTerminalPrintAssistantPrefix) {
  Terminal term;
  term.PrintAssistantPrefix();
}

TEST(ChatTest, TestTerminalPrintHelp) {
  Terminal term;
  term.PrintHelp();
}

TEST(ChatTest, TestTerminalPrintStats) {
  Terminal term;
  ChatSession session;
  session.RecordTurn(10, 5, 250.0);
  term.PrintStats(session);
}

TEST(ChatTest, TestTerminalPrintParams) {
  Terminal term;
  SamplingParams params;
  params.temperature = 0.7f;
  params.top_k = 40;
  params.seed = 42;
  term.PrintParams(params);
}

TEST(ChatTest, TestTerminalPrintMessage) {
  Terminal term;
  term.PrintMessage("test message");
}

TEST(ChatTest, TestTerminalNewLine) {
  Terminal term;
  term.NewLine();
}

// ---------------------------------------------------------------------------
