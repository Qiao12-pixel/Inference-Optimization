// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "tests/test_names.h"

namespace {

std::filesystem::path TestTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

std::vector<char> ReadBinaryFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
}

void WriteBinaryFile(const std::filesystem::path& path,
                     const std::vector<char>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void AppendU32Le(std::vector<char>& data, uint32_t value) {
  data.push_back(static_cast<char>(value & 0xff));
  data.push_back(static_cast<char>((value >> 8) & 0xff));
  data.push_back(static_cast<char>((value >> 16) & 0xff));
  data.push_back(static_cast<char>((value >> 24) & 0xff));
}

void AppendU64Le(std::vector<char>& data, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
  }
}

void AppendString(std::vector<char>& data, const std::string& value) {
  AppendU64Le(data, value.size());
  data.insert(data.end(), value.begin(), value.end());
}

void WriteU32Le(std::vector<char>& data, size_t offset, uint32_t value) {
  data[offset + 0] = static_cast<char>(value & 0xff);
  data[offset + 1] = static_cast<char>((value >> 8) & 0xff);
  data[offset + 2] = static_cast<char>((value >> 16) & 0xff);
  data[offset + 3] = static_cast<char>((value >> 24) & 0xff);
}

void WriteU64Le(std::vector<char>& data, size_t offset, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data[offset + i] = static_cast<char>((value >> (i * 8)) & 0xff);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// GGUF reader tests
// ---------------------------------------------------------------------------
TEST(GgufTest, TestGgufLoadTestFile) {
  GgufReader reader;
  bool ok = reader.Load("models/tiny/test.gguf");
  if (!ok) {
    FAIL() << "failed to load test.gguf: " << reader.load_error;
  }
  EXPECT_EQ(reader.version, 3);
  EXPECT_EQ(reader.n_tensors, 1);
  EXPECT_EQ(reader.n_metadata, 3);
}

TEST(GgufTest, TestGgufMetadata) {
  GgufReader reader;
  if (!reader.Load("models/tiny/test.gguf")) {
    FAIL() << reader.load_error;
  }

  EXPECT_EQ(reader.metadata.size(), 3);
  EXPECT_EQ(reader.metadata[0].key, "general.architecture");
  EXPECT_EQ(reader.metadata[0].value_str, "llama");
  EXPECT_EQ(reader.metadata[1].key, "general.name");
  EXPECT_EQ(reader.metadata[1].value_str, "tiny-test");
  EXPECT_EQ(reader.metadata[2].key, "general.alignment");
  EXPECT_EQ(reader.metadata[2].value_str, "64");
}

TEST(GgufTest, TestGgufTensorInfo) {
  GgufReader reader;
  if (!reader.Load("models/tiny/test.gguf")) {
    FAIL() << reader.load_error;
  }

  EXPECT_EQ(reader.tensors.size(), 1);
  EXPECT_EQ(reader.tensors[0].name, "token_embd.weight");
  EXPECT_EQ(reader.tensors[0].shape.size(), 2);
  EXPECT_EQ(reader.tensors[0].shape[0], 128);
  EXPECT_EQ(reader.tensors[0].shape[1], 32);
  EXPECT_EQ(reader.tensors[0].type, kGgmlTypeF32);
  EXPECT_EQ(reader.tensors[0].offset, 0);
}

TEST(GgufTest, TestGgufDataOffsetComputed) {
  GgufReader reader;
  if (!reader.Load("models/tiny/test.gguf")) {
    FAIL() << reader.load_error;
  }

  // Data offset should be > 0 and honor general.alignment.
  EXPECT_GT(reader.data_offset, 0);
  EXPECT_EQ(reader.data_offset % 64, 0);
}

TEST(GgufTest, TestGgufRepeatedLoadResetsState) {
  GgufReader reader;
  if (!reader.Load("models/tiny/test.gguf")) {
    FAIL() << reader.load_error;
  }

  bool ok = reader.Load("models/tiny/model.json");
  EXPECT_TRUE(!ok);
  EXPECT_EQ(reader.version, 0);
  EXPECT_EQ(reader.n_tensors, 0);
  EXPECT_EQ(reader.n_metadata, 0);
  EXPECT_TRUE(reader.metadata.empty());
  EXPECT_TRUE(reader.tensors.empty());
  EXPECT_EQ(reader.data_offset, 0);
  EXPECT_EQ(reader.load_error, "invalid GGUF magic");
}

TEST(GgufTest, TestGgufUnsupportedVersionRejected) {
  std::vector<char> data = ReadBinaryFile("models/tiny/test.gguf");
  WriteU32Le(data, 4, 2);

  std::filesystem::path path = TestTempPath("bad_version.gguf");
  WriteBinaryFile(path, data);

  GgufReader reader;
  bool ok = reader.Load(path.string());
  std::filesystem::remove(path);

  EXPECT_TRUE(!ok);
  EXPECT_EQ(reader.load_error, "unsupported GGUF version: 2");
}

TEST(GgufTest, TestGgufTruncatedFileRejected) {
  std::vector<char> data = ReadBinaryFile("models/tiny/test.gguf");
  data.resize(12);

  std::filesystem::path path = TestTempPath("truncated.gguf");
  WriteBinaryFile(path, data);

  GgufReader reader;
  bool ok = reader.Load(path.string());
  std::filesystem::remove(path);

  EXPECT_TRUE(!ok);
  EXPECT_TRUE(!reader.load_error.empty());
}

TEST(GgufTest, TestGgufMissingFile) {
  GgufReader reader;
  bool ok = reader.Load("models/tiny/nonexistent.gguf");
  EXPECT_TRUE(!ok);
  EXPECT_TRUE(!reader.load_error.empty());
}

TEST(GgufTest, TestGgufInvalidMagic) {
  GgufReader reader;
  bool ok = reader.Load("models/tiny/model.json");  // not a GGUF file
  EXPECT_TRUE(!ok);
  EXPECT_EQ(reader.load_error, "invalid GGUF magic");
}

TEST(GgufTest, TestGgufInvalidAlignmentRejected) {
  std::vector<char> data = ReadBinaryFile("models/tiny/test.gguf");

  // general.alignment is the third metadata value in the generated file.
  const std::string needle = "general.alignment";
  auto it = std::search(data.begin(), data.end(), needle.begin(), needle.end());
  EXPECT_TRUE(it != data.end());
  size_t alignment_value_offset =
      static_cast<size_t>(it - data.begin()) + needle.size() + 4;
  WriteU32Le(data, alignment_value_offset, 48);

  std::filesystem::path path = TestTempPath("bad_alignment.gguf");
  WriteBinaryFile(path, data);

  GgufReader reader;
  bool ok = reader.Load(path.string());
  std::filesystem::remove(path);

  EXPECT_TRUE(!ok);
  EXPECT_EQ(reader.load_error, "invalid general.alignment");
}

TEST(GgufTest, TestGgufDuplicateMetadataKeyRejected) {
  std::vector<char> data;
  data.insert(data.end(), {'G', 'G', 'U', 'F'});
  AppendU32Le(data, 3);
  AppendU64Le(data, 0);
  AppendU64Le(data, 2);

  AppendString(data, "general.name");
  AppendU32Le(data, kGgufTypeString);
  AppendString(data, "first");

  AppendString(data, "general.name");
  AppendU32Le(data, kGgufTypeString);
  AppendString(data, "second");

  std::filesystem::path path = TestTempPath("duplicate_metadata.gguf");
  WriteBinaryFile(path, data);

  GgufReader reader;
  bool ok = reader.Load(path.string());
  std::filesystem::remove(path);

  EXPECT_TRUE(!ok);
  EXPECT_EQ(reader.load_error, "duplicate metadata key: general.name");
}

TEST(GgufTest, TestGgufZeroTensorDimensionRejected) {
  std::vector<char> data = ReadBinaryFile("models/tiny/test.gguf");

  const std::string tensor_name = "token_embd.weight";
  auto it = std::search(data.begin(), data.end(), tensor_name.begin(),
                        tensor_name.end());
  EXPECT_TRUE(it != data.end());
  size_t first_dim_offset =
      static_cast<size_t>(it - data.begin()) + tensor_name.size() + 4;
  WriteU64Le(data, first_dim_offset, 0);

  std::filesystem::path path = TestTempPath("zero_dim.gguf");
  WriteBinaryFile(path, data);

  GgufReader reader;
  bool ok = reader.Load(path.string());
  std::filesystem::remove(path);

  EXPECT_TRUE(!ok);
  EXPECT_EQ(reader.load_error,
            "invalid zero tensor dimension for: token_embd.weight");
}

TEST(GgufTest, TestGgufTensorDataBoundsChecked) {
  std::vector<char> data = ReadBinaryFile("models/tiny/test.gguf");

  const std::string tensor_name = "token_embd.weight";
  auto it = std::search(data.begin(), data.end(), tensor_name.begin(),
                        tensor_name.end());
  EXPECT_TRUE(it != data.end());
  size_t offset_field = static_cast<size_t>(it - data.begin()) +
                        tensor_name.size() + 4 + 8 + 8 + 4;
  WriteU64Le(data, offset_field, static_cast<uint64_t>(data.size()));

  std::filesystem::path path = TestTempPath("bad_tensor_bounds.gguf");
  WriteBinaryFile(path, data);

  GgufReader reader;
  bool ok = reader.Load(path.string());
  std::filesystem::remove(path);

  EXPECT_TRUE(!ok);
  EXPECT_EQ(reader.load_error,
            "tensor data outside file bounds: token_embd.weight");
}

// ---------------------------------------------------------------------------
// Type name helpers
// ---------------------------------------------------------------------------
TEST(GgufTest, TestGgmlTypeName) {
  EXPECT_EQ(std::string(GgmlTypeName(kGgmlTypeF32)), "F32");
  EXPECT_EQ(std::string(GgmlTypeName(kGgmlTypeF16)), "F16");
  EXPECT_EQ(std::string(GgmlTypeName(kGgmlTypeQ40)), "Q4_0");
  EXPECT_EQ(std::string(GgmlTypeName(kGgmlTypeQ80)), "Q8_0");
  EXPECT_EQ(std::string(GgmlTypeName(999)), "UNKNOWN");
}

TEST(GgufTest, TestGgufValueTypeName) {
  EXPECT_EQ(std::string(GgufValueTypeName(kGgufTypeUint32)), "uint32");
  EXPECT_EQ(std::string(GgufValueTypeName(kGgufTypeString)), "string");
  EXPECT_EQ(std::string(GgufValueTypeName(kGgufTypeArray)), "array");
  EXPECT_EQ(std::string(GgufValueTypeName(999)), "unknown");
}

// ---------------------------------------------------------------------------
