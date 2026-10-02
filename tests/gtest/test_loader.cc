// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "mini_llama/loader.h"
#include "tests/test_names.h"

namespace {

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}
std::filesystem::path TestTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() / ("mini_llama_" + name);
}

void WriteTextFile(const std::filesystem::path& path,
                   const std::string& content) {
  std::ofstream out(path);
  out << content;
}

bool ReplaceOnce(std::string& text, const std::string& from,
                 const std::string& to) {
  size_t pos = text.find(from);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

bool ReplaceAfter(std::string& text, const std::string& marker,
                  const std::string& from, const std::string& to) {
  size_t marker_pos = text.find(marker);
  if (marker_pos == std::string::npos) {
    return false;
  }
  size_t pos = text.find(from, marker_pos);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

bool ParseManifestFails(const std::string& json, const std::string& filename) {
  std::filesystem::path path = TestTempPath(filename);
  WriteTextFile(path, json);
  try {
    ParseManifest(path.string());
    std::filesystem::remove(path);
    return false;
  } catch (const std::runtime_error&) {
    std::filesystem::remove(path);
    return true;
  }
}

std::string ReplaceTokenizerType(const std::string& json,
                                 const std::string& tokenizer_type) {
  std::string updated = json;
  if (!ReplaceOnce(updated, "\"type\": \"json_vocab\"",
                   "\"type\": \"" + tokenizer_type + "\"")) {
    throw std::runtime_error("failed to replace tokenizer type");
  }
  return updated;
}

std::string RemoveTokenizerBlock(const std::string& json) {
  const std::string tokenizer_block =
      "  \"tokenizer\": {\n"
      "    \"type\": \"json_vocab\",\n"
      "    \"path\": \"vocab.json\",\n"
      "    \"bos_id\": 1,\n"
      "    \"eos_id\": 2,\n"
      "    \"unk_id\": 0\n"
      "  },\n";
  std::string updated = json;
  if (!ReplaceOnce(updated, tokenizer_block, "")) {
    throw std::runtime_error("failed to remove tokenizer block");
  }
  return updated;
}

size_t CudaResidentWeightBytes(const MiniLlamaModel& model) {
  auto bytes_for = [](const QuantizedTensor& weight) -> size_t {
    if (weight.type != QuantType::kF32) {
      return 0;
    }
    return weight.f32_data.size() * sizeof(float);
  };
  auto tensor_bytes = [](const Tensor& tensor) -> size_t {
    return tensor.data.size() * sizeof(float);
  };

  size_t bytes = tensor_bytes(model.token_embedding) +
                 tensor_bytes(model.final_norm) + bytes_for(model.lm_head);
  for (const auto& layer : model.layers) {
    bytes += tensor_bytes(layer.attention_norm);
    bytes += bytes_for(layer.wq);
    bytes += bytes_for(layer.wk);
    bytes += bytes_for(layer.wv);
    bytes += tensor_bytes(layer.bq);
    bytes += tensor_bytes(layer.bk);
    bytes += tensor_bytes(layer.bv);
    bytes += bytes_for(layer.wo);
    bytes += tensor_bytes(layer.ffn_norm);
    bytes += bytes_for(layer.w_gate);
    bytes += bytes_for(layer.w_up);
    bytes += bytes_for(layer.w_down);
  }
  return bytes;
}

}  // namespace

TEST(LoaderTest, TestLoaderManifestParsesMetadata) {
  ModelManifest manifest = ParseManifest("models/tiny/model.json");
  EXPECT_EQ(manifest.config.vocab_size, 128);
  EXPECT_EQ(manifest.config.n_layers, 2);
  EXPECT_EQ(manifest.tensors.size(), 21);
  EXPECT_EQ(manifest.tensors[0].name, "token_embedding");
  EXPECT_EQ(manifest.tensors[0].dtype, "float32");
  EXPECT_EQ(manifest.tensors[0].offset, 0);
  EXPECT_EQ(manifest.tensors[0].byte_size, 16384);
  EXPECT_EQ(manifest.tokenizer.type, "json_vocab");
  EXPECT_EQ(manifest.tokenizer.path, "vocab.json");
  EXPECT_EQ(manifest.tokenizer.bos_id, 1);
  EXPECT_EQ(manifest.tokenizer.eos_id, 2);
  EXPECT_EQ(manifest.tokenizer.unk_id, 0);
}

TEST(LoaderTest, TestLoaderManifestDefaultsAsciiWithoutTokenizerBlock) {
  std::string json = ReadTextFile("models/tiny/model.json");
  std::string without_tokenizer = RemoveTokenizerBlock(json);
  std::filesystem::path path = TestTempPath("tokenizer_manifest.json");
  WriteTextFile(path, without_tokenizer);

  ModelManifest manifest = ParseManifest(path.string());
  std::filesystem::remove(path);

  EXPECT_EQ(manifest.tokenizer.type, "ascii");
  EXPECT_TRUE(manifest.tokenizer.path.empty());
  EXPECT_EQ(manifest.tokenizer.bos_id, 1);
  EXPECT_EQ(manifest.tokenizer.eos_id, 2);
  EXPECT_EQ(manifest.tokenizer.unk_id, 0);
}

TEST(LoaderTest, TestLoaderRejectsInvalidTokenizerType) {
  std::string json = ReadTextFile("models/tiny/model.json");
  std::string with_tokenizer = ReplaceTokenizerType(json, "bpe");
  EXPECT_TRUE(
      ParseManifestFails(with_tokenizer, "invalid_tokenizer_type.json"));
}

TEST(LoaderTest, TestLoaderRejectsDuplicateTensorName) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceOnce(json, "\"name\": \"final_norm\"",
                          "\"name\": \"token_embedding\""));
  EXPECT_TRUE(ParseManifestFails(json, "duplicate_tensor.json"));
}

TEST(LoaderTest, TestLoaderRejectsRequiredShapeMismatch) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                           "\"shape\": [\n        128,\n        32\n      ]",
                           "\"shape\": [\n        64,\n        64\n      ]"));
  EXPECT_TRUE(ParseManifestFails(json, "shape_mismatch.json"));
}

TEST(LoaderTest, TestLoaderRejectsOverlappingTensorRanges) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"", "\"offset\": 115840",
                           "\"offset\": 115712"));
  EXPECT_TRUE(ParseManifestFails(json, "overlap.json"));
}

TEST(LoaderTest, TestLoaderRejectsMissingDtype) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                           "      \"dtype\": \"float32\",\n", ""));
  EXPECT_TRUE(ParseManifestFails(json, "missing_dtype.json"));
}

TEST(LoaderTest, TestLoaderRejectsFractionalIntegerConfig) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(ReplaceOnce(json, "\"n_heads\": 4", "\"n_heads\": 4.5"));
  EXPECT_TRUE(ParseManifestFails(json, "fractional_integer_config.json"));
}

TEST(LoaderTest, TestLoaderRejectsNonfiniteFloatConfig) {
  std::string json = ReadTextFile("models/tiny/model.json");
  EXPECT_TRUE(
      ReplaceOnce(json, "\"rope_theta\": 10000.0", "\"rope_theta\": NaN"));
  EXPECT_TRUE(ParseManifestFails(json, "nonfinite_float_config.json"));
}

TEST(LoaderTest, TestLoaderRejectsTrailingWeightBytes) {
  std::filesystem::path weights_path = TestTempPath("trailing_model.bin");
  {
    std::ifstream in("models/tiny/model.bin", std::ios::binary);
    std::ofstream out(weights_path, std::ios::binary);
    out << in.rdbuf();
    char extra = 0;
    out.write(&extra, 1);
  }

  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", weights_path.string());
  std::filesystem::remove(weights_path);
  EXPECT_TRUE(!model.loaded);
  EXPECT_TRUE(model.load_error.find("trailing bytes") != std::string::npos);
}

TEST(LoaderTest, TestModelCudaWeightStorageState) {
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  ASSERT_TRUE(model.loaded);
  EXPECT_FALSE(ModelHasCudaWeights(model));
  EXPECT_EQ(ModelCudaUploadedWeightCount(model), 0);
  EXPECT_EQ(ModelCudaMemoryBytes(model), 0);

  if (!CudaRuntimeBuilt()) {
    EXPECT_THROW(
        {
          try {
            UploadModelWeightsToCuda(model, 0);
          } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("CUDA backend was not built"),
                      std::string::npos);
            throw;
          }
        },
        std::runtime_error);
    EXPECT_FALSE(ModelHasCudaWeights(model));
    return;
  }

  if (CudaDeviceCount() == 0) {
    return;
  }

  UploadModelWeightsToCuda(model, 0);
  EXPECT_TRUE(ModelHasCudaWeights(model));
  EXPECT_EQ(ModelCudaUploadedWeightCount(model), 21);
  EXPECT_EQ(ModelCudaMemoryBytes(model), CudaResidentWeightBytes(model));

  ClearModelCudaWeights(model);
  EXPECT_FALSE(ModelHasCudaWeights(model));
  EXPECT_EQ(ModelCudaUploadedWeightCount(model), 0);
  EXPECT_EQ(ModelCudaMemoryBytes(model), 0);
}
