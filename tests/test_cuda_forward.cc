// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/ops.h"
#include "mini_llama/sampler.h"
#include "tests/test_names.h"

namespace {

constexpr float kAbsTol = 2e-3f;
constexpr float kRelTol = 2e-3f;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool CloseEnough(float actual, float expected) {
  float abs_err = std::abs(actual - expected);
  float scale = std::max(1.0f, std::abs(expected));
  return abs_err <= kAbsTol || abs_err / scale <= kRelTol;
}

void RequireCloseTensor(const Tensor& actual, const Tensor& expected,
                        const std::string& label) {
  Require(actual.shape == expected.shape, label + ": shape mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!CloseEnough(actual.data[i], expected.data[i])) {
      throw std::runtime_error(
          label + ": value mismatch at " + std::to_string(i) +
          ", actual=" + std::to_string(actual.data[i]) +
          ", expected=" + std::to_string(expected.data[i]));
    }
  }
}

void RequireCudaNotBuilt() {
  Require(!CudaRuntimeBuilt(), "CudaRuntimeBuilt should be false in CPU build");
  Require(!CudaOpsBuilt(), "CudaOpsBuilt should be false in CPU build");
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(model.loaded, "tiny model should load");
  Require(ModelCudaActivationCalls(model) == 0,
          "CPU model should report zero CUDA activation calls");
}

void TestCudaForwardFullTinyPath() {
  Require(CudaRuntimeBuilt(), "CudaRuntimeBuilt should be true in CUDA build");
  Require(CudaOpsBuilt(), "CudaOpsBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  MiniLlamaModel cpu_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(cpu_model.loaded, "CPU tiny model should load");
  Require(cuda_model.loaded, "CUDA tiny model should load");

  MiniLlamaContext cpu_ctx(&cpu_model);
  MiniLlamaContext cuda_ctx(&cuda_model);
  MiniBatch batch = MiniBatch::FromTokens({1, 2, 3}, 0);

  Tensor cpu_logits = ForwardBatch(cpu_ctx, cpu_model, batch);

  UploadModelWeightsToCuda(cuda_model, 0);
  ResetModelCudaRuntimeStats(cuda_model);
  Tensor cuda_logits = ForwardBatch(cuda_ctx, cuda_model, batch);

  RequireCloseTensor(cuda_logits, cpu_logits, "cuda_forward logits");
  Require(ModelCudaLinearCalls(cuda_model) == 15,
          "batched Prefill should run one set of 15 CUDA Linear calls");
  Require(ModelCudaActivationCalls(cuda_model) == 13,
          "batched Prefill should run 13 CUDA activation calls");
  Require(ModelCudaAttentionCalls(cuda_model) == 2,
          "batched Prefill should run one CUDA attention call per layer");
  Require(!cuda_ctx.cuda_kv_cache.empty(),
          "CUDA forward should allocate GPU KV cache");

  MiniLlamaModel device_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(device_model.loaded, "tiny model should load for device logits");
  MiniLlamaContext device_ctx(&device_model);
  UploadModelWeightsToCuda(device_model, 0);
  ResetModelCudaRuntimeStats(device_model);
  CudaTensor device_logits;
  (void)ForwardBatch(device_ctx, device_model, batch, &device_logits);
  RequireCloseTensor(device_logits.Download(), cpu_logits,
                     "cuda_forward device logits");

  SamplingParams greedy_params;
  MiniSampler sampler(greedy_params);
  Require(sampler.Sample(device_logits, greedy_params, device_model) ==
              ArgMax(cpu_logits),
          "GPU ArgMax should select the same greedy token as CPU");
  Require(ModelCudaDeviceToHostCopies(device_model) == 1,
          "GPU greedy sampling should only copy one token id");
  Require(ModelCudaDeviceToHostBytes(device_model) == sizeof(int),
          "GPU greedy sampling should not download full logits");

  MiniLlamaModel fused_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaContext fused_ctx(&fused_model);
  UploadModelWeightsToCuda(fused_model, 0);
  ResetModelCudaRuntimeStats(fused_model);
  MiniLlamaContext cpu_single_ctx(&cpu_model);
  Tensor cpu_single_logits =
      ForwardBatch(cpu_single_ctx, cpu_model, MiniBatch::Single(1, 0));
  int fused_token = -1;
  Require(ForwardTokenGreedyCuda(fused_ctx, fused_model, 1, fused_token),
          "F32 tiny lm_head should support fused greedy Decode");
  Require(fused_token == ArgMax(cpu_single_logits),
          "fused greedy Decode should match CPU ArgMax");
  Require(ModelCudaLinearCalls(fused_model) == 15,
          "fused greedy Decode should retain one lm_head Linear accounting");
  Require(ModelCudaLinearArgMaxDispatchCalls(fused_model) == 1,
          "fused greedy Decode should record one Linear ArgMax dispatch");
  Require(ModelCudaDeviceToHostBytes(fused_model) == sizeof(int),
          "fused greedy Decode should only download one token id");

  MiniLlamaModel cpu_q8_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_q8_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  QuantizeModelToQ80(cpu_q8_model);
  QuantizeModelToQ80(cuda_q8_model);
  MiniLlamaContext cpu_q8_ctx(&cpu_q8_model);
  const Tensor cpu_q8_logits =
      ForwardBatch(cpu_q8_ctx, cpu_q8_model, MiniBatch::Single(1, 0));
  MiniLlamaContext cuda_q8_ctx(&cuda_q8_model);
  UploadModelWeightsToCuda(cuda_q8_model, 0);
  ResetModelCudaRuntimeStats(cuda_q8_model);
  int q8_token = -1;
  Require(ForwardTokenGreedyCuda(cuda_q8_ctx, cuda_q8_model, 1, q8_token),
          "Q8 lm_head should use the generic Linear ArgMax dispatcher");
  Require(q8_token == ArgMax(cpu_q8_logits),
          "Q8 fused Linear ArgMax should match CPU ArgMax");
  Require(ModelCudaLinearArgMaxDispatchCalls(cuda_q8_model) == 1,
          "Q8 fused Linear ArgMax should record one dispatcher call");

  MiniLlamaModel cpu_q4_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_q4_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  QuantizeModelToQ40(cpu_q4_model);
  QuantizeModelToQ40(cuda_q4_model);
  MiniLlamaContext cpu_q4_ctx(&cpu_q4_model);
  const Tensor cpu_q4_logits =
      ForwardBatch(cpu_q4_ctx, cpu_q4_model, MiniBatch::Single(1, 0));
  MiniLlamaContext cuda_q4_ctx(&cuda_q4_model);
  UploadModelWeightsToCuda(cuda_q4_model, 0);
  ResetModelCudaRuntimeStats(cuda_q4_model);
  int q4_token = -1;
  Require(ForwardTokenGreedyCuda(cuda_q4_ctx, cuda_q4_model, 1, q4_token),
          "Q4 lm_head should use the generic Linear ArgMax dispatcher");
  Require(q4_token == ArgMax(cpu_q4_logits),
          "Q4 fused Linear ArgMax should match CPU ArgMax");
  Require(ModelCudaLinearArgMaxDispatchCalls(cuda_q4_model) == 1,
          "Q4 fused Linear ArgMax should record one dispatcher call");
}

}  // namespace

int main() {
  try {
#ifdef MINI_LLAMA_USE_CUDA
    TestCudaForwardFullTinyPath();
    std::cout << "PASS cuda_forward\n";
#else
    RequireCudaNotBuilt();
    std::cout << "PASS cuda_forward_cpu_build\n";
#endif
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL cuda_forward: " << e.what() << "\n";
    return 1;
  }
}
