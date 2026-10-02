// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_MODEL_H_
#define INCLUDE_MINI_LLAMA_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

enum class RopeType {
  kNormal,
  kNeoX,
};

struct ModelConfig {
  int vocab_size = 128;
  int dim = 32;
  int hidden_dim = 86;
  int n_layers = 2;
  int n_heads = 4;
  int n_kv_heads = 4;
  int head_dim = 8;
  int max_seq_len = 128;
  float rope_theta = 10000.0f;
  float rms_norm_eps = 1e-5f;
  RopeType rope_type = RopeType::kNormal;
};

struct LayerWeights {
  Tensor attention_norm;   // [dim] F32
  QuantizedTensor wq;      // [n_heads * head_dim, dim]  F32/Q8_0/Q4_0
  QuantizedTensor wk;      // [n_kv_heads * head_dim, dim]
  QuantizedTensor wv;      // [n_kv_heads * head_dim, dim]
  Tensor bq;               // optional [n_heads * head_dim] F32
  Tensor bk;               // optional [n_kv_heads * head_dim] F32
  Tensor bv;               // optional [n_kv_heads * head_dim] F32
  QuantizedTensor wo;      // [dim, n_heads * head_dim]
  Tensor ffn_norm;         // [dim] F32
  QuantizedTensor w_gate;  // [hidden_dim, dim]
  QuantizedTensor w_up;    // [hidden_dim, dim]
  QuantizedTensor w_down;  // [dim, hidden_dim]
};

struct CudaUploadedWeight {
  std::string name;
  QuantType type = QuantType::kF32;
  bool linear = true;
  std::vector<int> shape;
  size_t block_count = 0;
  CudaDeviceBuffer buffer;
  // Optional F32 dequantized Q8_0 copy used only by the experimental batched
  // Prefill GEMM path. The original quantized buffer remains used for Decode.
  CudaDeviceBuffer prefill_f32_buffer;
  CudaDeviceBuffer prefill_f16_buffer;
};

struct CudaModelWeights {
  int device_id = 0;
  size_t uploaded_weight_count = 0;
  size_t uploaded_bytes = 0;
  size_t linear_calls = 0;
  size_t linear_argmax_dispatch_calls = 0;
  size_t activation_calls = 0;
  size_t attention_calls = 0;
  size_t attention_cpu_fallbacks = 0;
  size_t kv_cache_write_bytes = 0;
  size_t kv_cache_read_bytes = 0;
  size_t host_to_device_copies = 0;
  size_t device_to_host_copies = 0;
  size_t host_to_device_bytes = 0;
  size_t device_to_host_bytes = 0;
  bool q80_prefill_f32_enabled = false;
  bool q80_prefill_f16_enabled = false;
  std::vector<CudaUploadedWeight> weights;
};

struct MiniLlamaModel {
  ModelConfig config;
  bool loaded = false;
  std::string load_error;
  Tensor token_embedding;            // [vocab_size, dim] F32
  std::vector<LayerWeights> layers;  // [n_layers]
  Tensor final_norm;                 // [dim] F32
  QuantizedTensor lm_head;           // [vocab_size, dim] F32/Q8_0/Q4_0
  std::shared_ptr<CudaModelWeights> cuda_weights;
};

// Return the actual bytes consumed by model weights in their current format.
size_t ModelWeightBytes(const MiniLlamaModel& model);

// Return the bytes the same weights would consume if stored as F32.
size_t ModelWeightBytesF32(const MiniLlamaModel& model);

// Upload CUDA-supported resident weights and keep CPU weights intact.
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id = 0);

// Materialize Q8_0 Linear weights as additional GPU-resident F32 buffers.
// ForwardPrefill uses the F32 buffers through cuBLAS while Decode keeps the
// original Q8_0 kernel. This trades VRAM for batched Prefill throughput.
void EnableModelQ80PrefillF32(MiniLlamaModel& model);

// Materialize Q8_0 Linear weights as GPU-resident FP16 buffers for the
// experimental Tensor Core Prefill path. Decode remains Q8_0.
void EnableModelQ80PrefillF16(MiniLlamaModel& model);

void ClearModelCudaWeights(MiniLlamaModel& model);

bool ModelHasCudaWeights(const MiniLlamaModel& model);

size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model);

size_t ModelCudaMemoryBytes(const MiniLlamaModel& model);

size_t ModelCudaQ80PrefillF32Bytes(const MiniLlamaModel& model);
size_t ModelCudaQ80PrefillF16Bytes(const MiniLlamaModel& model);

// Bytes required to materialize F32 copies of all Q8_0 Transformer Linear
// weights used by Prefill. The lm_head remains on the Q8 Decode path.
size_t ModelQ80PrefillF32CacheBytes(const MiniLlamaModel& model);

void ResetModelCudaRuntimeStats(MiniLlamaModel& model);

size_t ModelCudaLinearCalls(const MiniLlamaModel& model);

size_t ModelCudaLinearArgMaxDispatchCalls(const MiniLlamaModel& model);

size_t ModelCudaActivationCalls(const MiniLlamaModel& model);

size_t ModelCudaAttentionCalls(const MiniLlamaModel& model);

size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model);

size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model);

size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model);

size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model);

size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model);

size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model);

size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model);

// Runtime instrumentation for CUDA paths that return a scalar token rather
// than a Tensor (for example, GPU-side ArgMax sampling).
void RecordModelCudaDeviceToHostCopy(const MiniLlamaModel& model,
                                     size_t bytes);

// Convert all Linear weight QuantizedTensors from F32 to Q8_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void QuantizeModelToQ80(MiniLlamaModel& model);

// Convert all Linear weight QuantizedTensors from F32 to Q4_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void QuantizeModelToQ40(MiniLlamaModel& model);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MODEL_H_
