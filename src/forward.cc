// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_linear_argmax.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

namespace mini_llama {

static void ValidateForwardInputs(const MiniLlamaContext& ctx,
                                  const MiniLlamaModel& model, int token) {
  const ModelConfig& c = model.config;
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }
  if (token < 0 || token >= c.vocab_size) {
    throw std::out_of_range("ForwardToken token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
    throw std::out_of_range("ForwardToken position out of range");
  }
  if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
    throw std::runtime_error(
        "ForwardToken layer count does not match model config");
  }
  if (model.token_embedding.data.size() <
      static_cast<size_t>(c.vocab_size * c.dim)) {
    throw std::runtime_error(
        "ForwardToken token embedding tensor is smaller than config");
  }
}

static void RequireShape(const Tensor& t, const std::vector<int>& expected,
                         const char* caller) {
  t.AssertShape(expected, caller);
}

static Tensor AddOptionalBias(const Tensor& x, const Tensor& bias,
                              const char* caller) {
  if (bias.data.empty()) {
    return x;
  }
  if (x.num_dims() != 1 || bias.num_dims() != 1 ||
      x.shape[0] != bias.shape[0]) {
    throw std::runtime_error(std::string(caller) +
                             ": bias shape mismatch x=" + x.ShapeStringShort() +
                             " bias=" + bias.ShapeStringShort());
  }

  Tensor y = x;
  for (int i = 0; i < x.shape[0]; ++i) {
    y.data[i] += bias.data[i];
  }
  return y;
}

static const CudaUploadedWeight* FindCudaWeight(const MiniLlamaModel& model,
                                                const std::string& name,
                                                QuantType type) {
  if (!model.cuda_weights) {
    return nullptr;
  }
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.name == name && weight.type == type) {
      return &weight;
    }
  }
  return nullptr;
}

static void RecordCudaLinearCopy(const MiniLlamaModel& model, const Tensor& x,
                                 const Tensor& y) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls += 1;
  model.cuda_weights->host_to_device_copies += 1;
  model.cuda_weights->device_to_host_copies += 1;
  model.cuda_weights->host_to_device_bytes += x.size() * sizeof(float);
  model.cuda_weights->device_to_host_bytes += y.size() * sizeof(float);
}

static void RecordCudaLinearCall(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls += 1;
}

static void RecordCudaHostToDeviceCopy(const MiniLlamaModel& model,
                                       size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->host_to_device_copies += 1;
  model.cuda_weights->host_to_device_bytes += bytes;
}

static void RecordCudaDeviceToHostCopy(const MiniLlamaModel& model,
                                       size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->device_to_host_copies += 1;
  model.cuda_weights->device_to_host_bytes += bytes;
}

static void RecordCudaActivationCall(const MiniLlamaModel& model,
                                     size_t calls = 1) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->activation_calls += calls;
}

static void RecordCudaAttentionCall(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->attention_calls += 1;
}

static void RecordCudaAttentionCpuFallback(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->attention_cpu_fallbacks += 1;
}

static void RecordCudaKvCacheWrite(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->kv_cache_write_bytes += bytes;
}

static void RecordCudaKvCacheRead(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->kv_cache_read_bytes += bytes;
}

static int CudaDeviceId(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_id : 0;
}

static bool CudaLinearSupported(const QuantizedTensor& weight) {
  return weight.type == QuantType::kF32 || weight.type == QuantType::kQ80 ||
         weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41;
}

static bool CudaLinearReady(const MiniLlamaModel& model,
                            const std::string& name,
                            const QuantizedTensor& weight) {
  return model.cuda_weights && CudaLinearSupported(weight) &&
         FindCudaWeight(model, name, weight.type) != nullptr;
}

static CudaTensor UploadCudaTensor(const MiniLlamaModel& model,
                                   const Tensor& x) {
  CudaTensor x_dev = CudaTensorFromHost(x, CudaDeviceId(model));
  RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
  return x_dev;
}

static Tensor DownloadCudaTensor(const MiniLlamaModel& model,
                                 const CudaTensor& x) {
  Tensor y = x.Download();
  RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
  return y;
}

static CudaTensor ForwardLinearDevice(const MiniLlamaModel& model,
                                      const std::string& name,
                                      const CudaTensor& x,
                                      const QuantizedTensor& weight) {
  if (!CudaLinearSupported(weight)) {
    throw std::runtime_error("CUDA forward Linear unsupported weight type: " +
                             name);
  }

  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, weight.type);
  if (cuda_weight == nullptr) {
    throw std::runtime_error("CUDA forward Linear missing uploaded weight: " +
                             name);
  }

  CudaTensor y;
  switch (weight.type) {
    case QuantType::kF32:
      y = CudaLinearDeviceInput(x, cuda_weight->buffer.data(),
                                cuda_weight->shape, CudaDeviceId(model));
      break;
    case QuantType::kQ80:
      if (x.num_dims() == 2 && !cuda_weight->prefill_f16_buffer.empty()) {
        y = CudaLinearF16DeviceInput(x, cuda_weight->prefill_f16_buffer.data(),
                                     cuda_weight->shape, CudaDeviceId(model));
      } else if (x.num_dims() == 2 &&
                 !cuda_weight->prefill_f32_buffer.empty()) {
        y = CudaLinearDeviceInput(x, cuda_weight->prefill_f32_buffer.data(),
                                  cuda_weight->shape, CudaDeviceId(model));
      } else {
        y = CudaQ80LinearDeviceInput(
            x, cuda_weight->buffer.data(), cuda_weight->block_count,
            cuda_weight->shape, CudaDeviceId(model));
      }
      break;
    case QuantType::kQ40:
      y = CudaQ40LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
    case QuantType::kQ41:
      y = CudaQ41LinearDeviceInput(x, cuda_weight->buffer.data(),
                                   cuda_weight->block_count, cuda_weight->shape,
                                   CudaDeviceId(model));
      break;
  }
  RecordCudaLinearCall(model);
  return y;
}

static Tensor ForwardLinear(const MiniLlamaModel& model,
                            const std::string& name, const Tensor& x,
                            const QuantizedTensor& weight) {
  if (!model.cuda_weights) {
    return Linear(x, weight);
  }

  if (!CudaLinearSupported(weight)) {
    return Linear(x, weight);
  }

  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, weight.type);
  if (cuda_weight == nullptr) {
    throw std::runtime_error("CUDA forward Linear missing uploaded weight: " +
                             name);
  }
  Tensor y;
  switch (weight.type) {
    case QuantType::kF32:
      y = CudaLinearDeviceWeight(x, cuda_weight->buffer.data(),
                                 cuda_weight->shape, nullptr,
                                 model.cuda_weights->device_id);
      break;
    case QuantType::kQ80:
      y = CudaQ80LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
    case QuantType::kQ40:
      y = CudaQ40LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
    case QuantType::kQ41:
      y = CudaQ41LinearDeviceWeight(
          x, cuda_weight->buffer.data(), cuda_weight->block_count,
          cuda_weight->shape, nullptr, model.cuda_weights->device_id);
      break;
  }
  RecordCudaLinearCopy(model, x, y);
  return y;
}

static void ForwardQkvProjection(const MiniLlamaModel& model,
                                 const std::string& layer_prefix,
                                 const Tensor& h, const LayerWeights& lw,
                                 Tensor& q_flat, Tensor& k_flat,
                                 Tensor& v_flat) {
  const std::string q_name = layer_prefix + "wq";
  const std::string k_name = layer_prefix + "wk";
  const std::string v_name = layer_prefix + "wv";
  if (CudaLinearReady(model, q_name, lw.wq) &&
      CudaLinearReady(model, k_name, lw.wk) &&
      CudaLinearReady(model, v_name, lw.wv)) {
    CudaTensor h_dev = UploadCudaTensor(model, h);
    CudaTensor q_dev = ForwardLinearDevice(model, q_name, h_dev, lw.wq);
    CudaTensor k_dev = ForwardLinearDevice(model, k_name, h_dev, lw.wk);
    CudaTensor v_dev = ForwardLinearDevice(model, v_name, h_dev, lw.wv);
    q_flat = AddOptionalBias(DownloadCudaTensor(model, q_dev), lw.bq,
                             "forward_layer q");
    k_flat = AddOptionalBias(DownloadCudaTensor(model, k_dev), lw.bk,
                             "forward_layer k");
    v_flat = AddOptionalBias(DownloadCudaTensor(model, v_dev), lw.bv,
                             "forward_layer v");
    return;
  }

  q_flat = AddOptionalBias(ForwardLinear(model, q_name, h, lw.wq), lw.bq,
                           "forward_layer q");
  k_flat = AddOptionalBias(ForwardLinear(model, k_name, h, lw.wk), lw.bk,
                           "forward_layer k");
  v_flat = AddOptionalBias(ForwardLinear(model, v_name, h, lw.wv), lw.bv,
                           "forward_layer v");
}

static CudaTensor AddOptionalBiasDevice(const MiniLlamaModel& model,
                                        const std::string& name, CudaTensor x,
                                        const Tensor& bias) {
  if (bias.data.empty()) {
    return x;
  }
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error("CUDA forward bias missing uploaded weight: " +
                             name);
  }
  if (cuda_weight->shape != bias.shape) {
    throw std::runtime_error(
        "CUDA forward bias uploaded weight shape mismatch: " + name);
  }
  CudaTensor y = x.num_dims() == 1
                     ? CudaElementwiseAddDeviceWeight(
                           x, cuda_weight->buffer.data(), cuda_weight->shape,
                           CudaDeviceId(model))
                     : CudaAddRowBiasDeviceWeight(
                           x, cuda_weight->buffer.data(), cuda_weight->shape,
                           CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static void ForwardQkvProjectionDevice(const MiniLlamaModel& model,
                                       const std::string& layer_prefix,
                                       const CudaTensor& h,
                                       const LayerWeights& lw, CudaTensor& q,
                                       CudaTensor& k, CudaTensor& v) {
  q = ForwardLinearDevice(model, layer_prefix + "wq", h, lw.wq);
  k = ForwardLinearDevice(model, layer_prefix + "wk", h, lw.wk);
  v = ForwardLinearDevice(model, layer_prefix + "wv", h, lw.wv);
  q = AddOptionalBiasDevice(model, layer_prefix + "bq", std::move(q), lw.bq);
  k = AddOptionalBiasDevice(model, layer_prefix + "bk", std::move(k), lw.bk);
  v = AddOptionalBiasDevice(model, layer_prefix + "bv", std::move(v), lw.bv);
}

static Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                             const Tensor& weight, float eps) {
  if (!model.cuda_weights) {
    return RmsNorm(x, weight, eps);
  }
  Tensor y = CudaRmsNorm(x, weight, eps, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static CudaTensor ForwardRmsNormDevice(const MiniLlamaModel& model,
                                       const std::string& name,
                                       const CudaTensor& x,
                                       const Tensor& weight,
                                       CudaDeviceBuffer& sum_workspace,
                                       float eps) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error("CUDA forward RMSNorm missing uploaded weight: " +
                             name);
  }
  if (cuda_weight->shape != weight.shape) {
    throw std::runtime_error(
        "CUDA forward RMSNorm uploaded weight shape mismatch: " + name);
  }
  CudaTensor y =
      CudaRmsNormDeviceWeight(x, cuda_weight->buffer.data(), cuda_weight->shape,
                              sum_workspace, eps, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static CudaTensor ForwardRmsNormBatchDevice(const MiniLlamaModel& model,
                                            const std::string& name,
                                            const CudaTensor& x,
                                            const Tensor& weight, float eps) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear ||
      cuda_weight->shape != weight.shape) {
    throw std::runtime_error("CUDA prefill RMSNorm missing uploaded weight: " +
                             name);
  }
  CudaTensor y = CudaRmsNormBatchDeviceWeight(
      x, cuda_weight->buffer.data(), cuda_weight->shape, eps,
      CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                            const Tensor& up) {
  if (!model.cuda_weights) {
    return SwiGlu(gate, up);
  }
  Tensor silu_gate = CudaSilu(gate, CudaDeviceId(model));
  Tensor y = CudaElementwiseMul(silu_gate, up, CudaDeviceId(model));
  RecordCudaActivationCall(model, 2);
  return y;
}

static CudaTensor ForwardAddDevice(const MiniLlamaModel& model,
                                   const CudaTensor& a, const CudaTensor& b) {
  CudaTensor y = CudaElementwiseAddDeviceInput(a, b, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static void ForwardAddRmsNormDevice(const MiniLlamaModel& model,
                                    const std::string& norm_name,
                                    const CudaTensor& a, const CudaTensor& b,
                                    const Tensor& weight, float eps,
                                    CudaTensor& sum_out,
                                    CudaTensor& norm_out) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, norm_name, QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear ||
      cuda_weight->shape != weight.shape) {
    throw std::runtime_error(
        "CUDA forward fused Add/RMSNorm missing uploaded weight: " +
        norm_name);
  }
  CudaAddRmsNormDeviceWeight(a, b, cuda_weight->buffer.data(),
                              cuda_weight->shape, eps, sum_out, norm_out,
                              CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

static Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a,
                         const Tensor& b) {
  if (!model.cuda_weights) {
    if (a.shape != b.shape) {
      throw std::runtime_error("forward_add: shape mismatch " +
                               a.ShapeStringShort() + " vs " +
                               b.ShapeStringShort());
    }
    Tensor y(a.shape, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
      y.data[i] = a.data[i] + b.data[i];
    }
    return y;
  }
  Tensor y = CudaElementwiseAdd(a, b, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
  if (!model.cuda_weights) {
    return Softmax(x);
  }
  Tensor y = CudaSoftmax(x, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

static void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k,
                        int pos, float theta, RopeType rope_type) {
  if (!model.cuda_weights) {
    Rope(q, k, pos, theta, rope_type);
    return;
  }
  CudaRope(q, k, pos, theta, rope_type, CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

static void ForwardRopeDevice(const MiniLlamaModel& model, CudaTensor& q,
                              CudaTensor& k, int n_heads, int n_kv_heads,
                              int head_dim, int pos, float theta,
                              RopeType rope_type) {
  CudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta,
                      rope_type, CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

// ---------------------------------------------------------------------------
// Embedding lookup: token_id -> x[dim]
// ---------------------------------------------------------------------------
static Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
  int dim = model.config.dim;
  Tensor x({dim}, 0.0f);
  for (int i = 0; i < dim; ++i) {
    x.data[i] = model.token_embedding.data[token_id * dim + i];
  }
  return x;
}

static CudaTensor EmbedTokenDevice(const MiniLlamaModel& model, int token_id) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, "token_embedding", QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear) {
    throw std::runtime_error(
        "CUDA forward embedding missing uploaded weight: token_embedding");
  }
  if (cuda_weight->shape != model.token_embedding.shape) {
    throw std::runtime_error(
        "CUDA forward embedding uploaded weight shape mismatch");
  }
  return CudaEmbeddingLookupDeviceWeight(cuda_weight->buffer.data(),
                                         cuda_weight->shape, token_id,
                                         CudaDeviceId(model));
}

static CudaTensor EmbedTokensDevice(const MiniLlamaModel& model,
                                    const std::vector<int>& tokens) {
  const CudaUploadedWeight* cuda_weight =
      FindCudaWeight(model, "token_embedding", QuantType::kF32);
  if (cuda_weight == nullptr || cuda_weight->linear ||
      cuda_weight->shape != model.token_embedding.shape) {
    throw std::runtime_error(
        "CUDA prefill embedding missing uploaded token_embedding");
  }
  return CudaEmbeddingLookupBatchDeviceWeight(
      cuda_weight->buffer.data(), cuda_weight->shape, tokens,
      CudaDeviceId(model));
}

// ---------------------------------------------------------------------------
// GQA head mapping: given a query head, return the corresponding kv head
// ---------------------------------------------------------------------------
static int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  return q_head / (n_heads / n_kv_heads);
}

// ---------------------------------------------------------------------------
// Attention forward for a single layer
//   q: [n_heads, head_dim]
//   k: [n_kv_heads, head_dim]
//   v: [n_kv_heads, head_dim]
//   pos: current position
//   layer: layer FlatIndex
//   kv_cache: global kv cache
//   n_heads, n_kv_heads, head_dim
// Returns attention output: [n_heads, head_dim]
// ---------------------------------------------------------------------------
static Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q,
                               const Tensor& k, const Tensor& v, int pos,
                               int layer, KvCache& kv_cache,
                               CudaKvCache* cuda_kv_cache, int n_heads,
                               int n_kv_heads, int head_dim) {
  // Write current k, v into cache
  kv_cache.Write(layer, pos, k, v);
  if (cuda_kv_cache != nullptr && !cuda_kv_cache->empty()) {
    cuda_kv_cache->Write(layer, pos, k, v);
    RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
    Tensor attn_out =
        CudaAttentionDecode(q, *cuda_kv_cache, layer, pos, n_heads, n_kv_heads,
                            head_dim, CudaDeviceId(model));
    RecordCudaAttentionCall(model);
    RecordCudaKvCacheRead(
        model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                   static_cast<size_t>(head_dim) * sizeof(float) * 2);
    RecordCudaHostToDeviceCopy(model, q.size() * sizeof(float));
    RecordCudaDeviceToHostCopy(model, attn_out.size() * sizeof(float));
    return attn_out;
  }
  if (model.cuda_weights) {
    RecordCudaAttentionCpuFallback(model);
  }

  Tensor attn_out({n_heads, head_dim}, 0.0f);
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  auto compute_head = [&](int h) {
    int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);

    std::vector<float> scores_data(pos + 1, 0.0f);
    for (int t = 0; t <= pos; ++t) {
      const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q.data[h * head_dim + d] * k_ptr[d];
      }
      scores_data[t] = dot * scale;
    }

    Tensor scores({pos + 1}, 0.0f);
    scores.data = std::move(scores_data);
    Tensor probs = ForwardSoftmax(model, scores);

    for (int d = 0; d < head_dim; ++d) {
      float val = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        const float* v_ptr = kv_cache.ValuePtr(layer, t, kv_head);
        val += probs.data[t] * v_ptr[d];
      }
      attn_out.data[h * head_dim + d] = val;
    }
  };

  if (model.cuda_weights) {
    for (int h = 0; h < n_heads; ++h) {
      compute_head(h);
    }
    return attn_out;
  }

  ParallelFor(n_heads, [&](int begin, int end) {
    for (int h = begin; h < end; ++h) {
      compute_head(h);
    }
  });

  return attn_out;
}

static CudaTensor AttentionForwardDevice(
    const MiniLlamaModel& model, const CudaTensor& q, const CudaTensor& k,
    const CudaTensor& v, int pos, int layer, CudaKvCache& cuda_kv_cache,
    CudaDeviceBuffer& scores_workspace, int n_heads, int n_kv_heads,
    int head_dim) {
  cuda_kv_cache.WriteDevice(layer, pos, k, v);
  RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
  CudaTensor attn_out =
      CudaAttentionDecodeDeviceInput(q, cuda_kv_cache, scores_workspace, layer,
                                     pos, n_heads, n_kv_heads, head_dim,
                                     CudaDeviceId(model));
  RecordCudaAttentionCall(model);
  RecordCudaKvCacheRead(
      model, static_cast<size_t>(pos + 1) * static_cast<size_t>(n_heads) *
                 static_cast<size_t>(head_dim) * sizeof(float) * 2);
  return attn_out;
}

static CudaTensor AttentionPrefillDevice(
    const MiniLlamaModel& model, const CudaTensor& q, const CudaTensor& k,
    const CudaTensor& v, int start_pos, int layer, CudaKvCache& cuda_kv_cache,
    int n_heads, int n_kv_heads, int head_dim) {
  cuda_kv_cache.WriteDeviceBatch(layer, start_pos, k, v);
  RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
  CudaTensor out = CudaAttentionPrefillDeviceInput(
      q, cuda_kv_cache, layer, start_pos, n_heads, n_kv_heads, head_dim,
      CudaDeviceId(model));
  RecordCudaAttentionCall(model);
  const size_t rows = static_cast<size_t>(q.shape()[0]);
  const size_t first = static_cast<size_t>(start_pos);
  const size_t last = first + rows - 1;
  RecordCudaKvCacheRead(model, rows * (first + last + 2) / 2 *
                                    static_cast<size_t>(n_heads) * head_dim *
                                    sizeof(float) * 2);
  return out;
}

// ---------------------------------------------------------------------------
// FFN forward (SwiGLU) for a single layer
//   h: [dim]  (already normalized)
//   lw: layer weights
// Returns FFN output: [dim]
// ---------------------------------------------------------------------------
static Tensor FfnForward(const MiniLlamaModel& model,
                         const std::string& layer_prefix, const Tensor& h,
                         const LayerWeights& lw) {
  const std::string gate_name = layer_prefix + "w_gate";
  const std::string up_name = layer_prefix + "w_up";
  const std::string down_name = layer_prefix + "w_down";
  if (CudaLinearReady(model, gate_name, lw.w_gate) &&
      CudaLinearReady(model, up_name, lw.w_up) &&
      CudaLinearReady(model, down_name, lw.w_down)) {
    CudaTensor h_dev = UploadCudaTensor(model, h);
    CudaTensor gate_dev =
        ForwardLinearDevice(model, gate_name, h_dev, lw.w_gate);
    CudaTensor up_dev = ForwardLinearDevice(model, up_name, h_dev, lw.w_up);
    CudaTensor silu_gate_dev =
        CudaSiluDeviceInput(gate_dev, CudaDeviceId(model));
    CudaTensor ff_dev = CudaElementwiseMulDeviceInput(silu_gate_dev, up_dev,
                                                      CudaDeviceId(model));
    RecordCudaActivationCall(model, 2);
    CudaTensor out_dev =
        ForwardLinearDevice(model, down_name, ff_dev, lw.w_down);
    return DownloadCudaTensor(model, out_dev);
  }

  Tensor gate = ForwardLinear(model, layer_prefix + "w_gate", h,
                              lw.w_gate);  // [hidden_dim]
  Tensor up =
      ForwardLinear(model, layer_prefix + "w_up", h, lw.w_up);  // [hidden_dim]

  Tensor ff =
      ForwardLinear(model, layer_prefix + "w_down",
                    ForwardSwiGlu(model, gate, up), lw.w_down);  // [dim]

  return ff;
}

static CudaTensor FfnForwardDevice(const MiniLlamaModel& model,
                                   const std::string& layer_prefix,
                                   const CudaTensor& h,
                                   const LayerWeights& lw) {
  CudaTensor gate =
      ForwardLinearDevice(model, layer_prefix + "w_gate", h, lw.w_gate);
  CudaTensor up = ForwardLinearDevice(model, layer_prefix + "w_up", h, lw.w_up);
  CudaTensor ff = CudaSiluMulDeviceInput(gate, up, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
}

// ---------------------------------------------------------------------------
// Single layer forward
//   x: input hidden state [dim]
//   layer: layer FlatIndex
//   ctx: inference context (for pos and kv_cache)
//   lw: layer weights
// Returns updated hidden state [dim]
// ---------------------------------------------------------------------------
static Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                           const Tensor& x, int layer,
                           const LayerWeights& lw, const ModelConfig& c) {
  int dim = c.dim;
  int n_heads = c.n_heads;
  int n_kv_heads = c.n_kv_heads;
  int head_dim = c.head_dim;
  int pos = ctx.pos;
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  // ---- Attention sublayer ----
  Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, c.rms_norm_eps);

  Tensor q_flat;
  Tensor k_flat;
  Tensor v_flat;
  ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);

  Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
  Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
  Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");

  // Apply RoPE to q and k
  ForwardRope(model, q, k, pos, c.rope_theta, c.rope_type);

  // Attention: compute + read/write KV cache
  Tensor attn_out =
      AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache,
                       model.cuda_weights ? &ctx.cuda_kv_cache : nullptr,
                       n_heads, n_kv_heads, head_dim);

  // Project and residual
  Tensor attn_out_flat = attn_out.ReshapeChecked({1, n_heads * head_dim},
                                                 "forward_layer attn_out");
  Tensor attn_proj =
      ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo);  // [dim]
  RequireShape(attn_proj, {1, dim}, "forward_layer attn_proj");
  Tensor attn_proj_1d =
      attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");

  Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

  // ---- FFN sublayer ----
  Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, c.rms_norm_eps);
  Tensor ff = FfnForward(model, layer_prefix, h2, lw);
  RequireShape(ff, {dim}, "forward_layer ffn");

  return ForwardAdd(model, x_attn, ff);
}

static CudaTensor ForwardLayerDevice(MiniLlamaContext& ctx,
                                     const MiniLlamaModel& model,
                                     const CudaTensor& x, int layer,
                                     const LayerWeights& lw,
                                     const ModelConfig& c) {
  int dim = c.dim;
  int n_heads = c.n_heads;
  int n_kv_heads = c.n_kv_heads;
  int head_dim = c.head_dim;
  int pos = ctx.pos;
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  CudaTensor h = ForwardRmsNormDevice(
      model, layer_prefix + "attention_norm", x, lw.attention_norm,
      ctx.cuda_rms_norm_workspace, c.rms_norm_eps);

  CudaTensor q;
  CudaTensor k;
  CudaTensor v;
  ForwardQkvProjectionDevice(model, layer_prefix, h, lw, q, k, v);
  ForwardRopeDevice(model, q, k, n_heads, n_kv_heads, head_dim, pos,
                    c.rope_theta, c.rope_type);

  CudaTensor attn_out =
      AttentionForwardDevice(model, q, k, v, pos, layer, ctx.cuda_kv_cache,
                             ctx.cuda_attention_scores_workspace, n_heads,
                             n_kv_heads, head_dim);

  CudaTensor attn_proj =
      ForwardLinearDevice(model, layer_prefix + "wo", attn_out, lw.wo);
  if (attn_proj.size() != static_cast<size_t>(dim)) {
    throw std::runtime_error(
        "forward_layer_device: attention projection shape mismatch");
  }
  CudaTensor x_attn;
  CudaTensor h2;
  ForwardAddRmsNormDevice(model, layer_prefix + "ffn_norm", x, attn_proj,
                           lw.ffn_norm, c.rms_norm_eps, x_attn, h2);
  CudaTensor ff = FfnForwardDevice(model, layer_prefix, h2, lw);
  if (ff.size() != static_cast<size_t>(dim)) {
    throw std::runtime_error("forward_layer_device: FFN output shape mismatch");
  }
  return ForwardAddDevice(model, x_attn, ff);
}

static CudaTensor ForwardLayerPrefillDevice(
    MiniLlamaContext& ctx, const MiniLlamaModel& model, const CudaTensor& x,
    int layer, const LayerWeights& lw, const ModelConfig& c, int start_pos) {
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  CudaTensor h = ForwardRmsNormBatchDevice(
      model, prefix + "attention_norm", x, lw.attention_norm, c.rms_norm_eps);
  CudaTensor q;
  CudaTensor k;
  CudaTensor v;
  ForwardQkvProjectionDevice(model, prefix, h, lw, q, k, v);
  CudaRopeBatchDeviceInput(q, k, c.n_heads, c.n_kv_heads, c.head_dim,
                           start_pos, c.rope_theta, c.rope_type,
                           CudaDeviceId(model));
  RecordCudaActivationCall(model);
  CudaTensor attn = AttentionPrefillDevice(
      model, q, k, v, start_pos, layer, ctx.cuda_kv_cache, c.n_heads,
      c.n_kv_heads, c.head_dim);
  CudaTensor projected = ForwardLinearDevice(model, prefix + "wo", attn, lw.wo);
  CudaTensor x_attn = ForwardAddDevice(model, x, projected);
  CudaTensor h2 = ForwardRmsNormBatchDevice(model, prefix + "ffn_norm", x_attn,
                                             lw.ffn_norm, c.rms_norm_eps);
  CudaTensor ff = FfnForwardDevice(model, prefix, h2, lw);
  return ForwardAddDevice(model, x_attn, ff);
}

// ---------------------------------------------------------------------------
// Compute logits from final hidden state
//   x: [dim]
// Returns logits: [vocab_size]
// ---------------------------------------------------------------------------
static Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                            const ModelConfig& c) {
  Tensor normed = ForwardRmsNorm(model, x, model.final_norm, c.rms_norm_eps);
  Tensor logits_flat =
      ForwardLinear(model, "lm_head", normed, model.lm_head);  // [vocab_size]
  return logits_flat.ReshapeChecked({c.vocab_size}, "compute_logits");
}

static CudaTensor ComputeLogitsDevice(const CudaTensor& x,
                                      const MiniLlamaModel& model,
                                      const ModelConfig& c,
                                      CudaDeviceBuffer& sum_workspace) {
  CudaTensor normed = ForwardRmsNormDevice(model, "final_norm", x,
                                           model.final_norm, sum_workspace,
                                           c.rms_norm_eps);
  CudaTensor logits_dev =
      ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
  if (logits_dev.shape() != std::vector<int>{c.vocab_size}) {
    throw std::runtime_error("compute_logits_device: logits shape mismatch");
  }
  return logits_dev;
}

// Run the CUDA Decode transformer body and retain only the final hidden state.
// Both the regular logits path and the greedy lm_head fast path share this
// sequence so their KV-cache and residual semantics remain identical.
static CudaTensor ForwardTokenDeviceHidden(MiniLlamaContext& ctx,
                                           const MiniLlamaModel& model,
                                           int token) {
  ValidateForwardInputs(ctx, model, token);
  const ModelConfig& c = model.config;
  if (ctx.cuda_kv_cache.empty()) {
    ctx.cuda_kv_cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim,
                            model.cuda_weights->device_id);
  }
  if (ctx.cuda_attention_scores_workspace.empty()) {
    const size_t score_elements =
        static_cast<size_t>(c.n_heads) * static_cast<size_t>(c.max_seq_len);
    ctx.cuda_attention_scores_workspace.Reset(score_elements * sizeof(float),
                                              model.cuda_weights->device_id);
  }
  if (ctx.cuda_rms_norm_workspace.empty()) {
    ctx.cuda_rms_norm_workspace.Reset(sizeof(float),
                                      model.cuda_weights->device_id);
  }
  if (ctx.cuda_argmax_workspace.empty()) {
    ctx.cuda_argmax_workspace.Reset(sizeof(int), model.cuda_weights->device_id);
  }

  CudaTensor x_dev = EmbedTokenDevice(model, token);
  for (int layer = 0; layer < c.n_layers; ++layer) {
    x_dev = ForwardLayerDevice(ctx, model, x_dev, layer, model.layers[layer], c);
  }
  return x_dev;
}

// ---------------------------------------------------------------------------
// Forward pass for a single token
// ---------------------------------------------------------------------------
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token, CudaTensor* device_logits) {
  ValidateForwardInputs(ctx, model, token);

  const ModelConfig& c = model.config;
  int n_layers = c.n_layers;

  // 1. Embedding lookup
  if (model.cuda_weights) {
    CudaTensor x_dev = ForwardTokenDeviceHidden(ctx, model, token);
    CudaTensor logits = ComputeLogitsDevice(x_dev, model, c,
                                            ctx.cuda_rms_norm_workspace);
    if (device_logits != nullptr) {
      *device_logits = std::move(logits);
      return Tensor();
    }
    return DownloadCudaTensor(model, logits);
  }

  if (device_logits != nullptr) {
    throw std::runtime_error(
        "ForwardToken: device logits requested for a CPU model");
  }

  Tensor x = EmbedToken(model, token);

  // 2. Transformer layers
  for (int layer = 0; layer < n_layers; ++layer) {
    x = ForwardLayer(ctx, model, x, layer, model.layers[layer], c);
  }

  // 3. Final norm + logits
  Tensor logits = ComputeLogits(x, model, c);

  return logits;
}

bool ForwardTokenGreedyCuda(MiniLlamaContext& ctx,
                            const MiniLlamaModel& model, int token,
                            int& next_token) {
  if (!model.cuda_weights) {
    return false;
  }
  const CudaUploadedWeight* lm_head =
      FindCudaWeight(model, "lm_head", model.lm_head.type);
  if (lm_head == nullptr || !lm_head->linear) {
    throw std::runtime_error(
        "ForwardTokenGreedyCuda: missing uploaded lm_head");
  }

  CudaTensor hidden = ForwardTokenDeviceHidden(ctx, model, token);
  CudaTensor normed = ForwardRmsNormDevice(
      model, "final_norm", hidden, model.final_norm,
      ctx.cuda_rms_norm_workspace, model.config.rms_norm_eps);
  next_token = CudaLinearArgMaxDeviceInput(
      normed, lm_head->type, lm_head->buffer.data(), lm_head->block_count,
      lm_head->shape,
      ctx.cuda_linear_argmax_workspace, ctx.cuda_argmax_workspace,
      CudaDeviceId(model));
  RecordCudaLinearCall(model);
  model.cuda_weights->linear_argmax_dispatch_calls += 1;
  RecordModelCudaDeviceToHostCopy(model, sizeof(next_token));
  return true;
}

CudaTensor ForwardPrefillDevice(MiniLlamaContext& ctx,
                                const MiniLlamaModel& model,
                                const MiniBatch& batch) {
  if (!model.cuda_weights || batch.num_tokens() <= 0 ||
      batch.tokens.size() != batch.positions.size()) {
    throw std::runtime_error("ForwardPrefillDevice: invalid CUDA batch");
  }
  const ModelConfig& c = model.config;
  const int start_pos = batch.positions.front();
  if (start_pos < 0 || start_pos + batch.num_tokens() > c.max_seq_len) {
    throw std::out_of_range(
        "ForwardPrefillDevice: batch exceeds the context window");
  }
  for (int i = 0; i < batch.num_tokens(); ++i) {
    if (batch.positions[i] != start_pos + i) {
      throw std::runtime_error(
          "ForwardPrefillDevice: batch positions must be contiguous");
    }
    ValidateForwardInputs(ctx, model, batch.tokens[i]);
  }
  if (ctx.cuda_kv_cache.empty()) {
    ctx.cuda_kv_cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim,
                            CudaDeviceId(model));
  }
  if (ctx.cuda_argmax_workspace.empty()) {
    ctx.cuda_argmax_workspace.Reset(sizeof(int), CudaDeviceId(model));
  }
  CudaTensor x = EmbedTokensDevice(model, batch.tokens);
  for (int layer = 0; layer < c.n_layers; ++layer) {
    x = ForwardLayerPrefillDevice(ctx, model, x, layer, model.layers[layer], c,
                                  start_pos);
  }
  // Generation samples only the final prompt position. Avoid projecting every
  // prefill hidden state to the full vocabulary [T, vocab_size].
  CudaTensor last_hidden = CudaLastRow(x, CudaDeviceId(model));
  if (ctx.cuda_rms_norm_workspace.empty()) {
    ctx.cuda_rms_norm_workspace.Reset(sizeof(float), CudaDeviceId(model));
  }
  CudaTensor normed = ForwardRmsNormDevice(
      model, "final_norm", last_hidden, model.final_norm,
      ctx.cuda_rms_norm_workspace, c.rms_norm_eps);
  return ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
}

}  // namespace mini_llama
