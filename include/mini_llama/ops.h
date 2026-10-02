// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_OPS_H_
#define INCLUDE_MINI_LLAMA_OPS_H_

#include <cmath>

#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// Matrix multiplication: c = a * b
// a: [m, k], b: [k, n], c: [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b);

// Linear projection: y = x * weight^T
// x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
// result keeps x rank: [out_features] or [1, out_features]
Tensor Linear(const Tensor& x, const Tensor& weight);

// Linear projection with quantized weight.
// Dispatches to F32, Q8_0, or Q4_0 path based on weight.type.
Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// x: [dim], weight: [dim], result: [dim]
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over a 1D tensor
// x: [n], result: [n]
Tensor Softmax(const Tensor& x);

// SiLU activation: Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x);

// Element-wise multiplication
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// SwiGLU: SwiGlu(gate, up) = Silu(gate) * up
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// RoPE (Rotary Position Embedding) applied to Q and k
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
// pos: current token position
void Rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType rope_type = RopeType::kNormal);

// Argmax: return FlatIndex of maximum value
int ArgMax(const Tensor& x);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_OPS_H_
