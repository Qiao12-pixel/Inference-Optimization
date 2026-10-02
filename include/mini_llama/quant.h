// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_QUANT_H_
#define INCLUDE_MINI_LLAMA_QUANT_H_

#include <vector>

#include "mini_llama/quantized_tensor.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Quantize / dequantize (Q8_0)
// ---------------------------------------------------------------------------
std::vector<BlockQ80> QuantizeToQ80(const Tensor& src);

// Dequantize Q8_0 blocks back to F32 tensor.
// `shape` must match the original source tensor shape.
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// Quantize / dequantize (Q4_0)
// ---------------------------------------------------------------------------
std::vector<BlockQ40> QuantizeToQ40(const Tensor& src);

// Dequantize Q4_0 blocks back to F32 tensor.
// `shape` must match the original source tensor shape.
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// Dequantize / Linear (Q4_1)
// ---------------------------------------------------------------------------
Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// True quantized Linear (block-level on-the-fly)
// weight is stored in quantized format; input and output are F32.
// No temporary F32 weight array is allocated.
// ---------------------------------------------------------------------------
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape);
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape);
Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape);

// ---------------------------------------------------------------------------
// Legacy pseudo-quantized Matmul (dequantizes to F32 then calls Matmul)
// Kept for backward compatibility with existing tests.
// ---------------------------------------------------------------------------
Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape);
Tensor MatmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape);

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
// Compare F32 Matmul vs Q8_0 Matmul for the same weight + input.
// Returns max absolute error.
float CompareMatmulError(const Tensor& weight, const Tensor& input);

// Compare F32 Linear vs Q4_0 Linear for the same weight + input.
// Returns max absolute error.
float CompareQ40Error(const Tensor& weight, const Tensor& input);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_QUANT_H_
