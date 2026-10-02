// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the CUDA implementation.
// clang-format off
#include "mini_llama/cuda_quant.h"
// clang-format on

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {

namespace {

constexpr int kBlockSize = 128;

void CheckCudaQuant(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA quant error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

void CheckLastKernel(const char* name) {
  CheckCudaQuant(cudaGetLastError(), name);
}

__device__ float HalfBitsToFloat(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(raw);
}

__global__ void Q80LinearKernel(const float* x, const BlockQ80* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (out >= out_features || row >= rows) {
    return;
  }

  const float* x_row = x + static_cast<size_t>(row) * in_features;
  const BlockQ80* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
  float sum = 0.0f;
  for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
    const BlockQ80& block = w_row[block_index];
    float d = HalfBitsToFloat(block.d);
    int base_k = block_index * kQ80BlockSize;
    int k_end = base_k + kQ80BlockSize < in_features ? base_k + kQ80BlockSize
                                                     : in_features;
    for (int k = base_k; k < k_end; ++k) {
      sum += d * static_cast<float>(block.qs[k - base_k]) * x_row[k];
    }
  }
  y[static_cast<size_t>(row) * out_features + out] = sum;
}

// Decode-specialized Q8_0 Linear. A warp cooperatively calculates one output
// row: each lane owns one element of every 32-value Q8 block, then a warp
// reduction produces the dot product. This replaces the old one-thread-per-
// output loop for rows == 1, where serial 896-dimension dot products left most
// lanes idle.
__global__ void Q80LinearDecodeWarpKernel(const float* x,
                                          const BlockQ80* weight, float* y,
                                          int in_features, int out_features,
                                          int blocks_per_row) {
  constexpr int kWarpSize = 32;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int out = blockIdx.x * warps_per_block + warp;
  if (out >= out_features) {
    return;
  }

  const BlockQ80* weight_row =
      weight + static_cast<size_t>(out) * blocks_per_row;
  float sum = 0.0f;
  for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
    const int k = block_index * kQ80BlockSize + lane;
    if (k < in_features) {
      const BlockQ80& block = weight_row[block_index];
      sum += HalfBitsToFloat(block.d) * static_cast<float>(block.qs[lane]) *
             x[k];
    }
  }

  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) {
    y[out] = sum;
  }
}

__device__ float Q40Value(const BlockQ40& block, int flat_index) {
  int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
                          : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return HalfBitsToFloat(block.d) * static_cast<float>(q - 8);
}

__device__ float Q41Value(const BlockQ41& block, int flat_index) {
  int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
                          : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return HalfBitsToFloat(block.d) * static_cast<float>(q) +
         HalfBitsToFloat(block.m);
}

struct QuantArgMaxCandidate {
  float value;
  int index;
};

__device__ bool IsBetterQuantArgMax(const QuantArgMaxCandidate& candidate,
                                    const QuantArgMaxCandidate& incumbent) {
  return candidate.value > incumbent.value ||
         (candidate.value == incumbent.value &&
          candidate.index < incumbent.index);
}

struct Q80Accessor {
  __device__ static float Value(const BlockQ80& block, int lane) {
    return HalfBitsToFloat(block.d) * static_cast<float>(block.qs[lane]);
  }
};

struct Q40Accessor {
  __device__ static float Value(const BlockQ40& block, int lane) {
    return Q40Value(block, lane);
  }
};

struct Q41Accessor {
  __device__ static float Value(const BlockQ41& block, int lane) {
    return Q41Value(block, lane);
  }
};

// One warp computes one vocabulary row. Instead of writing a full logits
// tensor, each CUDA block reduces its warp results to one candidate; a second
// kernel then computes the global ArgMax.
template <typename Block, typename Accessor>
__global__ void QuantLinearArgMaxCandidatesKernel(
    const float* x, const Block* weight, int in_features, int out_features,
    int blocks_per_row, QuantArgMaxCandidate* candidates) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
  __shared__ QuantArgMaxCandidate block_candidates[kWarpsPerBlock];

  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int out = blockIdx.x * kWarpsPerBlock + warp;
  float sum = 0.0f;
  if (out < out_features) {
    const Block* row = weight + static_cast<size_t>(out) * blocks_per_row;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
      const int k = block_index * kWarpSize + lane;
      if (k < in_features) {
        sum += Accessor::Value(row[block_index], lane) * x[k];
      }
    }
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
  }
  if (lane == 0) {
    block_candidates[warp] = {out < out_features ? sum : -3.402823466e+38F,
                              out < out_features ? out : out_features};
  }
  __syncthreads();
  if (tid == 0) {
    QuantArgMaxCandidate best = block_candidates[0];
    for (int candidate = 1; candidate < kWarpsPerBlock; ++candidate) {
      if (IsBetterQuantArgMax(block_candidates[candidate], best)) {
        best = block_candidates[candidate];
      }
    }
    candidates[blockIdx.x] = best;
  }
}

__global__ void QuantArgMaxCandidatesKernel(const QuantArgMaxCandidate* input,
                                            int n, int* result) {
  __shared__ QuantArgMaxCandidate reduced[kBlockSize];
  const int tid = threadIdx.x;
  QuantArgMaxCandidate best = {-3.402823466e+38F, n};
  for (int i = tid; i < n; i += blockDim.x) {
    if (IsBetterQuantArgMax(input[i], best)) {
      best = input[i];
    }
  }
  reduced[tid] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride && IsBetterQuantArgMax(reduced[tid + stride], reduced[tid])) {
      reduced[tid] = reduced[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *result = reduced[0].index;
  }
}

__global__ void Q40LinearKernel(const float* x, const BlockQ40* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (out >= out_features || row >= rows) {
    return;
  }

  const float* x_row = x + static_cast<size_t>(row) * in_features;
  const BlockQ40* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
  float sum = 0.0f;
  for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
    const BlockQ40& block = w_row[block_index];
    int base_k = block_index * kQ40BlockSize;
    int k_end = base_k + kQ40BlockSize < in_features ? base_k + kQ40BlockSize
                                                     : in_features;
    for (int k = base_k; k < k_end; ++k) {
      sum += Q40Value(block, k - base_k) * x_row[k];
    }
  }
  y[static_cast<size_t>(row) * out_features + out] = sum;
}

__global__ void Q41LinearKernel(const float* x, const BlockQ41* weight,
                                float* y, int rows, int in_features,
                                int out_features, int blocks_per_row) {
  int out = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (out >= out_features || row >= rows) {
    return;
  }

  const float* x_row = x + static_cast<size_t>(row) * in_features;
  const BlockQ41* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
  float sum = 0.0f;
  for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
    const BlockQ41& block = w_row[block_index];
    int base_k = block_index * kQ41BlockSize;
    int k_end = base_k + kQ41BlockSize < in_features ? base_k + kQ41BlockSize
                                                     : in_features;
    for (int k = base_k; k < k_end; ++k) {
      sum += Q41Value(block, k - base_k) * x_row[k];
    }
  }
  y[static_cast<size_t>(row) * out_features + out] = sum;
}

void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols,
                    const char* caller) {
  if (bias.num_dims() != 1 || bias.shape[0] != cols) {
    throw std::runtime_error(std::string(caller) +
                             ": expected bias shape [out_features]");
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
    }
  }
}

int Q80InFeaturesFromShape(const std::vector<int>& x_shape,
                           const std::string& x_shape_str) {
  if (x_shape.size() == 1) {
    return x_shape[0];
  }
  if (x_shape.size() == 2) {
    return x_shape[1];
  }
  throw std::runtime_error(
      "CudaQ80Linear: expected x shape [in_features] or [batch, in_features], "
      "got " +
      x_shape_str);
}

void ValidateQ80LinearShape(const std::vector<int>& x_shape,
                            const std::string& x_shape_str,
                            const std::vector<int>& weight_shape,
                            size_t block_count) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error(
        "CudaQ80Linear: expected weight shape [out_features, in_features]");
  }
  int in_features = Q80InFeaturesFromShape(x_shape, x_shape_str);
  if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
    throw std::runtime_error("CudaQ80Linear: empty tensors are not supported");
  }
  if (weight_shape[1] != in_features) {
    throw std::runtime_error(
        "CudaQ80Linear: dimension mismatch x=" + x_shape_str + " weight=[" +
        std::to_string(weight_shape[0]) + ", " +
        std::to_string(weight_shape[1]) + "]");
  }
  int blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
  size_t expected_blocks =
      static_cast<size_t>(weight_shape[0]) * blocks_per_row;
  if (block_count != expected_blocks) {
    throw std::runtime_error("CudaQ80Linear: block count mismatch: expected " +
                             std::to_string(expected_blocks) + ", got " +
                             std::to_string(block_count));
  }
}

void ValidateQ80LinearInputs(const Tensor& x,
                             const std::vector<int>& weight_shape,
                             size_t block_count) {
  ValidateQ80LinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                         block_count);
}

int QuantInFeaturesFromShape(const std::vector<int>& x_shape,
                             const std::string& x_shape_str,
                             const char* caller) {
  if (x_shape.size() == 1) {
    return x_shape[0];
  }
  if (x_shape.size() == 2) {
    return x_shape[1];
  }
  throw std::runtime_error(
      std::string(caller) +
      ": expected x shape [in_features] or [batch, in_features], got " +
      x_shape_str);
}

void ValidateQuantLinearShape(const std::vector<int>& x_shape,
                              const std::string& x_shape_str,
                              const std::vector<int>& weight_shape,
                              size_t block_count, int block_size,
                              const char* caller) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error(
        std::string(caller) +
        ": expected weight shape [out_features, in_features]");
  }
  int in_features = QuantInFeaturesFromShape(x_shape, x_shape_str, caller);
  if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
    throw std::runtime_error(std::string(caller) +
                             ": empty tensors are not supported");
  }
  if (weight_shape[1] != in_features) {
    throw std::runtime_error(std::string(caller) +
                             ": dimension mismatch x=" + x_shape_str +
                             " weight=[" + std::to_string(weight_shape[0]) +
                             ", " + std::to_string(weight_shape[1]) + "]");
  }
  int blocks_per_row = (in_features + block_size - 1) / block_size;
  size_t expected_blocks =
      static_cast<size_t>(weight_shape[0]) * blocks_per_row;
  if (block_count != expected_blocks) {
    throw std::runtime_error(std::string(caller) +
                             ": block count mismatch: expected " +
                             std::to_string(expected_blocks) + ", got " +
                             std::to_string(block_count));
  }
}

Tensor CudaQ80LinearWithDevicePointer(const Tensor& x, const void* q8_0_device,
                                      size_t q8_0_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
  if (q8_0_device == nullptr) {
    throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
  }
  ValidateQ80LinearInputs(x, weight_shape, q8_0_block_count);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;

  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  dim3 block(kBlockSize);
  if (rows == 1) {
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    dim3 grid((out_features + kWarpsPerBlock - 1) / kWarpsPerBlock);
    Q80LinearDecodeWarpKernel<<<grid, block>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<const BlockQ80*>(q8_0_device),
        static_cast<float*>(y_dev.data()), in_features, out_features,
        blocks_per_row);
    CheckLastKernel("Q80LinearDecodeWarpKernel");
  } else {
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q80LinearKernel<<<grid, block>>>(
        static_cast<const float*>(x_dev.data()),
        static_cast<const BlockQ80*>(q8_0_device),
        static_cast<float*>(y_dev.data()), rows, in_features, out_features,
        blocks_per_row);
    CheckLastKernel("Q80LinearKernel");
  }

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    AddBiasInPlace(y, *bias, rows, out_features, "CudaQ80Linear");
  }
  return y;
}

CudaTensor CudaQ80LinearWithDeviceInput(const CudaTensor& x,
                                        const void* q8_0_device,
                                        size_t q8_0_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
  if (q8_0_device == nullptr) {
    throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaQ80Linear: input tensor is on a different CUDA device");
  }
  ValidateQ80LinearShape(x.shape(), x.ShapeStringShort(), weight_shape,
                         q8_0_block_count);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape()[0];
  int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;

  CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

  dim3 block(kBlockSize);
  if (rows == 1) {
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    dim3 grid((out_features + kWarpsPerBlock - 1) / kWarpsPerBlock);
    Q80LinearDecodeWarpKernel<<<grid, block>>>(
        static_cast<const float*>(x.data()),
        static_cast<const BlockQ80*>(q8_0_device), static_cast<float*>(y.data()),
        in_features, out_features, blocks_per_row);
    CheckLastKernel("Q80LinearDecodeWarpKernel");
  } else {
    dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
    Q80LinearKernel<<<grid, block>>>(
        static_cast<const float*>(x.data()),
        static_cast<const BlockQ80*>(q8_0_device), static_cast<float*>(y.data()),
        rows, in_features, out_features, blocks_per_row);
    CheckLastKernel("Q80LinearKernel");
  }

  return y;
}

Tensor CudaQ40LinearWithDevicePointer(const Tensor& x, const void* q4_0_device,
                                      size_t q4_0_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
  static constexpr const char* kCaller = "CudaQ40Linear";
  if (q4_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           q4_0_block_count, kQ40BlockSize, kCaller);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;

  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  dim3 block(kBlockSize);
  dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
  Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()),
                                   static_cast<const BlockQ40*>(q4_0_device),
                                   static_cast<float*>(y_dev.data()), rows,
                                   in_features, out_features, blocks_per_row);
  CheckLastKernel("Q40LinearKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    AddBiasInPlace(y, *bias, rows, out_features, kCaller);
  }
  return y;
}

CudaTensor CudaQ40LinearWithDeviceInput(const CudaTensor& x,
                                        const void* q4_0_device,
                                        size_t q4_0_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
  static constexpr const char* kCaller = "CudaQ40Linear";
  if (q4_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(std::string(kCaller) +
                             ": input tensor is on a different CUDA device");
  }
  ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), weight_shape,
                           q4_0_block_count, kQ40BlockSize, kCaller);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape()[0];
  int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;

  CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

  dim3 block(kBlockSize);
  dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
  Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()),
                                   static_cast<const BlockQ40*>(q4_0_device),
                                   static_cast<float*>(y.data()), rows,
                                   in_features, out_features, blocks_per_row);
  CheckLastKernel("Q40LinearKernel");

  return y;
}

Tensor CudaQ41LinearWithDevicePointer(const Tensor& x, const void* q4_1_device,
                                      size_t q4_1_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
  static constexpr const char* kCaller = "CudaQ41Linear";
  if (q4_1_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           q4_1_block_count, kQ41BlockSize, kCaller);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features + kQ41BlockSize - 1) / kQ41BlockSize;

  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  dim3 block(kBlockSize);
  dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
  Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()),
                                   static_cast<const BlockQ41*>(q4_1_device),
                                   static_cast<float*>(y_dev.data()), rows,
                                   in_features, out_features, blocks_per_row);
  CheckLastKernel("Q41LinearKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    AddBiasInPlace(y, *bias, rows, out_features, kCaller);
  }
  return y;
}

CudaTensor CudaQ41LinearWithDeviceInput(const CudaTensor& x,
                                        const void* q4_1_device,
                                        size_t q4_1_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
  static constexpr const char* kCaller = "CudaQ41Linear";
  if (q4_1_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(std::string(kCaller) +
                             ": input tensor is on a different CUDA device");
  }
  ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), weight_shape,
                           q4_1_block_count, kQ41BlockSize, kCaller);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape()[0];
  int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features + kQ41BlockSize - 1) / kQ41BlockSize;

  CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

  dim3 block(kBlockSize);
  dim3 grid((out_features + kBlockSize - 1) / kBlockSize, rows);
  Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()),
                                   static_cast<const BlockQ41*>(q4_1_device),
                                   static_cast<float*>(y.data()), rows,
                                   in_features, out_features, blocks_per_row);
  CheckLastKernel("Q41LinearKernel");

  return y;
}

template <typename Block, typename Accessor>
int CudaQuantLinearArgMaxWithDeviceInput(
    const CudaTensor& x, const void* weight_data, size_t block_count,
    const std::vector<int>& weight_shape, int quant_block_size,
    const char* caller, CudaDeviceBuffer& candidate_workspace,
    CudaDeviceBuffer& result_workspace, int device_id) {
  if (weight_data == nullptr) {
    throw std::runtime_error(std::string(caller) + ": device weight pointer is null");
  }
  if (x.num_dims() != 1) {
    throw std::runtime_error(std::string(caller) +
                             ": greedy ArgMax requires a 1D Decode input");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(std::string(caller) +
                             ": input tensor is on a different CUDA device");
  }
  ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), weight_shape,
                           block_count, quant_block_size, caller);
  CudaSetDevice(device_id);
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  const int out_features = weight_shape[0];
  const int blocks_per_row =
      (weight_shape[1] + quant_block_size - 1) / quant_block_size;
  const int candidate_count =
      (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t candidate_bytes =
      static_cast<size_t>(candidate_count) * sizeof(QuantArgMaxCandidate);
  if (candidate_workspace.empty() ||
      candidate_workspace.device_id() != device_id ||
      candidate_workspace.bytes() < candidate_bytes) {
    candidate_workspace.Reset(candidate_bytes, device_id);
  }
  if (result_workspace.empty() || result_workspace.device_id() != device_id ||
      result_workspace.bytes() < sizeof(int)) {
    throw std::runtime_error(std::string(caller) +
                             ": result workspace is too small or on a "
                             "different CUDA device");
  }

  QuantLinearArgMaxCandidatesKernel<Block, Accessor><<<candidate_count,
                                                        kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<const Block*>(weight_data),
      weight_shape[1], out_features, blocks_per_row,
      static_cast<QuantArgMaxCandidate*>(candidate_workspace.data()));
  CheckLastKernel("QuantLinearArgMaxCandidatesKernel");
  QuantArgMaxCandidatesKernel<<<1, kBlockSize>>>(
      static_cast<const QuantArgMaxCandidate*>(candidate_workspace.data()),
      candidate_count, static_cast<int*>(result_workspace.data()));
  CheckLastKernel("QuantArgMaxCandidatesKernel");

  int result = 0;
  result_workspace.Download(&result, sizeof(result));
  return result;
}

}  // namespace

Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  ValidateQ80LinearInputs(x, weight_shape, weight.size());
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ80), device_id);
  w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ80));
  return CudaQ80LinearWithDevicePointer(x, w_dev.data(), weight.size(),
                                        weight_shape, bias, device_id);
}

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  return CudaQ80LinearWithDevicePointer(x, q8_0_device, q8_0_block_count,
                                        weight_shape, bias, device_id);
}

CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x,
                                    const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  return CudaQ80LinearWithDeviceInput(x, q8_0_device, q8_0_block_count,
                                      weight_shape, device_id);
}

int CudaQ80LinearArgMaxDeviceInput(
    const CudaTensor& x, const void* q8_0_device, size_t q8_0_block_count,
    const std::vector<int>& weight_shape, CudaDeviceBuffer& candidate_workspace,
    CudaDeviceBuffer& result_workspace, int device_id) {
  return CudaQuantLinearArgMaxWithDeviceInput<BlockQ80, Q80Accessor>(
      x, q8_0_device, q8_0_block_count, weight_shape, kQ80BlockSize,
      "CudaQ80LinearArgMax", candidate_workspace, result_workspace, device_id);
}

Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           weight.size(), kQ40BlockSize, "CudaQ40Linear");
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ40), device_id);
  w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ40));
  return CudaQ40LinearWithDevicePointer(x, w_dev.data(), weight.size(),
                                        weight_shape, bias, device_id);
}

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  return CudaQ40LinearWithDevicePointer(x, q4_0_device, q4_0_block_count,
                                        weight_shape, bias, device_id);
}

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  return CudaQ40LinearWithDeviceInput(x, q4_0_device, q4_0_block_count,
                                      weight_shape, device_id);
}

int CudaQ40LinearArgMaxDeviceInput(
    const CudaTensor& x, const void* q4_0_device, size_t q4_0_block_count,
    const std::vector<int>& weight_shape, CudaDeviceBuffer& candidate_workspace,
    CudaDeviceBuffer& result_workspace, int device_id) {
  return CudaQuantLinearArgMaxWithDeviceInput<BlockQ40, Q40Accessor>(
      x, q4_0_device, q4_0_block_count, weight_shape, kQ40BlockSize,
      "CudaQ40LinearArgMax", candidate_workspace, result_workspace, device_id);
}

Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           weight.size(), kQ41BlockSize, "CudaQ41Linear");
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ41), device_id);
  w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ41));
  return CudaQ41LinearWithDevicePointer(x, w_dev.data(), weight.size(),
                                        weight_shape, bias, device_id);
}

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  return CudaQ41LinearWithDevicePointer(x, q4_1_device, q4_1_block_count,
                                        weight_shape, bias, device_id);
}

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  return CudaQ41LinearWithDeviceInput(x, q4_1_device, q4_1_block_count,
                                      weight_shape, device_id);
}

int CudaQ41LinearArgMaxDeviceInput(
    const CudaTensor& x, const void* q4_1_device, size_t q4_1_block_count,
    const std::vector<int>& weight_shape, CudaDeviceBuffer& candidate_workspace,
    CudaDeviceBuffer& result_workspace, int device_id) {
  return CudaQuantLinearArgMaxWithDeviceInput<BlockQ41, Q41Accessor>(
      x, q4_1_device, q4_1_block_count, weight_shape, kQ41BlockSize,
      "CudaQ41LinearArgMax", candidate_workspace, result_workspace, device_id);
}

}  // namespace mini_llama
