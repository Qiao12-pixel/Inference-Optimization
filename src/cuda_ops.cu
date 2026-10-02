// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the CUDA implementation.
// clang-format off
#include "mini_llama/cuda_ops.h"
// clang-format on

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {

namespace {

constexpr int kBlockSize = 256;

void CheckCudaOps(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA ops error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

void CheckLastKernel(const char* name) {
  CheckCudaOps(cudaGetLastError(), name);
}

__global__ void SumSquaresKernel(const float* x, float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(sum, x[i] * x[i]);
  }
}

__global__ void RmsNormKernel(const float* x, const float* weight, float* y,
                              const float* sum, int n, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float scale = rsqrtf((*sum / static_cast<float>(n)) + eps);
    y[i] = x[i] * scale * weight[i];
  }
}

__global__ void RmsNormFusedKernel(const float* x, const float* weight,
                                   float* y, int n, float eps) {
  __shared__ float partial[kBlockSize];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    sum += x[i] * x[i];
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / static_cast<float>(n) + eps);
  for (int i = tid; i < n; i += blockDim.x) {
    y[i] = x[i] * scale * weight[i];
  }
}

__global__ void AddRmsNormFusedKernel(const float* a, const float* b,
                                      const float* weight, float* sum_out,
                                      float* norm_out, int n, float eps) {
  __shared__ float partial[kBlockSize];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    const float value = a[i] + b[i];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / static_cast<float>(n) + eps);
  for (int i = tid; i < n; i += blockDim.x) {
    const float value = a[i] + b[i];
    sum_out[i] = value;
    norm_out[i] = value * scale * weight[i];
  }
}

__global__ void RmsNormBatchKernel(const float* x, const float* weight,
                                   float* y, int dim, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float partial[kBlockSize];
  float sum = 0.0f;
  for (int d = tid; d < dim; d += blockDim.x) {
    const float value = x[static_cast<size_t>(row) * dim + d];
    sum += value * value;
  }
  partial[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / static_cast<float>(dim) + eps);
  for (int d = tid; d < dim; d += blockDim.x) {
    const size_t index = static_cast<size_t>(row) * dim + d;
    y[index] = x[index] * scale * weight[d];
  }
}

__global__ void SiluKernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

__global__ void SiluMulKernel(const float* gate, const float* up, float* y,
                              int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float value = gate[i];
    y[i] = (value / (1.0f + expf(-value))) * up[i];
  }
}

__global__ void ElementwiseMulKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] * b[i];
  }
}

__global__ void ElementwiseAddKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] + b[i];
  }
}

__global__ void AddRowBiasKernel(const float* x, const float* bias, float* y,
                                 int rows, int cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * cols) y[index] = x[index] + bias[index % cols];
}

__global__ void EmbeddingLookupKernel(const float* embedding, float* y,
                                      int token_id, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    y[i] = embedding[token_id * dim + i];
  }
}

__global__ void EmbeddingLookupBatchKernel(const float* embedding,
                                           const int* token_ids, float* y,
                                           int dim) {
  const int token = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < dim) {
    y[static_cast<size_t>(token) * dim + d] =
        embedding[static_cast<size_t>(token_ids[token]) * dim + d];
  }
}

__global__ void FloatToHalfKernel(const float* src, __half* dst, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    dst[index] = __float2half(src[index]);
  }
}

__global__ void SoftmaxMaxKernel(const float* x, float* max_out, int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  float local_max = -3.402823466e+38F;
  for (int i = tid; i < n; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    *max_out = shared[0];
  }
}

__global__ void SoftmaxExpSumKernel(const float* x, float* y,
                                    const float* max_value, float* sum_out,
                                    int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  float local_sum = 0.0f;
  float max_v = *max_value;
  for (int i = tid; i < n; i += blockDim.x) {
    float e = expf(x[i] - max_v);
    y[i] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *sum_out = shared[0];
  }
}

__global__ void SoftmaxNormKernel(float* y, const float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] /= *sum;
  }
}

__global__ void ArgMaxKernel(const float* x, int n, int* result) {
  __shared__ float values[kBlockSize];
  __shared__ int indices[kBlockSize];

  const int tid = threadIdx.x;
  // Keep this independent of CUDA-version-specific infinity macros.
  float best_value = -3.402823466e+38F;
  int best_index = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    const float value = x[i];
    if (value > best_value || (value == best_value && i < best_index)) {
      best_value = value;
      best_index = i;
    }
  }
  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_value = values[tid + stride];
      const int other_index = indices[tid + stride];
      if (other_value > values[tid] ||
          (other_value == values[tid] && other_index < indices[tid])) {
        values[tid] = other_value;
        indices[tid] = other_index;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    *result = indices[0];
  }
}

struct ArgMaxCandidate {
  float value;
  int index;
};

__device__ bool IsBetterArgMaxCandidate(const ArgMaxCandidate& candidate,
                                        const ArgMaxCandidate& incumbent) {
  return candidate.value > incumbent.value ||
         (candidate.value == incumbent.value &&
          candidate.index < incumbent.index);
}

// Each warp owns one vocabulary row. This is intentionally restricted to the
// F32 lm_head Decode GEMV: it avoids writing and rereading a full logits vector
// when only greedy ArgMax is required.
__global__ void F32LinearArgMaxCandidatesKernel(
    const float* x, const float* weight, int in_features, int out_features,
    ArgMaxCandidate* candidates) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
  __shared__ ArgMaxCandidate block_candidates[kWarpsPerBlock];

  const int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  const int out = blockIdx.x * kWarpsPerBlock + warp;

  float sum = 0.0f;
  if (out < out_features) {
    const float* row = weight + static_cast<size_t>(out) * in_features;
    for (int k = lane; k < in_features; k += kWarpSize) {
      sum += row[k] * x[k];
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
    ArgMaxCandidate best = block_candidates[0];
    for (int candidate = 1; candidate < kWarpsPerBlock; ++candidate) {
      if (IsBetterArgMaxCandidate(block_candidates[candidate], best)) {
        best = block_candidates[candidate];
      }
    }
    candidates[blockIdx.x] = best;
  }
}

__global__ void ArgMaxCandidatesKernel(const ArgMaxCandidate* candidates,
                                       int n, int* result) {
  __shared__ ArgMaxCandidate reduced[kBlockSize];
  const int tid = threadIdx.x;
  ArgMaxCandidate best = {-3.402823466e+38F, n};
  for (int i = tid; i < n; i += blockDim.x) {
    if (IsBetterArgMaxCandidate(candidates[i], best)) {
      best = candidates[i];
    }
  }
  reduced[tid] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride &&
        IsBetterArgMaxCandidate(reduced[tid + stride], reduced[tid])) {
      reduced[tid] = reduced[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *result = reduced[0].index;
  }
}

__global__ void RopeNormalKernel(float* x, int n_heads, int head_dim, int pos,
                                 float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs_per_head = head_dim / 2;
  int total_pairs = n_heads * pairs_per_head;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / pairs_per_head;
  int pair = pair_index % pairs_per_head;
  int dim = pair * 2;
  int base = head * head_dim + dim;
  float freq = 1.0f / powf(theta, static_cast<float>(dim) /
                                      static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base];
  float x1 = x[base + 1];
  x[base] = x0 * cos_val - x1 * sin_val;
  x[base + 1] = x0 * sin_val + x1 * cos_val;
}

__global__ void RopeNeoXKernel(float* x, int n_heads, int head_dim, int pos,
                               float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  int total_pairs = n_heads * half_dim;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / half_dim;
  int pair = pair_index % half_dim;
  int base = head * head_dim;
  float freq = 1.0f / powf(theta, static_cast<float>(2 * pair) /
                                      static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base + pair];
  float x1 = x[base + half_dim + pair];
  x[base + pair] = x0 * cos_val - x1 * sin_val;
  x[base + half_dim + pair] = x0 * sin_val + x1 * cos_val;
}

__global__ void RopeBatchNormalKernel(float* x, int n_heads, int head_dim,
                                      int start_pos, float theta,
                                      int total_pairs) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_pairs) return;
  const int pairs_per_token = n_heads * (head_dim / 2);
  const int token = index / pairs_per_token;
  const int within = index % pairs_per_token;
  const int head = within / (head_dim / 2);
  const int pair = within % (head_dim / 2);
  const int dim = pair * 2;
  const size_t base =
      (static_cast<size_t>(token) * n_heads + head) * head_dim + dim;
  const float freq = 1.0f / powf(theta, static_cast<float>(dim) / head_dim);
  const float angle = static_cast<float>(start_pos + token) * freq;
  const float c = cosf(angle);
  const float s = sinf(angle);
  const float x0 = x[base];
  const float x1 = x[base + 1];
  x[base] = x0 * c - x1 * s;
  x[base + 1] = x0 * s + x1 * c;
}

__global__ void RopeBatchNeoXKernel(float* x, int n_heads, int head_dim,
                                    int start_pos, float theta,
                                    int total_pairs) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_pairs) return;
  const int half_dim = head_dim / 2;
  const int pairs_per_token = n_heads * half_dim;
  const int token = index / pairs_per_token;
  const int within = index % pairs_per_token;
  const int head = within / half_dim;
  const int pair = within % half_dim;
  const size_t base =
      (static_cast<size_t>(token) * n_heads + head) * head_dim;
  const float freq =
      1.0f / powf(theta, static_cast<float>(2 * pair) / head_dim);
  const float angle = static_cast<float>(start_pos + token) * freq;
  const float c = cosf(angle);
  const float s = sinf(angle);
  const float x0 = x[base + pair];
  const float x1 = x[base + half_dim + pair];
  x[base + pair] = x0 * c - x1 * s;
  x[base + half_dim + pair] = x0 * s + x1 * c;
}

int GridFor(int n) { return (n + kBlockSize - 1) / kBlockSize; }

void RequireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
  if (a.shape != b.shape) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
}

void RequireSameShape(const CudaTensor& a, const CudaTensor& b,
                      const char* caller) {
  if (a.shape() != b.shape()) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
  if (a.device_id() != b.device_id()) {
    throw std::runtime_error(std::string(caller) +
                             ": tensors are on different CUDA devices");
  }
}

void Require1D(const Tensor& x, const char* caller) {
  if (x.num_dims() != 1) {
    throw std::runtime_error(std::string(caller) +
                             ": expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

void Require1D(const CudaTensor& x, const char* caller) {
  if (x.num_dims() != 1) {
    throw std::runtime_error(std::string(caller) +
                             ": expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

void ValidateRopeInputs(const Tensor& q, const Tensor& k, int pos,
                        float theta) {
  if (q.num_dims() != 2 || k.num_dims() != 2) {
    throw std::runtime_error("CudaRope: expected 2D tensors");
  }
  if (pos < 0) {
    throw std::out_of_range("CudaRope: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("CudaRope: theta must be finite and positive");
  }
  if (q.shape[1] != k.shape[1]) {
    throw std::runtime_error("CudaRope: q and k head_dim mismatch " +
                             q.ShapeStringShort() + " vs " +
                             k.ShapeStringShort());
  }
  if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
    throw std::runtime_error("CudaRope: head_dim must be positive and even");
  }
}

void ValidateRopeDeviceInputs(const CudaTensor& q, const CudaTensor& k,
                              int n_heads, int n_kv_heads, int head_dim,
                              int pos, float theta, int device_id) {
  if (q.device_id() != device_id || k.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRopeDeviceInput: input tensor is on a different CUDA device");
  }
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("CudaRopeDeviceInput: invalid head shape");
  }
  if (q.size() !=
      static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("CudaRopeDeviceInput: q shape mismatch");
  }
  if (k.size() !=
      static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("CudaRopeDeviceInput: k shape mismatch");
  }
  if (pos < 0) {
    throw std::out_of_range(
        "CudaRopeDeviceInput: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error(
        "CudaRopeDeviceInput: theta must be finite and positive");
  }
}

void ValidateEmbeddingDeviceWeightInputs(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id) {
  if (embedding_data == nullptr) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: embedding data is null");
  }
  if (embedding_shape.size() != 2) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: expected 2D embedding weight");
  }
  int vocab_size = embedding_shape[0];
  int dim = embedding_shape[1];
  if (vocab_size <= 0 || dim <= 0) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: embedding shape must be positive");
  }
  if (token_id < 0 || token_id >= vocab_size) {
    throw std::out_of_range(
        "CudaEmbeddingLookupDeviceWeight: token id out of range");
  }
}

void ValidateRmsNormDeviceWeightInputs(const CudaTensor& x,
                                       const void* weight_data,
                                       const std::vector<int>& weight_shape,
                                       float eps, int device_id) {
  Require1D(x, "CudaRmsNormDeviceWeight");
  if (weight_data == nullptr) {
    throw std::runtime_error("CudaRmsNormDeviceWeight: weight data is null");
  }
  if (x.shape() != weight_shape) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: x and weight shape mismatch");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: eps must be finite and positive");
  }
}

void ValidateAddDeviceWeightInputs(const CudaTensor& a, const void* b_data,
                                   const std::vector<int>& b_shape,
                                   int device_id) {
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: input tensor is on a different CUDA "
        "device");
  }
  if (b_data == nullptr) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: weight data is null");
  }
  if (a.shape() != b_shape) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: input and weight shape mismatch");
  }
}

}  // namespace

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id) {
  Require1D(x, "CudaRmsNorm");
  Require1D(weight, "CudaRmsNorm");
  if (x.shape != weight.shape) {
    throw std::runtime_error("CudaRmsNorm: x and weight shape mismatch");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error("CudaRmsNorm: eps must be finite and positive");
  }

  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));
  w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));
  CheckCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)),
               "cudaMemset(RmsNorm sum)");

  int n = static_cast<int>(x.size());
  SumSquaresKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(sum_dev.data()), n);
  CheckLastKernel("SumSquaresKernel");
  RmsNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<const float*>(w_dev.data()),
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(sum_dev.data()), n, eps);
  CheckLastKernel("RmsNormKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

Tensor CudaSilu(const Tensor& x, int device_id) {
  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());
  SiluKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("SiluKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseMul");
  CudaSetDevice(device_id);
  Tensor y(a.shape, 0.0f);
  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.Upload(a.data.data(), a.size() * sizeof(float));
  b_dev.Upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  ElementwiseMulKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("ElementwiseMulKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

CudaTensor CudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id) {
  ValidateEmbeddingDeviceWeightInputs(embedding_data, embedding_shape,
                                      token_id);
  CudaSetDevice(device_id);

  int dim = embedding_shape[1];
  CudaTensor y({dim}, device_id);
  EmbeddingLookupKernel<<<GridFor(dim), kBlockSize>>>(
      static_cast<const float*>(embedding_data), static_cast<float*>(y.data()),
      token_id, dim);
  CheckLastKernel("EmbeddingLookupKernel");
  return y;
}

CudaTensor CudaEmbeddingLookupBatchDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    const std::vector<int>& token_ids, int device_id) {
  if (token_ids.empty()) {
    throw std::runtime_error("CudaEmbeddingLookupBatch: token ids are empty");
  }
  for (int token_id : token_ids) {
    ValidateEmbeddingDeviceWeightInputs(embedding_data, embedding_shape,
                                        token_id);
  }
  CudaSetDevice(device_id);
  const int rows = static_cast<int>(token_ids.size());
  const int dim = embedding_shape[1];
  CudaDeviceBuffer ids(token_ids.size() * sizeof(int), device_id);
  ids.Upload(token_ids.data(), token_ids.size() * sizeof(int));
  CudaTensor y({rows, dim}, device_id);
  EmbeddingLookupBatchKernel<<<dim3(GridFor(dim), rows), kBlockSize>>>(
      static_cast<const float*>(embedding_data),
      static_cast<const int*>(ids.data()), static_cast<float*>(y.data()), dim);
  CheckLastKernel("EmbeddingLookupBatchKernel");
  return y;
}

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id) {
  Require1D(x, "CudaRmsNormDeviceInput");
  Require1D(weight, "CudaRmsNormDeviceInput");
  if (x.shape() != weight.shape) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: x and weight shape mismatch");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: eps must be finite and positive");
  }

  CudaSetDevice(device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));
  return CudaRmsNormDeviceWeight(x, w_dev.data(), weight.shape, eps, device_id);
}

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id) {
  CudaDeviceBuffer sum_workspace(sizeof(float), device_id);
  return CudaRmsNormDeviceWeight(x, weight_data, weight_shape, sum_workspace,
                                 eps, device_id);
}

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   CudaDeviceBuffer& sum_workspace, float eps,
                                   int device_id) {
  ValidateRmsNormDeviceWeightInputs(x, weight_data, weight_shape, eps,
                                    device_id);
  if (sum_workspace.empty() || sum_workspace.device_id() != device_id ||
      sum_workspace.bytes() < sizeof(float)) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: sum workspace is too small or on a "
        "different CUDA device");
  }
  CudaSetDevice(device_id);

  CudaTensor y(x.shape(), device_id);
  int n = static_cast<int>(x.size());
  RmsNormFusedKernel<<<1, kBlockSize>>>(
      static_cast<const float*>(x.data()),
      static_cast<const float*>(weight_data), static_cast<float*>(y.data()), n,
      eps);
  CheckLastKernel("RmsNormFusedKernel");

  return y;
}

void CudaAddRmsNormDeviceWeight(const CudaTensor& a, const CudaTensor& b,
                                const void* weight_data,
                                const std::vector<int>& weight_shape,
                                float eps, CudaTensor& sum_out,
                                CudaTensor& norm_out, int device_id) {
  RequireSameShape(a, b, "CudaAddRmsNormDeviceWeight");
  Require1D(a, "CudaAddRmsNormDeviceWeight");
  if (a.device_id() != device_id || weight_data == nullptr ||
      weight_shape != a.shape() || !std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error("CudaAddRmsNormDeviceWeight: invalid inputs");
  }
  CudaSetDevice(device_id);
  sum_out.Reset(a.shape(), device_id);
  norm_out.Reset(a.shape(), device_id);
  AddRmsNormFusedKernel<<<1, kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<const float*>(weight_data),
      static_cast<float*>(sum_out.data()), static_cast<float*>(norm_out.data()),
      static_cast<int>(a.size()), eps);
  CheckLastKernel("AddRmsNormFusedKernel");
}

CudaTensor CudaRmsNormBatchDeviceWeight(
    const CudaTensor& x, const void* weight_data,
    const std::vector<int>& weight_shape, float eps, int device_id) {
  if (x.num_dims() != 2 || weight_shape.size() != 1 ||
      x.shape()[1] != weight_shape[0] || x.device_id() != device_id ||
      weight_data == nullptr || !std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error("CudaRmsNormBatchDeviceWeight: invalid inputs");
  }
  CudaSetDevice(device_id);
  CudaTensor y(x.shape(), device_id);
  RmsNormBatchKernel<<<x.shape()[0], kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<const float*>(weight_data),
      static_cast<float*>(y.data()), x.shape()[1], eps);
  CheckLastKernel("RmsNormBatchKernel");
  return y;
}

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id) {
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaSiluDeviceInput: input tensor is on a different CUDA device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(x.shape(), device_id);

  int n = static_cast<int>(x.size());
  SiluKernel<<<GridFor(n), kBlockSize>>>(static_cast<const float*>(x.data()),
                                         static_cast<float*>(y.data()), n);
  CheckLastKernel("SiluKernel");

  return y;
}

CudaTensor CudaSiluMulDeviceInput(const CudaTensor& gate,
                                  const CudaTensor& up, int device_id) {
  RequireSameShape(gate, up, "CudaSiluMulDeviceInput");
  if (gate.device_id() != device_id) {
    throw std::runtime_error(
        "CudaSiluMulDeviceInput: input tensor is on a different CUDA device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(gate.shape(), device_id);

  const int n = static_cast<int>(gate.size());
  SiluMulKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(gate.data()), static_cast<const float*>(up.data()),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("SiluMulKernel");
  return y;
}

CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseMulDeviceInput");
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseMulDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseMulKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseMulKernel");

  return y;
}

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseAddDeviceInput");
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  return y;
}

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id) {
  ValidateAddDeviceWeightInputs(a, b_data, b_shape, device_id);
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b_data),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  return y;
}

CudaTensor CudaAddRowBiasDeviceWeight(const CudaTensor& x,
                                      const void* bias_data,
                                      const std::vector<int>& bias_shape,
                                      int device_id) {
  if (x.num_dims() != 2 || x.device_id() != device_id ||
      bias_data == nullptr || bias_shape != std::vector<int>{x.shape()[1]}) {
    throw std::runtime_error("CudaAddRowBiasDeviceWeight: invalid inputs");
  }
  CudaSetDevice(device_id);
  CudaTensor y(x.shape(), device_id);
  AddRowBiasKernel<<<GridFor(static_cast<int>(x.size())), kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<const float*>(bias_data),
      static_cast<float*>(y.data()), x.shape()[0], x.shape()[1]);
  CheckLastKernel("AddRowBiasKernel");
  return y;
}

CudaTensor CudaLastRow(const CudaTensor& x, int device_id) {
  if (x.num_dims() != 2 || x.device_id() != device_id || x.shape()[0] <= 0) {
    throw std::runtime_error("CudaLastRow: expected non-empty 2D CUDA tensor");
  }
  CudaSetDevice(device_id);
  const int cols = x.shape()[1];
  CudaTensor y({cols}, device_id);
  const char* src = static_cast<const char*>(x.data()) +
                    static_cast<size_t>(x.shape()[0] - 1) * cols * sizeof(float);
  CudaMemcpyBytes(y.data(), src, static_cast<size_t>(cols) * sizeof(float),
                  CudaMemcpyKind::kDeviceToDevice);
  return y;
}

void CudaConvertTensorToF16(const CudaTensor& src, CudaDeviceBuffer& dst,
                            int device_id) {
  if (src.device_id() != device_id || dst.device_id() != device_id ||
      dst.bytes() < src.size() * sizeof(uint16_t)) {
    throw std::runtime_error("CudaConvertTensorToF16: invalid device buffers");
  }
  CudaSetDevice(device_id);
  FloatToHalfKernel<<<GridFor(static_cast<int>(src.size())), kBlockSize>>>(
      static_cast<const float*>(src.data()), static_cast<__half*>(dst.data()),
      static_cast<int>(src.size()));
  CheckLastKernel("FloatToHalfKernel");
}

void CudaUploadTensorAsF16(const Tensor& src, CudaDeviceBuffer& dst,
                           int device_id) {
  if (src.size() == 0 || dst.device_id() != device_id ||
      dst.bytes() < src.size() * sizeof(uint16_t)) {
    throw std::runtime_error("CudaUploadTensorAsF16: invalid device buffer");
  }
  CudaDeviceBuffer temporary(src.size() * sizeof(float), device_id);
  temporary.Upload(src.data.data(), src.size() * sizeof(float));
  CudaSetDevice(device_id);
  FloatToHalfKernel<<<GridFor(static_cast<int>(src.size())), kBlockSize>>>(
      static_cast<const float*>(temporary.data()),
      static_cast<__half*>(dst.data()), static_cast<int>(src.size()));
  CheckLastKernel("FloatToHalfKernel(upload)");
}

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseAdd");
  CudaSetDevice(device_id);
  Tensor y(a.shape, 0.0f);
  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.Upload(a.data.data(), a.size() * sizeof(float));
  b_dev.Upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RopeType rope_type, int device_id) {
  ValidateRopeDeviceInputs(q, k, n_heads, n_kv_heads, head_dim, pos, theta,
                           device_id);
  CudaSetDevice(device_id);

  int q_pairs = n_heads * (head_dim / 2);
  int k_pairs = n_kv_heads * (head_dim / 2);
  if (rope_type == RopeType::kNeoX) {
    RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNeoXKernel(q)");
    RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNeoXKernel(k)");
  } else {
    RopeNormalKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNormalKernel(q)");
    RopeNormalKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNormalKernel(k)");
  }
}

void CudaRopeBatchDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                              int n_kv_heads, int head_dim, int start_pos,
                              float theta, RopeType rope_type, int device_id) {
  if (q.num_dims() != 2 || k.num_dims() != 2 || q.device_id() != device_id ||
      k.device_id() != device_id || q.shape()[0] != k.shape()[0] ||
      q.shape()[1] != n_heads * head_dim ||
      k.shape()[1] != n_kv_heads * head_dim || start_pos < 0 ||
      head_dim <= 0 || head_dim % 2 != 0 || !std::isfinite(theta) ||
      theta <= 0.0f) {
    throw std::runtime_error("CudaRopeBatchDeviceInput: invalid inputs");
  }
  CudaSetDevice(device_id);
  const int rows = q.shape()[0];
  const int q_pairs = rows * n_heads * (head_dim / 2);
  const int k_pairs = rows * n_kv_heads * (head_dim / 2);
  if (rope_type == RopeType::kNeoX) {
    RopeBatchNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, start_pos, theta,
        q_pairs);
    CheckLastKernel("RopeBatchNeoXKernel(q)");
    RopeBatchNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, start_pos, theta,
        k_pairs);
    CheckLastKernel("RopeBatchNeoXKernel(k)");
  } else {
    RopeBatchNormalKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, start_pos, theta,
        q_pairs);
    CheckLastKernel("RopeBatchNormalKernel(q)");
    RopeBatchNormalKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, start_pos, theta,
        k_pairs);
    CheckLastKernel("RopeBatchNormalKernel(k)");
  }
}

Tensor CudaSoftmax(const Tensor& x, int device_id) {
  Require1D(x, "CudaSoftmax");
  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer max_dev(sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());
  SoftmaxMaxKernel<<<1, kBlockSize>>>(static_cast<const float*>(x_dev.data()),
                                      static_cast<float*>(max_dev.data()), n);
  CheckLastKernel("SoftmaxMaxKernel");
  SoftmaxExpSumKernel<<<1, kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(max_dev.data()),
      static_cast<float*>(sum_dev.data()), n);
  CheckLastKernel("SoftmaxExpSumKernel");
  SoftmaxNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(sum_dev.data()), n);
  CheckLastKernel("SoftmaxNormKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

int CudaArgMax(const CudaTensor& x, int device_id) {
  CudaDeviceBuffer result_workspace(sizeof(int), device_id);
  return CudaArgMax(x, result_workspace, device_id);
}

int CudaArgMax(const CudaTensor& x, CudaDeviceBuffer& result_workspace,
               int device_id) {
  Require1D(x, "CudaArgMax");
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaArgMax: input tensor is on a different CUDA device");
  }
  CudaSetDevice(device_id);
  if (result_workspace.empty() || result_workspace.device_id() != device_id ||
      result_workspace.bytes() < sizeof(int)) {
    throw std::runtime_error(
        "CudaArgMax: result workspace is too small or on a different CUDA "
        "device");
  }
  ArgMaxKernel<<<1, kBlockSize>>>(static_cast<const float*>(x.data()),
                                  static_cast<int>(x.size()),
                                  static_cast<int*>(result_workspace.data()));
  CheckLastKernel("ArgMaxKernel");

  int result = 0;
  result_workspace.Download(&result, sizeof(result));
  return result;
}

int CudaF32LinearArgMaxDeviceInput(
    const CudaTensor& x, const void* weight_data,
    const std::vector<int>& weight_shape, CudaDeviceBuffer& candidate_workspace,
    CudaDeviceBuffer& result_workspace, int device_id) {
  Require1D(x, "CudaF32LinearArgMaxDeviceInput");
  if (weight_data == nullptr || weight_shape.size() != 2 ||
      weight_shape[0] <= 0 || weight_shape[1] <= 0 ||
      weight_shape[1] != x.shape()[0]) {
    throw std::runtime_error(
        "CudaF32LinearArgMaxDeviceInput: invalid lm_head shape");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaF32LinearArgMaxDeviceInput: input is on a different CUDA device");
  }
  CudaSetDevice(device_id);
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  const int candidate_count =
      (weight_shape[0] + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const size_t candidate_bytes =
      static_cast<size_t>(candidate_count) * sizeof(ArgMaxCandidate);
  if (candidate_workspace.empty() ||
      candidate_workspace.device_id() != device_id ||
      candidate_workspace.bytes() < candidate_bytes) {
    candidate_workspace.Reset(candidate_bytes, device_id);
  }
  if (result_workspace.empty() || result_workspace.device_id() != device_id ||
      result_workspace.bytes() < sizeof(int)) {
    throw std::runtime_error(
        "CudaF32LinearArgMaxDeviceInput: result workspace is too small or on "
        "a different CUDA device");
  }

  F32LinearArgMaxCandidatesKernel<<<candidate_count, kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<const float*>(weight_data),
      weight_shape[1], weight_shape[0],
      static_cast<ArgMaxCandidate*>(candidate_workspace.data()));
  CheckLastKernel("F32LinearArgMaxCandidatesKernel");
  ArgMaxCandidatesKernel<<<1, kBlockSize>>>(
      static_cast<const ArgMaxCandidate*>(candidate_workspace.data()),
      candidate_count, static_cast<int*>(result_workspace.data()));
  CheckLastKernel("ArgMaxCandidatesKernel");

  int result = 0;
  result_workspace.Download(&result, sizeof(result));
  return result;
}

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type,
              int device_id) {
  ValidateRopeInputs(q, k, pos, theta);
  CudaSetDevice(device_id);
  CudaDeviceBuffer q_dev(q.size() * sizeof(float), device_id);
  CudaDeviceBuffer k_dev(k.size() * sizeof(float), device_id);
  q_dev.Upload(q.data.data(), q.size() * sizeof(float));
  k_dev.Upload(k.data.data(), k.size() * sizeof(float));

  int q_pairs = q.shape[0] * (q.shape[1] / 2);
  int k_pairs = k.shape[0] * (k.shape[1] / 2);
  if (rope_type == RopeType::kNeoX) {
    RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    CheckLastKernel("RopeNeoXKernel(q)");
    RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    CheckLastKernel("RopeNeoXKernel(k)");
  } else {
    RopeNormalKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    CheckLastKernel("RopeNormalKernel(q)");
    RopeNormalKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    CheckLastKernel("RopeNormalKernel(k)");
  }

  q_dev.Download(q.data.data(), q.size() * sizeof(float));
  k_dev.Download(k.data.data(), k.size() * sizeof(float));
}

}  // namespace mini_llama
