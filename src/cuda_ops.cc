// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/cuda_ops.h"

#include <stdexcept>
#include <vector>

namespace mini_llama {

namespace {

std::runtime_error CudaOpsNotBuiltError() {
  return std::runtime_error(
      "CUDA ops were not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a "
      "NVIDIA CUDA machine.");
}

}  // namespace

bool CudaOpsBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id) {
  (void)x;
  (void)weight;
  (void)eps;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

Tensor CudaSilu(const Tensor& x, int device_id) {
  (void)x;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
  (void)a;
  (void)b;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id) {
  (void)embedding_data;
  (void)embedding_shape;
  (void)token_id;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaEmbeddingLookupBatchDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    const std::vector<int>& token_ids, int device_id) {
  (void)embedding_data;
  (void)embedding_shape;
  (void)token_ids;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id) {
  (void)x;
  (void)weight;
  (void)eps;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id) {
  (void)x;
  (void)weight_data;
  (void)weight_shape;
  (void)eps;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

void CudaAddRmsNormDeviceWeight(const CudaTensor& a, const CudaTensor& b,
                                const void* weight_data,
                                const std::vector<int>& weight_shape,
                                float eps, CudaTensor& sum_out,
                                CudaTensor& norm_out, int device_id) {
  (void)a;
  (void)b;
  (void)weight_data;
  (void)weight_shape;
  (void)eps;
  (void)sum_out;
  (void)norm_out;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaRmsNormBatchDeviceWeight(
    const CudaTensor& x, const void* weight_data,
    const std::vector<int>& weight_shape, float eps, int device_id) {
  (void)x;
  (void)weight_data;
  (void)weight_shape;
  (void)eps;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   CudaDeviceBuffer& sum_workspace, float eps,
                                   int device_id) {
  (void)x;
  (void)weight_data;
  (void)weight_shape;
  (void)sum_workspace;
  (void)eps;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id) {
  (void)x;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaSiluMulDeviceInput(const CudaTensor& gate,
                                  const CudaTensor& up, int device_id) {
  (void)gate;
  (void)up;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
  (void)a;
  (void)b;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
  (void)a;
  (void)b;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id) {
  (void)a;
  (void)b_data;
  (void)b_shape;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaAddRowBiasDeviceWeight(const CudaTensor& x,
                                      const void* bias_data,
                                      const std::vector<int>& bias_shape,
                                      int device_id) {
  (void)x;
  (void)bias_data;
  (void)bias_shape;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

CudaTensor CudaLastRow(const CudaTensor& x, int device_id) {
  (void)x;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

void CudaUploadTensorAsF16(const Tensor& src, CudaDeviceBuffer& dst,
                           int device_id) {
  (void)src;
  (void)dst;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

void CudaConvertTensorToF16(const CudaTensor& src, CudaDeviceBuffer& dst,
                            int device_id) {
  (void)src;
  (void)dst;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
  (void)a;
  (void)b;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

Tensor CudaSoftmax(const Tensor& x, int device_id) {
  (void)x;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

int CudaArgMax(const CudaTensor& x, int device_id) {
  (void)x;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

int CudaArgMax(const CudaTensor& x, CudaDeviceBuffer& result_workspace,
               int device_id) {
  (void)x;
  (void)result_workspace;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

int CudaF32LinearArgMaxDeviceInput(
    const CudaTensor& x, const void* weight_data,
    const std::vector<int>& weight_shape, CudaDeviceBuffer& candidate_workspace,
    CudaDeviceBuffer& result_workspace, int device_id) {
  (void)x;
  (void)weight_data;
  (void)weight_shape;
  (void)candidate_workspace;
  (void)result_workspace;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type,
              int device_id) {
  (void)q;
  (void)k;
  (void)pos;
  (void)theta;
  (void)rope_type;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RopeType rope_type, int device_id) {
  (void)q;
  (void)k;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)pos;
  (void)theta;
  (void)rope_type;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

void CudaRopeBatchDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                              int n_kv_heads, int head_dim, int start_pos,
                              float theta, RopeType rope_type, int device_id) {
  (void)q;
  (void)k;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)start_pos;
  (void)theta;
  (void)rope_type;
  (void)device_id;
  throw CudaOpsNotBuiltError();
}

#endif

}  // namespace mini_llama
