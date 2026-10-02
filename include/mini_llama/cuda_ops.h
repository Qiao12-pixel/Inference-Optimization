// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_OPS_H_
#define INCLUDE_MINI_LLAMA_CUDA_OPS_H_

#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool CudaOpsBuilt();

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id = 0);

Tensor CudaSilu(const Tensor& x, int device_id = 0);

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaSoftmax(const Tensor& x, int device_id = 0);

// Returns the lowest index whose value is maximal. Only the int token id is
// copied back to the host.
int CudaArgMax(const CudaTensor& x, int device_id = 0);

int CudaArgMax(const CudaTensor& x, CudaDeviceBuffer& result_workspace,
               int device_id = 0);

// Greedy-only F32 lm_head projection. Computes the matrix-vector product and
// ArgMax without materializing a full [vocab_size] logits tensor. The caller
// owns reusable candidate/result workspaces; only the winning token id is
// copied to the host. Ties select the lower token id, matching CudaArgMax.
int CudaF32LinearArgMaxDeviceInput(
    const CudaTensor& x, const void* weight_data,
    const std::vector<int>& weight_shape, CudaDeviceBuffer& candidate_workspace,
    CudaDeviceBuffer& result_workspace, int device_id = 0);

CudaTensor CudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id = 0);

CudaTensor CudaEmbeddingLookupBatchDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    const std::vector<int>& token_ids, int device_id = 0);

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id = 0);

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id = 0);

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   CudaDeviceBuffer& sum_workspace, float eps,
                                   int device_id = 0);

// Fuses residual add and RMSNorm for 1D Decode activations. Both the residual
// sum and its normalized view are produced in one kernel because the FFN
// consumes the normalized tensor while the final residual still needs the sum.
void CudaAddRmsNormDeviceWeight(const CudaTensor& a, const CudaTensor& b,
                                const void* weight_data,
                                const std::vector<int>& weight_shape,
                                float eps, CudaTensor& sum_out,
                                CudaTensor& norm_out, int device_id = 0);

CudaTensor CudaRmsNormBatchDeviceWeight(
    const CudaTensor& x, const void* weight_data,
    const std::vector<int>& weight_shape, float eps, int device_id = 0);

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id = 0);

// Fused SiLU(gate) * up used by SwiGLU FFNs. This avoids materializing the
// intermediate SiLU tensor and one kernel launch.
CudaTensor CudaSiluMulDeviceInput(const CudaTensor& gate,
                                  const CudaTensor& up,
                                  int device_id = 0);

CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id = 0);

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b,
                                         int device_id = 0);

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id = 0);

CudaTensor CudaAddRowBiasDeviceWeight(const CudaTensor& x,
                                      const void* bias_data,
                                      const std::vector<int>& bias_shape,
                                      int device_id = 0);

CudaTensor CudaLastRow(const CudaTensor& x, int device_id = 0);

// Convert F32 activations/weights to raw IEEE FP16 device storage. The raw
// buffer is consumed by the experimental Tensor Core Prefill GEMM path.
void CudaUploadTensorAsF16(const Tensor& src, CudaDeviceBuffer& dst,
                           int device_id = 0);
void CudaConvertTensorToF16(const CudaTensor& src, CudaDeviceBuffer& dst,
                            int device_id = 0);

void CudaRope(Tensor& q, Tensor& k, int pos, float theta,
              RopeType rope_type = RopeType::kNormal, int device_id = 0);

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RopeType rope_type = RopeType::kNormal,
                         int device_id = 0);

void CudaRopeBatchDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                              int n_kv_heads, int head_dim, int start_pos,
                              float theta,
                              RopeType rope_type = RopeType::kNormal,
                              int device_id = 0);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_OPS_H_
