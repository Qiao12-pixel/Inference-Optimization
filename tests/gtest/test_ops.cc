// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "mini_llama/ops.h"
#include "mini_llama/tensor.h"
#include "tests/test_names.h"

// ==========================================================================
// Matmul
// ==========================================================================
TEST(OpsTest, TestMatmulIdentity) {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1.0f;
  a[1] = 0.0f;
  a[2] = 0.0f;
  a[3] = 1.0f;
  Tensor b({2, 2}, 0.0f);
  b[0] = 1.0f;
  b[1] = 2.0f;
  b[2] = 3.0f;
  b[3] = 4.0f;
  Tensor c = Matmul(a, b);
  EXPECT_EQ(c.shape[0], 2);
  EXPECT_EQ(c.shape[1], 2);
  EXPECT_NEAR(c[0], 1.0f, 1e-5f);
  EXPECT_NEAR(c[1], 2.0f, 1e-5f);
  EXPECT_NEAR(c[2], 3.0f, 1e-5f);
  EXPECT_NEAR(c[3], 4.0f, 1e-5f);
}
TEST(OpsTest, TestMatmulSimple) {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;
  Tensor b({2, 2}, 0.0f);
  b[0] = 5.0f;
  b[1] = 6.0f;
  b[2] = 7.0f;
  b[3] = 8.0f;
  Tensor c = Matmul(a, b);
  EXPECT_NEAR(c[0], 19.0f, 1e-5f);
  EXPECT_NEAR(c[1], 22.0f, 1e-5f);
  EXPECT_NEAR(c[2], 43.0f, 1e-5f);
  EXPECT_NEAR(c[3], 50.0f, 1e-5f);
}

TEST(OpsTest, TestMatmulRectangular) {
  // a: [2, 3], b: [3, 4] -> c: [2, 4]
  Tensor a({2, 3}, 0.0f);
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;
  a[4] = 5.0f;
  a[5] = 6.0f;
  Tensor b({3, 4}, 0.0f);
  for (int i = 0; i < 12; ++i) {
    b[i] = static_cast<float>(i);
  }
  Tensor c = Matmul(a, b);
  EXPECT_EQ(c.shape[0], 2);
  EXPECT_EQ(c.shape[1], 4);
  // c[0,0] = 1*0 + 2*4 + 3*8 = 0 + 8 + 24 = 32
  EXPECT_NEAR(c.At2(0, 0), 32.0f, 1e-5f);
  // c[1,0] = 4*0 + 5*4 + 6*8 = 0 + 20 + 48 = 68
  EXPECT_NEAR(c.At2(1, 0), 68.0f, 1e-5f);
}

TEST(OpsTest, TestMatmulShapeMismatch) {
  Tensor a({2, 3}, 0.0f);
  Tensor b({2, 3}, 0.0f);
  try {
    Matmul(a, b);
    FAIL() << "expected exception for Matmul shape mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ==========================================================================
// Linear
// ==========================================================================
TEST(OpsTest, TestLinear1d) {
  Tensor x({3}, 0.0f);
  x[0] = 1.0f;
  x[1] = 2.0f;
  x[2] = 3.0f;
  Tensor weight({3, 3}, 0.0f);
  weight[0] = 1.0f;
  weight[4] = 1.0f;
  weight[8] = 1.0f;
  Tensor y = Linear(x, weight);
  EXPECT_EQ(y.num_dims(), 1);
  EXPECT_EQ(y.shape[0], 3);
  EXPECT_NEAR(y[0], 1.0f, 1e-5f);
  EXPECT_NEAR(y[1], 2.0f, 1e-5f);
  EXPECT_NEAR(y[2], 3.0f, 1e-5f);
}

TEST(OpsTest, TestLinear2d) {
  Tensor x({1, 3}, 0.0f);
  x[0] = 1.0f;
  x[1] = 2.0f;
  x[2] = 3.0f;
  Tensor weight({2, 3}, 0.0f);
  weight[0] = 1.0f;
  weight[1] = 0.0f;
  weight[2] = 0.0f;
  weight[3] = 0.0f;
  weight[4] = 1.0f;
  weight[5] = 0.0f;
  Tensor y = Linear(x, weight);
  EXPECT_EQ(y.num_dims(), 2);
  EXPECT_EQ(y.shape[0], 1);
  EXPECT_EQ(y.shape[1], 2);
  EXPECT_NEAR(y[0], 1.0f, 1e-5f);
  EXPECT_NEAR(y[1], 2.0f, 1e-5f);
}

TEST(OpsTest, TestLinearShapeMismatch) {
  Tensor x({3}, 0.0f);
  Tensor weight({2, 4}, 0.0f);
  try {
    Linear(x, weight);
    FAIL() << "expected exception for Linear shape mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ==========================================================================
// RmsNorm
// ==========================================================================
TEST(OpsTest, TestRmsNormUnit) {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  EXPECT_EQ(y.shape[0], 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(y[i], 1.0f, 1e-4f);
  }
}

TEST(OpsTest, TestRmsNormScaled) {
  Tensor x({4}, 0.0f);
  x[0] = 2.0f;
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  EXPECT_NEAR(y[0], 2.0f, 1e-4f);
  EXPECT_NEAR(y[1], 0.0f, 1e-4f);
}

TEST(OpsTest, TestRmsNormWithWeight) {
  // x = [1, 1, 1, 1], weight = [2, 2, 2, 2]
  // rms = 1, scale = 1, y = x * scale * weight = [2, 2, 2, 2]
  Tensor x({4}, 1.0f);
  Tensor w({4}, 2.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(y[i], 2.0f, 1e-4f);
  }
}

TEST(OpsTest, TestRmsNormShapeMismatch) {
  Tensor x({4}, 0.0f);
  Tensor w({3}, 0.0f);
  try {
    RmsNorm(x, w, 1e-5f);
    FAIL() << "expected exception for RmsNorm shape mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ==========================================================================
// Softmax
// ==========================================================================
TEST(OpsTest, TestSoftmax) {
  Tensor x({3}, 0.0f);
  x[0] = 1.0f;
  x[1] = 2.0f;
  x[2] = 3.0f;
  Tensor y = Softmax(x);
  EXPECT_EQ(y.shape[0], 3);
  EXPECT_NEAR(y[0], 0.09003057f, 1e-5f);
  EXPECT_NEAR(y[1], 0.24472847f, 1e-5f);
  EXPECT_NEAR(y[2], 0.66524096f, 1e-5f);
  float sum = y[0] + y[1] + y[2];
  EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(OpsTest, TestSoftmaxUniform) {
  // Softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
  Tensor x({3}, 0.0f);
  Tensor y = Softmax(x);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(y[i], 1.0f / 3.0f, 1e-5f);
  }
}

TEST(OpsTest, TestSoftmaxWrongDim) {
  Tensor x({2, 3}, 0.0f);
  try {
    Softmax(x);
    FAIL() << "expected exception for Softmax on 2D tensor";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(OpsTest, TestSoftmaxEmpty) {
  Tensor x;
  x.shape = {0};
  try {
    Softmax(x);
    FAIL() << "expected exception for Softmax on empty tensor";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ==========================================================================
// Silu
// ==========================================================================
TEST(OpsTest, TestSilu) {
  Tensor x({1}, 0.0f);
  Tensor y = Silu(x);
  EXPECT_NEAR(y[0], 0.0f, 1e-5f);

  x[0] = 1.0f;
  y = Silu(x);
  EXPECT_NEAR(y[0], 0.7310586f, 1e-5f);
}

// ==========================================================================
// SwiGlu
// ==========================================================================
TEST(OpsTest, TestSwiglu) {
  // SwiGlu(gate, up) = Silu(gate) * up
  Tensor gate({3}, 0.0f);
  gate[0] = 0.0f;
  gate[1] = 1.0f;
  gate[2] = 2.0f;
  Tensor up({3}, 0.0f);
  up[0] = 1.0f;
  up[1] = 2.0f;
  up[2] = 3.0f;
  Tensor y = SwiGlu(gate, up);

  // Silu(0) = 0, so SwiGlu(0, 1) = 0
  EXPECT_NEAR(y[0], 0.0f, 1e-5f);
  // Silu(1) ≈ 0.731, so SwiGlu(1, 2) ≈ 0.731 * 2 ≈ 1.462
  EXPECT_NEAR(y[1], 0.7310586f * 2.0f, 1e-5f);
  // Silu(2) ≈ 1.761, so SwiGlu(2, 3) ≈ 1.761 * 3 ≈ 5.284
  EXPECT_NEAR(y[2], 1.7615942f * 3.0f, 1e-4f);
}

TEST(OpsTest, TestSwigluShapeMismatch) {
  Tensor a({3}, 0.0f);
  Tensor b({4}, 0.0f);
  try {
    SwiGlu(a, b);
    FAIL() << "expected exception for SwiGlu shape mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ==========================================================================
// ElementwiseMul
// ==========================================================================
TEST(OpsTest, TestElementwiseMul) {
  Tensor a({3}, 0.0f);
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  Tensor b({3}, 0.0f);
  b[0] = 4.0f;
  b[1] = 5.0f;
  b[2] = 6.0f;
  Tensor y = ElementwiseMul(a, b);
  EXPECT_NEAR(y[0], 4.0f, 1e-5f);
  EXPECT_NEAR(y[1], 10.0f, 1e-5f);
  EXPECT_NEAR(y[2], 18.0f, 1e-5f);
}

TEST(OpsTest, TestElementwiseMulShapeMismatch) {
  Tensor a({3}, 0.0f);
  Tensor b({4}, 0.0f);
  try {
    ElementwiseMul(a, b);
    FAIL() << "expected exception for ElementwiseMul shape mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ==========================================================================
// ArgMax
// ==========================================================================
TEST(OpsTest, TestArgMax) {
  Tensor x({5}, 0.0f);
  x[0] = 1.0f;
  x[1] = 5.0f;
  x[2] = 3.0f;
  x[3] = 5.0f;
  x[4] = 2.0f;
  int idx = ArgMax(x);
  EXPECT_EQ(idx, 1);
}

TEST(OpsTest, TestArgMaxEmpty) {
  Tensor x;
  try {
    ArgMax(x);
    FAIL() << "expected exception for ArgMax on empty tensor";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ==========================================================================
// Rope
// ==========================================================================
TEST(OpsTest, TestRopePreservesNorm) {
  int n_heads = 2;
  int head_dim = 4;
  Tensor q({n_heads, head_dim}, 1.0f);
  Tensor k({n_heads, head_dim}, 1.0f);
  Rope(q, k, 5, 10000.0f);
  for (int h = 0; h < n_heads; ++h) {
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      EXPECT_TRUE(std::isfinite(q.At2(h, d)));
      EXPECT_TRUE(std::isfinite(k.At2(h, d)));
      q_norm += q.At2(h, d) * q.At2(h, d);
      k_norm += k.At2(h, d) * k.At2(h, d);
    }
    EXPECT_NEAR(q_norm, 4.0f, 1e-5f);
    EXPECT_NEAR(k_norm, 4.0f, 1e-5f);
  }
}

TEST(OpsTest, TestRopeNeoXMatchesSplitHalfPairs) {
  Tensor q({1, 4}, 0.0f);
  Tensor k({1, 4}, 0.0f);
  q[0] = 1.0f;
  q[1] = 2.0f;
  q[2] = 3.0f;
  q[3] = 4.0f;
  k = q;

  Rope(q, k, 1, 10000.0f, RopeType::kNeoX);

  float c0 = std::cos(1.0f);
  float s0 = std::sin(1.0f);
  float c1 = std::cos(0.01f);
  float s1 = std::sin(0.01f);

  EXPECT_NEAR(q[0], 1.0f * c0 - 3.0f * s0, 1e-5f);
  EXPECT_NEAR(q[2], 1.0f * s0 + 3.0f * c0, 1e-5f);
  EXPECT_NEAR(q[1], 2.0f * c1 - 4.0f * s1, 1e-5f);
  EXPECT_NEAR(q[3], 2.0f * s1 + 4.0f * c1, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(k[i], q[i], 1e-5f);
  }
}

TEST(OpsTest, TestRopeWrongDim) {
  Tensor q({4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  try {
    Rope(q, k, 0, 10000.0f);
    FAIL() << "expected exception for Rope with 1D q";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(OpsTest, TestRopeRejectsMismatchedHeadDim) {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 6}, 0.0f);
  try {
    Rope(q, k, 0, 10000.0f);
    FAIL() << "expected exception for mismatched RoPE head_dim";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(OpsTest, TestRopeRejectsOddHeadDim) {
  Tensor q({2, 3}, 0.0f);
  Tensor k({2, 3}, 0.0f);
  try {
    Rope(q, k, 0, 10000.0f);
    FAIL() << "expected exception for odd RoPE head_dim";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(OpsTest, TestRopeRejectsNegativePosition) {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  try {
    Rope(q, k, -1, 10000.0f);
    FAIL() << "expected exception for negative RoPE position";
  } catch (const std::out_of_range&) {
    // expected
  }
}

TEST(OpsTest, TestRopeRejectsNonpositiveTheta) {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  try {
    Rope(q, k, 0, 0.0f);
    FAIL() << "expected exception for nonpositive RoPE theta";
  } catch (const std::runtime_error&) {
    // expected
  }
}
