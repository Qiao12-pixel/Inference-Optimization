// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
#include "mini_llama/tensor.h"
#include "tests/test_names.h"

TEST(QuantTest, TestQ80BlockLayout) {
  EXPECT_EQ(kQ80BlockSize, 32);
  EXPECT_EQ(sizeof(BlockQ80), 34);
}

// ---------------------------------------------------------------------------
// Q8_0 quantization / dequantization
// ---------------------------------------------------------------------------
TEST(QuantTest, TestQ80RoundtripIdentity) {
  Tensor src({32}, 0.0f);
  for (int i = 0; i < 32; ++i) {
    src.data[i] = static_cast<float>(i) * 0.1f;
  }

  auto blocks = QuantizeToQ80(src);
  EXPECT_EQ(blocks.size(), 1);

  Tensor dst = DequantizeFromQ80(blocks, src.shape);
  EXPECT_EQ(dst.shape.size(), 1);
  EXPECT_EQ(dst.shape[0], 32);

  // Should be close but not exact due to quantization.
  // Q8_0 roundtrip error can be up to ~0.5 * scale per element.
  float max_err = 0.0f;
  for (size_t i = 0; i < src.size(); ++i) {
    float err = std::abs(src.data[i] - dst.data[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  EXPECT_TRUE(max_err < 6e-2f);
}

TEST(QuantTest, TestQ80AllZeros) {
  Tensor src({64}, 0.0f);
  auto blocks = QuantizeToQ80(src);
  EXPECT_EQ(blocks.size(), 2);

  Tensor dst = DequantizeFromQ80(blocks, src.shape);
  for (size_t i = 0; i < dst.size(); ++i) {
    EXPECT_NEAR(dst.data[i], 0.0f, 1e-6f);
  }
}

TEST(QuantTest, TestQ80MultiBlock) {
  Tensor src({100}, 0.0f);
  for (int i = 0; i < 100; ++i) {
    src.data[i] = static_cast<float>(i) * 0.05f;
  }

  auto blocks = QuantizeToQ80(src);
  EXPECT_EQ(blocks.size(), 4);  // ceil(100 / 32) = 4

  Tensor dst = DequantizeFromQ80(blocks, src.shape);
  EXPECT_EQ(dst.shape[0], 100);

  float max_err = 0.0f;
  for (size_t i = 0; i < src.size(); ++i) {
    float err = std::abs(src.data[i] - dst.data[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  EXPECT_TRUE(max_err < 6e-2f);
}

TEST(QuantTest, TestQ80EmptyTensor) {
  try {
    Tensor src({0}, 0.0f);
    (void)src;
    FAIL() << "expected exception for zero-dimension tensor";
  } catch (const std::runtime_error&) {
    // expected: Tensor rejects zero dimension
  }
}

// ---------------------------------------------------------------------------
// Q8_0 Matmul
// ---------------------------------------------------------------------------
TEST(QuantTest, TestMatmulQ80MatchesF32) {
  // Small exact test
  Tensor a({2, 3}, 0.0f);
  a.data[0] = 1.0f;
  a.data[1] = 2.0f;
  a.data[2] = 3.0f;
  a.data[3] = 4.0f;
  a.data[4] = 5.0f;
  a.data[5] = 6.0f;

  Tensor b({3, 2}, 0.0f);
  b.data[0] = 1.0f;
  b.data[1] = 2.0f;
  b.data[2] = 3.0f;
  b.data[3] = 4.0f;
  b.data[4] = 5.0f;
  b.data[5] = 6.0f;

  auto qA = QuantizeToQ80(a);
  Tensor c_f32 = Matmul(a, b);
  Tensor c_q8 = MatmulQ80(qA, b, a.shape);

  EXPECT_EQ(c_f32.shape.size(), 2);
  EXPECT_EQ(c_q8.shape.size(), 2);
  EXPECT_EQ(c_f32.shape[0], c_q8.shape[0]);
  EXPECT_EQ(c_f32.shape[1], c_q8.shape[1]);

  for (size_t i = 0; i < c_f32.size(); ++i) {
    // Q8_0 Matmul accumulates per-element quantization error;
    // tolerance must be looser than raw roundtrip.
    EXPECT_NEAR(c_f32.data[i], c_q8.data[i], 2e-1f);
  }
}

TEST(QuantTest, TestCompareMatmulError) {
  Tensor a({16, 16}, 0.0f);
  Tensor b({16, 16}, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    a.data[i] = static_cast<float>(i) * 0.01f - 1.0f;
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
  }

  float err = CompareMatmulError(a, b);
  EXPECT_TRUE(err >= 0.0f);
  EXPECT_TRUE(err < 1e-1f);  // Q8_0 error should be small
}

TEST(QuantTest, TestDequantizeRejectsBadShape) {
  Tensor src({32}, 1.0f);
  auto blocks = QuantizeToQ80(src);

  try {
    (void)DequantizeFromQ80(blocks, {-1});
    FAIL() << "expected exception for negative shape";
  } catch (const std::runtime_error&) {
    // expected
  }

  try {
    (void)DequantizeFromQ80(blocks, {0});
    FAIL() << "expected exception for zero shape";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(QuantTest, TestDequantizeRejectsBlockCountMismatch) {
  Tensor src({64}, 1.0f);
  auto blocks = QuantizeToQ80(src);
  blocks.pop_back();

  try {
    (void)DequantizeFromQ80(blocks, src.shape);
    FAIL() << "expected exception for block count mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ---------------------------------------------------------------------------
// Q4_0 quantization / dequantization
// ---------------------------------------------------------------------------
TEST(QuantTest, TestQ40BlockLayout) {
  EXPECT_EQ(kQ40BlockSize, 32);
  EXPECT_EQ(sizeof(BlockQ40), 18);
}

TEST(QuantTest, TestQ40RoundtripIdentity) {
  Tensor src({32}, 0.0f);
  for (int i = 0; i < 32; ++i) {
    src.data[i] = static_cast<float>(i) * 0.1f;
  }

  auto blocks = QuantizeToQ40(src);
  EXPECT_EQ(blocks.size(), 1);

  Tensor dst = DequantizeFromQ40(blocks, src.shape);
  EXPECT_EQ(dst.shape[0], 32);

  float max_err = 0.0f;
  for (size_t i = 0; i < src.size(); ++i) {
    float err = std::abs(src.data[i] - dst.data[i]);
    if (err > max_err) {
      max_err = err;
    }
  }
  EXPECT_TRUE(max_err < 3e-1f);
}

TEST(QuantTest, TestQ40AllZeros) {
  Tensor src({64}, 0.0f);
  auto blocks = QuantizeToQ40(src);
  EXPECT_EQ(blocks.size(), 2);

  Tensor dst = DequantizeFromQ40(blocks, src.shape);
  for (size_t i = 0; i < dst.size(); ++i) {
    EXPECT_NEAR(dst.data[i], 0.0f, 1e-6f);
  }
}

TEST(QuantTest, TestQ40LinearMatchesF32) {
  // Use in_features divisible by kQ40BlockSize (32)
  Tensor weight({4, 32}, 0.0f);
  for (size_t i = 0; i < weight.size(); ++i) {
    weight.data[i] = static_cast<float>(i) * 0.05f - 1.0f;
  }

  Tensor x({32}, 0.0f);
  for (size_t i = 0; i < x.size(); ++i) {
    x.data[i] = static_cast<float>(i) * 0.1f - 0.4f;
  }

  auto qW = QuantizeToQ40(weight);
  Tensor y_f32 = Linear(x, weight);
  Tensor y_q4 = LinearQ40(x, qW, weight.shape);

  EXPECT_EQ(y_f32.shape, y_q4.shape);
  for (size_t i = 0; i < y_f32.size(); ++i) {
    EXPECT_NEAR(y_f32.data[i], y_q4.data[i], 2.0f);
  }
}

TEST(QuantTest, TestCompareQ40Error) {
  // Use dimensions divisible by kQ40BlockSize (32)
  Tensor weight({8, 32}, 0.0f);
  Tensor x({32}, 0.0f);
  for (size_t i = 0; i < weight.size(); ++i) {
    weight.data[i] = static_cast<float>(i) * 0.01f - 1.0f;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    x.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
  }

  float err = CompareQ40Error(weight, x);
  EXPECT_TRUE(err >= 0.0f);
  EXPECT_TRUE(err < 2.0f);
}

// ---------------------------------------------------------------------------
