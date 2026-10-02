// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "mini_llama/tensor.h"
#include "tests/test_names.h"

// ---------------------------------------------------------------------------
// Basic construction and properties
// ---------------------------------------------------------------------------
TEST(TensorTest, TestTensorShapeAndSize) {
  Tensor t1({3, 4, 5}, 0.0f);
  EXPECT_EQ(t1.num_dims(), 3);
  EXPECT_EQ(t1.size(), 60);
  EXPECT_EQ(t1.num_elements(), 60);
  EXPECT_EQ(t1.shape[0], 3);
  EXPECT_EQ(t1.shape[1], 4);
  EXPECT_EQ(t1.shape[2], 5);
}

TEST(TensorTest, TestTensorFill) {
  Tensor t({10}, 3.14f);
  EXPECT_EQ(t.size(), 10);
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.data[i], 3.14f, 1e-6f);
  }
}

TEST(TensorTest, TestTensorRejectsZeroDimension) {
  try {
    Tensor t({2, 0}, 0.0f);
    FAIL() << "expected exception for zero tensor dimension";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(TensorTest, TestTensorRejectsNegativeDimension) {
  try {
    Tensor t({2, -3}, 0.0f);
    FAIL() << "expected exception for negative tensor dimension";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(TensorTest, TestTensorIndexing) {
  Tensor t({2, 3}, 0.0f);
  t.At({0, 0}) = 1.0f;
  t.At({0, 1}) = 2.0f;
  t.At({1, 2}) = 6.0f;
  EXPECT_NEAR(t.At({0, 0}), 1.0f, 1e-6f);
  EXPECT_NEAR(t.At({0, 1}), 2.0f, 1e-6f);
  EXPECT_NEAR(t.At({1, 2}), 6.0f, 1e-6f);
}

TEST(TensorTest, TestTensorIndexWrongRank) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({1});
    FAIL() << "expected exception for wrong FlatIndex rank";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(TensorTest, TestTensorIndexNegative) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({-1, 0});
    FAIL() << "expected exception for negative FlatIndex";
  } catch (const std::out_of_range&) {
    // expected
  }
}

TEST(TensorTest, TestTensorIndexTooLarge) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At({0, 3});
    FAIL() << "expected exception for FlatIndex above shape bound";
  } catch (const std::out_of_range&) {
    // expected
  }
}

TEST(TensorTest, TestTensor1dAccess) {
  Tensor t({5}, 0.0f);
  t[2] = 42.0f;
  EXPECT_NEAR(t[2], 42.0f, 1e-6f);
}

TEST(TensorTest, TestMakeTensorHelpers) {
  auto t1 = MakeTensor1D(10, 1.0f);
  EXPECT_EQ(t1.num_dims(), 1);
  EXPECT_EQ(t1.shape[0], 10);

  auto t2 = MakeTensor2D(3, 4, 1.0f);
  EXPECT_EQ(t2.num_dims(), 2);
  EXPECT_EQ(t2.shape[0], 3);
  EXPECT_EQ(t2.shape[1], 4);

  auto t3 = MakeTensor3D(2, 3, 4, 1.0f);
  EXPECT_EQ(t3.num_dims(), 3);
  EXPECT_EQ(t3.size(), 24);

  auto t4 = MakeTensor4D(2, 3, 4, 5, 1.0f);
  EXPECT_EQ(t4.num_dims(), 4);
  EXPECT_EQ(t4.size(), 120);
}

// ---------------------------------------------------------------------------
// At1 / At2 / At3 / At4 convenience accessors
// ---------------------------------------------------------------------------
TEST(TensorTest, TestTensorAt1) {
  Tensor t({5}, 0.0f);
  t.At1(2) = 7.0f;
  EXPECT_NEAR(t.At1(2), 7.0f, 1e-6f);
}

TEST(TensorTest, TestTensorAt1WrongDim) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At1(0);
    FAIL() << "expected exception for At1 on 2D tensor";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(TensorTest, TestTensorAt2) {
  Tensor t({2, 3}, 0.0f);
  t.At2(0, 1) = 5.0f;
  t.At2(1, 2) = 9.0f;
  EXPECT_NEAR(t.At2(0, 1), 5.0f, 1e-6f);
  EXPECT_NEAR(t.At2(1, 2), 9.0f, 1e-6f);
}

TEST(TensorTest, TestTensorAt2OutOfRange) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.At2(2, 0);
    FAIL() << "expected exception for At2 row out of range";
  } catch (const std::out_of_range&) {
    // expected
  }
}

TEST(TensorTest, TestTensorAt3) {
  Tensor t({2, 2, 2}, 0.0f);
  t.At3(1, 0, 1) = 3.0f;
  EXPECT_NEAR(t.At3(1, 0, 1), 3.0f, 1e-6f);
}

TEST(TensorTest, TestTensorAt4) {
  Tensor t({2, 2, 2, 2}, 0.0f);
  t.At4(1, 1, 0, 0) = 4.0f;
  EXPECT_NEAR(t.At4(1, 1, 0, 0), 4.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// row_ptr
// ---------------------------------------------------------------------------
TEST(TensorTest, TestTensorRowPtr) {
  Tensor t({3, 4}, 0.0f);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      t.At2(i, j) = static_cast<float>(i * 10 + j);
    }
  }
  const float* row1 = t.RowPtr(1);
  EXPECT_NEAR(row1[0], 10.0f, 1e-6f);
  EXPECT_NEAR(row1[3], 13.0f, 1e-6f);
}

TEST(TensorTest, TestTensorRowPtrWrongDim) {
  Tensor t({5}, 0.0f);
  try {
    t.RowPtr(0);
    FAIL() << "expected exception for row_ptr on 1D tensor";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(TensorTest, TestTensorRowPtrOutOfRange) {
  Tensor t({3, 4}, 0.0f);
  try {
    t.RowPtr(3);
    FAIL() << "expected exception for row_ptr row out of range";
  } catch (const std::out_of_range&) {
    // expected
  }
}

// ---------------------------------------------------------------------------
// AssertShape
// ---------------------------------------------------------------------------
TEST(TensorTest, TestTensorAssertShapePass) {
  Tensor t({2, 3, 4}, 0.0f);
  t.AssertShape({2, 3, 4}, "TestPass");
}

TEST(TensorTest, TestTensorAssertShapeFail) {
  Tensor t({2, 3, 4}, 0.0f);
  try {
    t.AssertShape({2, 3, 5}, "TestFail");
    FAIL() << "expected exception for shape mismatch";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ---------------------------------------------------------------------------
// ReshapeChecked
// ---------------------------------------------------------------------------
TEST(TensorTest, TestTensorReshapeCheckedPass) {
  Tensor t({2, 3}, 0.0f);
  for (int i = 0; i < 6; ++i) {
    t[i] = static_cast<float>(i);
  }
  Tensor r = t.ReshapeChecked({3, 2}, "TestReshape");
  EXPECT_EQ(r.num_dims(), 2);
  EXPECT_EQ(r.shape[0], 3);
  EXPECT_EQ(r.shape[1], 2);
  EXPECT_NEAR(r.At2(0, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(r.At2(2, 1), 5.0f, 1e-6f);
}

TEST(TensorTest, TestTensorReshapeCheckedFail) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.ReshapeChecked({4, 2}, "TestReshapeFail");
    FAIL() << "expected exception for incompatible reshape";
  } catch (const std::runtime_error&) {
    // expected
  }
}

TEST(TensorTest, TestTensorReshapeCheckedNegativeDim) {
  Tensor t({2, 3}, 0.0f);
  try {
    t.ReshapeChecked({-1, 6}, "TestReshapeNeg");
    FAIL() << "expected exception for negative dimension";
  } catch (const std::runtime_error&) {
    // expected
  }
}

// ---------------------------------------------------------------------------
// ShapeString alias
// ---------------------------------------------------------------------------
TEST(TensorTest, TestTensorShapeStringAlias) {
  Tensor t({2, 3, 4}, 0.0f);
  EXPECT_TRUE(t.ShapeStringShort() == t.ShapeString());
}

// ---------------------------------------------------------------------------
