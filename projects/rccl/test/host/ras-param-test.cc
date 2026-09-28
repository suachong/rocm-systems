/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/ras_param.cc.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>

#define ncclParamRasTimeoutFactor RasParamTestNcclParamRasTimeoutFactor
#define rasTimeoutFactorNs RasParamTestRasTimeoutFactorNs
#define rasTimeoutFactorSec RasParamTestRasTimeoutFactorSec

#define NCCL_RAS_CLIENT
#include RAS_PARAM_CC_PATH

namespace {

class RasParamMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { unsetenv("NCCL_RAS_TIMEOUT_FACTOR"); }
  void TearDown() override { unsetenv("NCCL_RAS_TIMEOUT_FACTOR"); }
};

TEST_F(RasParamMicrotest, LoadTimeoutFactorMissingOrEmptyUsesDefault) {
  EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor());

  ASSERT_EQ(0, setenv("NCCL_RAS_TIMEOUT_FACTOR", "", 1));
  EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor());
}

TEST_F(RasParamMicrotest, LoadTimeoutFactorAcceptsPositiveFiniteValues) {
  for (const char* value : {"0.25", "1", "2.5", "1e3"}) {
    ASSERT_EQ(0, setenv("NCCL_RAS_TIMEOUT_FACTOR", value, 1));
    EXPECT_FLOAT_EQ(std::strtof(value, nullptr), rasLoadTimeoutFactor()) << value;
  }
}

TEST_F(RasParamMicrotest, LoadTimeoutFactorRejectsInvalidValues) {
  for (const char* value : {"abc", "1x", "0", "-1", "nan", "inf", "1e9999", "1e-9999"}) {
    ASSERT_EQ(0, setenv("NCCL_RAS_TIMEOUT_FACTOR", value, 1));
    EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor()) << value;
  }
}

TEST_F(RasParamMicrotest, LoadTimeoutFactorRejectsTrailingWhitespace) {
  ASSERT_EQ(0, setenv("NCCL_RAS_TIMEOUT_FACTOR", "2.5 ", 1));
  EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor());
}

TEST_F(RasParamMicrotest, PublicAccessorsScaleAndCacheTheFactor) {
  ASSERT_EQ(0, setenv("NCCL_RAS_TIMEOUT_FACTOR", "2.5", 1));
  EXPECT_FLOAT_EQ(2.5f, ncclParamRasTimeoutFactor());
  EXPECT_EQ(7500000000LL, rasTimeoutFactorNs(3));
  EXPECT_DOUBLE_EQ(10.0, rasTimeoutFactorSec(4));

  ASSERT_EQ(0, setenv("NCCL_RAS_TIMEOUT_FACTOR", "9", 1));
  EXPECT_FLOAT_EQ(2.5f, ncclParamRasTimeoutFactor());
}

}  // namespace
