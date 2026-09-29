/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <hip/hip_runtime.h>

namespace RcclUnitTesting
{

// Runtime driver-version gate mirroring ncclCeImplemented(). Keep the bounds
// in one place for the CE host eligibility suites.
inline bool isCeRuntimeDriverSupported()
{
    int driverVer = 0;
    if(hipDriverGetVersion(&driverVer) != hipSuccess)
        return false;
    return (driverVer >= 71200000) ||
           (driverVer >= 70051831 && driverVer < 70060000);
}

} // namespace RcclUnitTesting
