// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu::transcendental {
enum class F32Operation { Log, Exp, Sin, Cos, Rcp, Rsq, Sqrt };

/// Whether the host supports the exact eight-lane integer implementation.
bool supports_f32_simd();

/// Evaluate complete eight-lane chunks, preserving host floating-point state.
/// Input and output may be identical; otherwise they must not overlap.
/// Other counts use the unchanged scalar mapping. Rcp/Rsq retain unconditional
/// NaN quieting and denormal flushing, independent of denorm and quiet_snan.
void evaluate_f32_simd(F32Operation operation, const uint32_t *input, uint32_t *output,
                       std::size_t count, unsigned denorm, bool quiet_snan);
} // namespace rocjitsu::amdgpu::transcendental
