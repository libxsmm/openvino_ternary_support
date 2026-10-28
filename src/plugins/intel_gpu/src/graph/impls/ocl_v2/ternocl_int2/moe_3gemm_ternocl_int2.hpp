// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "program_node.h"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace ocl {

// moe_3gemm_fused_compressed with ternary (u2, zero point 1, 128-groups) routed experts on TernOCL's
// MoE kernels: expert-indexed int2 GEMVs below kTernoclMoePrefillTokens tokens, and from there rows
// grouped by expert on the device with per-expert M-tiled int2 GEMMs (int2 x int8 DPAS with
// GPU_TERNOCL_INT2_INT8_PREFILL). The expert weights are packed at build time into
// format::b_y_xs_f_xsv16_ternary (see TernoclInt2WeightsReorderImplementationManager); the scales
// arrive as [E, K/128, N] (byfx, prepare_quantization).
struct TernoclInt2MoEImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("TernoclInt2MoEImplementationManager")
    explicit TernoclInt2MoEImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override;
};

}  // namespace ocl
}  // namespace cldnn
