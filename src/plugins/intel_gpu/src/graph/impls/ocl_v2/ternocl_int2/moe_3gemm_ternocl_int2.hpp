// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace ocl {

// Fused MoE (gate / up / down experts) with ternary u2 weights, on TernOCL's
// expert-indexed int2 GEMVs (int2_moe.cl): one launch per projection covers every
// (token, expert) pair and the routing stays on the device.
struct TernoclInt2MoEImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("TernoclInt2MoEImplementationManager")
    TernoclInt2MoEImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override;
};

}  // namespace ocl
}  // namespace cldnn
