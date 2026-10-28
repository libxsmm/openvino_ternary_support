// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_OCL
#    include "impls/ocl_v2/moe/moe_3gemm_swiglu_opt.hpp"
#    include "impls/ocl_v2/ternocl_int2/moe_3gemm_ternocl_int2.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<moe_3gemm_fused_compressed>::get_implementations() {
    // Ternary (u2) experts run on the TernOCL impl; moe_3gemm_swiglu_opt rejects u2.
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(cldnn::ocl::TernoclInt2MoEImplementationManager, shape_types::any)
            OV_GPU_CREATE_INSTANCE_OCL(ocl::moe_3gemm_swiglu_opt, shape_types::any)};

    return impls;
}

}  // namespace ov::intel_gpu
