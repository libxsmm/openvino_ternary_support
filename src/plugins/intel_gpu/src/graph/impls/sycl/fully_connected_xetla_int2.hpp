// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "data_inst.h"
#include "activation_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "registry/implementation_manager.hpp"
#include "xetla/xetla_int2_gemv.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace cldnn {
namespace sycl {

// OV_XETLA_INT2_FOLD_GATES=bias|sigmoid|none|1 selects which GatedDeltaNet gate
// epilogues are folded; bit 0 is the bias, bit 1 the sigmoid.
inline int xetla_int2_fold_gates() {
    static const int v = [] {
        const char* e = std::getenv("OV_XETLA_INT2_FOLD_GATES");
        if (e == nullptr)
            return 3;
        const std::string s(e);
        if (s == "none" || s == "0")
            return 0;
        if (s == "bias")
            return 1;
        if (s == "sigmoid")
            return 2;
        return 3;
    }();
    return v;
}

// Weight-only-quantized FullyConnected on 2-bit weights, executed by the XeTLA
// int2 GEMV. OpenVINO has no signed 2-bit type, so ternary weights arrive as u2
// codes {0,1,2} plus a zero point of 1; the codes are re-encoded to the
// kernel's {0,1,3} and packed to VNNI16 once, when the impl is created.
struct XetlaInt2FCImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("XetlaInt2FCImplementationManager")
    XetlaInt2FCImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::sycl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());
        const auto& fc_node = node.as<fully_connected>();
        const auto& fc_prim = fc_node.get_primitive();
        const bool dbg = std::getenv("OV_XETLA_INT2_DEBUG") != nullptr;
        if (std::getenv("OV_XETLA_INT2_DISABLE") != nullptr)
            return false;
#define XETLA_REJECT(reason)                                                     \
    do {                                                                         \
        if (dbg)                                                                 \
            std::cerr << "[xetla-int2] reject: " << (reason) << " node=" << fc_node.id() << std::endl;         \
        /* a fused Hadamard input exists only here; another impl would ignore it */ \
        OPENVINO_ASSERT(fc_prim->hadamard_block == 0,                            \
                        "[GPU] xetla int2: ", fc_node.id(), " carries a Hadamard input transform but ", \
                        "the XeTLA impl rejected it: ", (reason));               \
        return false;                                                            \
    } while (0)

        if (!fc_prim->compressed_weights)
            XETLA_REJECT("not compressed_weights");
        const int fold_gates = xetla_int2_fold_gates();
        // A constant bias folds into the epilogue, but only on the up-convert
        // path and only when nothing else is fused after it.
        if (fc_prim->bias.is_valid()) {
            if ((fold_gates & 1) == 0)
                XETLA_REJECT("bias");
            if (!fc_node.bias().is_type<data>())
                XETLA_REJECT("bias is not a constant");
            if (!fc_node.get_fused_primitives().empty())
                XETLA_REJECT("bias combined with fused post-ops");
            if (ov::intel_gpu::xetla_int2::variant_from_env() !=
                ov::intel_gpu::xetla_int2::Variant::upcvt_fp16)
                XETLA_REJECT("bias needs the up-convert epilogue");
        }
        // Only a trailing SiLU or sigmoid is folded; anything else still goes
        // to the OCL path.
        for (const auto& f : fc_node.get_fused_primitives()) {
            const auto act = std::dynamic_pointer_cast<const activation>(f.desc);
            const auto elt = std::dynamic_pointer_cast<const eltwise>(f.desc);
            const bool ok_act = act && f.total_num_deps == 1 &&
                                (act->activation_function == activation_func::swish ||
                                 ((fold_gates & 2) &&
                                  act->activation_function == activation_func::logistic));
            const bool ok_elt = elt && f.total_num_deps == 2 &&
                                (elt->mode == eltwise_mode::sum || elt->mode == eltwise_mode::prod) &&
                                f.has_outer_dep() &&
                                static_cast<size_t>(f.outer_dep_start_idx) < fc_node.get_dependencies().size() &&
                                fc_node.get_dependency(f.outer_dep_start_idx).get_output_layout(false).data_type ==
                                    data_types::f16;
            if (!ok_act && !ok_elt) {
                if (std::getenv("OV_XETLA_INT2_DEBUG") != nullptr) {
                    std::cerr << "[xetla-int2] postop bias=" << fc_prim->bias.is_valid();
                    for (const auto& g : fc_node.get_fused_primitives()) {
                        std::cerr << " [" << g.desc->type_string() << " deps=" << g.total_num_deps;
                        if (const auto a = std::dynamic_pointer_cast<const activation>(g.desc))
                            std::cerr << " func=" << static_cast<int>(a->activation_function);
                        std::cerr << "]";
                    }
                    std::cerr << " node=" << fc_node.id() << std::endl;
                }
                XETLA_REJECT("unsupported fused primitive");
            }
        }

        const auto& in_layout = fc_node.get_input_layout(0);
        const auto& out_layout = fc_node.get_output_layout(0);
        // lm_head emits f32 logits; the GEMM stays f16 and the result is converted.
        if (in_layout.data_type != data_types::f16)
            XETLA_REJECT("activations are not f16");
        if (out_layout.data_type != data_types::f16 && out_layout.data_type != data_types::f32)
            XETLA_REJECT("output is not f16 or f32");
        if (fc_node.weights().get_output_layout(false).data_type != data_types::u2)
            XETLA_REJECT("weights are not u2");
        if (in_layout.format != format::bfyx || out_layout.format != format::bfyx)
            XETLA_REJECT("unsupported activation format");
        if (in_layout.data_padding || out_layout.data_padding)
            XETLA_REJECT("padded activations");

        // Packing happens at compile time, so the weights have to be constant.
        if (!fc_node.weights().is_type<data>())
            XETLA_REJECT("weights are not a constant");

        // Weight layouts are canonicalized to 4D, so [N, K] arrives as [N, K, 1, 1].
        const auto& wei_pshape = fc_node.weights().get_output_layout(false).get_partial_shape();
        if (wei_pshape.is_dynamic() || wei_pshape.size() < 2)
            XETLA_REJECT("weights shape is not a static [N, K]: " + wei_pshape.to_string());
        for (size_t i = 2; i < wei_pshape.size(); ++i) {
            if (wei_pshape[i].get_length() != 1)
                XETLA_REJECT("weights shape is not a static [N, K]: " + wei_pshape.to_string());
        }
        const auto K = wei_pshape[1].get_length();
        if (K % static_cast<int64_t>(ov::intel_gpu::xetla_int2::kGroupSize) != 0)
            XETLA_REJECT("K is not a multiple of the group size");

        // Debug aid: OV_XETLA_INT2_ONLY_N=6144,24576 restricts the impl to the
        // listed output widths so a misbehaving layer can be bisected.
        if (const char* only = std::getenv("OV_XETLA_INT2_ONLY_N")) {
            const std::string want = "," + std::string(only) + ",";
            if (want.find("," + std::to_string(wei_pshape[0].get_length()) + ",") == std::string::npos)
                XETLA_REJECT("N not in OV_XETLA_INT2_ONLY_N");
        }

        if (dbg)
            std::cerr << "[xetla-int2] accepted " << fc_node.id() << std::endl;
        return true;
#undef XETLA_REJECT
    }
};

}  // namespace sycl
}  // namespace cldnn
