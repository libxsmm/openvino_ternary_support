// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>

namespace ov {
namespace intel_gpu {
namespace xetla_int2 {

// Weights must already be in the packed VNNI16 form the kernel expects:
// B is [K/16, N] int32, one uint32 per 16 K-rows of a single N column, with the
// 2-bit code for row (kb * 16 + j) at bits [2j, 2j+1]. Codes are {0, 1, 3}
// meaning {0, +1, -1}; code 2 is forbidden. ScaleB is fp16 [K/kGroupSize, N].
constexpr size_t kGroupSize = 128;
constexpr size_t kPackFactor = 16;

// `slot` identifies the caller so each FullyConnected gets private k-slicing
// scratch; OpenVINO's queue is out-of-order and independent GEMMs overlap.
// `postop` selects the fused epilogue: 0 none, 1 silu*other, 2 +other.
sycl::event gemv_f16(sycl::queue& q, size_t M, size_t N, size_t K,
                     void* A, void* B, void* C, void* ScaleB, size_t slot = 0,
                     int postop = 0, void* other = nullptr, bool out_f32 = false);

// Experimental native ZE entry point used to validate direct command-list
// launch of the named up-convert kernel.
bool gemv_f16_ze_probe(void* list, void* context, void* device, size_t M, size_t N, size_t K,
                       void* A, void* B, void* C, void* ScaleB, size_t slot = 0,
                       int postop = 0, void* other = nullptr, bool out_f32 = false);

// Same operands, but quantizes the activations to int8 and uses the int2 x int8
// DPAS instead of up-converting the weights to fp16. Twice the compute peak, at
// the cost of an activation abs-max pass.
sycl::event gemv_f16_dpas(sycl::queue& q, size_t M, size_t N, size_t K,
                          void* A, void* B, void* C, void* ScaleB,
                          int postop = 0, void* other = nullptr);

enum class Variant { upcvt_fp16, dpas_i2xi8, dpas_prefill_only };

// XETLA_INT2_KERNEL=upcvt|dpas|dpas_prefill selects the variant. Both run on
// DPAS: up-convert dequantizes the weights to fp16 and uses fp16 DPAS, while
// dpas quantizes activations to int8 and uses integer DPAS. Up-convert is the
// default because it wins for the batch-1 decode GEMVs this impl targets.
// dpas_prefill uses the int2 x int8 variant only for M>1.
Variant variant_from_env();

}  // namespace xetla_int2
}  // namespace intel_gpu
}  // namespace ov
