// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Fused sign flip + blockwise (1024) normalised Walsh-Hadamard transform:
//   y = H_1024 (s * x) / 32   per 1024-wide block along K of a [rows, K] f16 tensor.
// Input pre-transform of the Hadamard-folded (rotated basis) checkpoints such as
// Bonsai 2. One work-group of 128 items handles one block: the 10 radix-2 stages
// run as four passes over registers (three radix-8 gathers through SLM and a
// final radix-2 pass that stores), fp32 accumulation, fp16 only at load/store.

#include "xetla_int2_gemv.hpp"

#include <sycl/sycl.hpp>

namespace ov::intel_gpu::xetla_int2 {

namespace {

constexpr int kBlock = 1024;
constexpr int kWG = 128;
constexpr int kPerItem = kBlock / kWG;

inline void wht8(float* v) {
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
    }
}

}  // namespace

::sycl::event hadamard_fwht_1024(::sycl::queue& q, size_t rows, size_t K, const void* x_in,
                                 const void* signs_in, void* y_out) {
    const auto* x = static_cast<const ::sycl::half*>(x_in);
    const auto* signs = static_cast<const int8_t*>(signs_in);
    auto* y = static_cast<::sycl::half*>(y_out);
    const size_t nblocks = rows * (K / kBlock);
    const float scale = 1.0f / 32.0f;
    return q.submit([&](::sycl::handler& h) {
        ::sycl::local_accessor<float, 1> slm(::sycl::range<1>(kBlock), h);
        h.parallel_for(
            ::sycl::nd_range<1>(::sycl::range<1>(nblocks * kWG), ::sycl::range<1>(kWG)),
            [=](::sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const size_t blk = it.get_group(0);
                const int lid = static_cast<int>(it.get_local_id(0));
                const size_t blocks_per_row = K / kBlock;
                const size_t row = blk / blocks_per_row;
                const size_t col0 = (blk % blocks_per_row) * kBlock;
                const ::sycl::half* xb = x + row * K + col0;
                ::sycl::half* yb = y + row * K + col0;
                const int8_t* sb = signs ? signs + col0 : nullptr;

                float v[kPerItem];
                {
                    const int base = lid * kPerItem;
                    const auto xv = *reinterpret_cast<const ::sycl::vec<::sycl::half, 8>*>(xb + base);
                    if (sb) {
#pragma unroll
                        for (int j = 0; j < kPerItem; ++j)
                            v[j] = static_cast<float>(xv[j]) * static_cast<float>(sb[base + j]);
                    } else {
#pragma unroll
                        for (int j = 0; j < kPerItem; ++j)
                            v[j] = static_cast<float>(xv[j]);
                    }
                    wht8(v);
#pragma unroll
                    for (int j = 0; j < kPerItem; ++j)
                        slm[base + j] = v[j];
                }
                it.barrier(::sycl::access::fence_space::local_space);
                {
                    const int base = (lid & 7) | ((lid >> 3) << 6);
#pragma unroll
                    for (int j = 0; j < kPerItem; ++j)
                        v[j] = slm[base + (j << 3)];
                    wht8(v);
#pragma unroll
                    for (int j = 0; j < kPerItem; ++j)
                        slm[base + (j << 3)] = v[j];
                }
                it.barrier(::sycl::access::fence_space::local_space);
                {
                    const int base = (lid & 63) | ((lid >> 6) << 9);
#pragma unroll
                    for (int j = 0; j < kPerItem; ++j)
                        v[j] = slm[base + (j << 6)];
                    wht8(v);
#pragma unroll
                    for (int j = 0; j < kPerItem; ++j)
                        slm[base + (j << 6)] = v[j];
                }
                it.barrier(::sycl::access::fence_space::local_space);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int i = lid + j * kWG;
                    const float a = slm[i], b = slm[i + 512];
                    yb[i] = static_cast<::sycl::half>((a + b) * scale);
                    yb[i + 512] = static_cast<::sycl::half>((a - b) * scale);
                }
            });
    });
}

}  // namespace ov::intel_gpu::xetla_int2
