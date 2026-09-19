// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_xetla_int2.hpp"

#include "data_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/runtime/memory.hpp"
#ifdef OV_GPU_WITH_ZE_RT
#    include "ze/ze_engine.hpp"
#    include "ze/ze_common.hpp"
#    include "ze/ze_stream.hpp"
#else
#include "ocl/sycl_engine.hpp"
#include "ocl/sycl_stream.hpp"
#endif
#include "primitive_sycl_base.h"
#include "reorder_inst.h"
#include "registry/implementation_map.hpp"
#include "xetla/xetla_int2_gemv.hpp"

#include <sycl/sycl.hpp>
#ifdef OV_GPU_WITH_ZE_RT
#    include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cldnn {
namespace sycl {

namespace {

using ov::intel_gpu::xetla_int2::kGroupSize;
using ov::intel_gpu::xetla_int2::kPackFactor;

// OpenVINO stores u2 little-endian inside each byte: value j at bits [2j, 2j+1].
inline uint8_t read_u2(const uint8_t* base, size_t index) {
    return static_cast<uint8_t>((base[index >> 2] >> (2 * (index & 0x3))) & 0x3);
}

::sycl::queue xetla_queue(stream& stream) {
#ifdef OV_GPU_WITH_ZE_RT
    auto& ze_stream = downcast<ze::ze_stream>(stream);
    const auto cmd_list = ze_stream.get_queue();
    const auto& engine = ze_stream.get_engine();

    auto device = ::sycl::make_device<::sycl::backend::ext_oneapi_level_zero>(engine.get_device().handle());
    ::sycl::backend_input_t<::sycl::backend::ext_oneapi_level_zero, ::sycl::context> context_input{
        engine.get_context().handle(), {device}, ::sycl::ext::oneapi::level_zero::ownership::keep};
    auto context = ::sycl::make_context<::sycl::backend::ext_oneapi_level_zero>(context_input);
    ::sycl::backend_input_t<::sycl::backend::ext_oneapi_level_zero, ::sycl::queue> queue_input{
        cmd_list, device, ::sycl::ext::oneapi::level_zero::ownership::keep};
    return ::sycl::make_queue<::sycl::backend::ext_oneapi_level_zero>(queue_input, context);
#else
    return downcast<ocl::sycl_stream>(stream).get_sycl_queue();
#endif
}

// [N, K] u2 codes -> [K/16, N] int32, re-encoding (code - zp) to the kernel's
// two's-complement {0, +1, -1} = {0, 1, 3}.
// `n_stride` may exceed N: padding N up to a multiple of 4 keeps the kernel on
// its aligned epilogue, whose tail handling would otherwise write past N.
void pack_weights_vnni16(const uint8_t* src, int32_t* dst, size_t N, size_t n_stride, size_t K, int32_t zp) {
    const size_t k_blocks = K / kPackFactor;
    std::fill(dst, dst + k_blocks * n_stride, 0);
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            const int32_t code = static_cast<int32_t>(read_u2(src, n * K + k)) - zp;
            OPENVINO_ASSERT(code >= -1 && code <= 1,
                            "[GPU] xetla int2: weight code ", code, " is outside the ternary range; ",
                            "the kernel encodes only {-1, 0, +1}.");
            dst[(k / kPackFactor) * n_stride + n] |= (code & 0x3) << (2 * (k % kPackFactor));
        }
    }
}

// The zero point carries whatever precision the model used, so read it by type.
float read_scalar(const memory::ptr& mem, stream& s) {
    switch (mem->get_layout().data_type) {
    case data_types::f16: {
        mem_lock<uint16_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(ov::float16::from_bits(l.data()[0]));
    }
    case data_types::f32: {
        mem_lock<float, mem_lock_type::read> l{mem, s};
        return l.data()[0];
    }
    case data_types::u8: {
        mem_lock<uint8_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(l.data()[0]);
    }
    case data_types::i8: {
        mem_lock<int8_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(l.data()[0]);
    }
    case data_types::u2: {
        mem_lock<uint8_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(read_u2(l.data(), 0));
    }
    default:
        OPENVINO_THROW("[GPU] xetla int2: unsupported zero point type ", mem->get_layout().data_type);
    }
}

// Constant inputs may be fed through a reorder inserted by the graph optimizer.
// Only the data node behind it can be read at compile time, so walk to it.
const program_node* const_source(const program_node* node, bool* via_reorder = nullptr) {
    if (via_reorder != nullptr)
        *via_reorder = false;
    while (node != nullptr && !node->is_type<data>()) {
        if (!node->is_type<reorder>() || node->get_dependencies().size() != 1)
            return nullptr;
        if (via_reorder != nullptr)
            *via_reorder = true;
        node = &node->get_dependency(0);
    }
    return node;
}

}  // namespace

// OpenVINO caches primitive_impl objects by kernel_impl_params, so every
// FullyConnected with the same shape shares one impl. Stock impls are stateless
// and bind weights per execution, so the packed buffers have to be keyed by the
// weights allocation rather than stored on the impl.
struct XetlaInt2Packed {
    memory::ptr weights;
    memory::ptr scales;
    memory::ptr staging;  // padded f16 output, only when N is padded or C is f32
    memory::ptr staging_f32;  // decode-only f32 staging, when prefill and decode use different kernels
    size_t n_pad = 0;
    size_t staging_rows = 0;
    // Hadamard input pre-transform: +-1 signs [K] (i8, may be null) and the
    // rotated activation the GEMV reads instead of the node's input.
    memory::ptr had_signs;
    memory::ptr had_input;
    size_t had_rows = 0;
};

static std::mutex& xetla_int2_packed_mutex() {
    static std::mutex m;
    return m;
}

static std::unordered_map<std::string, XetlaInt2Packed>& xetla_int2_packed_cache() {
    // The cache owns ZE allocations, but the plugin can be unloaded after the
    // ZE context. Keep it process-lifetime to avoid freeing stale allocations
    // during C++ static destruction.
    static auto* c = new std::unordered_map<std::string, XetlaInt2Packed>;
    return *c;
}

struct fully_connected_xetla_int2 : typed_primitive_sycl_impl<fully_connected> {
    using parent = typed_primitive_sycl_impl<fully_connected>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl::fully_connected_xetla_int2)

    memory::ptr _packed_weights;
    memory::ptr _scales;
    size_t _N = 0;
    size_t _K = 0;
    size_t _n_pad = 0;
    memory::ptr _staging;
    // Identifies this node's private k-slicing scratch inside the kernel.
    size_t _slot = 0;
    memory::ptr _had_signs;
    memory::ptr _had_input;
    size_t _had_rows = 0;

    fully_connected_xetla_int2(const engine& engine, const ExecutionConfig& config,
                               memory::ptr packed_weights, memory::ptr scales, size_t N, size_t K,
                               size_t n_pad, memory::ptr staging,
                               memory::ptr had_signs = nullptr, memory::ptr had_input = nullptr)
        : parent(engine, config),
          _packed_weights(std::move(packed_weights)),
          _scales(std::move(scales)),
          _N(N),
          _K(K),
          _n_pad(n_pad),
          _staging(std::move(staging)),
          _had_signs(std::move(had_signs)),
          _had_input(std::move(had_input)),
          _had_rows(_had_input ? 1 : 0) {
        static std::atomic<size_t> next_slot{1};
        _slot = next_slot++;
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<fully_connected_xetla_int2>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<fully_connected>& instance) override {
        // getenv is not free and this runs once per FullyConnected per token.
        static const bool dbg = std::getenv("OV_XETLA_INT2_DEBUG") != nullptr;
        auto& network = instance.get_network();
        auto& stream = network.get_stream();

        const auto& params = instance.get_impl_params();
        const auto out_shape = params->output_layouts[0].get_shape();
        const size_t M = ov::shape_size(out_shape) / _N;

        if (dbg) {
            const auto& in_l = params->input_layouts[0];
            const auto& out_l = params->output_layouts[0];
            std::cerr << "[xetla-int2] exec N=" << _N << " n_pad=" << _n_pad << " K=" << _K << " M=" << M
                      << " slot=" << _slot
                      << " id=" << instance.id()
                      << " in " << in_l.get_shape().to_string() << " buf=" << instance.input_memory_ptr(0)->size()
                      << " (dense " << M * _K * 2 << ")"
                      << " pad=" << static_cast<bool>(in_l.data_padding)
                      << " | out " << out_l.get_shape().to_string() << " buf=" << instance.output_memory_ptr(0)->size()
                      << " (dense " << M * _N * 2 << ")"
                      << " pad=" << static_cast<bool>(out_l.data_padding) << std::endl;
        }

        // Producers that are OpenCL primitives are waited on via their events.
        for (auto& e : events) {
            if (e)
                e->wait();
        }

        void* out_ptr = instance.output_memory_ptr(0)->buffer_ptr();

        // This impl may be shared with other FullyConnected nodes of the same
        // shape, so resolve the packed buffers from this node's own weights.
        memory::ptr packed_weights = _packed_weights;
        memory::ptr scales = _scales;
        // Derived, not cached: a cache miss must never silently change the
        // stride the weights were packed with.
        size_t n_pad = _n_pad;
        memory::ptr staging = _staging;
        memory::ptr staging_f32;
        memory::ptr had_signs;
        memory::ptr had_input;
        const size_t had_block = params->typed_desc<fully_connected>()->hadamard_block;
        {
            std::lock_guard<std::mutex> lock(xetla_int2_packed_mutex());
            auto it = xetla_int2_packed_cache().find(params->desc->id);
            if (it != xetla_int2_packed_cache().end()) {
                packed_weights = it->second.weights;
                scales = it->second.scales;
                staging = it->second.staging;
                staging_f32 = it->second.staging_f32;
                n_pad = it->second.n_pad;
                if (staging && M > it->second.staging_rows) {
                    // Only a longer prefill can reach this; decode never reallocates.
                    auto st_layout = layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(n_pad)},
                                            staging->get_layout().data_type, format::bfyx};
                    staging = instance.get_network().get_engine().allocate_memory(st_layout,
                                                                                 allocation_type::usm_device, false);
                    it->second.staging = staging;
                    it->second.staging_rows = M;
                }
                if (had_block != 0) {
                    if (!it->second.had_input || M > it->second.had_rows) {
                        auto hl = layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(_K)},
                                         data_types::f16, format::bfyx};
                        it->second.had_input = instance.get_network().get_engine().allocate_memory(
                            hl, allocation_type::usm_device, false);
                        it->second.had_rows = M;
                    }
                    had_signs = it->second.had_signs;
                    had_input = it->second.had_input;
                }
            } else if (dbg) {
                std::cerr << "[xetla-int2] MISS desc=" << params->desc->id << " inst=" << instance.id() << std::endl;
            }
        }
        if (had_block != 0 && !had_input) {
            // Cache miss (the node was renamed after compile): fall back to the
            // impl's own buffers, growing the scratch for a longer prefill.
            if (!_had_input || M > _had_rows) {
                auto hl = layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(_K)},
                                 data_types::f16, format::bfyx};
                _had_input = instance.get_network().get_engine().allocate_memory(hl, allocation_type::usm_device, false);
                _had_rows = M;
            }
            had_signs = _had_signs;
            had_input = _had_input;
        }
        OPENVINO_ASSERT(had_block == 0 || had_input, "[GPU] xetla int2: hadamard scratch missing for ", instance.id());
        const bool out_f32 = params->output_layouts[0].data_type == data_types::f32;
        // The fused epilogue is only implemented for the up-convert path, and the
        // int2 x int8 path always writes f16, so it keeps the separate post-op pass.
        // When only prefill uses it, decode still gets its own f32 staging buffer
        // and keeps the native f32 output.
        const auto variant = ov::intel_gpu::xetla_int2::variant_from_env();
        const bool use_dpas = variant == ov::intel_gpu::xetla_int2::Variant::dpas_i2xi8 ||
                              (variant == ov::intel_gpu::xetla_int2::Variant::dpas_prefill_only && M > 1);
        if (!use_dpas && out_f32 && staging_f32)
            staging = staging_f32;
        // Derived from the buffer actually in use, so the two can never disagree.
        const bool native_f32 =
            out_f32 && !use_dpas && staging && staging->get_layout().data_type == data_types::f32;
        void* gemm_out = staging ? staging->buffer_ptr() : out_ptr;
        // Post-ops are replayed in graph order: SiLU then the binary eltwise.
        bool fuse_silu = false;
        bool fuse_sigmoid = false;
        const ::sycl::half* elt_other = nullptr;
        bool elt_is_prod = false;
        for (const auto& f : params->fused_desc) {
            if (const auto a = std::dynamic_pointer_cast<const activation>(f.desc)) {
                if (a->activation_function == activation_func::logistic)
                    fuse_sigmoid = true;
                else
                    fuse_silu = true;
            } else if (const auto e = std::dynamic_pointer_cast<const eltwise>(f.desc)) {
                elt_other =
                    static_cast<const ::sycl::half*>(instance.dep_memory_ptr(f.outer_dep_start_idx)->buffer_ptr());
                elt_is_prod = e->mode == eltwise_mode::prod;
            }
        }

        // Fold into the XeTLA epilogue when the chain matches a compiled variant;
        // anything else falls back to the separate pass below. The int2 x int8
        // kernel supports the fused epilogue too, but its large tiles make it
        // measurably slower than the separate pass, so it is opt-in there.
        static const bool dpas_fold = std::getenv("OV_XETLA_INT2_DPAS_FOLD") != nullptr;
        const bool has_bias = instance.bias_term();
        int postop = 0;
        if (elt_other != nullptr && (!use_dpas || dpas_fold)) {
            if (fuse_silu && elt_is_prod)
                postop = 1;
            else if (!fuse_silu && !elt_is_prod)
                postop = 2;
        } else if (elt_other == nullptr && !use_dpas) {
            // GatedDeltaNet gates: in_proj_b is a bare sigmoid, in_proj_a a bare bias.
            const int fold_gates = xetla_int2_fold_gates();
            if ((fold_gates & 2) && fuse_sigmoid && !fuse_silu)
                postop = 4;
            else if ((fold_gates & 1) && !fuse_sigmoid && !fuse_silu && has_bias)
                postop = 3;
        }
        void* postop_bias = postop == 3 ? instance.bias_memory()->buffer_ptr() : nullptr;
        void* postop_other = postop != 0 ? const_cast<::sycl::half*>(elt_other) : nullptr;
        if (postop != 0) {
            fuse_silu = false;
            fuse_sigmoid = false;
            elt_other = nullptr;
        }

#ifdef OV_GPU_WITH_ZE_RT
    if (had_block == 0 && (std::getenv("OV_XETLA_INT2_ZE_DIRECT") != nullptr ||
        std::getenv("OV_XETLA_INT2_ZE_DIRECT_QKV") != nullptr)) {
            auto& ze_stream = downcast<ze::ze_stream>(stream);
            // A shape or post-op the direct table does not cover falls through
            // to the SYCL submission below rather than failing the inference.
            const bool ok = ov::intel_gpu::xetla_int2::gemv_f16_ze_probe(
                ze_stream.get_queue(), ze_stream.get_engine().get_context().handle(),
                ze_stream.get_engine().get_device().handle(), M, n_pad, _K,
                instance.input_memory_ptr(0)->buffer_ptr(), packed_weights->buffer_ptr(),
                gemm_out, scales->buffer_ptr(), _slot, postop, postop_other, native_f32,
                postop_bias, _N);
            if (ok) {
                if (staging) {
                    const size_t element_size = native_f32 ? sizeof(float) : sizeof(::sycl::half);
                    const size_t row_bytes = _N * element_size;
                    const size_t padded_row_bytes = n_pad * element_size;
                    auto* src = static_cast<const char*>(gemm_out);
                    auto* dst = static_cast<char*>(out_ptr);
                    for (size_t row = 0; row < M; ++row) {
                        OV_ZE_EXPECT(ze::zeCommandListAppendMemoryCopy(
                            ze_stream.get_queue(), dst + row * row_bytes, src + row * padded_row_bytes,
                            row_bytes, nullptr, 0, nullptr));
                    }
                }
                return nullptr;
            }
        }
#endif

        auto sycl_queue = xetla_queue(stream);
        // Ordering against other xetla FCs comes from the in-order queue. Kept as
        // an escape hatch: OV_XETLA_INT2_BARRIER=1 forces an explicit barrier.
        static const bool force_barrier = std::getenv("OV_XETLA_INT2_BARRIER") != nullptr;
        if (force_barrier)
            sycl_queue.submit([=](::sycl::handler& cgh) { cgh.ext_oneapi_barrier(); });

        void* gemm_in = instance.input_memory_ptr(0)->buffer_ptr();
        if (had_block != 0) {
            // Rotated-basis checkpoint: the GEMV consumes H(s*x) rather than x.
            ov::intel_gpu::xetla_int2::hadamard_fwht_1024(
                sycl_queue, M, _K, gemm_in, had_signs ? had_signs->buffer_ptr() : nullptr,
                had_input->buffer_ptr());
            gemm_in = had_input->buffer_ptr();
        }

        auto ev = use_dpas
                      ? ov::intel_gpu::xetla_int2::gemv_f16_dpas(sycl_queue, M, n_pad, _K,
                                                                gemm_in,
                                                                packed_weights->buffer_ptr(),
                                                                gemm_out,
                                                                scales->buffer_ptr(),
                                                                postop, postop_other)
                      : ov::intel_gpu::xetla_int2::gemv_f16(sycl_queue, M, n_pad, _K,
                                                           gemm_in,
                                                           packed_weights->buffer_ptr(),
                                                           gemm_out,
                                                           scales->buffer_ptr(),
                                                           _slot, postop, postop_other, native_f32,
                                                           postop_bias, _N);

        if (staging) {
            const size_t n = _N;
            const size_t np = n_pad;
            const auto* src = static_cast<const ::sycl::half*>(gemm_out);
            if (native_f32) {
                // The epilogue already emitted f32; this only un-pads the row stride.
                const auto* fsrc = static_cast<const float*>(gemm_out);
                auto* dst = static_cast<float*>(out_ptr);
                ev = sycl_queue.parallel_for(::sycl::range<1>(M * n), ev, [=](::sycl::id<1> i) {
                    const size_t idx = i[0];
                    dst[idx] = fsrc[(idx / n) * np + (idx % n)];
                });
            } else if (out_f32) {
                auto* dst = static_cast<float*>(out_ptr);
                ev = sycl_queue.parallel_for(::sycl::range<1>(M * n), ev, [=](::sycl::id<1> i) {
                    const size_t idx = i[0];
                    dst[idx] = static_cast<float>(src[(idx / n) * np + (idx % n)]);
                });
            } else {
                auto* dst = static_cast<::sycl::half*>(out_ptr);
                ev = sycl_queue.parallel_for(::sycl::range<1>(M * n), ev, [=](::sycl::id<1> i) {
                    const size_t idx = i[0];
                    float v = static_cast<float>(src[(idx / n) * np + (idx % n)]);
                    if (fuse_silu)
                        v = v / (1.0f + ::sycl::exp(-v));
                    if (elt_other) {
                        const float o = static_cast<float>(elt_other[idx]);
                        v = elt_is_prod ? v * o : v + o;
                    }
                    dst[idx] = static_cast<::sycl::half>(v);
                });
            }
        } else if (fuse_silu || fuse_sigmoid || elt_other) {
            auto* dst = static_cast<::sycl::half*>(out_ptr);
            ev = sycl_queue.parallel_for(::sycl::range<1>(M * _N), ev, [=](::sycl::id<1> i) {
                const size_t idx = i[0];
                float v = static_cast<float>(dst[idx]);
                if (fuse_silu)
                    v = v / (1.0f + ::sycl::exp(-v));
                if (fuse_sigmoid)
                    v = 1.0f / (1.0f + ::sycl::exp(-v));
                if (elt_other) {
                    const float o = static_cast<float>(elt_other[idx]);
                    v = elt_is_prod ? v * o : v + o;
                }
                dst[idx] = static_cast<::sycl::half>(v);
            });
        }
        return to_ocl_event(stream, ev);
    }

    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& prog = arg.get_program();
        auto& engine = prog.get_engine();
        auto& stream = prog.get_stream();

        if (std::getenv("OV_XETLA_INT2_DEBUG") != nullptr) {
            for (size_t i = 0; i < arg.get_dependencies().size(); ++i) {
                const auto& dep = arg.get_dependency(i);
                std::cerr << "[xetla-int2] dep[" << i << "] id=" << dep.id()
                          << " type=" << dep.get_primitive()->type_string()
                          << " is_data=" << dep.is_type<data>()
                          << " dt=" << dep.get_output_layout(false).data_type
                          << " shape=" << dep.get_output_layout(false).get_partial_shape().to_string() << std::endl;
            }
        }

        const auto wei_layout = arg.weights().get_output_layout(false);
        const auto wei_shape = wei_layout.get_shape();
        const size_t N = wei_shape[0];
        const size_t K = wei_shape[1];

        const auto& desc = arg.get_primitive();
        // Dependencies are input, weights, [bias], [scale], [zero point], so a
        // bias shifts the decompression operands one slot along.
        const size_t scale_dep_idx = desc->bias.is_valid() ? 3 : 2;
        const size_t zp_dep_idx = scale_dep_idx + 1;
        // OpenVINO has no i2, so ternary arrives as u2 codes offset by a zero point.
        int32_t zp = 0;
        if (desc->decompression_zero_point_scalar.has_value()) {
            zp = static_cast<int32_t>(std::lround(desc->decompression_zero_point_scalar.value()));
        } else if (desc->decompression_zero_point.is_valid()) {
            const auto* zp_node = const_source(&arg.get_dependency(zp_dep_idx));
            OPENVINO_ASSERT(zp_node != nullptr, "[GPU] xetla int2: zero point is not constant");
            zp = static_cast<int32_t>(std::lround(read_scalar(zp_node->as<data>().get_attached_memory_ptr(), stream)));
        }

        auto wei_mem = arg.weights().as<data>().get_attached_memory_ptr();
        // mem_lock on a large device constant can expose the host pointer before
        // the data is resident; copy explicitly so packing reads stable bytes.
        std::vector<uint8_t> wei_host(wei_mem->size());
        wei_mem->copy_to(stream, wei_host.data(), true);
        // The packer indexes the weights        // The packer indexes the weights as dense row-major [N, K]; a padded or
        // otherwise strided buffer would silently be read wrong.
        const size_t wei_dense_bytes = (N * K) / 4;
        if (std::getenv("OV_XETLA_INT2_DEBUG") != nullptr) {
            std::cerr << "[xetla-int2] N=" << N << " K=" << K << " weights " << wei_mem->size()
                      << " B (dense " << wei_dense_bytes << " B)" << std::endl;
        }
        OPENVINO_ASSERT(wei_mem->size() >= wei_dense_bytes,
                        "[GPU] xetla int2: weight buffer ", wei_mem->size(),
                        " B is smaller than the dense ", wei_dense_bytes, " B this impl assumes");

        const size_t k_blocks = K / kPackFactor;
        // Padding N here, once, keeps the kernel on the aligned epilogue instead of
        // its tail path, which writes past N.
        // Pad to the workgroup tile width (WGN=32, SGN=16), not just to 4: a
        // partial tile puts the kernel on its tail path.
        const size_t n_pad = (N + 63) & ~static_cast<size_t>(63);
        auto packed_layout = layout{ov::PartialShape{static_cast<int64_t>(k_blocks), static_cast<int64_t>(n_pad)},
                                    data_types::i32, format::bfyx};
        auto packed_weights = engine.allocate_memory(packed_layout, allocation_type::usm_device, false);
        std::vector<int32_t> packed_host(k_blocks * n_pad);
        pack_weights_vnni16(wei_host.data(), packed_host.data(), N, n_pad, K, zp);
        packed_weights->copy_from(stream, packed_host.data(), true);

        // The kernel wants scales as [K/group_size, N]; OpenVINO stores them per
        // output channel, i.e. [N, K/group_size].
        const size_t groups = K / kGroupSize;
        bool scales_via_reorder = false;
        const auto* scale_node = const_source(&arg.get_dependency(scale_dep_idx), &scales_via_reorder);
        OPENVINO_ASSERT(scale_node != nullptr, "[GPU] xetla int2: decompression scale is not constant");
        auto scale_mem = scale_node->as<data>().get_attached_memory_ptr();
        OPENVINO_ASSERT(scale_mem->get_layout().data_type == data_types::f16,
                        "[GPU] xetla int2: decompression scale must be f16");
        // Same residency hazard as the weights: copy instead of mapping.
        std::vector<uint16_t> scale_src(scale_mem->size() / sizeof(uint16_t));
        scale_mem->copy_to(stream, scale_src.data(), true);
        const auto scale_shape = scale_mem->get_layout().get_shape();
        if (std::getenv("OV_XETLA_INT2_DEBUG") != nullptr) {
            const auto sl = scale_mem->get_layout();
            std::cerr << "[xetla-int2] SCALES N=" << N << " groups=" << K / kGroupSize
                      << " shape=" << scale_shape.to_string() << " fmt=" << sl.format.to_string()
                      << " bytes=" << scale_mem->size() << " dense=" << (N * (K / kGroupSize) * 2)
                      << " via_reorder=" << scales_via_reorder << std::endl;
        }
        // A reorder on the scales already materialises them as [groups, N];
        // the reported shape still reads [N, groups], so it cannot be used here.
        const bool scales_are_n_major = scale_shape.size() >= 2 && scale_shape[0] == N;

        auto scales_layout = layout{ov::PartialShape{static_cast<int64_t>(groups), static_cast<int64_t>(n_pad)},
                                    data_types::f16, format::bfyx};
        auto scales = engine.allocate_memory(scales_layout, allocation_type::usm_device, false);
        std::vector<uint16_t> scale_host(groups * n_pad, 0);
        {
            const auto* src = scale_src.data();
            for (size_t g = 0; g < groups; ++g)
                for (size_t n = 0; n < N; ++n)
                    scale_host[g * n_pad + n] = scales_are_n_major ? src[n * groups + g] : src[g * N + n];
        }
        scales->copy_from(stream, scale_host.data(), true);
        {
            // Allocated here, at compile time, so inference never allocates.
            const bool out_f32 = impl_params.output_layouts[0].data_type == data_types::f32;
            const auto variant = ov::intel_gpu::xetla_int2::variant_from_env();
            const bool any_dpas = variant != ov::intel_gpu::xetla_int2::Variant::upcvt_fp16;
            // The shared staging buffer must hold whatever the kernel that runs at
            // this M writes: f16 for any DPAS mode, f32 otherwise.
            const bool native_f32 = out_f32 && !any_dpas;
            memory::ptr staging;
            if (n_pad != N || out_f32) {
                auto st_layout = layout{ov::PartialShape{1, static_cast<int64_t>(n_pad)},
                                        native_f32 ? data_types::f32 : data_types::f16, format::bfyx};
                staging = engine.allocate_memory(st_layout, allocation_type::usm_device, false);
            }
            // With DPAS on prefill only, decode still runs the up-convert kernel and
            // keeps its native f32 output; decode is always a single row.
            memory::ptr staging_f32;
            if (out_f32 && variant == ov::intel_gpu::xetla_int2::Variant::dpas_prefill_only) {
                auto st_layout = layout{ov::PartialShape{1, static_cast<int64_t>(n_pad)},
                                        data_types::f32, format::bfyx};
                staging_f32 = engine.allocate_memory(st_layout, allocation_type::usm_device, false);
            }
            std::lock_guard<std::mutex> lock(xetla_int2_packed_mutex());
            if (std::getenv("OV_XETLA_INT2_DEBUG") != nullptr)
                std::cerr << "[xetla-int2] REGISTER key=" << arg.id() << std::endl;
            XetlaInt2Packed entry{packed_weights, scales, staging, staging_f32, n_pad, 1};
            if (desc->hadamard_block != 0) {
                OPENVINO_ASSERT(desc->hadamard_block == 1024 && K % 1024 == 0,
                                "[GPU] xetla int2: hadamard block must be 1024 and divide K");
                OPENVINO_ASSERT(desc->hadamard_signs.empty() || desc->hadamard_signs.size() == K,
                                "[GPU] xetla int2: hadamard signs length mismatch");
                if (!desc->hadamard_signs.empty()) {
                    auto sl = layout{ov::PartialShape{static_cast<int64_t>(K)}, data_types::i8, format::bfyx};
                    entry.had_signs = engine.allocate_memory(sl, allocation_type::usm_device, false);
                    entry.had_signs->copy_from(stream, desc->hadamard_signs.data(), true);
                }
                auto hl = layout{ov::PartialShape{1, static_cast<int64_t>(K)}, data_types::f16, format::bfyx};
                entry.had_input = engine.allocate_memory(hl, allocation_type::usm_device, false);
                entry.had_rows = 1;
            }
            // A second impl can be created for a node that is already executing
            // (new shape bucket). Replacing the entry would swap the buffers out
            // from under the live impl, which resolves them by node id per call.
            auto res = xetla_int2_packed_cache().try_emplace(arg.id(), std::move(entry));
            if (!res.second) {
                packed_weights = res.first->second.weights;
                scales = res.first->second.scales;
                if (std::getenv("OV_XETLA_INT2_DEBUG") != nullptr)
                    std::cerr << "[xetla-int2] REUSE key=" << arg.id() << std::endl;
            }
        }
        static int verify_budget = 3;
        if (std::getenv("OV_XETLA_INT2_VERIFY") != nullptr && verify_budget-- > 0) {
            // Round-trip the packed buffers back to dequantized weights and compare
            // against the IR constant, to separate packer bugs from kernel bugs.
            size_t mismatches = 0;
            double max_diff = 0.0;
            const size_t n_lim = std::min<size_t>(N, 4096);
            for (size_t n = 0; n < n_lim; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    const int raw = static_cast<int>(read_u2(wei_host.data(), n * K + k));
                    const float ref_ternary = static_cast<float>(raw - zp);
                    const float ref_scale = static_cast<float>(ov::float16::from_bits(scale_src.data()[
                        scales_are_n_major ? n * groups + (k / kGroupSize) : (k / kGroupSize) * N + n]));

                    const uint32_t word = static_cast<uint32_t>(packed_host[(k / kPackFactor) * n_pad + n]);
                    const uint32_t code = (word >> (2 * (k % kPackFactor))) & 0x3u;
                    const float got_ternary = code == 0 ? 0.0f : (code == 1 ? 1.0f : -1.0f);
                    const float got_scale = static_cast<float>(
                        ov::float16::from_bits(scale_host[(k / kGroupSize) * n_pad + n]));

                    const double d = std::abs(static_cast<double>(ref_ternary) * ref_scale -
                                              static_cast<double>(got_ternary) * got_scale);
                    if (d > 1e-6) {
                        if (mismatches < 8)
                            std::cerr << "[xetla-int2] VERIFY n=" << n << " k=" << k << " raw=" << raw
                                      << " zp=" << zp << " ref=" << ref_ternary * ref_scale
                                      << " got=" << got_ternary * got_scale << " code=" << code << std::endl;
                        ++mismatches;
                        max_diff = std::max(max_diff, d);
                    }
                }
            }
            std::cerr << "[xetla-int2] VERIFY N=" << N << " K=" << K << " zp=" << zp
                      << " scale_shape=" << scale_shape.to_string()
                      << " n_major=" << scales_are_n_major
                      << " mismatches=" << mismatches << "/" << n_lim * K
                      << " max_diff=" << max_diff << std::endl;
        }

        memory::ptr own_staging, own_had_signs, own_had_input;
        {
            std::lock_guard<std::mutex> lock(xetla_int2_packed_mutex());
            auto& e = xetla_int2_packed_cache()[arg.id()];
            own_staging = e.staging;
            own_had_signs = e.had_signs;
            own_had_input = e.had_input;
        }
        return std::make_unique<fully_connected_xetla_int2>(engine, prog.get_config(),
                                                           packed_weights, scales, N, K,
                                                           n_pad, own_staging, own_had_signs, own_had_input);
    }
};

std::unique_ptr<primitive_impl> XetlaInt2FCImplementationManager::create_impl(const program_node& node,
                                                                             const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return fully_connected_xetla_int2::create(static_cast<const fully_connected_node&>(node), params);
}

}  // namespace sycl
}  // namespace cldnn
