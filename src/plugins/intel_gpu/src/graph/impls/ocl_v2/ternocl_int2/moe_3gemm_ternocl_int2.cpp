// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_ternocl_int2.hpp"

#ifdef OV_GPU_WITH_OCL_RT

#    include <algorithm>
#    include <array>
#    include <atomic>
#    include <cmath>
#    include <cstdint>
#    include <cstring>
#    include <map>
#    include <mutex>
#    include <set>
#    include <sstream>
#    include <string>
#    include <unordered_map>
#    include <vector>

#    include "data_inst.h"
#    include "fully_connected_ternocl_int2.hpp"
#    include "intel_gpu/runtime/debug_configuration.hpp"
#    include "intel_gpu/runtime/kernel_args.hpp"
#    include "intel_gpu/runtime/memory.hpp"
#    include "moe_3gemm_fused_inst.h"
#    include "openvino/core/parallel.hpp"
#    include "primitive_inst.h"
#    include "reorder_inst.h"
#    include "runtime/ocl/ocl_engine.hpp"
#    include "runtime/ocl/ocl_kernel.hpp"
#    include "ternocl_int2_kernels.inc"  // generated from TERNOCL_ROOT (CMakeLists.txt)

namespace cldnn {
namespace ocl {

namespace {

using MoEIdx = moe_3gemm_fused_compressed::input_index;

constexpr size_t idx(MoEIdx i) {
    return static_cast<size_t>(i);
}

constexpr std::array<MoEIdx, 3> kWeights = {MoEIdx::weight_0, MoEIdx::weight_1, MoEIdx::weight_2};
constexpr std::array<MoEIdx, 3> kScales = {MoEIdx::scale_0, MoEIdx::scale_1, MoEIdx::scale_2};
constexpr std::array<MoEIdx, 3> kZps = {MoEIdx::zp_0, MoEIdx::zp_1, MoEIdx::zp_2};

// Decode tiles (TernOCL int2_moe sweep.sh, Arc Pro B70, Qwen3-30B-A3B experts at T = 1).
struct MoETile {
    int nsg, ls, u;
};
constexpr MoETile kGateUpTile{2, 4, 1};
constexpr MoETile kDownTile{1, 1, 1};

// From this many tokens the rows are grouped by expert and run as M-tiled GEMMs, which
// read an expert's weights once per row tile rather than once per token.
constexpr size_t kPrefillMinTokens = 16;

// Prefill tiles (TernOCL int2_moe sweep_prefill.sh, Arc Pro B70): mt_m, mt_n, wg_m, wg_n.
// Rows per tile (mt_m * wg_m) are shared by the three GEMMs of one launch set.
struct MtTile {
    int mt_m, mt_n, wg_m, wg_n;
};
constexpr std::array<MtTile, 3> kPrefillTiles = {MtTile{16, 32, 1, 8}, MtTile{32, 32, 1, 4}, MtTile{32, 32, 1, 8}};

size_t prefill_band(size_t T) {
    return T <= 64 ? 0 : (T <= 512 ? 1 : 2);
}

const program_node* const_source(const program_node* node) {
    while (node != nullptr && !node->is_type<data>()) {
        if (!node->is_type<reorder>() || node->get_dependencies().size() != 1)
            return nullptr;
        node = &node->get_dependency(0);
    }
    return node;
}

// u2 code c of OpenVINO (value - zero point 1) -> the kernel's two's-complement (c - 1) & 3,
// applied to the four 2-bit fields of a byte; 0xFF marks a byte holding the unused code 3.
const std::array<uint16_t, 256>& u2_byte_lut() {
    static const auto lut = [] {
        std::array<uint16_t, 256> t{};
        for (int b = 0; b < 256; ++b) {
            uint16_t out = 0;
            bool bad = false;
            for (int j = 0; j < 4; ++j) {
                const int c = (b >> (2 * j)) & 3;
                bad |= c == 3;
                out |= static_cast<uint16_t>(((c - 1) & 3) << (2 * j));
            }
            t[b] = bad ? 0x100 : out;
        }
        return t;
    }();
    return lut;
}

// [E, N, K] u2 (16 codes of a row = one little-endian word) -> [E, K/16, N] uint32 kernel codes.
void pack_experts(const uint8_t* src, uint32_t* dst, size_t E, size_t N, size_t K) {
    const auto& lut = u2_byte_lut();
    const size_t kw = K / kTernoclPackFactor;
    std::atomic<bool> bad{false};
    ov::parallel_for(E, [&](size_t e) {
        const uint8_t* s = src + e * N * K / 4;
        uint32_t* d = dst + e * kw * N;
        for (size_t n = 0; n < N; ++n) {
            const uint8_t* row = s + n * K / 4;
            for (size_t w = 0; w < kw; ++w) {
                uint32_t v = 0;
                for (size_t b = 0; b < 4; ++b) {
                    const uint16_t m = lut[row[w * 4 + b]];
                    if (m & 0x100)
                        bad = true;
                    v |= static_cast<uint32_t>(m & 0xFF) << (8 * b);
                }
                d[w * N + n] = v;
            }
        }
    });
    OPENVINO_ASSERT(!bad, "[GPU] ternocl int2 MoE: expert weights hold u2 code 3, outside the ternary range");
}

cl::Program get_program(const ocl_engine& engine, const char* src, const std::string& opts) {
    static std::mutex m;
    static auto* cache = new std::map<std::string, cl::Program>;
    const std::string key =
        std::to_string(reinterpret_cast<uintptr_t>(engine.get_cl_context().get())) + "|" + std::to_string(reinterpret_cast<uintptr_t>(src)) + "|" + opts;
    std::lock_guard<std::mutex> lock(m);
    auto it = cache->find(key);
    if (it != cache->end())
        return it->second;
    cl::Program prog(engine.get_cl_context(), std::string(src));
    try {
        prog.build(std::vector<cl::Device>{engine.get_cl_device()}, opts.c_str());
    } catch (const cl::Error&) {
        OPENVINO_THROW("[GPU] ternocl int2 MoE: kernel build failed (", opts, "):\n", prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(engine.get_cl_device()));
    }
    return cache->emplace(key, prog).first->second;
}

kernel::ptr make_kernel(const ocl_engine& engine, const cl::Program& prog, const char* name) {
    return std::make_shared<ocl_kernel>(ocl_kernel_type(cl::Kernel(prog, name), engine.get_usm_helper()), name);
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

std::mutex& moe_mutex() {
    static std::mutex m;
    return m;
}

// Transposed scales per program and node; process lifetime, as for the dense impl.
std::unordered_map<std::string, std::array<memory::ptr, 3>>& moe_scales_cache() {
    static auto* c = new std::unordered_map<std::string, std::array<memory::ptr, 3>>;
    return *c;
}

// Weight buffers already repacked in place (program id | buffer address).
std::set<std::string>& moe_packed_buffers() {
    static auto* c = new std::set<std::string>;
    return *c;
}

}  // namespace

struct moe_3gemm_ternocl_int2 : typed_primitive_impl<moe_3gemm_fused_compressed> {
    using parent = typed_primitive_impl<moe_3gemm_fused_compressed>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::moe_3gemm_ternocl_int2)

    const ocl_engine* _engine = nullptr;
    std::array<memory::ptr, 3> _scales;  // gate, up, down: f16 [E, K/128, N]
    size_t _hidden = 0;
    size_t _inter = 0;
    size_t _topk = 0;
    size_t _experts = 0;
    kernel::ptr _gate_up, _down, _reduce;
    struct Prefill {
        kernel::ptr route, gather, up, gate, down, combine;
    };
    std::array<Prefill, kPrefillTiles.size()> _prefill;

    moe_3gemm_ternocl_int2() : parent("ternocl_int2_moe") {}
    moe_3gemm_ternocl_int2(const ocl_engine& engine, std::array<memory::ptr, 3> scales, size_t hidden, size_t inter, size_t topk, size_t experts)
        : parent("ternocl_int2_moe"),
          _engine(&engine),
          _scales(std::move(scales)),
          _hidden(hidden),
          _inter(inter),
          _topk(topk),
          _experts(experts) {
        const std::string base = "-cl-std=CL3.0 -DTOPK=" + std::to_string(topk);
        auto tile = [](const MoETile& t) {
            return " -DNSG_N=" + std::to_string(t.nsg) + " -DLS=" + std::to_string(t.ls) + " -DU=" + std::to_string(t.u);
        };
        const auto gu = get_program(engine, kTernoclMoeSource, base + tile(kGateUpTile));
        const auto dn = get_program(engine, kTernoclMoeSource, base + tile(kDownTile));
        _gate_up = make_kernel(engine, gu, "int2_moe_gate_up");
        _down = make_kernel(engine, dn, "int2_moe_down");
        _reduce = make_kernel(engine, dn, "int2_moe_reduce");
    }

    Prefill& get_prefill(size_t band) {
        auto& p = _prefill[band];
        if (p.up)
            return p;
        const auto& t = kPrefillTiles[band];
        const std::string opts = "-cl-std=CL3.0 -DTOPK=" + std::to_string(_topk) + " -DNUM_EXPERTS=" + std::to_string(_experts) +
                                 " -DMT_M=" + std::to_string(t.mt_m) + " -DMT_N=" + std::to_string(t.mt_n) + " -DWG_M=" + std::to_string(t.wg_m) +
                                 " -DWG_N=" + std::to_string(t.wg_n) + " -cl-intel-256-GRF-per-thread";
        const auto up = get_program(*_engine, kTernoclMoePrefillSource, opts);
        const auto gate = get_program(*_engine, kTernoclMoePrefillSource, opts + " -DPOSTOP=1");
        const auto down = get_program(*_engine, kTernoclMoePrefillSource, opts + " -DOUT_F32");
        p.route = make_kernel(*_engine, up, "int2_moe_route");
        p.gather = make_kernel(*_engine, up, "int2_moe_gather");
        p.up = make_kernel(*_engine, up, "int2_moe_gemm_mt");
        p.gate = make_kernel(*_engine, gate, "int2_moe_gemm_mt");
        p.down = make_kernel(*_engine, down, "int2_moe_gemm_mt");
        p.combine = make_kernel(*_engine, down, "int2_moe_combine");
        return p;
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_3gemm_ternocl_int2>(*this);
    }

    bool is_cpu() const override {
        return false;
    }
    bool is_onednn() const override {
        return false;
    }

    // [G, I] f16 SwiGLU output and [G, H] f32 expert outputs (G = T * TOPK) from the memory pool; prefill
    // adds the routing meta, permutation and inverse, expert-sorted activations and the up projection.
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        const auto& in = params.input_layouts[0];
        if (in.is_dynamic() || _hidden == 0)
            return {};
        const size_t T = ov::shape_size(in.get_shape()) / _hidden;
        const size_t rows = T * _topk;
        std::vector<BufferDescriptor> d = {BufferDescriptor(rows * _inter, ov::element::f16), BufferDescriptor(rows * _hidden, ov::element::f32)};
        if (T >= kPrefillMinTokens) {
            d.emplace_back(2 * _experts + 2, ov::element::i32);
            d.emplace_back(rows, ov::element::i32);
            d.emplace_back(rows, ov::element::i32);
            d.emplace_back(rows * _hidden, ov::element::f16);
            d.emplace_back(rows * _inter, ov::element::f16);
        }
        return d;
    }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void set_arguments_impl(typed_primitive_inst<moe_3gemm_fused_compressed>&) override {}
    void set_arguments_impl(typed_primitive_inst<moe_3gemm_fused_compressed>&, kernel_arguments_data&) override {}

    static scalars_desc ints(std::initializer_list<size_t> v) {
        scalars_desc sc(v.size());
        size_t i = 0;
        for (auto x : v) {
            sc[i].t = scalar_desc::Types::INT32;
            sc[i++].v.s32 = static_cast<int32_t>(x);
        }
        return sc;
    }

    static kernel_arguments_desc args_desc(size_t n_in, size_t n_scalars, std::array<size_t, 2> global, std::array<size_t, 2> local) {
        kernel_arguments_desc d;
        d.workGroups.global = {global[0], global[1], 1};
        d.workGroups.local = {local[0], local[1], 1};
        for (size_t i = 0; i < n_in; ++i)
            d.arguments.push_back({argument_desc::Types::INPUT, static_cast<uint32_t>(i)});
        d.arguments.push_back({argument_desc::Types::OUTPUT, 0});
        for (size_t i = 0; i < n_scalars; ++i)
            d.arguments.push_back({argument_desc::Types::SCALAR, static_cast<uint32_t>(i)});
        return d;
    }

    // every buffer as a positional argument, then the scalars
    static kernel_arguments_desc bufs_desc(size_t n_bufs, size_t n_scalars, std::array<size_t, 2> global, std::array<size_t, 2> local) {
        kernel_arguments_desc d;
        d.workGroups.global = {global[0], global[1], 1};
        d.workGroups.local = {local[0], local[1], 1};
        for (size_t i = 0; i < n_bufs; ++i)
            d.arguments.push_back({argument_desc::Types::INPUT, static_cast<uint32_t>(i)});
        for (size_t i = 0; i < n_scalars; ++i)
            d.arguments.push_back({argument_desc::Types::SCALAR, static_cast<uint32_t>(i)});
        return d;
    }

    event::ptr launch(stream& stream,
                      kernel& k,
                      std::vector<memory::cptr> bufs,
                      scalars_desc sc,
                      std::array<size_t, 2> global,
                      std::array<size_t, 2> local,
                      const std::vector<event::ptr>& deps,
                      bool is_output = false) {
        const auto d = bufs_desc(bufs.size(), sc.size(), global, local);
        kernel_arguments_data a;
        a.inputs = std::move(bufs);
        a.scalars = &sc;
        stream.set_arguments(k, d, a);
        return stream.enqueue_kernel(k, d, a, deps, is_output);
    }

    event::ptr execute_prefill(const std::vector<event::ptr>& events, typed_primitive_inst<moe_3gemm_fused_compressed>& instance, size_t T) {
        auto& stream = instance.get_network().get_stream();
        const size_t G = T * _topk;
        const auto band = prefill_band(T);
        auto& p = get_prefill(band);
        const auto& t = kPrefillTiles[band];
        const auto& tmp = instance.get_intermediates_memories();
        const auto &h = tmp.at(0), &y = tmp.at(1), &meta = tmp.at(2), &perm = tmp.at(3), &inv = tmp.at(4), &xs = tmp.at(5), &u = tmp.at(6);
        const auto ids = instance.input_memory_ptr(idx(MoEIdx::topk_indices));

        auto ev = launch(stream, *p.route, {ids, meta, perm, inv}, ints({G}), {1024, 1}, {1024, 1}, events);
        ev = launch(stream, *p.gather, {instance.input_memory_ptr(idx(MoEIdx::hidden_states)), perm, xs}, ints({_hidden}), {256, G}, {256, 1}, {ev});

        // row tiles: at most one partial tile per expert beyond the full ones
        const size_t rows_per_tile = static_cast<size_t>(t.mt_m * t.wg_m);
        const size_t tiles = ceil_div(G, rows_per_tile) + _experts;
        const size_t wg = 16 * static_cast<size_t>(t.wg_m * t.wg_n);
        auto grid = [&](size_t N) {
            return std::array<size_t, 2>{ceil_div(N, static_cast<size_t>(t.mt_n * t.wg_n)) * wg, tiles};
        };
        // up; H = silu(gate) * up; Y = H . down (fp32); args: meta, A, B, S, C, Other, N, K
        ev = launch(stream,
                    *p.up,
                    {meta, xs, instance.input_memory_ptr(idx(MoEIdx::weight_1)), _scales[1], u, xs},
                    ints({_inter, _hidden}),
                    grid(_inter),
                    {wg, 1},
                    {ev});
        ev = launch(stream,
                    *p.gate,
                    {meta, xs, instance.input_memory_ptr(idx(MoEIdx::weight_0)), _scales[0], h, u},
                    ints({_inter, _hidden}),
                    grid(_inter),
                    {wg, 1},
                    {ev});
        ev = launch(stream,
                    *p.down,
                    {meta, h, instance.input_memory_ptr(idx(MoEIdx::weight_2)), _scales[2], y, h},
                    ints({_hidden, _inter}),
                    grid(_hidden),
                    {wg, 1},
                    {ev});
        return launch(stream,
                      *p.combine,
                      {y, inv, instance.input_memory_ptr(idx(MoEIdx::topk_weights)), instance.output_memory_ptr(0)},
                      ints({_hidden}),
                      {ceil_div(_hidden, 64) * 64, T},
                      {64, 1},
                      {ev},
                      instance.is_output());
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<moe_3gemm_fused_compressed>& instance) override {
        auto& stream = instance.get_network().get_stream();
        const auto& in_layout = instance.get_impl_params()->input_layouts[0];
        const size_t T = ov::shape_size(in_layout.get_shape()) / _hidden;
        if (T >= kPrefillMinTokens)
            return execute_prefill(events, instance, T);
        const size_t G = T * _topk;
        const auto& tmp = instance.get_intermediates_memories();
        const auto ids = instance.input_memory_ptr(idx(MoEIdx::topk_indices));

        // H[g] = silu(x . gate_e) * (x . up_e)
        const size_t gu_wg = 16 * static_cast<size_t>(kGateUpTile.nsg);
        auto d = args_desc(6, 2, {ceil_div(_inter, gu_wg) * gu_wg * kGateUpTile.ls, G}, {gu_wg * kGateUpTile.ls, 1});
        auto sc = ints({_inter, _hidden});
        kernel_arguments_data a;
        a.inputs = {ids,
                    instance.input_memory_ptr(idx(MoEIdx::hidden_states)),
                    instance.input_memory_ptr(idx(MoEIdx::weight_0)),
                    _scales[0],
                    instance.input_memory_ptr(idx(MoEIdx::weight_1)),
                    _scales[1]};
        a.outputs = {tmp.at(0)};
        a.scalars = &sc;
        stream.set_arguments(*_gate_up, d, a);
        auto ev = stream.enqueue_kernel(*_gate_up, d, a, events, false);

        // Y[g] = rw[g] * (H[g] . down_e)
        const size_t dn_wg = 16 * static_cast<size_t>(kDownTile.nsg);
        d = args_desc(5, 2, {ceil_div(_hidden, dn_wg) * dn_wg * kDownTile.ls, G}, {dn_wg * kDownTile.ls, 1});
        sc = ints({_hidden, _inter});
        a.inputs = {ids, tmp.at(0), instance.input_memory_ptr(idx(MoEIdx::topk_weights)), instance.input_memory_ptr(idx(MoEIdx::weight_2)), _scales[2]};
        a.outputs = {tmp.at(1)};
        stream.set_arguments(*_down, d, a);
        ev = stream.enqueue_kernel(*_down, d, a, {ev}, false);

        // out[t] = sum of the token's TOPK rows of Y
        d = args_desc(1, 1, {ceil_div(_hidden, 64) * 64, T}, {64, 1});
        sc = ints({_hidden});
        a.inputs = {tmp.at(1)};
        a.outputs = {instance.output_memory_ptr(0)};
        stream.set_arguments(*_reduce, d, a);
        return stream.enqueue_kernel(*_reduce, d, a, {ev}, instance.is_output());
    }

public:
    static std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params&) {
        auto& prog = node.get_program();
        auto& engine = prog.get_engine();
        auto& stream = prog.get_stream();
        const auto& cfg = node.as<moe_3gemm_fused_compressed>().get_primitive()->_config;
        const size_t E = cfg.num_expert, H = cfg.hidden_size, I = cfg.inter_size;

        const std::string key = std::to_string(prog.get_id()) + "|" + node.id();
        std::array<memory::ptr, 3> scales;
        {
            std::lock_guard<std::mutex> lock(moe_mutex());
            auto it = moe_scales_cache().find(key);
            if (it != moe_scales_cache().end())
                scales = it->second;
        }
        if (!scales[0]) {
            for (size_t p = 0; p < 3; ++p) {
                const size_t N = p == 2 ? H : I, K = p == 2 ? I : H, groups = K / kTernoclGroupSize;

                // Weights: the kernel layout has the constant's byte size, so it is written in place.
                auto wei_mem = const_source(&node.get_dependency(idx(kWeights[p])))->as<data>().get_attached_memory_ptr();
                std::ostringstream buffer_key;
                buffer_key << prog.get_id() << "|" << wei_mem->buffer_ptr();
                bool packed;
                {
                    std::lock_guard<std::mutex> lock(moe_mutex());
                    packed = !moe_packed_buffers().insert(buffer_key.str()).second;
                }
                if (!packed) {
                    OPENVINO_ASSERT(wei_mem->size() == E * N * K / 4, "[GPU] ternocl int2 MoE: unexpected expert weight size ", wei_mem->size());
                    std::vector<uint8_t> host(wei_mem->size());
                    wei_mem->copy_to(stream, host.data(), true);
                    std::vector<uint32_t> out(E * (K / kTernoclPackFactor) * N);
                    pack_experts(host.data(), out.data(), E, N, K);
                    wei_mem->copy_from(stream, out.data(), 0, 0, wei_mem->size(), true);
                }

                // Scales: OpenVINO's [E, N, groups, 1] -> the kernel's [E, groups, N].
                auto sc_mem = const_source(&node.get_dependency(idx(kScales[p])))->as<data>().get_attached_memory_ptr();
                OPENVINO_ASSERT(sc_mem->size() == E * N * groups * 2, "[GPU] ternocl int2 MoE: unexpected scale size ", sc_mem->size());
                std::vector<uint16_t> src(E * N * groups), dst(E * N * groups);
                sc_mem->copy_to(stream, src.data(), true);
                ov::parallel_for(E, [&](size_t e) {
                    for (size_t n = 0; n < N; ++n)
                        for (size_t g = 0; g < groups; ++g)
                            dst[(e * groups + g) * N + n] = src[(e * N + n) * groups + g];
                });
                scales[p] =
                    engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(E * groups), static_cast<int64_t>(N)}, data_types::f16, format::bfyx},
                                           allocation_type::usm_device,
                                           false);
                scales[p]->copy_from(stream, dst.data(), true);
            }
            std::lock_guard<std::mutex> lock(moe_mutex());
            scales = moe_scales_cache().try_emplace(key, scales).first->second;
        }
        GPU_DEBUG_TRACE << "[ternocl-int2] create MoE " << node.id() << " E=" << E << " H=" << H << " I=" << I << " top_k=" << cfg.top_k << std::endl;
        return std::make_unique<moe_3gemm_ternocl_int2>(downcast<const ocl_engine>(engine), scales, H, I, cfg.top_k, E);
    }
};

bool TernoclInt2MoEImplementationManager::validate_impl(const program_node& node) const {
    const auto& config = node.get_program().get_config();
    if (!config.get_enable_ternocl_int2() || node.get_program().get_engine().runtime_type() != runtime_types::ocl)
        return false;
    const auto& info = node.get_program().get_engine().get_device_info();
    if (info.arch < gpu_arch::xe2 || !info.supports_immad)
        return false;
    // Weights are repacked in place, which an exported blob (cache_dir) must not see.
    if (!config.get_cache_dir().empty())
        return false;

    const auto& desc = node.as<moe_3gemm_fused_compressed>().get_primitive();
    const auto& cfg = desc->_config;
    if (cfg.expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU || cfg.activation_type != ov::op::internal::MOE::Activation_type::SWIGLU ||
        cfg.num_shared_expert != 0 || cfg.group_size != kTernoclGroupSize || !cfg.has_zp || cfg.scale_factor.has_value())
        return false;
    if (cfg.hidden_size % kTernoclGroupSize != 0 || cfg.inter_size % kTernoclGroupSize != 0 || desc->_otd.lru_expert_num != 0)
        return false;
    if (node.get_dependencies().size() != moe_3gemm_fused_compressed::input_count)
        return false;

    const auto& in = node.get_input_layout(idx(MoEIdx::hidden_states));
    const auto& out = node.get_output_layout(0);
    if (in.data_type != data_types::f16 || out.data_type != data_types::f16 || in.format != format::bfyx || out.format != format::bfyx)
        return false;
    if (node.get_input_layout(idx(MoEIdx::topk_weights)).data_type != data_types::f16 ||
        node.get_input_layout(idx(MoEIdx::topk_indices)).data_type != data_types::i32)
        return false;
    for (size_t p = 0; p < 3; ++p) {
        const auto* w = const_source(&node.get_dependency(idx(kWeights[p])));
        const auto* s = const_source(&node.get_dependency(idx(kScales[p])));
        const auto* z = const_source(&node.get_dependency(idx(kZps[p])));
        if (w == nullptr || s == nullptr || z == nullptr || w->get_users().size() != 1)
            return false;
        if (w->get_output_layout().data_type != data_types::u2 || s->get_output_layout().data_type != data_types::f16)
            return false;
        // Ternary: a zero point of 1, as a scalar or broadcast per expert.
        const auto& zl = z->get_output_layout();
        if (zl.count() > cfg.num_expert || zl.data_type != data_types::f16)
            return false;
        std::vector<uint16_t> zp(zl.count());
        z->as<data>().get_attached_memory_ptr()->copy_to(node.get_program().get_stream(), zp.data(), 0, 0, zp.size() * sizeof(uint16_t), true);
        if (std::any_of(zp.begin(), zp.end(), [](uint16_t v) {
                return v != 0x3C00;
            }))
            return false;
    }
    return true;
}

std::unique_ptr<primitive_impl> TernoclInt2MoEImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(validate_impl(node), "[GPU] ternocl int2 MoE: ", node.id(), " is outside the supported configuration");
    return moe_3gemm_ternocl_int2::create(node, params);
}

}  // namespace ocl
}  // namespace cldnn

#endif  // OV_GPU_WITH_OCL_RT
