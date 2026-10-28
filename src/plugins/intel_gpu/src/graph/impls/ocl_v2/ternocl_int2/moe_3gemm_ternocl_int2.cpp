// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_ternocl_int2.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "fully_connected_ternocl_int2.hpp"
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "impls/ocl_v2/utils/kernel_generator.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "moe_3gemm_fused_inst.h"
#include "primitive_inst.h"
#include "registry/registry.hpp"
#include "ternocl_int2_kernels.inc"  // generated from TERNOCL_ROOT (CMakeLists.txt)

namespace cldnn {
namespace ocl {

namespace {

using ov::intel_gpu::ArgumentDescriptor;
using ov::intel_gpu::Arguments;
using ov::intel_gpu::DispatchDataFunc;
using ov::intel_gpu::ImplRuntimeParams;
using ov::intel_gpu::KernelData;
using ov::intel_gpu::KernelLanguage;
using ov::intel_gpu::KernelString;
using ov::intel_gpu::RuntimeParams;
using ov::intel_gpu::ScalarDescriptor;
using ov::intel_gpu::ocl::KernelGenerator;
using ov::intel_gpu::ocl::PrimitiveImplOCL;
using ov::intel_gpu::ocl::Stage;

using MoEIdx = moe_3gemm_fused_compressed::input_index;

constexpr size_t idx(MoEIdx i) {
    return static_cast<size_t>(i);
}

// Decode tiles (TernOCL int2_moe/sweep.sh, Arc Pro B70, Qwen3-30B-A3B experts at T = 1):
// sub-groups per work-group NSG_N, local k-slicing LS, loads issued ahead U.
struct GemvTile {
    int nsg, ls, u;
};
constexpr GemvTile kGateUpTile{2, 4, 1};
constexpr GemvTile kDownTile{1, 1, 1};

// From this many tokens the rows are grouped by expert and run as M-tiled GEMMs, which
// read an expert's weights once per row tile rather than once per token.
constexpr size_t kPrefillTokens = 16;

// Prefill tiles (TernOCL int2_moe/sweep_prefill.sh, Arc Pro B70) for T <= 64, <= 512 and larger,
// and the single int8 one (GPU_TERNOCL_INT2_INT8_PREFILL). Rows per tile (mt_m * wg_m) are
// shared by the routing and the three GEMMs of one set.
struct MtTile {
    int mt_m, mt_n, wg_m, wg_n;
};
constexpr std::array<MtTile, 3> kPrefillTiles = {MtTile{16, 32, 1, 8}, MtTile{32, 32, 1, 4}, MtTile{32, 32, 1, 8}};
constexpr MtTile kPrefillInt8Tile{16, 64, 1, 4};
constexpr size_t kInt8Set = kPrefillTiles.size();
constexpr size_t kPrefillSets = kPrefillTiles.size() + 1;

size_t prefill_band(size_t T) {
    return T <= 64 ? 0 : (T <= 512 ? 1 : 2);
}

const MtTile& prefill_tile(size_t set) {
    return set == kInt8Set ? kPrefillInt8Tile : kPrefillTiles[set];
}

// Activation scale rows are padded to 32 (int2_int8_dpas.cl LDSA).
size_t ldsa(size_t rows) {
    return (rows + 31) & ~size_t{31};
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

struct MoEShape {
    size_t E, H, I, topk;
};

MoEShape moe_shape(const RuntimeParams& params) {
    const auto& cfg = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    return {cfg.num_expert, cfg.hidden_size, cfg.inter_size, cfg.top_k};
}

size_t num_tokens(const RuntimeParams& params) {
    return ov::shape_size(params.get_input_layout(idx(MoEIdx::hidden_states)).get_shape()) / moe_shape(params).H;
}

enum class MoEKernel { gate_up, down, reduce, route, gather, gemm_up, gemm_gate, gemm_down, combine, quant };

// One kernel per program (TERNOCL_SELECT_KERNELS). Arguments and work sizes are set at each launch.
class TernoclMoEStageGenerator : public KernelGenerator {
public:
    TernoclMoEStageGenerator(MoEKernel kind, size_t set) : KernelGenerator("ternocl_int2_moe", suffix(kind, set)), m_kind(kind), m_set(set) {}

    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override {
        const std::string name = kernel_name();
        std::string selector = name;
        std::transform(selector.begin(), selector.end(), selector.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });

        KernelData kd;
        kd.code = std::make_shared<KernelString>();
        kd.code->language = KernelLanguage::OCLC_V2;
        kd.code->entry_point = get_entry_point(params);
        kd.code->options = get_build_options(params);
        kd.code->batch_compilation = false;
        kd.code->str =
            "#define TERNOCL_SELECT_KERNELS\n#define TERNOCL_KERNEL_" + selector + "\n#define " + name + " " + kd.code->entry_point + "\n" + source();
        kd.params.layerID = params.desc->id;
        kd.update_dispatch_data_func = get_dispatch_data_func();
        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] " << params.desc->id << " " << name << ": " << kd.code->options << std::endl;
        return kd;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        const auto s = moe_shape(params);
        std::string opts = "-cl-std=CL3.0 -DTOPK=" + std::to_string(s.topk);
        if (is_decode()) {
            const auto& t = m_kind == MoEKernel::gate_up ? kGateUpTile : kDownTile;
            return opts + " -DNSG_N=" + std::to_string(t.nsg) + " -DLS=" + std::to_string(t.ls) + " -DU=" + std::to_string(t.u);
        }
        const auto& t = prefill_tile(m_set);
        opts += " -DNUM_EXPERTS=" + std::to_string(s.E) + " -DMT_M=" + std::to_string(t.mt_m) + " -DMT_N=" + std::to_string(t.mt_n) +
                " -DWG_M=" + std::to_string(t.wg_m) + " -DWG_N=" + std::to_string(t.wg_n);
        if (m_set == kInt8Set)
            opts += " -cl-fp32-correctly-rounded-divide-sqrt -DQMODE=0 -DWRITE_Q";
        // Only the GEMMs take 256 GRF, which halves the work-group limit (routing runs one 1024-wide work-group).
        if (m_kind == MoEKernel::gemm_up || m_kind == MoEKernel::gemm_gate || m_kind == MoEKernel::gemm_down)
            opts += " -cl-intel-256-GRF-per-thread";
        if (m_kind == MoEKernel::gemm_gate)
            opts += " -DPOSTOP=1";
        if (m_kind == MoEKernel::gemm_down)
            opts += " -DOUT_F32";
        return opts;
    }

private:
    [[nodiscard]] bool is_decode() const {
        return m_kind == MoEKernel::gate_up || m_kind == MoEKernel::down || m_kind == MoEKernel::reduce;
    }

    static std::string suffix(MoEKernel kind, size_t set) {
        static const std::array<const char*, 10> names = {"gate_up", "down", "reduce", "route", "gather", "up", "gate", "down_mt", "combine", "quant"};
        const std::string base = names[static_cast<size_t>(kind)];
        if (kind == MoEKernel::gate_up || kind == MoEKernel::down || kind == MoEKernel::reduce)
            return base;
        return base + (set == kInt8Set ? std::string("_i8") : std::to_string(set));
    }

    [[nodiscard]] std::string kernel_name() const {
        const bool int8 = m_set == kInt8Set;
        switch (m_kind) {
        case MoEKernel::gate_up:
            return "int2_moe_gate_up";
        case MoEKernel::down:
            return "int2_moe_down";
        case MoEKernel::reduce:
            return "int2_moe_reduce";
        case MoEKernel::route:
            return "int2_moe_route";
        case MoEKernel::gather:
            return int8 ? "int2_moe_gather_q" : "int2_moe_gather";
        case MoEKernel::gemm_up:
        case MoEKernel::gemm_gate:
        case MoEKernel::gemm_down:
            return int8 ? "int2_moe_gemm_mt_i8" : "int2_moe_gemm_mt";
        case MoEKernel::combine:
            return "int2_moe_combine";
        case MoEKernel::quant:
            return "quant_a";
        }
        return {};
    }

    [[nodiscard]] const char* source() const {
        if (is_decode())
            return kTernoclMoeSource;
        return m_set == kInt8Set ? kTernoclMoePrefillInt8Source : kTernoclMoePrefillSource;
    }

    MoEKernel m_kind;
    size_t m_set;
};

}  // namespace

// Decode (T < kPrefillTokens): H[g] = silu(x . gate_e) * (x . up_e), Y[g] = rw[g] * (H[g] . down_e) over
// all G = T * TOPK (token, expert) rows, then the per-token sum. Prefill: route on the device, gather
// the rows expert-sorted, up / gate (silu * up) / down as per-expert M-tiled GEMMs, combine. A dynamic
// impl builds the decode stages and either the three fp16 prefill bands or the int8 set.
class moe_3gemm_ternocl_int2 : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::moe_3gemm_ternocl_int2)

    struct PrefillStages {
        Stage::Ptr route, gather, up, gate, down, combine, quant;
    };

    Stage::Ptr gate_up = make_stage<TernoclMoEStageGenerator>(MoEKernel::gate_up, 0);
    Stage::Ptr down = make_stage<TernoclMoEStageGenerator>(MoEKernel::down, 0);
    Stage::Ptr reduce = make_stage<TernoclMoEStageGenerator>(MoEKernel::reduce, 0);
    std::array<PrefillStages, kPrefillSets> prefill = make_prefill_stages();

    moe_3gemm_ternocl_int2() : PrimitiveImplOCL("ternocl_int2_moe") {}
    moe_3gemm_ternocl_int2(const program_node& node, const RuntimeParams& params) : moe_3gemm_ternocl_int2() {
        // The weights inputs are the u2 constants at build time and the packed ones once these reorders are
        // folded. The packed experts are [E, N, K] whatever the source rank.
        const auto& wei_layout = params.get_input_layout(idx(MoEIdx::weight_0));
        const auto s = moe_shape(params);
        const layout packed_layout{ov::PartialShape{static_cast<int64_t>(s.E), static_cast<int64_t>(s.I), static_cast<int64_t>(s.H)},
                                   data_types::u2,
                                   format::b_y_xs_f_xsv16_ternary};
        _weights_reorder_params = std::make_shared<WeightsReorderParams>(wei_layout, packed_layout);

        auto add = [&](Stage::Ptr& stage) {
            add_stage(stage, params);
            OPENVINO_ASSERT(has_stage(stage), "[GPU] ternocl int2 MoE: failed to generate a kernel for ", node.id());
        };
        auto add_set = [&](size_t set) {
            auto& p = prefill[set];
            for (auto* stage : {&p.route, &p.gather, &p.up, &p.gate, &p.down, &p.combine})
                add(*stage);
            if (set == kInt8Set)
                add(p.quant);
        };
        const bool int8 = node.get_program().get_config().get_ternocl_int2_int8_prefill();
        const bool dynamic = params.is_dynamic();
        const size_t T = dynamic ? 0 : num_tokens(params);
        if (dynamic || T < kPrefillTokens) {
            add(gate_up);
            add(down);
            add(reduce);
        }
        if (dynamic && int8) {
            add_set(kInt8Set);
        } else if (dynamic) {
            for (size_t set = 0; set < kPrefillTiles.size(); ++set)
                add_set(set);
        } else if (T >= kPrefillTokens) {
            add_set(int8 ? kInt8Set : prefill_band(T));
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<moe_3gemm_ternocl_int2>(this);
    }

    // From the network memory pool: [G, I] f16 SwiGLU output and [G, H] f32 expert outputs; prefill adds
    // the routing meta, permutation and inverse, the expert-sorted activations and the up projection, and
    // int8 prefill the int8 tokens, their scales, the sorted rows' scales and the int8 SwiGLU output.
    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        if (params.get_input_layout(idx(MoEIdx::hidden_states)).is_dynamic())
            return {};
        const auto s = moe_shape(params);
        const size_t T = num_tokens(params);
        const size_t G = T * s.topk;
        std::vector<BufferDescriptor> d = {BufferDescriptor(G * s.I, ov::element::f16), BufferDescriptor(G * s.H, ov::element::f32)};
        if (T >= kPrefillTokens) {
            const bool int8 = runs_int8();
            d.emplace_back(2 * s.E + 2, ov::element::i32);
            d.emplace_back(G, ov::element::i32);
            d.emplace_back(G, ov::element::i32);
            d.emplace_back(G * s.H, int8 ? ov::element::i8 : ov::element::f16);
            d.emplace_back(G * s.I, ov::element::f16);
            if (int8) {
                const size_t gh = s.H / kTernoclGroupSize, gi = s.I / kTernoclGroupSize;
                d.emplace_back(T * s.H, ov::element::i8);
                d.emplace_back(gh * ldsa(T), ov::element::f16);
                d.emplace_back(gh * ldsa(G), ov::element::f16);
                d.emplace_back(gi * ldsa(G), ov::element::f16);
                d.emplace_back(G * s.I, ov::element::i8);
            }
        }
        return d;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        auto& stream = instance.get_network().get_stream();
        const auto& params = *instance.get_impl_params();
        const size_t T = num_tokens(params);
        if (instance.can_be_optimized() || T == 0)
            return stream.aggregate_events(events, events.size() > 1, instance.is_output());
        return T >= kPrefillTokens ? execute_prefill(events, instance, T) : execute_decode(events, instance, T);
    }

private:
    static std::array<PrefillStages, kPrefillSets> make_prefill_stages_impl(moe_3gemm_ternocl_int2& self) {
        std::array<PrefillStages, kPrefillSets> sets;
        for (size_t set = 0; set < kPrefillSets; ++set) {
            auto& p = sets[set];
            p.route = self.make_stage<TernoclMoEStageGenerator>(MoEKernel::route, set);
            p.gather = self.make_stage<TernoclMoEStageGenerator>(MoEKernel::gather, set);
            p.up = self.make_stage<TernoclMoEStageGenerator>(MoEKernel::gemm_up, set);
            p.gate = self.make_stage<TernoclMoEStageGenerator>(MoEKernel::gemm_gate, set);
            p.down = self.make_stage<TernoclMoEStageGenerator>(MoEKernel::gemm_down, set);
            p.combine = self.make_stage<TernoclMoEStageGenerator>(MoEKernel::combine, set);
            if (set == kInt8Set)
                p.quant = self.make_stage<TernoclMoEStageGenerator>(MoEKernel::quant, set);
        }
        return sets;
    }

    std::array<PrefillStages, kPrefillSets> make_prefill_stages() {
        return make_prefill_stages_impl(*this);
    }

    [[nodiscard]] bool runs_int8() const {
        return has_stage(prefill[kInt8Set].up);
    }

    event::ptr launch(stream& stream,
                      Stage& stage,
                      const std::vector<memory::ptr>& bufs,
                      const std::vector<size_t>& ints,
                      std::array<size_t, 2> global,
                      std::array<size_t, 2> local,
                      const std::vector<event::ptr>& deps,
                      bool completion_event = false) const {
        kernel_arguments_desc desc;
        kernel_arguments_data args;
        for (uint32_t i = 0; i < bufs.size(); ++i) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(bufs[i]);
        }
        scalars_desc scalars(ints.size());
        for (uint32_t i = 0; i < ints.size(); ++i) {
            desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, i});
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(ints[i]);
        }
        args.scalars = &scalars;
        desc.workGroups.global = {global[0], global[1], 1};
        desc.workGroups.local = {local[0], local[1], 1};
        stream.set_arguments(*stage.kernel, desc, args);
        kernel_dump_info.add_entry_point(stage.kernel->get_id());
        return stream.enqueue_kernel(*stage.kernel, desc, args, deps, completion_event);
    }

    event::ptr execute_decode(const std::vector<event::ptr>& events, primitive_inst& instance, size_t T) {
        OPENVINO_ASSERT(has_stage(gate_up), "[GPU] ternocl int2 MoE: ", instance.id(), " has no decode kernels for T=", T);
        auto& stream = instance.get_network().get_stream();
        const auto s = moe_shape(*instance.get_impl_params());
        const size_t G = T * s.topk;
        const auto& tmp = instance.get_intermediates_memories();
        const auto in = [&](MoEIdx i) {
            return instance.input_memory_ptr(idx(i));
        };
        const auto ids = in(MoEIdx::topk_indices);

        const size_t gu_wg = 16 * static_cast<size_t>(kGateUpTile.nsg);
        const size_t gu_ls = static_cast<size_t>(kGateUpTile.ls);
        auto ev = launch(stream,
                         *gate_up,
                         {ids, in(MoEIdx::hidden_states), in(MoEIdx::weight_0), in(MoEIdx::scale_0), in(MoEIdx::weight_1), in(MoEIdx::scale_1), tmp.at(0)},
                         {s.I, s.H},
                         {ceil_div(s.I, gu_wg) * gu_wg * gu_ls, G},
                         {gu_wg * gu_ls, 1},
                         events);
        const size_t dn_wg = 16 * static_cast<size_t>(kDownTile.nsg);
        const size_t dn_ls = static_cast<size_t>(kDownTile.ls);
        ev = launch(stream,
                    *down,
                    {ids, tmp.at(0), in(MoEIdx::topk_weights), in(MoEIdx::weight_2), in(MoEIdx::scale_2), tmp.at(1)},
                    {s.H, s.I},
                    {ceil_div(s.H, dn_wg) * dn_wg * dn_ls, G},
                    {dn_wg * dn_ls, 1},
                    {ev});
        return launch(stream,
                      *reduce,
                      {tmp.at(1), instance.output_memory_ptr(0)},
                      {s.H},
                      {ceil_div(s.H, 64) * 64, T},
                      {64, 1},
                      {ev},
                      instance.needs_completion_event());
    }

    event::ptr execute_prefill(const std::vector<event::ptr>& events, primitive_inst& instance, size_t T) {
        const bool int8 = runs_int8();
        const size_t set = int8 ? kInt8Set : prefill_band(T);
        auto& p = prefill[set];
        OPENVINO_ASSERT(has_stage(p.up), "[GPU] ternocl int2 MoE: ", instance.id(), " has no prefill kernels for T=", T);
        auto& stream = instance.get_network().get_stream();
        const auto s = moe_shape(*instance.get_impl_params());
        const size_t G = T * s.topk;
        const auto& t = prefill_tile(set);
        const auto& tmp = instance.get_intermediates_memories();
        const auto &h = tmp.at(0), &y = tmp.at(1), &meta = tmp.at(2), &perm = tmp.at(3), &inv = tmp.at(4), &xs = tmp.at(5), &u = tmp.at(6);
        const auto in = [&](MoEIdx i) {
            return instance.input_memory_ptr(idx(i));
        };
        const auto x = in(MoEIdx::hidden_states);

        // One work-group counts the rows per expert and writes the permutation.
        const size_t route_wg = std::min<size_t>(1024, instance.get_network().get_engine().get_device_info().max_work_group_size);
        auto ev = launch(stream, *p.route, {in(MoEIdx::topk_indices), meta, perm, inv}, {G}, {route_wg, 1}, {route_wg, 1}, events);

        // Row tiles: at most one partial tile per expert beyond the full ones.
        const size_t rows_per_tile = static_cast<size_t>(t.mt_m * t.wg_m);
        const size_t tiles = ceil_div(G, rows_per_tile) + s.E;
        const size_t wg = 16 * static_cast<size_t>(t.wg_m * t.wg_n);
        const auto grid = [&](size_t N) {
            return std::array<size_t, 2>{ceil_div(N, static_cast<size_t>(t.mt_n * t.wg_n)) * wg, tiles};
        };

        if (int8) {
            // xs holds the int8 sorted rows; the fp16 A argument is unused in QMODE 0.
            const auto &xq = tmp.at(7), &sa_t = tmp.at(8), &sa_x = tmp.at(9), &sa_h = tmp.at(10), &hq = tmp.at(11);
            const size_t gk = kTernoclGroupSize / 16;
            // The T tokens are quantized once, before the gather duplicates them TOPK times.
            ev = launch(stream, *p.quant, {x, sa_t, xq}, {T, s.H}, {s.H / gk, T}, {16, 1}, {ev});
            ev = launch(stream, *p.gather, {xq, sa_t, perm, xs, sa_x}, {T, G, s.H}, {256, G}, {256, 1}, {ev});
            ev = launch(stream, *p.up, {meta, h, xs, sa_x, in(MoEIdx::weight_1), in(MoEIdx::scale_1), u, u}, {G, s.I, s.H}, grid(s.I), {wg, 1}, {ev});
            ev = launch(stream, *p.gate, {meta, h, xs, sa_x, in(MoEIdx::weight_0), in(MoEIdx::scale_0), h, u}, {G, s.I, s.H}, grid(s.I), {wg, 1}, {ev});
            ev = launch(stream, *p.quant, {h, sa_h, hq}, {G, s.I}, {s.I / gk, G}, {16, 1}, {ev});
            ev = launch(stream, *p.down, {meta, h, hq, sa_h, in(MoEIdx::weight_2), in(MoEIdx::scale_2), y, h}, {G, s.H, s.I}, grid(s.H), {wg, 1}, {ev});
        } else {
            ev = launch(stream, *p.gather, {x, perm, xs}, {s.H}, {256, G}, {256, 1}, {ev});
            // up; H = silu(gate) * up; Y = H . down (fp32)
            ev = launch(stream, *p.up, {meta, xs, in(MoEIdx::weight_1), in(MoEIdx::scale_1), u, xs}, {s.I, s.H}, grid(s.I), {wg, 1}, {ev});
            ev = launch(stream, *p.gate, {meta, xs, in(MoEIdx::weight_0), in(MoEIdx::scale_0), h, u}, {s.I, s.H}, grid(s.I), {wg, 1}, {ev});
            ev = launch(stream, *p.down, {meta, h, in(MoEIdx::weight_2), in(MoEIdx::scale_2), y, h}, {s.H, s.I}, grid(s.H), {wg, 1}, {ev});
        }
        return launch(stream,
                      *p.combine,
                      {y, inv, in(MoEIdx::topk_weights), instance.output_memory_ptr(0)},
                      {s.H},
                      {ceil_div(s.H, 64) * 64, T},
                      {64, 1},
                      {ev},
                      instance.needs_completion_event());
    }
};

bool TernoclInt2MoEImplementationManager::validate_impl(const program_node& node) const {
    assert(node.is_type<moe_3gemm_fused_compressed>());
#define TERNOCL_MOE_REJECT(reason)                                                                            \
    do {                                                                                                      \
        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] reject MoE " << node.id() << ": " << (reason) << std::endl; \
        return false;                                                                                         \
    } while (0)

    const auto& config = node.get_program().get_config();
    if (!config.get_enable_ternocl_int2())
        return false;
    // SIMD16 DPAS and 2D block loads, validated on the OpenCL and Level Zero runtimes.
    const auto& engine = node.get_program().get_engine();
    const auto& info = engine.get_device_info();
    const auto runtime = engine.runtime_type();
    if (info.arch < gpu_arch::xe2 || !info.supports_immad || (runtime != runtime_types::ocl && runtime != runtime_types::ze))
        return false;

    const auto& desc = node.as<moe_3gemm_fused_compressed>().get_primitive();
    const auto& cfg = desc->_config;
    if (node.get_input_layout(idx(MoEIdx::weight_0)).data_type != data_types::u2)
        return false;
    if (cfg.expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU || cfg.activation_type != ov::op::internal::MOE::Activation_type::SWIGLU)
        TERNOCL_MOE_REJECT("not a gate/up/down SwiGLU expert");
    if (cfg.num_shared_expert != 0 || node.get_dependencies().size() != moe_3gemm_fused_compressed::input_count)
        TERNOCL_MOE_REJECT("shared experts");
    if (cfg.group_size != kTernoclGroupSize || !cfg.has_zp || cfg.scale_factor.has_value())
        TERNOCL_MOE_REJECT("not 128-groups with a zero point and no scale factor");
    // Ternary {-1, 0, +1} arrives as u2 codes {0, 1, 2} with zero point 1 (recorded when the op was created).
    if (!desc->decompression_zero_point_scalar.has_value() || std::lround(desc->decompression_zero_point_scalar.value()) != 1)
        TERNOCL_MOE_REJECT("zero points are not all 1");
    if (cfg.hidden_size % kTernoclGroupSize != 0 || cfg.inter_size % kTernoclGroupSize != 0 || cfg.top_k == 0 || cfg.num_expert == 0)
        TERNOCL_MOE_REJECT("hidden or intermediate size is not a multiple of the group size");
    if (desc->_otd.lru_expert_num != 0)
        TERNOCL_MOE_REJECT("experts offloaded to disk");

    const auto& in = node.get_input_layout(idx(MoEIdx::hidden_states));
    const auto& out = node.get_output_layout(0);
    if (in.data_type != data_types::f16 || out.data_type != data_types::f16 || in.format != format::bfyx || out.format != format::bfyx)
        TERNOCL_MOE_REJECT("activations are not f16 bfyx");
    if (in.data_padding || out.data_padding)
        TERNOCL_MOE_REJECT("padded activations");
    if (node.get_input_layout(idx(MoEIdx::topk_weights)).data_type != data_types::f16 ||
        node.get_input_layout(idx(MoEIdx::topk_indices)).data_type != data_types::i32)
        TERNOCL_MOE_REJECT("routing weights are not f16 or indices are not i32");
    for (const auto [w, sc] :
         {std::pair{MoEIdx::weight_0, MoEIdx::scale_0}, std::pair{MoEIdx::weight_1, MoEIdx::scale_1}, std::pair{MoEIdx::weight_2, MoEIdx::scale_2}}) {
        // [E, N, K] or [E, N, K/128, 128]
        const auto& wl = node.get_input_layout(idx(w));
        const auto rank = wl.get_partial_shape().size();
        if (!node.get_dependency(idx(w)).is_constant() || wl.data_type != data_types::u2 || wl.format != format::bfyx || wl.is_dynamic() ||
            (rank != 3 && rank != 4))
            TERNOCL_MOE_REJECT("expert weights are not a static u2 [E, N, K] constant");
        // [E, K/128, N] f16, as prepare_quantization lays out the MoE scales (with one group bfyx is the same).
        const auto& sl = node.get_input_layout(idx(sc));
        const auto& sp = sl.get_partial_shape();
        const bool one_group = sl.format == format::bfyx && sp.size() == 4 && sp[2].is_static() && sp[2].get_length() == 1;
        if (sl.data_type != data_types::f16 || (sl.format != format::byfx && !one_group) || sl.data_padding)
            TERNOCL_MOE_REJECT("scales are not f16 [E, K/128, N]: " + sl.to_short_string());
    }
    GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] accept MoE " << node.id() << std::endl;
    return true;
#undef TERNOCL_MOE_REJECT
}

std::unique_ptr<primitive_impl> TernoclInt2MoEImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(validate_impl(node), "[GPU] ternocl int2 MoE: ", node.id(), " is outside the supported configuration");
    const auto s = moe_shape(params);
    GPU_DEBUG_TRACE << "[ternocl-int2] create MoE " << node.id() << " E=" << s.E << " H=" << s.H << " I=" << s.I << " top_k=" << s.topk
                    << " int8_prefill=" << node.get_program().get_config().get_ternocl_int2_int8_prefill() << (params.is_dynamic() ? " dynamic" : " static")
                    << std::endl;
    return std::make_unique<moe_3gemm_ternocl_int2>(node, params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::moe_3gemm_ternocl_int2)
