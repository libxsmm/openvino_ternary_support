// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_ternocl_int2.hpp"

#ifdef OV_GPU_WITH_OCL_RT

#include "data_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/parallel.hpp"
#include "primitive_inst.h"
#include "reorder_inst.h"
#include "runtime/ocl/ocl_engine.hpp"
#include "runtime/ocl/ocl_kernel.hpp"
#include "ternocl_int2_kernels.inc"  // generated from TERNOCL_ROOT (CMakeLists.txt)

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace cldnn {
namespace ocl {

namespace {

// OpenVINO stores u2 little-endian inside each byte: value j at bits [2j, 2j+1].
inline uint8_t read_u2(const uint8_t* base, size_t index) {
    return static_cast<uint8_t>((base[index >> 2] >> (2 * (index & 0x3))) & 0x3);
}

// [N, K] u2 codes -> [K/16, N] uint32, re-encoding (code - zp) to the kernel's
// two's-complement {0, +1, -1} = {0, 1, 3}; row 16*kb+j of column n sits at bits [2j, 2j+1].
void pack_weights(const uint8_t* src, uint32_t* dst, size_t N, size_t K, int32_t zp) {
    std::fill(dst, dst + (K / kTernoclPackFactor) * N, 0u);
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            const int32_t code = static_cast<int32_t>(read_u2(src, n * K + k)) - zp;
            OPENVINO_ASSERT(code >= -1 && code <= 1, "[GPU] ternocl int2: weight code ", code,
                            " is outside the ternary range {-1, 0, +1}");
            dst[(k / kTernoclPackFactor) * N + n] |= static_cast<uint32_t>(code & 0x3) << (2 * (k % kTernoclPackFactor));
        }
    }
}

// ---------------------------------------------------------------------------
// BITCOS (experimental, OV_TERNOCL_INT2_BITCOS=1): decode GEMVs read a
// (2 - z)-bit/weight layout instead of int2; prefill keeps the int2 copy.
// One uint32 buffer, three planes back to back (TernOCL common/bitcos.hpp):
//   bitmap  [K/32][N]  bit c of word (kp, n) = weight(kp*32+c, n) != 0
//   offsets [N]        first sign word of column n
//   signs              one bit per non-zero, column-major in k, 1 -> -1
// plus kBitcosPad words (the sign gather reads up to two words past a run), and
// per local-K-slice count LS: slice_ranks [LS-1][N], the non-zeros of column n
// above each boundary s*K/LS.
constexpr size_t kBitcosPad = 4;
constexpr int kBitcosLs[3] = {2, 4, 8};

bool bitcos_enabled() {
    static const bool v = std::getenv("OV_TERNOCL_INT2_BITCOS") != nullptr && kTernoclBitcosSource[0] != '\0';
    return v;
}

struct BitcosHost {
    std::vector<uint32_t> buf;
    std::array<std::vector<uint32_t>, 3> sr;  // LS = 2, 4, 8
    uint64_t nnz = 0;
};

BitcosHost pack_bitcos(const uint8_t* src, size_t N, size_t K, int32_t zp) {
    BitcosHost b;
    const size_t kw = K / 32, bw = kw * N;
    std::vector<uint32_t> bmp(bw), nnz(N);
    ov::parallel_for(N, [&](size_t n) {
        uint32_t cnt = 0;
        for (size_t kp = 0; kp < kw; ++kp) {
            uint32_t w = 0;
            for (size_t c = 0; c < 32; ++c)
                w |= static_cast<uint32_t>(read_u2(src, n * K + kp * 32 + c) != zp) << c;
            bmp[kp * N + n] = w;
            cnt += static_cast<uint32_t>(__builtin_popcount(w));
        }
        nnz[n] = cnt;
    });
    std::vector<uint32_t> off(N);
    size_t words = 0;
    for (size_t n = 0; n < N; ++n) {
        off[n] = static_cast<uint32_t>(words);
        words += (nnz[n] + 31) / 32;
        b.nnz += nnz[n];
    }
    OPENVINO_ASSERT(bw + N + words + kBitcosPad < (size_t{1} << 32), "[GPU] ternocl bitcos: layout exceeds 32-bit offsets");
    b.buf.assign(bw + N + words + kBitcosPad, 0u);
    std::copy(bmp.begin(), bmp.end(), b.buf.begin());
    std::copy(off.begin(), off.end(), b.buf.begin() + bw);
    uint32_t* signs = b.buf.data() + bw + N;
    ov::parallel_for(N, [&](size_t n) {
        uint32_t* sg = signs + off[n];
        uint32_t rank = 0;
        for (size_t k = 0; k < K; ++k) {
            const int32_t code = static_cast<int32_t>(read_u2(src, n * K + k)) - zp;
            OPENVINO_ASSERT(code >= -1 && code <= 1, "[GPU] ternocl bitcos: weight code ", code, " is not ternary");
            if (code == 0)
                continue;
            sg[rank / 32] |= static_cast<uint32_t>(code < 0) << (rank % 32);
            ++rank;
        }
    });
    for (size_t i = 0; i < 3; ++i) {
        const size_t ls = static_cast<size_t>(kBitcosLs[i]);
        if (K % (64 * ls) != 0)
            continue;
        auto& r = b.sr[i];
        r.assign((ls - 1) * N, 0u);
        const size_t slice_kw = K / ls / 32;
        ov::parallel_for(N, [&](size_t n) {
            uint32_t s = 0;
            size_t kp = 0;
            for (size_t sl = 1; sl < ls; ++sl) {
                for (; kp < sl * slice_kw; ++kp)
                    s += static_cast<uint32_t>(__builtin_popcount(bmp[kp * N + n]));
                r[(sl - 1) * N + n] = s;
            }
        });
    }
    return b;
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
        OPENVINO_THROW("[GPU] ternocl int2: unsupported zero point type ", mem->get_layout().data_type);
    }
}

// Constant inputs may be fed through a reorder inserted by the graph optimizer;
// only the data node behind it can be read at compile time.
const program_node* const_source(const program_node* node) {
    while (node != nullptr && !node->is_type<data>()) {
        if (!node->is_type<reorder>() || node->get_dependencies().size() != 1)
            return nullptr;
        node = &node->get_dependency(0);
    }
    return node;
}

// ---------------------------------------------------------------------------
// Tile selection. GEMV (M <= 8): sub-group rows SGM, work-group width WGN
// (16 per sub-group), local k-slicing LS, loads issued ahead U. M-tiled GEMM
// (M > 8): sub-group tile MT_M x MT_N, work-group WG_M x WG_N sub-groups.
// Tuned with TernOCL's bench.sh (>= 2 GiB rotating weights, median of 3).
struct GemvTile {
    int wgn, ls, u;
};
struct MtTile {
    int mt_m, mt_n, wg_m, wg_n;
};

bool env_ints(const char* name, int* v, int n) {
    const char* e = std::getenv(name);
    if (e == nullptr)
        return false;
    std::stringstream ss(e);
    for (int i = 0; i < n; ++i) {
        if (!(ss >> v[i]))
            return false;
        ss.ignore(1, ',');
    }
    return true;
}

GemvTile gemv_tile(size_t K, size_t N, bool integrated) {
    GemvTile t{0, 0, 0};
    if (env_ints("OV_TERNOCL_INT2_GEMV", &t.wgn, 3))  // "wgn,ls,u" for sweeps
        return t;
    struct Entry {
        size_t k, n;
        GemvTile t;
    };
    // Arc Pro B70, M = 1.
    static const Entry discrete[] = {
        {4096, 6144, {16, 4, 1}},     // 8B qkv
        {4096, 4096, {32, 8, 1}},     // 8B o_proj
        {4096, 24576, {64, 1, 2}},    // 8B gate_up (merged)
        {12288, 4096, {32, 8, 1}},    // 8B down
        {4096, 151680, {16, 8, 1}},   // 8B lm_head
        {5120, 34816, {16, 4, 2}},    // 27B gate_up (merged)
        {17408, 5120, {32, 4, 2}},    // 27B down
        {5120, 16384, {16, 2, 1}},    // 27B in_proj_qkvz
        {6144, 5120, {32, 6, 1}},     // 27B out_proj / o_proj
        {5120, 14336, {16, 2, 1}},    // 27B qkv
        {5120, 248320, {16, 4, 2}},   // 27B lm_head
    };
    // Arc 140V (Lunar Lake), M = 1, paced sweep.
    static const Entry igpu[] = {
        {5120, 34816, {16, 4, 2}},
        {17408, 5120, {64, 1, 2}},
        {5120, 16384, {16, 1, 2}},
        {6144, 5120, {64, 1, 2}},
        {5120, 14336, {16, 4, 2}},
        {5120, 248320, {32, 2, 2}},
    };
    if (integrated) {
        for (const auto& e : igpu)
            if (e.k == K && e.n == N)
                return e.t;
    }
    for (const auto& e : discrete)
        if (e.k == K && e.n == N)
            return e.t;
    return N <= 8192 ? GemvTile{32, 4, 2} : GemvTile{16, 2, 1};
}

// BITCOS GEMV tile (wgn, ls; u unused), per SGM.
GemvTile bitcos_tile(size_t K, size_t N, int sgm) {
    GemvTile t{0, 0, 0};
    if (env_ints("OV_TERNOCL_INT2_BITCOS_TILE", &t.wgn, 2))  // "wgn,ls" for sweeps
        return t;
    struct Entry {
        size_t k, n;
        GemvTile t[4];  // SGM 1, 2, 4, 8
    };
    // Arc Pro B70, Bonsai 2 27B shapes, swept at M = 1 / 4 / 8 at z = 0.33 (SGM 2 uses the M = 4 tile).
    static const Entry discrete[] = {
        {5120, 34816, {{16, 8, 0}, {64, 8, 0}, {64, 8, 0}, {128, 8, 0}}},
        {17408, 5120, {{32, 4, 0}, {64, 4, 0}, {64, 4, 0}, {32, 8, 0}}},
        {5120, 16384, {{32, 4, 0}, {64, 8, 0}, {64, 8, 0}, {128, 8, 0}}},
        {6144, 5120, {{32, 4, 0}, {64, 4, 0}, {64, 4, 0}, {32, 8, 0}}},
        {5120, 14336, {{64, 2, 0}, {64, 8, 0}, {64, 8, 0}, {128, 8, 0}}},
        {5120, 248320, {{32, 4, 0}, {64, 8, 0}, {64, 8, 0}, {64, 8, 0}}},
    };
    const int i = sgm == 1 ? 0 : (sgm == 2 ? 1 : (sgm == 4 ? 2 : 3));
    for (const auto& e : discrete)
        if (e.k == K && e.n == N)
            t = e.t[i];
    if (t.wgn == 0)
        t = N <= 8192 ? GemvTile{32, 8, 0} : GemvTile{64, 4, 0};
    while (t.ls > 1 && K % (64 * static_cast<size_t>(t.ls)) != 0)
        t.ls /= 2;
    return t;
}

// BITCOS M-tiled tile for M > 8: Arc Pro B70, Bonsai 2 27B shapes at z = 0.4,
// swept at M = 20 (used up to M = 32) and M = 512 (above).
MtTile bitcos_mt_tile(size_t K, size_t N, size_t M) {
    MtTile t{0, 0, 0, 0};
    if (env_ints("OV_TERNOCL_INT2_BITCOS_MT", &t.mt_m, 4))  // "mt_m,mt_n,wg_m,wg_n"
        return t;
    struct Entry {
        size_t k, n;
        MtTile prompt, bulk;
    };
    static const Entry table[] = {
        {5120, 34816, {32, 32, 1, 4}, {32, 32, 4, 2}},
        {17408, 5120, {16, 16, 2, 4}, {32, 32, 4, 4}},
        {5120, 16384, {32, 16, 1, 4}, {32, 32, 4, 2}},
        {6144, 5120, {16, 16, 2, 4}, {32, 32, 4, 2}},
        {5120, 14336, {32, 16, 1, 4}, {32, 32, 2, 4}},
        {5120, 248320, {32, 32, 1, 8}, {64, 16, 4, 2}},
    };
    for (const auto& e : table)
        if (e.k == K && e.n == N)
            return M <= 32 ? e.prompt : e.bulk;
    return M <= 32 ? MtTile{32, 16, 1, 4} : MtTile{32, 32, 4, 2};
}

MtTile mt_tile(size_t K, size_t N, size_t M, bool integrated) {
    MtTile t{0, 0, 0, 0};
    if (M < 64) {
        if (env_ints("OV_TERNOCL_INT2_MID", &t.mt_m, 4))  // "mt_m,mt_n,wg_m,wg_n"
            return t;
        // Sweep at M = 12 / 20 / 32 / 48 over the 27B shapes; the winners group
        // by output width (narrow N <= 8192, head N >= 65536).
        const int band = M <= 16 ? 0 : (M <= 32 ? 1 : 2);
        static const MtTile narrow[3] = {{16, 16, 2, 2}, {32, 16, 2, 4}, {32, 16, 1, 4}};  // Arc Pro B70
        static const MtTile wide[3] = {{16, 16, 1, 8}, {32, 16, 1, 4}, {64, 16, 1, 8}};
        static const MtTile head[3] = {{16, 16, 1, 4}, {64, 32, 1, 8}, {64, 32, 1, 8}};
        static const MtTile i_narrow[3] = {{16, 32, 1, 4}, {32, 32, 1, 8}, {64, 32, 1, 8}};  // Arc 140V
        static const MtTile i_wide[3] = {{16, 16, 1, 8}, {64, 32, 1, 4}, {64, 32, 1, 4}};
        static const MtTile i_head[3] = {{16, 16, 1, 4}, {32, 16, 1, 8}, {64, 32, 1, 4}};
        if (integrated)
            return N <= 8192 ? i_narrow[band] : (N >= 65536 ? i_head[band] : i_wide[band]);
        return N <= 8192 ? narrow[band] : (N >= 65536 ? head[band] : wide[band]);
    }
    if (env_ints("OV_TERNOCL_INT2_MT", &t.mt_m, 4))
        return t;
    struct Entry {
        size_t k, n;
        MtTile t;
    };
    // Arc Pro B70, M = 1024.
    static const Entry table[] = {
        {4096, 6144, {64, 32, 4, 4}},
        {4096, 4096, {64, 32, 1, 8}},
        {4096, 24576, {64, 32, 4, 4}},
        {12288, 4096, {64, 32, 1, 8}},
        {4096, 151680, {128, 16, 2, 2}},
        {5120, 34816, {64, 32, 4, 4}},
        {17408, 5120, {64, 32, 4, 2}},
        {5120, 16384, {64, 32, 8, 1}},
        {6144, 5120, {64, 32, 2, 4}},
        {5120, 14336, {64, 32, 4, 4}},
        {5120, 248320, {128, 16, 2, 4}},
    };
    // Arc 140V (Lunar Lake), M = 512.
    static const Entry igpu[] = {
        {5120, 34816, {128, 16, 2, 2}},
        {17408, 5120, {64, 32, 4, 2}},
        {5120, 16384, {128, 16, 2, 4}},
        {6144, 5120, {128, 16, 2, 4}},
        {5120, 14336, {128, 16, 2, 4}},
        {5120, 248320, {128, 16, 2, 2}},
    };
    if (integrated) {
        for (const auto& e : igpu)
            if (e.k == K && e.n == N)
                return e.t;
        return MtTile{128, 16, 2, 4};
    }
    for (const auto& e : table)
        if (e.k == K && e.n == N)
            return e.t;
    return MtTile{64, 32, 4, 4};
}

// ---------------------------------------------------------------------------
// Programs are built once per (context, source, options) and shared; every impl
// creates its own cl_kernel from them so argument state is never shared.
cl::Program get_program(const ocl_engine& engine, const char* src, const std::string& opts) {
    static std::mutex m;
    static auto* cache = new std::map<std::string, cl::Program>;
    const std::string key = std::to_string(reinterpret_cast<uintptr_t>(engine.get_cl_context().get())) + "|" +
                            std::to_string(reinterpret_cast<uintptr_t>(src)) + "|" + opts;
    std::lock_guard<std::mutex> lock(m);
    auto it = cache->find(key);
    if (it != cache->end())
        return it->second;
    cl::Program prog(engine.get_cl_context(), std::string(src));
    try {
        prog.build(std::vector<cl::Device>{engine.get_cl_device()}, opts.c_str());
    } catch (const cl::Error&) {
        OPENVINO_THROW("[GPU] ternocl int2: kernel build failed (", opts, "):\n",
                       prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(engine.get_cl_device()));
    }
    if (std::getenv("OV_TERNOCL_INT2_CFG_DEBUG") != nullptr)
        std::cerr << "[ternocl-int2] built " << opts << std::endl;
    return cache->emplace(key, prog).first->second;
}

kernel::ptr make_kernel(const ocl_engine& engine, const cl::Program& prog, const char* name) {
    return std::make_shared<ocl_kernel>(ocl_kernel_type(cl::Kernel(prog, name), engine.get_usm_helper()), name);
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

}  // namespace

// OpenVINO caches primitive_impl objects by kernel_impl_params, so one impl can
// serve several FullyConnected nodes of identical shape: packed weights are
// resolved by node id per execution.
struct TernoclInt2Packed {
    memory::ptr weights;
    memory::ptr scales;
    memory::ptr had_signs;  // i8 [K] +-1, or null
    memory::ptr had_input;  // f16 [rows, K] rotated activation
    size_t had_rows = 0;
    memory::ptr bitcos;                    // BITCOS planes, when OV_TERNOCL_INT2_BITCOS is set
    std::array<memory::ptr, 3> bitcos_sr;  // slice ranks for LS = 2, 4, 8
};

static std::mutex& ternocl_packed_mutex() {
    static std::mutex m;
    return m;
}

// Process lifetime: the plugin can be unloaded after the context is gone.
static std::unordered_map<std::string, TernoclInt2Packed>& ternocl_packed_cache() {
    static auto* c = new std::unordered_map<std::string, TernoclInt2Packed>;
    return *c;
}

struct fully_connected_ternocl_int2 : typed_primitive_impl<fully_connected> {
    using parent = typed_primitive_impl<fully_connected>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::fully_connected_ternocl_int2)

    const ocl_engine* _engine = nullptr;
    TernoclInt2Packed _own;  // fallback when the node-id lookup misses
    size_t _N = 0;
    size_t _K = 0;
    int _postop = 0;
    size_t _other_dep = 0;
    bool _out_f32 = false;
    size_t _had_block = 0;
    bool _integrated = false;

    // GEMV for M = 1, 2, <= 4, <= 8; M-tiled for M <= 16, <= 32, < 64, >= 64.
    struct Launch {
        kernel::ptr k;
        GemvTile g{};
        MtTile t{};
    };
    std::array<Launch, 8> _launch;
    std::array<Launch, 4> _bitcos_launch;     // GEMV, SGM 1, 2, 4, 8
    std::array<Launch, 4> _bitcos_mt_launch;  // M-tiled, M <= 16, <= 32, < 64, >= 64
    kernel::ptr _fwht;

    fully_connected_ternocl_int2() : parent("ternocl_int2") {}
    fully_connected_ternocl_int2(const ocl_engine& engine, TernoclInt2Packed own, size_t N, size_t K, int postop,
                                 size_t other_dep, bool out_f32, size_t had_block)
        : parent("ternocl_int2"),
          _engine(&engine),
          _own(std::move(own)),
          _N(N),
          _K(K),
          _postop(postop),
          _other_dep(other_dep),
          _out_f32(out_f32),
          _had_block(had_block),
          _integrated(engine.get_device_info().dev_type == device_type::integrated_gpu) {}

    std::unique_ptr<primitive_impl> clone() const override {
        // Shares the kernel handles like the stock OCL impls: OpenVINO clones
        // cached impls on every shape update, and rebuilding costs tens of us per FC.
        return std::make_unique<fully_connected_ternocl_int2>(*this);
    }

    bool is_cpu() const override {
        return false;
    }
    bool is_onednn() const override {
        return false;
    }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void set_arguments_impl(typed_primitive_inst<fully_connected>&) override {}
    void set_arguments_impl(typed_primitive_inst<fully_connected>&, kernel_arguments_data&) override {}

    std::string epi_opts() const {
        return " -DPOSTOP=" + std::to_string(_postop) + (_out_f32 ? " -DOUT_F32" : "");
    }

    static size_t launch_class(size_t M) {
        if (M <= 8)
            return M == 1 ? 0 : (M == 2 ? 1 : (M <= 4 ? 2 : 3));
        return M <= 16 ? 4 : (M <= 32 ? 5 : (M < 64 ? 6 : 7));
    }

    Launch& get_launch(size_t M) {
        auto& l = _launch[launch_class(M)];
        if (l.k)
            return l;
        std::string opts = "-cl-std=CL3.0";
        if (M <= 8) {
            const int sgm = M == 1 ? 1 : (M == 2 ? 2 : (M <= 4 ? 4 : 8));
            l.g = gemv_tile(_K, _N, _integrated);
            opts += " -DSGM=" + std::to_string(sgm) + " -DNSG_N=" + std::to_string(l.g.wgn / 16) +
                    " -DLS=" + std::to_string(l.g.ls) + " -DU=" + std::to_string(l.g.u) + " -DPF=0";
            l.k = make_kernel(*_engine, get_program(*_engine, kTernoclUpcvtSource, opts + epi_opts()),
                              "int2_fp16_upcvt_gemm");
        } else {
            l.t = mt_tile(_K, _N, M, _integrated);
            opts += " -DMT_M=" + std::to_string(l.t.mt_m) + " -DMT_N=" + std::to_string(l.t.mt_n) +
                    " -DWG_M=" + std::to_string(l.t.wg_m) + " -DWG_N=" + std::to_string(l.t.wg_n) +
                    " -cl-intel-256-GRF-per-thread";
            l.k = make_kernel(*_engine, get_program(*_engine, kTernoclUpcvtSource, opts + epi_opts()),
                              "int2_fp16_upcvt_gemm_mt");
        }
        if (std::getenv("OV_TERNOCL_INT2_CFG_DEBUG") != nullptr)
            std::cerr << "[ternocl-int2] K=" << _K << " N=" << _N << " M-class " << launch_class(M) << ": "
                      << opts << epi_opts() << std::endl;
        return l;
    }

    static int gemv_sgm(size_t M) {
        return M == 1 ? 1 : (M == 2 ? 2 : (M <= 4 ? 4 : 8));
    }

    Launch& get_bitcos_launch(size_t M) {
        const int sgm = gemv_sgm(M);
        auto& l = _bitcos_launch[sgm == 1 ? 0 : (sgm == 2 ? 1 : (sgm == 4 ? 2 : 3))];
        if (l.k)
            return l;
        l.g = bitcos_tile(_K, _N, sgm);
        const std::string opts = "-cl-std=CL3.0 -DSGM=" + std::to_string(sgm) + " -DNSG_N=" +
                                 std::to_string(l.g.wgn / 16) + " -DLS=" + std::to_string(l.g.ls) + epi_opts();
        l.k = make_kernel(*_engine, get_program(*_engine, kTernoclBitcosSource, opts), "bitcos_fp16_upcvt_gemv");
        if (std::getenv("OV_TERNOCL_INT2_CFG_DEBUG") != nullptr)
            std::cerr << "[ternocl-bitcos] K=" << _K << " N=" << _N << " SGM " << sgm << ": " << opts << std::endl;
        return l;
    }

    Launch& get_bitcos_mt_launch(size_t M) {
        const int band = M <= 16 ? 0 : (M <= 32 ? 1 : (M < 64 ? 2 : 3));
        auto& l = _bitcos_mt_launch[band];
        if (l.k)
            return l;
        l.t = bitcos_mt_tile(_K, _N, M);
        const std::string opts = "-cl-std=CL3.0 -DMT_M=" + std::to_string(l.t.mt_m) + " -DMT_N=" +
                                 std::to_string(l.t.mt_n) + " -DWG_M=" + std::to_string(l.t.wg_m) + " -DWG_N=" +
                                 std::to_string(l.t.wg_n) + " -cl-intel-256-GRF-per-thread" + epi_opts();
        l.k = make_kernel(*_engine, get_program(*_engine, kTernoclBitcosSource, opts), "bitcos_fp16_upcvt_gemm_mt");
        if (std::getenv("OV_TERNOCL_INT2_CFG_DEBUG") != nullptr)
            std::cerr << "[ternocl-bitcos] K=" << _K << " N=" << _N << " M-band " << band << ": " << opts << std::endl;
        return l;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<fully_connected>& instance) override {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        const auto& params = instance.get_impl_params();
        const size_t M = ov::shape_size(params->output_layouts[0].get_shape()) / _N;

        TernoclInt2Packed* pk = &_own;
        {
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            auto it = ternocl_packed_cache().find(params->desc->id);
            if (it != ternocl_packed_cache().end())
                pk = &it->second;
        }

        memory::cptr in = instance.input_memory_ptr(0);
        std::vector<event::ptr> deps = events;
        if (_had_block != 0) {
            // Rotated-basis checkpoint: the GEMM consumes H_1024(s * x) / 32.
            if (!pk->had_input || M > pk->had_rows) {
                const auto hl = layout{ov::PartialShape{static_cast<int64_t>(M), static_cast<int64_t>(_K)},
                                       data_types::f16, format::bfyx};
                pk->had_input = network.get_engine().allocate_memory(hl, allocation_type::usm_device, false);
                pk->had_rows = M;
            }
            if (!_fwht)
                _fwht = make_kernel(*_engine, get_program(*_engine, kTernoclFwhtSource, "-cl-std=CL3.0"),
                                    "hadamard_fwht_1024");
            kernel_arguments_desc d;
            d.workGroups.global = {M * (_K / 1024) * 128, 1, 1};
            d.workGroups.local = {128, 1, 1};
            d.arguments = {{argument_desc::Types::INPUT, 0},
                           {argument_desc::Types::INPUT, 1},
                           {argument_desc::Types::OUTPUT, 0},
                           {argument_desc::Types::SCALAR, 0},
                           {argument_desc::Types::SCALAR, 1}};
            scalars_desc sc(2);
            sc[0].t = sc[1].t = scalar_desc::Types::INT32;
            sc[0].v.s32 = static_cast<int32_t>(_K);
            sc[1].v.s32 = pk->had_signs ? 1 : 0;
            kernel_arguments_data a;
            a.inputs = {in, pk->had_signs ? memory::cptr(pk->had_signs) : in};
            a.outputs = {pk->had_input};
            a.scalars = &sc;
            stream.set_arguments(*_fwht, d, a);
            deps = {stream.enqueue_kernel(*_fwht, d, a, deps, false)};
            in = pk->had_input;
        }

        // Unused Other / Bias / slice-rank slots are never read, but need a valid buffer.
        memory::cptr other = (_postop == 1 || _postop == 2) ? instance.dep_memory_ptr(_other_dep) : in;
        memory::cptr bias = _postop == 3 ? instance.bias_memory() : in;
        if (pk->bitcos) {
            const bool mt = M > 8;
            auto& l = mt ? get_bitcos_mt_launch(M) : get_bitcos_launch(M);
            const size_t wgn = static_cast<size_t>(l.g.wgn), ls = static_cast<size_t>(l.g.ls);
            memory::cptr sr = in;
            if (!mt && ls > 1) {
                sr = pk->bitcos_sr[ls == 2 ? 0 : (ls == 4 ? 1 : 2)];
                OPENVINO_ASSERT(sr, "[GPU] ternocl bitcos: no slice ranks for LS=", ls);
            }
            kernel_arguments_desc d;
            if (mt) {
                const size_t tn = static_cast<size_t>(l.t.mt_n * l.t.wg_n), tm = static_cast<size_t>(l.t.mt_m * l.t.wg_m);
                d.workGroups.local = {16 * static_cast<size_t>(l.t.wg_n * l.t.wg_m), 1, 1};
                d.workGroups.global = {ceil_div(_N, tn) * d.workGroups.local[0], ceil_div(M, tm), 1};
            } else {
                d.workGroups.local = {wgn * ls, 1, 1};
                d.workGroups.global = {ceil_div(_N, wgn) * wgn * ls, ceil_div(M, static_cast<size_t>(gemv_sgm(M))), 1};
            }
            // A, B, S, SR, C, Other, Bias, M, N, K
            d.arguments = {{argument_desc::Types::INPUT, 0},  {argument_desc::Types::INPUT, 1},
                           {argument_desc::Types::INPUT, 2},  {argument_desc::Types::INPUT, 3},
                           {argument_desc::Types::OUTPUT, 0}, {argument_desc::Types::INPUT, 4},
                           {argument_desc::Types::INPUT, 5},  {argument_desc::Types::SCALAR, 0},
                           {argument_desc::Types::SCALAR, 1}, {argument_desc::Types::SCALAR, 2}};
            scalars_desc sc(3);
            for (auto& s : sc)
                s.t = scalar_desc::Types::INT32;
            sc[0].v.s32 = static_cast<int32_t>(M);
            sc[1].v.s32 = static_cast<int32_t>(_N);
            sc[2].v.s32 = static_cast<int32_t>(_K);
            kernel_arguments_data a;
            a.inputs = {in, pk->bitcos, pk->scales, sr, other, bias};
            a.outputs = {instance.output_memory_ptr(0)};
            a.scalars = &sc;
            stream.set_arguments(*l.k, d, a);
            return stream.enqueue_kernel(*l.k, d, a, deps, instance.is_output());
        }

        auto& l = get_launch(M);
        kernel_arguments_desc d;
        if (M <= 8) {
            const size_t sgm = M == 1 ? 1 : (M == 2 ? 2 : (M <= 4 ? 4 : 8));
            const size_t wgn = static_cast<size_t>(l.g.wgn);
            d.workGroups.local = {wgn * static_cast<size_t>(l.g.ls), 1, 1};
            d.workGroups.global = {ceil_div(_N, wgn) * d.workGroups.local[0], ceil_div(M, sgm), 1};
        } else {
            const size_t tn = static_cast<size_t>(l.t.mt_n * l.t.wg_n), tm = static_cast<size_t>(l.t.mt_m * l.t.wg_m);
            d.workGroups.local = {16 * static_cast<size_t>(l.t.wg_n * l.t.wg_m), 1, 1};
            d.workGroups.global = {ceil_div(_N, tn) * d.workGroups.local[0], ceil_div(M, tm), 1};
        }
        // A, B, S, C, Other, Bias, M, N, K
        d.arguments = {{argument_desc::Types::INPUT, 0},  {argument_desc::Types::INPUT, 1},
                       {argument_desc::Types::INPUT, 2},  {argument_desc::Types::OUTPUT, 0},
                       {argument_desc::Types::INPUT, 3},  {argument_desc::Types::INPUT, 4},
                       {argument_desc::Types::SCALAR, 0}, {argument_desc::Types::SCALAR, 1},
                       {argument_desc::Types::SCALAR, 2}};
        scalars_desc sc(3);
        for (auto& s : sc)
            s.t = scalar_desc::Types::INT32;
        sc[0].v.s32 = static_cast<int32_t>(M);
        sc[1].v.s32 = static_cast<int32_t>(_N);
        sc[2].v.s32 = static_cast<int32_t>(_K);
        kernel_arguments_data a;
        a.inputs = {in, pk->weights, pk->scales, other, bias};
        a.outputs = {instance.output_memory_ptr(0)};
        a.scalars = &sc;
        stream.set_arguments(*l.k, d, a);
        return stream.enqueue_kernel(*l.k, d, a, deps, instance.is_output());
    }

public:
    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg,
                                                  const kernel_impl_params& impl_params) {
        auto& prog = arg.get_program();
        auto& engine = prog.get_engine();
        auto& stream = prog.get_stream();
        const bool dbg = std::getenv("OV_TERNOCL_INT2_DEBUG") != nullptr;

        const auto wei_shape = arg.weights().get_output_layout(false).get_shape();
        const size_t N = wei_shape[0];
        const size_t K = wei_shape[1];
        const auto& desc = arg.get_primitive();

        size_t other_dep = 0;
        const int postop = ternocl_int2_postop(impl_params.fused_desc, desc->bias.is_valid(), &other_dep);
        OPENVINO_ASSERT(postop >= 0, "[GPU] ternocl int2: ", arg.id(), " has no folded epilogue for its fused chain");
        const bool out_f32 = impl_params.output_layouts[0].data_type == data_types::f32;

        TernoclInt2Packed own;
        {
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            auto it = ternocl_packed_cache().find(arg.id());
            if (it != ternocl_packed_cache().end())
                own = it->second;
        }
        if (!own.weights) {
            // Dependencies are input, weights, [bias], scale, [zero point].
            const size_t scale_dep_idx = desc->bias.is_valid() ? 3 : 2;
            int32_t zp = 0;
            if (desc->decompression_zero_point_scalar.has_value()) {
                zp = static_cast<int32_t>(std::lround(desc->decompression_zero_point_scalar.value()));
            } else if (desc->decompression_zero_point.is_valid()) {
                const auto* zp_node = const_source(&arg.get_dependency(scale_dep_idx + 1));
                OPENVINO_ASSERT(zp_node != nullptr, "[GPU] ternocl int2: zero point is not constant");
                zp = static_cast<int32_t>(std::lround(read_scalar(zp_node->as<data>().get_attached_memory_ptr(), stream)));
            }

            // Blocking copies: mapping a large device constant can expose it before it is resident.
            auto wei_mem = arg.weights().as<data>().get_attached_memory_ptr();
            OPENVINO_ASSERT(wei_mem->size() >= N * K / 4, "[GPU] ternocl int2: weight buffer ", wei_mem->size(),
                            " B is smaller than the dense ", N * K / 4, " B");
            std::vector<uint8_t> wei_host(wei_mem->size());
            wei_mem->copy_to(stream, wei_host.data(), true);

            // One weight copy: BITCOS (decode GEMV + M-tiled prefill) or int2.
            if (bitcos_enabled() && K % 64 == 0) {
                const BitcosHost bh = pack_bitcos(wei_host.data(), N, K, zp);
                auto upload = [&](const std::vector<uint32_t>& v) {
                    auto m = engine.allocate_memory(
                        layout{ov::PartialShape{static_cast<int64_t>(v.size())}, data_types::i32, format::bfyx},
                        allocation_type::usm_device, false);
                    m->copy_from(stream, v.data(), true);
                    return m;
                };
                own.bitcos = upload(bh.buf);
                for (size_t i = 0; i < 3; ++i)
                    if (!bh.sr[i].empty())
                        own.bitcos_sr[i] = upload(bh.sr[i]);
                if (dbg)
                    std::cerr << "[ternocl-bitcos] " << arg.id() << " z=" << 1.0 - static_cast<double>(bh.nnz) / (N * K)
                              << " " << 32.0 * bh.buf.size() / (N * K) << " bits/weight" << std::endl;
            } else {
                std::vector<uint32_t> packed((K / kTernoclPackFactor) * N);
                pack_weights(wei_host.data(), packed.data(), N, K, zp);
                own.weights = engine.allocate_memory(
                    layout{ov::PartialShape{static_cast<int64_t>(K / kTernoclPackFactor), static_cast<int64_t>(N)},
                           data_types::i32, format::bfyx},
                    allocation_type::usm_device, false);
                own.weights->copy_from(stream, packed.data(), true);
            }

            // OpenVINO keeps scales per output channel, [N, groups]; the kernel wants [groups, N].
            const size_t groups = K / kTernoclGroupSize;
            const auto* scale_node = const_source(&arg.get_dependency(scale_dep_idx));
            OPENVINO_ASSERT(scale_node != nullptr, "[GPU] ternocl int2: decompression scale is not constant");
            auto scale_mem = scale_node->as<data>().get_attached_memory_ptr();
            OPENVINO_ASSERT(scale_mem->get_layout().data_type == data_types::f16,
                            "[GPU] ternocl int2: decompression scale must be f16");
            std::vector<uint16_t> scale_src(scale_mem->size() / sizeof(uint16_t));
            scale_mem->copy_to(stream, scale_src.data(), true);
            // const_source() skipped any reorder, so this is the constant's own [N, groups] layout.
            const auto scale_shape = scale_mem->get_layout().get_shape();
            const bool n_major = scale_shape.size() >= 2 && scale_shape[0] == N;
            std::vector<uint16_t> scale_host(groups * N);
            for (size_t g = 0; g < groups; ++g)
                for (size_t n = 0; n < N; ++n)
                    scale_host[g * N + n] = n_major ? scale_src[n * groups + g] : scale_src[g * N + n];
            own.scales = engine.allocate_memory(
                layout{ov::PartialShape{static_cast<int64_t>(groups), static_cast<int64_t>(N)}, data_types::f16,
                       format::bfyx},
                allocation_type::usm_device, false);
            own.scales->copy_from(stream, scale_host.data(), true);

            if (desc->hadamard_block != 0 && !desc->hadamard_signs.empty()) {
                OPENVINO_ASSERT(desc->hadamard_signs.size() == K, "[GPU] ternocl int2: hadamard signs length mismatch");
                own.had_signs = engine.allocate_memory(
                    layout{ov::PartialShape{static_cast<int64_t>(K)}, data_types::i8, format::bfyx},
                    allocation_type::usm_device, false);
                own.had_signs->copy_from(stream, desc->hadamard_signs.data(), true);
            }
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            // A second impl for a node that is already executing must not swap its buffers.
            own = ternocl_packed_cache().try_emplace(arg.id(), own).first->second;
        }
        if (dbg)
            std::cerr << "[ternocl-int2] create " << arg.id() << " N=" << N << " K=" << K << " postop=" << postop
                      << " out_f32=" << out_f32 << " hadamard=" << desc->hadamard_block << std::endl;

        return std::make_unique<fully_connected_ternocl_int2>(downcast<const ocl_engine>(engine), own, N, K, postop,
                                                              other_dep, out_f32, desc->hadamard_block);
    }
};

std::unique_ptr<primitive_impl> TernoclInt2FCImplementationManager::create_impl(const program_node& node,
                                                                                const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    return fully_connected_ternocl_int2::create(static_cast<const fully_connected_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

#endif  // OV_GPU_WITH_OCL_RT
