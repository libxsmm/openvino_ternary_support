// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// int2 x int8 -> int32 DPAS GEMM with fp16 activations, built on the XeTLA
// header-only template library.
//
// Takes the same VNNI16-packed B and fp16 ScaleB as the up-convert kernel, but
// quantizes the activations to int8 and uses the int2 x int8 DPAS, which has
// twice the compute peak of the fp16 path.

#include "xetla_int2_gemv.hpp"

#include <cfloat>
#include <sycl/sycl.hpp>

#include "xetla.hpp"

using namespace gpu::xetla;
using namespace gpu::xetla::subgroup;
using namespace sycl;

using namespace sycl;

namespace {

// Group size (K dimension) for both A and B scales.
static constexpr int kScaleGS = 128;

// ---------------------------------------------------------------------------
// Per-row, per-K-group absmax reduction. Writes scale_A[g, row] = 127/absmax
// in fp16. Mirrors `absmax_to_inv_scale_reduction` from the reference test.
// twidth must equal the K-group size (kScaleGS). theight is rows per group.
template <typename in_dtype, typename out_dtype, int twidth, int theight>
struct absmax_to_inv_scale_fp16_reduction {
  static KERNEL_FUNC inline void run(
      sycl::nd_item<1> it, in_dtype* a_base, int swidth, int sheight,
      int spitch, int scale_bs, out_dtype* c_base) {
    const int group_id = static_cast<int>(it.get_group(0));
    const int local_id = static_cast<int>(it.get_local_id(0));
    const int total_rows = sheight;
    const int total_cols = swidth;
    const int tiles_per_row = ((total_cols + twidth - 1) / twidth) / scale_bs;
    const int group_base = group_id * theight;
    const int row = group_base + local_id;
    if (row >= total_rows)
      return;

    using row_tile_desc_t =
        tile_desc_t<twidth, 1, twidth, 1, reg_layout::tiled>;
    using row_tile_t = tile_t<in_dtype, row_tile_desc_t>;
    using row_payload_t = mem_payload_t<
        mem_desc_t<in_dtype, mem_layout::row_major, mem_space::global>,
        row_tile_desc_t, msg_type_v<row_tile_desc_t, mem_space::global>,
        gpu_arch::Xe>;
    using row_prefetch_payload_t = subgroup::prefetch_payload_t<
        mem_desc_t<in_dtype, mem_layout::row_major, mem_space::global>,
        row_tile_desc_t, twidth, gpu_arch::Xe>;

    row_tile_t row_tile;
    row_payload_t row_payload(a_base + row * spitch, swidth, 1, spitch, 0, 0);
    row_prefetch_payload_t row_payload_prefetch(
        a_base + row * spitch, swidth, 1, spitch, 0, 0);

    xetla_vector<float, twidth> src_reg_f32;
    xetla_mask<row_tile_t::tile_size_x> mask_max;

    for (int igs = 0; igs < scale_bs; igs++) {
      xetla_vector<float, twidth> acc_vec(FLT_EPSILON);
#pragma unroll
      for (int t = 0; t < tiles_per_row; ++t) {
        subgroup::tile_prefetch<cache_hint::cached, cache_hint::cached>(
            row_payload_prefetch);
        row_payload_prefetch.template update_tdesc<tdesc_update_dir::x_dir>(
            row_tile_t::tile_size_x);
      }
#pragma unroll
      for (int t = 0; t < tiles_per_row; ++t) {
        subgroup::tile_load<cache_hint::cached, cache_hint::cached>(
            row_tile, row_payload);
        row_payload.template update_tdesc<tdesc_update_dir::x_dir>(
            row_tile_t::tile_size_x);
        src_reg_f32 = xetla_abs<float, row_tile_t::tile_size_x>(
            xetla_cvt<float, in_dtype, row_tile_t::tile_size_x>(
                row_tile.reg.xetla_select<row_tile_t::tile_size_x, 1>(0)));
        mask_max = src_reg_f32 > acc_vec;
        acc_vec.xetla_merge(src_reg_f32, acc_vec, mask_max);
      }

      float row_max =
          xetla_reduce<float, float, twidth, reduce_op::max>(acc_vec);
      // Store 127/absmax cast to fp16. Matches the reference test's host-side
      // formula and the GEMM's scaleA expectation (forward-quant factor).
      float scale_val = 127.0f / row_max;
      c_base[igs * sheight + row] = static_cast<out_dtype>(scale_val);
    }
  }
};

template <typename in_dtype, typename out_dtype, int twidth, int theight, int POSTOP>
class absmax_to_inv_scale_fp16_kernel;

// ---------------------------------------------------------------------------
// Single fixed-tile DPAS GEMM config (the user-requested tuning):
//   wg_m=64, sg_m=8, wg_n=256, sg_n=128, sg_k=64, mma_xmx_m=8,
//   use_external_scale_a=true (kslicing's internal SLM scale_a is hardcoded
//   to float, so the fp16-scales path requires external scale_a).
// POSTOP selects the epilogue, matching the up-convert path: 1 = silu * other,
// 2 = + other, 0 = plain store.
template <int POSTOP = 0>
struct DpasFp16Cfg {
  static constexpr size_t wg_tile_m = 64;
  static constexpr size_t wg_tile_n = 256;
  static constexpr size_t sg_tile_m = 8;
  static constexpr size_t sg_tile_n = 128;
  static constexpr size_t sg_tile_k = 64;
  static constexpr int mma_xmx_m = 8;
  static constexpr uint32_t global_kslicing = 1;
  static constexpr uint32_t local_kslicing = 1;
  static constexpr bool use_external_scale_a = true;
  static constexpr uint32_t prefetch_distance = 4;
  static constexpr uint32_t periodic_sync_interval = 1;

  static constexpr gpu_arch arch_tag = gpu_arch::Xe;

  using fp16_t = gpu::xetla::fp16;
  using data_type_a = fp16_t;
  using data_type_b = int2x16;
  using data_type_c = fp16_t;
  using data_type_acc = int32_t;
  using scale_dtype = fp16_t;

  using data_type_mma_a = int8_t;
  using data_type_mma_b = int8_t;

  using tile_shape = gpu::xetla::group::tile_shape_t<
      wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
  using mem_desc_a_t = gpu::xetla::mem_desc_t<
      data_type_a, mem_layout::row_major, mem_space::global>;
  using mem_desc_b_t = gpu::xetla::mem_desc_t<
      data_type_b, mem_layout::row_major, mem_space::global>;
  using mem_desc_c_t = gpu::xetla::mem_desc_t<
      data_type_c, mem_layout::row_major, mem_space::global>;
  using compute_attr = gpu::xetla::group::compute_attr_t<
      data_type_mma_a, data_type_mma_b, int32_t>;
  using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<
      sg_tile_k, prefetch_distance, periodic_sync_interval>;
  using compute_policy = gpu::xetla::group::compute_policy_int2_fp16_dpas_xmx<
      compute_attr, perf_tuning_knob, scale_dtype, scale_dtype, mma_xmx_m,
      arch_tag>;
  using gemm_t = gpu::xetla::group::gemm_t<
      compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using silu_t = gpu::xetla::subgroup::silu_op_t;
  using prod_t = gpu::xetla::subgroup::elemwise_reduce_op_t<
      gpu::xetla::reduce_op::prod, fp16_t, arch_tag>;
  using sum_t = gpu::xetla::subgroup::elemwise_reduce_op_t<
      gpu::xetla::reduce_op::sum, fp16_t, arch_tag>;
  using tile_op_t = std::conditional_t<
      POSTOP == 1, gpu::xetla::subgroup::chained_tile_op_t<silu_t, prod_t>,
      gpu::xetla::subgroup::chained_tile_op_t<sum_t>>;
  using epilogue_policy_t = std::conditional_t<
      POSTOP != 0,
      gpu::xetla::group::epilogue_policy_tile_op<tile_op_t, arch_tag>,
      gpu::xetla::group::epilogue_policy_default<arch_tag>>;
  using epilogue_t = gpu::xetla::group::epilogue_t<
      epilogue_policy_t, tile_shape, mem_desc_c_t>;
  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;
  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      gpu::xetla::kernel::dispatch_policy_int2_fp16_dpas_kslicing<
          group_swizzle, global_kslicing, local_kslicing,
          use_external_scale_a>,
      gemm_t, epilogue_t>;
};

template <int POSTOP>
class int2_fp16_dpas_kernel_tag;

// ---------------------------------------------------------------------------
// Per-process scratch (scale_A, Acc, Cnt). Reuses one allocation, grows on
// demand.
struct DpasScratch {
  sycl::half* scale_a = nullptr;
  int32_t* acc = nullptr;
  uint32_t* cnt = nullptr;
  size_t scale_a_bytes = 0;
  size_t acc_bytes = 0;
  size_t cnt_bytes = 0;
};

DpasScratch& get_dpas_scratch(
    sycl::queue& q, size_t need_scale_a, size_t need_acc, size_t need_cnt) {
  static DpasScratch s;
  if (s.scale_a_bytes < need_scale_a) {
    if (s.scale_a)
      sycl::free(s.scale_a, q.get_context());
    s.scale_a = static_cast<sycl::half*>(sycl::aligned_alloc_device(
        256, need_scale_a, q.get_device(), q.get_context()));
    s.scale_a_bytes = need_scale_a;
  }
  if (s.acc_bytes < need_acc) {
    if (s.acc)
      sycl::free(s.acc, q.get_context());
    s.acc = static_cast<int32_t*>(sycl::aligned_alloc_device(
        256, need_acc, q.get_device(), q.get_context()));
    s.acc_bytes = need_acc;
    q.memset(s.acc, 0, need_acc);
  }
  if (s.cnt_bytes < need_cnt) {
    if (s.cnt)
      sycl::free(s.cnt, q.get_context());
    s.cnt = static_cast<uint32_t*>(sycl::aligned_alloc_device(
        256, need_cnt, q.get_device(), q.get_context()));
    s.cnt_bytes = need_cnt;
    q.memset(s.cnt, 0, need_cnt);
  }
  return s;
}

} // namespace

// Public entry point. M, K, N must satisfy K % kScaleGS == 0; otherwise the
// caller should fall back to the upcvt path. scale_gs (= K/kScaleGS) is set
// at runtime in gemm_arg.
template <int POSTOP>
sycl::event int2_fp16_dpas_gemm_run(
    sycl::queue& q, const size_t M, const size_t N, const size_t K,
    sycl::half* A, int32_t* B, sycl::half* C, sycl::half* ScaleB,
    sycl::half* Other = nullptr) {
  using Cfg = DpasFp16Cfg<POSTOP>;
  using gemm_op_t = typename Cfg::gemm_op_t;
  using fp16_t = typename Cfg::fp16_t;

  const uint32_t scale_gs = static_cast<uint32_t>(K / kScaleGS);

  size_t need_scale_a = M * scale_gs * sizeof(sycl::half);
  size_t acc_elems = gemm_op_t::get_acc_buf_size(M, N);
  size_t cnt_elems = gemm_op_t::get_cnt_buf_size(M, N);
  size_t need_acc = acc_elems * sizeof(int32_t);
  size_t need_cnt = cnt_elems * sizeof(uint32_t);
  auto& sc = get_dpas_scratch(q, need_scale_a, need_acc, need_cnt);

  // ----- 1) Compute scaleA = 127/absmax per (row, K-group), in fp16 -----
  {
    constexpr uint32_t TILE_X = kScaleGS; // == K-group size
    constexpr uint32_t TILE_Y = 32;
    const size_t local_size = TILE_Y;
    const size_t num_groups = (M + TILE_Y - 1) / TILE_Y;
    const size_t global_size = num_groups * local_size;
    auto* a_dev = A;
    auto* scale_dev = sc.scale_a;
    const int rows = static_cast<int>(M);
    const int cols = static_cast<int>(K);
    const int sgs = static_cast<int>(scale_gs);
    auto e_scale = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class absmax_to_inv_scale_fp16_kernel<
          fp16_t, fp16_t, TILE_X, TILE_Y, POSTOP>>(
          sycl::nd_range<1>(
              sycl::range<1>(global_size), sycl::range<1>(local_size)),
          [=](sycl::nd_item<1> it) {
            absmax_to_inv_scale_fp16_reduction<
                fp16_t, fp16_t, TILE_X, TILE_Y>::
                run(it, reinterpret_cast<fp16_t*>(a_dev), cols, rows, cols,
                    sgs, reinterpret_cast<fp16_t*>(scale_dev));
          });
    });
    // Note: do NOT wait on e_scale here. The XPU queue from
    // The queue is in-order, so the subsequent GEMM submission is naturally
    // serialized after this kernel. An explicit event.wait() would additionally
    // be invalid while a command graph is being recorded.
    (void)e_scale;
  }

  // ----- 2) Run the int2 x fp16 DPAS GEMM with external scaleA -----
  uint32_t lda = static_cast<uint32_t>(K);
  uint32_t ldb = static_cast<uint32_t>(N);
  uint32_t ldc = static_cast<uint32_t>(N);
  uint32_t scale_a_ld = static_cast<uint32_t>(M);
  uint32_t scale_b_ld = static_cast<uint32_t>(N);

  auto make_args = [&]() {
    if constexpr (POSTOP == 0) {
      return typename gemm_op_t::arguments_t(
          static_cast<uint32_t>(M), static_cast<uint32_t>(K),
          static_cast<uint32_t>(N), reinterpret_cast<fp16_t*>(A), lda,
          reinterpret_cast<int2x16*>(B), ldb, reinterpret_cast<fp16_t*>(C), ldc,
          reinterpret_cast<fp16_t*>(sc.scale_a), scale_a_ld,
          reinterpret_cast<fp16_t*>(ScaleB), scale_b_ld, sc.acc, sc.cnt);
    } else {
      using tile_op_t = typename Cfg::tile_op_t;
      using epilogue_t = typename Cfg::epilogue_t;
      using reduce_t =
          std::conditional_t<POSTOP == 1, typename Cfg::prod_t, typename Cfg::sum_t>;
      typename reduce_t::arguments_t other_args(
          reinterpret_cast<fp16_t*>(Other),
          {static_cast<uint32_t>(N), static_cast<uint32_t>(M),
           static_cast<uint32_t>(N)});
      typename tile_op_t::arguments_t tile_args;
      // silu occupies slot 0 when present, so the reduce op shifts to slot 1.
      tile_args.template set<POSTOP == 1 ? 1 : 0>(other_args);
      typename epilogue_t::arguments_t epilogue_args(tile_args);
      return typename gemm_op_t::arguments_t(
          static_cast<uint32_t>(M), static_cast<uint32_t>(K),
          static_cast<uint32_t>(N), reinterpret_cast<fp16_t*>(A), lda,
          reinterpret_cast<int2x16*>(B), ldb, reinterpret_cast<fp16_t*>(C), ldc,
          reinterpret_cast<fp16_t*>(sc.scale_a), scale_a_ld,
          reinterpret_cast<fp16_t*>(ScaleB), scale_b_ld, sc.acc, sc.cnt,
          epilogue_args);
    }
  };
  typename gemm_op_t::arguments_t gemm_arg = make_args();
  gemm_arg.scale_gs = scale_gs;

  if (!gemm_op_t::can_implement(gemm_arg)) {
    std::cout << "int2_fp16_dpas: arguments not supported (M=" << M
              << " N=" << N << " K=" << K << " scale_gs=" << scale_gs << ")\n";
    std::exit(1);
  }
  sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);

  return q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<int2_fp16_dpas_kernel_tag<POSTOP>>(
        nd_range, [=](sycl::nd_item<3> item) KERNEL_MAIN {
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(item, gemm_arg);
        });
  });
}

namespace ov {
namespace intel_gpu {
namespace xetla_int2 {

sycl::event gemv_f16_dpas(sycl::queue& q, size_t M, size_t N, size_t K,
                          void* A, void* B, void* C, void* ScaleB, int postop,
                          void* other) {
  auto* a = static_cast<sycl::half*>(A);
  auto* b = static_cast<int32_t*>(B);
  auto* c = static_cast<sycl::half*>(C);
  auto* s = static_cast<sycl::half*>(ScaleB);
  auto* o = static_cast<sycl::half*>(other);
  switch (postop) {
    case 1:
      return int2_fp16_dpas_gemm_run<1>(q, M, N, K, a, b, c, s, o);
    case 2:
      return int2_fp16_dpas_gemm_run<2>(q, M, N, K, a, b, c, s, o);
    default:
      return int2_fp16_dpas_gemm_run<0>(q, M, N, K, a, b, c, s);
  }
}

}  // namespace xetla_int2
}  // namespace intel_gpu
}  // namespace ov
