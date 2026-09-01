// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// int2 weight-only GEMM/GEMV with fp16 activations, built on the XeTLA
// header-only template library.
//
//   A      : fp16, row-major, [M, K]
//   B      : int2x16, K-packed, [K/16, N] (uint32 per 16 K-rows of one N col)
//   ScaleB : fp16, row-major, [K/gs, N]   (gs = 128)
//   C      : fp16, row-major, [M, N]
// int2 codes use {0, 1, 3}: 0 -> 0, 1 -> +1, 3 -> -1 (code 2 is forbidden).

#include "xetla_int2_gemv.hpp"

#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
#include <tuple>
#include <utility>

#ifdef OV_GPU_WITH_ZE_RT
#    include <level_zero/ze_api.h>
#    include <sycl/ext/oneapi/backend/level_zero.hpp>
#    include "ze/ze_common.hpp"
#endif

#include "xetla.hpp"

using namespace gpu::xetla;
using namespace sycl;

namespace {

// Fixed K-group for the fp16-scales upcvt kernel (matches the xetla test).
static constexpr int kScaleGS = 128;

// XeTLA's own sigmoid_op_t predates chained_tile_op_t and takes no arguments_t,
// so it cannot be chained. This mirrors silu_op_t without the final multiply.
struct sigmoid_chain_op_t {
  struct arguments_t {};
  template <typename matAcc_t, typename coord_t>
  __XETLA_API KERNEL_FUNC void operator()(
      matAcc_t& matAcc, [[maybe_unused]] const coord_t& coord,
      [[maybe_unused]] const arguments_t& args,
      [[maybe_unused]] uint32_t slm_base = 0,
      [[maybe_unused]] uint32_t nbarrier_base = 0) {
    constexpr int elems = matAcc_t::tile_desc::block_elems;
    constexpr int rounds = matAcc_t::tile_desc::tile_elems / elems;
#pragma unroll
    for (int i = 0; i < rounds; ++i) {
      auto sub_vec = matAcc.reg.xetla_select<elems, 1>(elems * i);
      sub_vec = xetla_sigmoid<typename matAcc_t::dtype, elems>(sub_vec);
    }
    constexpr int remaining_elems = matAcc_t::tile_desc::tile_elems % elems;
    if constexpr (remaining_elems != 0) {
      auto sub_vec = matAcc.reg.xetla_select<remaining_elems, 1>(
          elems * (matAcc_t::tile_elems / elems));
      sub_vec = xetla_sigmoid<typename matAcc_t::dtype, remaining_elems>(sub_vec);
    }
  }
};

// Unique kernel-name tag to avoid mangled-name collisions between
// (XT, WGM, WGN, SGM, SGN, SGK, KS, LS, kUnaligned) instantiations.
template <
    typename XT, int WGM, int WGN, int SGM, int SGN, int SGK, int KS, int LS,
    bool kUnaligned, typename CT, int POSTOP>
class int2_upcvt_kernel_tag;

// XT is the activation / scale element type: gpu::xetla::fp16 or
// gpu::xetla::bf16. CT is the output type (fp32 for lm_head logits). POSTOP
// selects the fused epilogue: 0 none, 1 silu*other, 2 +other.
template <
    typename XT, int WGM, int WGN, int SGM, int SGN, int SGK, int KS, int LS,
    bool kUnaligned, typename CT = XT, int POSTOP = 0>
sycl::event int2_upcvt_gemm_impl(
    sycl::queue& queue, const size_t M, const size_t N, const size_t K,
    XT* A_d, int32_t* B_d, CT* C_d, XT* ScaleB_d,
    float* Acc_d, uint32_t* Cnt_d, XT* Other_d = nullptr, CT* Bias_d = nullptr,
    size_t BiasN = 0) {
  using data_type_a = XT;
  using data_type_b = int2x16;
  using data_type_c = CT;
  using data_type_acc = float;
  using data_type_scale = XT;

  constexpr int wg_tile_m = WGM;
  constexpr int wg_tile_n = WGN;
  constexpr int sg_tile_m = SGM;
  constexpr int sg_tile_n = SGN;
  constexpr int sg_tile_k = SGK;
  constexpr uint32_t prefetch_distance = 0;
  constexpr uint32_t periodic_sync_interval = 0;
  constexpr uint32_t global_kslicing = KS;
  constexpr uint32_t local_kslicing = LS;

  static constexpr gpu_arch arch_tag = gpu_arch::Xe;

  using tile_shape = gpu::xetla::group::tile_shape_t<
      wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
  using mem_desc_a_t = gpu::xetla::mem_desc_t<
      data_type_a, mem_layout::row_major, mem_space::global>;
  using mem_desc_b_t = gpu::xetla::mem_desc_t<
      data_type_b, mem_layout::row_major, mem_space::global>;
  using mem_desc_c_t = gpu::xetla::mem_desc_t<
      data_type_c, mem_layout::row_major, mem_space::global>;

  using compute_attr = gpu::xetla::group::compute_attr_t<
      data_type_a, data_type_a, data_type_acc>;
  using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<
      sg_tile_k, prefetch_distance, periodic_sync_interval>;
  using compute_policy = gpu::xetla::group::compute_policy_int2_fp16_upcvt_xmx<
      compute_attr, perf_tuning_knob, data_type_scale, kScaleGS,
      /*mma_xmx_m=*/SGM, arch_tag, /*use_unaligned_n=*/kUnaligned>;
  using gemm_t = gpu::xetla::group::gemm_t<
      compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;

  using silu_t = gpu::xetla::subgroup::silu_op_t;
  using sigmoid_t = sigmoid_chain_op_t;
  using bias_t = gpu::xetla::subgroup::bias_add_op_t<CT, arch_tag>;
  using prod_t = gpu::xetla::subgroup::elemwise_reduce_op_t<
      gpu::xetla::reduce_op::prod, XT, arch_tag>;
  using sum_t = gpu::xetla::subgroup::elemwise_reduce_op_t<
      gpu::xetla::reduce_op::sum, XT, arch_tag>;
  // 3 and 4 cover the GatedDeltaNet gates: in_proj_a carries a bias, in_proj_b
  // a fused sigmoid.
  using tile_op_t = std::conditional_t<
      POSTOP == 1, gpu::xetla::subgroup::chained_tile_op_t<silu_t, prod_t>,
      std::conditional_t<
          POSTOP == 2, gpu::xetla::subgroup::chained_tile_op_t<sum_t>,
          std::conditional_t<
              POSTOP == 3, gpu::xetla::subgroup::chained_tile_op_t<bias_t>,
              gpu::xetla::subgroup::chained_tile_op_t<sigmoid_t>>>>;

  // Post-ops need the aligned epilogue; N is padded to a multiple of 4 at pack
  // time so that always holds when POSTOP != 0.
  using epilogue_policy_t = std::conditional_t<
      POSTOP != 0, gpu::xetla::group::epilogue_policy_tile_op<tile_op_t, arch_tag>,
      std::conditional_t<
          kUnaligned, gpu::xetla::group::epilogue_policy_unaligned<arch_tag>,
          gpu::xetla::group::epilogue_policy_default<arch_tag>>>;
  using mem_desc_c_epilogue_t = std::conditional_t<
      kUnaligned,
      gpu::xetla::mem_desc_t<
          data_type_c, mem_layout::row_major, mem_space::global,
          /*alignment=*/1>,
      mem_desc_c_t>;
  using epilogue_t = gpu::xetla::group::epilogue_t<
      epilogue_policy_t, tile_shape, mem_desc_c_epilogue_t>;

  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;
  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      gpu::xetla::kernel::dispatch_policy_int2_fp16_upcvt_kslicing<
          group_swizzle, global_kslicing, local_kslicing>,
      gemm_t, epilogue_t>;

  uint32_t lda = static_cast<uint32_t>(K);
  uint32_t ldb = static_cast<uint32_t>(N);
  uint32_t ldc = static_cast<uint32_t>(N);
  uint32_t lds = static_cast<uint32_t>(N);

  auto make_args = [&]() {
    if constexpr (POSTOP == 0) {
      return typename gemm_op_t::arguments_t(
          static_cast<uint32_t>(M), static_cast<uint32_t>(K),
          static_cast<uint32_t>(N), A_d, lda, reinterpret_cast<int2x16*>(B_d),
          ldb, C_d, ldc, ScaleB_d, lds, Acc_d, Cnt_d);
    } else {
      typename tile_op_t::arguments_t tile_args;
      if constexpr (POSTOP == 1 || POSTOP == 2) {
        using reduce_t = std::conditional_t<POSTOP == 1, prod_t, sum_t>;
        typename reduce_t::arguments_t other_args(
            Other_d, {static_cast<uint32_t>(N), static_cast<uint32_t>(M),
                      static_cast<uint32_t>(N)});
        // silu occupies slot 0 when present, so the reduce op shifts to slot 1.
        tile_args.template set<POSTOP == 1 ? 1 : 0>(other_args);
      } else if constexpr (POSTOP == 3) {
        // Bound by the real N so lanes in the padded tail read as masked-off.
        const uint32_t bn = static_cast<uint32_t>(BiasN != 0 ? BiasN : N);
        typename bias_t::arguments_t bias_args(Bias_d, {bn, 1, bn});
        tile_args.template set<0>(bias_args);
      }
      typename epilogue_t::arguments_t epilogue_args(tile_args);
      return typename gemm_op_t::arguments_t(
          static_cast<uint32_t>(M), static_cast<uint32_t>(K),
          static_cast<uint32_t>(N), A_d, lda, reinterpret_cast<int2x16*>(B_d),
          ldb, C_d, ldc, ScaleB_d, lds, Acc_d, Cnt_d, epilogue_args);
    }
  };
  typename gemm_op_t::arguments_t gemm_arg = make_args();

  if (!gemm_op_t::can_implement(gemm_arg)) {
    std::cout << "int2_fp16_upcvt: arguments not supported (M=" << M
              << " N=" << N << " K=" << K << ")\n";
    exit(1);
  }
  sycl::nd_range<3> nd_range = gemm_op_t::get_nd_range(gemm_arg);

  using kernel_name_t =
      int2_upcvt_kernel_tag<XT, WGM, WGN, SGM, SGN, SGK, KS, LS, kUnaligned, CT, POSTOP>;

  return queue.submit([&](handler& cgh) {
    cgh.parallel_for<kernel_name_t>(
        nd_range, [=](nd_item<3> item) KERNEL_MAIN {
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(item, gemm_arg);
        });
  });
}

#ifdef OV_GPU_WITH_ZE_RT
template <
    typename XT, int WGM, int WGN, int SGM, int SGN, int SGK, int KS, int LS,
    bool kUnaligned, typename CT = XT, int POSTOP = 0>
bool int2_upcvt_gemm_ze_probe(
  ze_command_list_handle_t list, const sycl::context& context, const sycl::device& device,
  const size_t M, const size_t N, const size_t K,
    XT* A_d, int32_t* B_d, CT* C_d, XT* ScaleB_d,
    float* Acc_d, uint32_t* Cnt_d, XT* Other_d = nullptr, CT* Bias_d = nullptr,
    size_t BiasN = 0) {
  using data_type_a = XT;
  using data_type_b = int2x16;
  using data_type_c = CT;
  using data_type_acc = float;
  using data_type_scale = XT;
  constexpr gpu_arch arch_tag = gpu_arch::Xe;
  using tile_shape = gpu::xetla::group::tile_shape_t<WGN, WGM, SGN, SGM>;
  using mem_desc_a_t = gpu::xetla::mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
  using mem_desc_b_t = gpu::xetla::mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>;
  using mem_desc_c_t = gpu::xetla::mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;
  using compute_attr = gpu::xetla::group::compute_attr_t<data_type_a, data_type_a, data_type_acc>;
  using perf_tuning_knob = gpu::xetla::group::perf_tuning_knob_t<SGK, 0, 0>;
  using compute_policy = gpu::xetla::group::compute_policy_int2_fp16_upcvt_xmx<
      compute_attr, perf_tuning_knob, data_type_scale, kScaleGS, SGM, arch_tag, kUnaligned>;
  using gemm_t = gpu::xetla::group::gemm_t<compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;
    using silu_t = gpu::xetla::subgroup::silu_op_t;
    using sigmoid_t = sigmoid_chain_op_t;
    using bias_t = gpu::xetla::subgroup::bias_add_op_t<CT, arch_tag>;
    using prod_t = gpu::xetla::subgroup::elemwise_reduce_op_t<
      gpu::xetla::reduce_op::prod, XT, arch_tag>;
    using sum_t = gpu::xetla::subgroup::elemwise_reduce_op_t<
      gpu::xetla::reduce_op::sum, XT, arch_tag>;
    using tile_op_t = std::conditional_t<
      POSTOP == 1, gpu::xetla::subgroup::chained_tile_op_t<silu_t, prod_t>,
      std::conditional_t<
        POSTOP == 2, gpu::xetla::subgroup::chained_tile_op_t<sum_t>,
        std::conditional_t<
          POSTOP == 3, gpu::xetla::subgroup::chained_tile_op_t<bias_t>,
          gpu::xetla::subgroup::chained_tile_op_t<sigmoid_t>>>>;
    using epilogue_policy_t = std::conditional_t<
      POSTOP != 0, gpu::xetla::group::epilogue_policy_tile_op<tile_op_t, arch_tag>,
      std::conditional_t<
        kUnaligned, gpu::xetla::group::epilogue_policy_unaligned<arch_tag>,
        gpu::xetla::group::epilogue_policy_default<arch_tag>>>;
    using mem_desc_c_epilogue_t = std::conditional_t<
      kUnaligned,
      gpu::xetla::mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global, 1>,
      mem_desc_c_t>;
    using epilogue_t = gpu::xetla::group::epilogue_t<epilogue_policy_t, tile_shape, mem_desc_c_epilogue_t>;
  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;
  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      gpu::xetla::kernel::dispatch_policy_int2_fp16_upcvt_kslicing<group_swizzle, KS, LS>, gemm_t, epilogue_t>;
  using kernel_name_t = int2_upcvt_kernel_tag<XT, WGM, WGN, SGM, SGN, SGK, KS, LS, kUnaligned, CT, POSTOP>;

    auto make_args = [&]() {
    if constexpr (POSTOP == 0) {
      return typename gemm_op_t::arguments_t(
        static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N),
        A_d, static_cast<uint32_t>(K), reinterpret_cast<int2x16*>(B_d), static_cast<uint32_t>(N),
        C_d, static_cast<uint32_t>(N), ScaleB_d, static_cast<uint32_t>(N), Acc_d, Cnt_d);
    } else {
      typename tile_op_t::arguments_t tile_args;
      if constexpr (POSTOP == 1 || POSTOP == 2) {
        using reduce_t = std::conditional_t<POSTOP == 1, prod_t, sum_t>;
        typename reduce_t::arguments_t other_args(
          Other_d, {static_cast<uint32_t>(N), static_cast<uint32_t>(M), static_cast<uint32_t>(N)});
        tile_args.template set<POSTOP == 1 ? 1 : 0>(other_args);
      } else if constexpr (POSTOP == 3) {
        const uint32_t bn = static_cast<uint32_t>(BiasN != 0 ? BiasN : N);
        typename bias_t::arguments_t bias_args(Bias_d, {bn, 1, bn});
        tile_args.template set<0>(bias_args);
      }
      typename epilogue_t::arguments_t epilogue_args(tile_args);
      return typename gemm_op_t::arguments_t(
        static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N),
        A_d, static_cast<uint32_t>(K), reinterpret_cast<int2x16*>(B_d), static_cast<uint32_t>(N),
        C_d, static_cast<uint32_t>(N), ScaleB_d, static_cast<uint32_t>(N), Acc_d, Cnt_d, epilogue_args);
    }
    };
    typename gemm_op_t::arguments_t args = make_args();
  if (!gemm_op_t::can_implement(args))
    return false;
  const auto range = gemm_op_t::get_nd_range(args);
  using bundle_t = sycl::kernel_bundle<sycl::bundle_state::executable>;
  static std::mutex kernel_mutex;
  static std::optional<bundle_t> bundle;
  static ze_context_handle_t bundle_context = nullptr;
  static ze_kernel_handle_t kernel = nullptr;
  {
    std::lock_guard<std::mutex> lock(kernel_mutex);
    const auto native_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context);
    if (kernel == nullptr || bundle_context != native_context) {
      bundle.emplace(sycl::get_kernel_bundle<sycl::bundle_state::executable>(
          context, {device}, {sycl::get_kernel_id<kernel_name_t>()}));
      auto sycl_kernel = bundle->get_kernel(sycl::get_kernel_id<kernel_name_t>());
      kernel = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_kernel);
      bundle_context = native_context;
    }
  }
  std::lock_guard<std::mutex> launch_lock(kernel_mutex);
  const auto local = range.get_local_range();
  const auto global = range.get_global_range();
  cldnn::ze::zeKernelSetGroupSize(kernel, local[2], local[1], local[0]);
  cldnn::ze::zeKernelSetArgumentValue(kernel, 0, sizeof(args), &args);
  ze_group_count_t groups = {static_cast<uint32_t>(global[2] / local[2]),
                             static_cast<uint32_t>(global[1] / local[1]),
                             static_cast<uint32_t>(global[0] / local[0])};
  const auto result = cldnn::ze::zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0, nullptr);
  return result == ZE_RESULT_SUCCESS;
}
#endif

// Get a scratch (Acc, Cnt) buffer pair sized for the worst case across the
// instantiations we precompile (KS, LS up to 4). The kslicing kernel writes
// per-WG accumulators into Acc and uses Cnt as a barrier counter, so we
// allocate once per device and reuse across calls. Sizes returned by
// `gemm_op_t::get_acc_buf_size(M, N)` are the same across our (KS, LS)
// pairs in elements (but we conservatively over-allocate).
struct UpcvtScratch {
  float* acc = nullptr;
  uint32_t* cnt = nullptr;
  size_t acc_bytes = 0;
  size_t cnt_bytes = 0;
};

UpcvtScratch& get_scratch(sycl::queue& q, size_t need_acc_bytes, size_t need_cnt_bytes) {
  // One scratch buffer per process. Note this only
  // grows, so it must never be split per caller: the prefill tier asks for 0
  // bytes and would then be handed a null Acc.
  static UpcvtScratch s;
  // XETLA_SCRATCH_DEBUG reports every growth: this scratch is allocated
  // directly from SYCL and is not visible to the plugin's memory pool.
  static const bool dbg = std::getenv("XETLA_SCRATCH_DEBUG") != nullptr;
  if (s.acc_bytes < need_acc_bytes) {
    if (s.acc) {
      // Kernels enqueued earlier may still be reading the old buffer.
      q.wait();
      sycl::free(s.acc, q.get_context());
    }
    s.acc = static_cast<float*>(sycl::aligned_alloc_device(
        256, need_acc_bytes, q.get_device(), q.get_context()));
    s.acc_bytes = need_acc_bytes;
    q.memset(s.acc, 0, need_acc_bytes).wait();
    if (dbg)
      fprintf(stderr, "[xetla] int2 acc scratch -> %.3f GiB\n",
              need_acc_bytes / 1073741824.0);
  }
  if (s.cnt_bytes < need_cnt_bytes) {
    if (s.cnt) {
      q.wait();
      sycl::free(s.cnt, q.get_context());
    }
    s.cnt = static_cast<uint32_t*>(sycl::aligned_alloc_device(
        256, need_cnt_bytes, q.get_device(), q.get_context()));
    s.cnt_bytes = need_cnt_bytes;
    q.memset(s.cnt, 0, need_cnt_bytes).wait();
    if (dbg)
      fprintf(stderr, "[xetla] int2 cnt scratch -> %.3f GiB\n",
              need_cnt_bytes / 1073741824.0);
  }
  return s;
}

// Function-pointer type for one compiled tile/kslicing variant.
template <typename XT>
using upcvt_fn_t = sycl::event (*)(
    sycl::queue&, size_t, size_t, size_t, XT*, int32_t*, XT*,
    XT*, float*, uint32_t*);

// Adapter that also computes the scratch sizes (which depend on the same
// compile-time params via gemm_op_t::get_acc_buf_size / get_cnt_buf_size).
template <
    typename XT, int WGM, int WGN, int SGM, int SGN, int SGK, int KS, int LS,
    bool kUnaligned, typename CT = XT, int POSTOP = 0>
sycl::event upcvt_dispatch(
    sycl::queue& q, size_t M, size_t N, size_t K, XT* A, int32_t* B,
    CT* C, XT* SB, float* acc, uint32_t* cnt, XT* other = nullptr,
    CT* bias = nullptr, size_t bias_n = 0) {
  return int2_upcvt_gemm_impl<XT, WGM, WGN, SGM, SGN, SGK, KS, LS, kUnaligned, CT, POSTOP>(
      q, M, N, K, A, B, C, SB, acc, cnt, other, bias, bias_n);
}

// Compute scratch byte requirements for a given (M, N, KS, LS) by
// instantiating a throw-away gemm_op_t to call its static helpers.
template <
    typename XT, int WGM, int WGN, int SGM, int SGN, int SGK, int KS, int LS,
    bool kUnaligned>
std::pair<size_t, size_t> upcvt_scratch_bytes(size_t M, size_t N) {
  using data_type_a = XT;
  using data_type_b = int2x16;
  using data_type_c = XT;
  using data_type_acc = float;
  using data_type_scale = XT;
  static constexpr gpu_arch arch_tag = gpu_arch::Xe;
  using tile_shape =
      gpu::xetla::group::tile_shape_t<WGN, WGM, SGN, SGM>;
  using mem_desc_a_t = gpu::xetla::mem_desc_t<
      data_type_a, mem_layout::row_major, mem_space::global>;
  using mem_desc_b_t = gpu::xetla::mem_desc_t<
      data_type_b, mem_layout::row_major, mem_space::global>;
  using mem_desc_c_t = gpu::xetla::mem_desc_t<
      data_type_c, mem_layout::row_major, mem_space::global>;
  using compute_attr = gpu::xetla::group::compute_attr_t<
      data_type_a, data_type_a, data_type_acc>;
  using perf_tuning_knob =
      gpu::xetla::group::perf_tuning_knob_t<SGK, 0, 0>;
  using compute_policy = gpu::xetla::group::compute_policy_int2_fp16_upcvt_xmx<
      compute_attr, perf_tuning_knob, data_type_scale, kScaleGS, SGM, arch_tag,
      kUnaligned>;
  using gemm_t = gpu::xetla::group::gemm_t<
      compute_policy, tile_shape, mem_desc_a_t, mem_desc_b_t>;
  using epilogue_policy_t = std::conditional_t<
      kUnaligned, gpu::xetla::group::epilogue_policy_unaligned<arch_tag>,
      gpu::xetla::group::epilogue_policy_default<arch_tag>>;
  using mem_desc_c_epilogue_t = std::conditional_t<
      kUnaligned,
      gpu::xetla::mem_desc_t<
          data_type_c, mem_layout::row_major, mem_space::global, 1>,
      mem_desc_c_t>;
  using epilogue_t = gpu::xetla::group::epilogue_t<
      epilogue_policy_t, tile_shape, mem_desc_c_epilogue_t>;
  using group_swizzle = gpu::xetla::kernel::group_swizzle_default<arch_tag>;
  using gemm_op_t = gpu::xetla::kernel::gemm_universal_t<
      gpu::xetla::kernel::dispatch_policy_int2_fp16_upcvt_kslicing<
          group_swizzle, KS, LS>,
      gemm_t, epilogue_t>;
  size_t acc_elems = gemm_op_t::get_acc_buf_size(M, N);
  size_t cnt_elems = gemm_op_t::get_cnt_buf_size(M, N);
  return {acc_elems * sizeof(data_type_acc), cnt_elems * sizeof(uint32_t)};
}

#ifdef OV_GPU_WITH_ZE_RT
bool get_direct_ze_scratch(ze_command_list_handle_t list,
                           ze_context_handle_t context,
                           ze_device_handle_t device,
                           size_t acc_bytes,
                           size_t cnt_bytes,
                           float*& acc,
                           uint32_t*& cnt) {
  struct scratch_t {
    ze_context_handle_t context = nullptr;
    float* acc = nullptr;
    uint32_t* cnt = nullptr;
    size_t acc_bytes = 0;
    size_t cnt_bytes = 0;
  };
  static scratch_t scratch;
  static std::mutex scratch_mutex;
  std::lock_guard<std::mutex> lock(scratch_mutex);

  if (scratch.context != context || scratch.acc_bytes < acc_bytes || scratch.cnt_bytes < cnt_bytes) {
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    void* new_acc = nullptr;
    void* new_cnt = nullptr;
    if (cldnn::ze::zeMemAllocDevice(context, &desc, acc_bytes, 256, device, &new_acc) != ZE_RESULT_SUCCESS ||
        cldnn::ze::zeMemAllocDevice(context, &desc, cnt_bytes, 256, device, &new_cnt) != ZE_RESULT_SUCCESS)
      return false;

    const uint32_t zero = 0;
    if (cldnn::ze::zeCommandListAppendMemoryFill(list, new_acc, &zero, sizeof(zero), acc_bytes, nullptr, 0, nullptr) !=
            ZE_RESULT_SUCCESS ||
        cldnn::ze::zeCommandListAppendMemoryFill(list, new_cnt, &zero, sizeof(zero), cnt_bytes, nullptr, 0, nullptr) !=
            ZE_RESULT_SUCCESS)
      return false;

    scratch.context = context;
    scratch.acc = static_cast<float*>(new_acc);
    scratch.cnt = static_cast<uint32_t*>(new_cnt);
    scratch.acc_bytes = acc_bytes;
    scratch.cnt_bytes = cnt_bytes;
  }

  acc = scratch.acc;
  cnt = scratch.cnt;
  return true;
}

template <typename XT, int WGN, int KS, int LS, typename CT = XT, int POSTOP = 0>
bool int2_upcvt_gemm_ze_run(ze_command_list_handle_t list,
                            ze_context_handle_t ze_context,
                            ze_device_handle_t ze_device,
                            const sycl::context& context,
                            const sycl::device& device,
                            size_t M,
                            size_t N,
                            size_t K,
                            XT* A,
                            int32_t* B,
                            CT* C,
                            XT* ScaleB,
                            XT* Other = nullptr,
                            CT* Bias = nullptr,
                            size_t BiasN = 0) {
  const auto need = upcvt_scratch_bytes<XT, 1, WGN, 1, 16, 128, KS, LS, false>(M, N);
  float* acc = nullptr;
  uint32_t* cnt = nullptr;
  if (!get_direct_ze_scratch(list, ze_context, ze_device, need.first, need.second, acc, cnt))
    return false;
  return int2_upcvt_gemm_ze_probe<XT, 1, WGN, 1, 16, 128, KS, LS, false, CT, POSTOP>(
      list, context, device, M, N, K, A, B, C, ScaleB, acc, cnt, Other, Bias, BiasN);
}
#endif

bool is_integrated_gpu(const sycl::device& device) {
  const auto name = device.get_info<sycl::info::device::name>();
  return name.find("Arc(TM) Graphics") != std::string::npos &&
         name.find("Pro") == std::string::npos;
}

} // namespace

// -- Public entry point: pick a precompiled variant based on (M, N) ----------
//
// We precompile a small set of useful variants (matching the M=1 GEMV-tuned
// tiers from the upcvt test driver). For M>1 we currently use a single GEMV
// fallback (KS=1, LS=1, WGN=128); this is suboptimal for prefill but works
// correctly for any (M, K, N) with K divisible by 128 and 16.
template <typename XT, typename CT = XT, int POSTOP = 0>
sycl::event int2_upcvt_gemm_run(
    sycl::queue& q, const size_t M, const size_t N, const size_t K,
    XT* A, int32_t* B, CT* C, XT* ScaleB, const size_t slot, XT* Other = nullptr,
    CT* Bias = nullptr, size_t BiasN = 0) {
  const bool aligned_n = (N % 4 == 0);

  // Bound the scratch by the tiers this M can actually reach. Prefill only ever
  // dispatches (KS=1, LS=1) below, so sizing it for the decode worst case
  // reserved 13.3 GiB of non-torch device memory on a 1024-token prefill
  // against the 27B's 248k-wide lm_head, which left no room for a KV cache.
  size_t need_acc = 0, need_cnt = 0;
  auto widen = [&](std::pair<size_t, size_t> p) {
    need_acc = std::max(need_acc, p.first);
    need_cnt = std::max(need_cnt, p.second);
  };
#define INT2_SCRATCH(KS_, LS_)                                                \
  widen(upcvt_scratch_bytes<XT, /*WGM*/ 1, /*WGN*/ 128, /*SGM*/ 1,            \
                            /*SGN*/ 16, /*SGK*/ 128, KS_, LS_,                \
                            /*kUnaligned*/ false>(M, N));                     \
  widen(upcvt_scratch_bytes<XT, /*WGM*/ 1, /*WGN*/ 128, /*SGM*/ 1,            \
                            /*SGN*/ 16, /*SGK*/ 128, KS_, LS_,                \
                            /*kUnaligned*/ true>(M, N))
  if (M == 1) {
    INT2_SCRATCH(4, 8);
  } else {
    INT2_SCRATCH(1, 1);
  }
#undef INT2_SCRATCH
  auto& sc = get_scratch(q, need_acc, need_cnt);

  // Dispatch table. We instantiate one compiled kernel per (KS, LS, kUnaligned)
  // triple that we want to use; tile shape is fixed at WGM=1/SGM=1/SGN=16/SGK=128
  // and WGN tracked separately.
// Prefill wants the XMX GEMM tile, not the GEMV one: XeTLA's own harness only
// uses the GEMV shape at M==1 and switches to sg_m=8/sg_n=128 above that.
#define DISPATCH_GEMM(WGM_, SGM_, WGN_, SGN_, SGK_)                           \
  do {                                                                        \
    if (aligned_n)                                                            \
      return upcvt_dispatch<XT, WGM_, WGN_, SGM_, SGN_, SGK_, 1, 1,           \
                            /*kUnaligned*/ false, CT, POSTOP>(                \
          q, M, N, K, A, B, C, ScaleB, sc.acc, sc.cnt, Other, Bias, BiasN);   \
    else                                                                      \
      return upcvt_dispatch<XT, WGM_, WGN_, SGM_, SGN_, SGK_, 1, 1,           \
                            /*kUnaligned*/ true, CT, POSTOP>(                 \
          q, M, N, K, A, B, C, ScaleB, sc.acc, sc.cnt, Other, Bias, BiasN);   \
  } while (0)

#define DISPATCH_RAW(WGN_, KS_, LS_)                                          \
  do {                                                                        \
    if (aligned_n)                                                            \
      return upcvt_dispatch<                                                  \
          XT, /*WGM*/ 1, WGN_, /*SGM*/ 1, /*SGN*/ 16, /*SGK*/ 128, KS_, LS_,  \
          /*kUnaligned*/ false, CT, POSTOP>(q, M, N, K, A, B, C, ScaleB,      \
                                            sc.acc, sc.cnt, Other, Bias, BiasN); \
    else                                                                      \
      return upcvt_dispatch<                                                  \
          XT, /*WGM*/ 1, WGN_, /*SGM*/ 1, /*SGN*/ 16, /*SGK*/ 128, KS_, LS_,  \
          /*kUnaligned*/ true, CT, POSTOP>(q, M, N, K, A, B, C, ScaleB,       \
                                           sc.acc, sc.cnt, Other, Bias, BiasN);  \
  } while (0)

// K is split KS ways globally and LS ways locally, each slice consuming whole
// SGK=128 chunks, so K must divide evenly by KS*LS*128. It silently produced
// NaNs otherwise (e.g. MoE down_proj with K=768 under the KS=2,LS=4 tier), so
// step down to a slicing that divides before falling back to no slicing.
#define K_DIVIDES(KS_, LS_) ((K % ((KS_) * (LS_) * 128)) == 0)
#define DISPATCH(WGN_, KS_, LS_)                                              \
  do {                                                                        \
    if (K_DIVIDES(KS_, LS_))     { DISPATCH_RAW(WGN_, KS_, LS_); }            \
    else if (K_DIVIDES(1, 2))    { DISPATCH_RAW(WGN_, 1, 2); }                \
    else                         { DISPATCH_RAW(WGN_, 1, 1); }                \
  } while (0)

  if (M == 1) {
    // Sweep hook: the tier table below is tuned for discrete parts, whose
    // bandwidth differs enough from integrated ones that the winning tile does
    // not carry over. -1 keeps the table.
    static const int dcfg = [] {
      const char* e = std::getenv("XETLA_INT2_DECODE_CFG");
      return e ? std::atoi(e) : -1;
    }();
    if (dcfg >= 0) {
      switch (dcfg) {
        case 0: DISPATCH(32, 1, 1);
        case 1: DISPATCH(32, 1, 2);
        case 2: DISPATCH(32, 1, 4);
        case 3: DISPATCH(32, 1, 8);
        case 4: DISPATCH(64, 1, 1);
        case 5: DISPATCH(64, 1, 2);
        case 6: DISPATCH(64, 1, 4);
        case 7: DISPATCH(64, 1, 8);
        case 8: DISPATCH(128, 1, 1);
        case 9: DISPATCH(128, 1, 2);
        case 10: DISPATCH(128, 1, 4);
        case 11: DISPATCH(128, 1, 8);
        default: break;
      }
    }
    // The integrated Xe2 device has substantially less bandwidth and needs
    // shallower K slicing than the discrete B70 tiles below.
    if (is_integrated_gpu(q.get_device())) {
      if (K == 4096 && N == 6144)        { DISPATCH(32, 1, 1); }
      else if (K == 4096 && N == 24576)  { DISPATCH(256, 1, 1); }
      else if (K == 12288 && N == 4096)  { DISPATCH(32, 1, 2); }
      else if (K == 4096 && N == 151680) { DISPATCH(128, 1, 2); }
    }
    // ----- Tuned GEMV shapes for a 4096-wide hidden size.
    // Format: (K, N) -> (WGN, KS, LS).
    //   (4096,   6144)  qkv_proj
    //   (4096,  12288)  gate/up_proj
    //   (4096,  24576)  gate_up_proj (fused)
    //   (4096, 151680)  lm_head
    //   (12288,  4096)  down_proj
    if (K == 4096 && N == 4096) {
      // o_proj has no tuned entry and falls to the generic tier; sweep hook.
      static const int ocfg = [] {
        const char* e = std::getenv("XETLA_INT2_OPROJ_CFG");
        return e ? std::atoi(e) : -1;
      }();
      switch (ocfg) {
        case 0: DISPATCH(32, 1, 1);
        case 1: DISPATCH(32, 1, 2);
        case 2: DISPATCH(32, 1, 4);
        case 3: DISPATCH(32, 1, 8);
        case 4: DISPATCH(64, 1, 1);
        case 5: DISPATCH(64, 1, 2);
        case 6: DISPATCH(64, 1, 4);
        case 7: DISPATCH(64, 1, 8);
        case 8: DISPATCH(128, 1, 1);
        case 9: DISPATCH(32, 2, 4);
        default: break;
      }
    }

    if (K == 4096 && N == 6144)        { DISPATCH(32, 1, 4); }
    // o_proj: tuned entry beats the generic N<=4096 tier (32,2,4).
    else if (K == 4096 && N == 4096)   { DISPATCH(32, 1, 8); }
    else if (K == 4096 && N == 12288)  { DISPATCH(32, 1, 2); }
    else if (K == 4096 && N == 24576)  { DISPATCH(32, 1, 4); }
    else if (K == 4096 && N == 151680) { DISPATCH(32, 1, 4); }
    else if (K == 12288 && N == 4096)  { DISPATCH(32, 1, 8); }
    // ----- Tuned GEMV shapes for a 5120-wide hidden size. The generic tiers
    // below would put these on wg_n=64/128; the (32, 1, *) tier wins for every
    // one of them, with the local k-slicing factor refined per shape.
    else if (K == 5120 && N == 34816)   { DISPATCH(32, 1, 8); }
    else if (K == 17408 && N == 5120)   { DISPATCH(32, 1, 4); }
    else if (K == 5120 && N == 16384)   { DISPATCH(32, 1, 2); }
    else if (K == 6144 && N == 5120)    { DISPATCH(32, 1, 4); }
    else if (K == 5120 && N == 14336)   { DISPATCH(32, 1, 2); }
    else if (K == 5120 && N == 248320)  { DISPATCH(32, 1, 4); }
    // Generic GEMV fallback tiers (kept from upcvt test driver).
    else if (N <= 4096) {
      DISPATCH(32, 2, 4);
    } else if (N <= 8192) {
      DISPATCH(64, 1, 4);
    } else if (N <= 16384) {
      DISPATCH(64, 1, 2);
    } else {
      DISPATCH(128, 1, 1);
    }
  }
  // M>1 prefill. The GEMV tile stays ahead for chat-length prompts (M=21:
  // 0.41 s) but scales at ~17 ms/token, while the XMX GEMM tile costs
  // ~7.5 ms/token and takes over around M~100 (M=1024: 7.7 s vs 17.6 s).
  // n_pad is a multiple of 16, not always of the 128-wide wg tile.
  if (M >= 128 && N % 128 == 0) {
    DISPATCH_GEMM(32, 8, 128, 128, 32);
  }
  DISPATCH_RAW(128, 1, 1);
#undef DISPATCH_GEMM
#undef DISPATCH
#undef DISPATCH_RAW
#undef K_DIVIDES
}

// Concrete entry points. fp16 covers checkpoints exported as fp16;
// bf16 is for checkpoints that are natively bf16, where converting to fp16
// would be an extra round trip.
sycl::event int2_fp16_upcvt_gemm_run(
    sycl::queue& q, const size_t M, const size_t N, const size_t K,
    gpu::xetla::fp16* A, int32_t* B, gpu::xetla::fp16* C,
    gpu::xetla::fp16* ScaleB, const size_t slot = 0) {
  return int2_upcvt_gemm_run<gpu::xetla::fp16>(q, M, N, K, A, B, C, ScaleB, slot);
}

sycl::event int2_bf16_upcvt_gemm_run(
    sycl::queue& q, const size_t M, const size_t N, const size_t K,
    gpu::xetla::bf16* A, int32_t* B, gpu::xetla::bf16* C,
    gpu::xetla::bf16* ScaleB, const size_t slot = 0) {
  return int2_upcvt_gemm_run<gpu::xetla::bf16>(q, M, N, K, A, B, C, ScaleB, slot);
}

namespace ov {
namespace intel_gpu {
namespace xetla_int2 {

Variant variant_from_env() {
    static const Variant v = [] {
        const char* e = std::getenv("XETLA_INT2_KERNEL");
        if (e == nullptr)
            return Variant::upcvt_fp16;
        const std::string s(e);
        if (s == "dpas")
            return Variant::dpas_i2xi8;
        if (s == "dpas_prefill")
            return Variant::dpas_prefill_only;
        return Variant::upcvt_fp16;
    }();
    return v;
}

sycl::event gemv_f16(sycl::queue& q, size_t M, size_t N, size_t K,
                     void* A, void* B, void* C, void* ScaleB, size_t slot,
                     int postop, void* other, bool out_f32, void* bias,
                     size_t bias_n) {
  using fp16 = gpu::xetla::fp16;
  auto* a = static_cast<fp16*>(A);
  auto* b = static_cast<int32_t*>(B);
  auto* s = static_cast<fp16*>(ScaleB);
  auto* o = static_cast<fp16*>(other);
  if (out_f32) {
    // lm_head and the f32 GatedDeltaNet gates: f32 straight out of the
    // fp32 accumulator, so the bias is f32 too.
    auto* cf = static_cast<float*>(C);
    auto* bf = static_cast<float*>(bias);
    switch (postop) {
      case 3:
        return int2_upcvt_gemm_run<fp16, float, 3>(q, M, N, K, a, b, cf, s, slot, nullptr, bf, bias_n);
      case 4:
        return int2_upcvt_gemm_run<fp16, float, 4>(q, M, N, K, a, b, cf, s, slot, nullptr, nullptr, 0);
      default:
        return int2_upcvt_gemm_run<fp16, float, 0>(q, M, N, K, a, b, cf, s, slot, o);
    }
  }
  auto* c = static_cast<fp16*>(C);
  auto* bi = static_cast<fp16*>(bias);
  switch (postop) {
    case 1:
      return int2_upcvt_gemm_run<fp16, fp16, 1>(q, M, N, K, a, b, c, s, slot, o);
    case 2:
      return int2_upcvt_gemm_run<fp16, fp16, 2>(q, M, N, K, a, b, c, s, slot, o);
    case 3:
      return int2_upcvt_gemm_run<fp16, fp16, 3>(q, M, N, K, a, b, c, s, slot, o, bi, bias_n);
    case 4:
      return int2_upcvt_gemm_run<fp16, fp16, 4>(q, M, N, K, a, b, c, s, slot, o);
    default:
      return int2_upcvt_gemm_run<fp16, fp16, 0>(q, M, N, K, a, b, c, s, slot, o);
  }
}

bool gemv_f16_ze_probe(void* list, void* context_handle, void* device_handle,
                       size_t M, size_t N, size_t K,
                       void* A, void* B, void* C, void* ScaleB, size_t,
                       int postop, void* other, bool out_f32, void* bias,
                       size_t bias_n) {
#ifdef OV_GPU_WITH_ZE_RT
  using fp16 = gpu::xetla::fp16;
  static std::mutex context_mutex;
  static ze_context_handle_t cached_context_handle = nullptr;
  static std::optional<sycl::device> cached_device;
  static std::optional<sycl::context> cached_context;
  std::lock_guard<std::mutex> lock(context_mutex);
  const auto ze_context = static_cast<ze_context_handle_t>(context_handle);
  const auto ze_device = static_cast<ze_device_handle_t>(device_handle);
  if (cached_context_handle != ze_context) {
    cached_device.emplace(sycl::make_device<sycl::backend::ext_oneapi_level_zero>(ze_device));
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> context_input{
      ze_context, {*cached_device},
      sycl::ext::oneapi::level_zero::ownership::keep};
    cached_context.emplace(sycl::make_context<sycl::backend::ext_oneapi_level_zero>(context_input));
    cached_context_handle = ze_context;
  }
  const bool integrated_gpu = is_integrated_gpu(*cached_device);
#define DIRECT_RUN(WGN_, KS_, LS_, CT_, POSTOP_)                                           \
  return int2_upcvt_gemm_ze_run<fp16, WGN_, KS_, LS_, CT_, POSTOP_>(                       \
      static_cast<ze_command_list_handle_t>(list), ze_context, ze_device,                  \
      *cached_context, *cached_device, M, N, K,                                             \
      static_cast<fp16*>(A), static_cast<int32_t*>(B), static_cast<CT_*>(C),               \
      static_cast<fp16*>(ScaleB), static_cast<fp16*>(other),                              \
      static_cast<CT_*>(bias), bias_n)
#define DIRECT_F16(WGN_, KS_, LS_)                                                          \
  switch (postop) {                                                                         \
    case 1: DIRECT_RUN(WGN_, KS_, LS_, fp16, 1);                                            \
    case 2: DIRECT_RUN(WGN_, KS_, LS_, fp16, 2);                                            \
    case 3: DIRECT_RUN(WGN_, KS_, LS_, fp16, 3);                                            \
    case 4: DIRECT_RUN(WGN_, KS_, LS_, fp16, 4);                                            \
    default: DIRECT_RUN(WGN_, KS_, LS_, fp16, 0);                                           \
  }

  if (out_f32) {
    if (K == 4096 && N == 151680) {
      if (integrated_gpu) {
        DIRECT_RUN(128, 1, 2, float, 0);
      } else {
        DIRECT_RUN(32, 1, 4, float, 0);
      }
    }
    // Any other f32-out lm_head, e.g. the 27B's K=5120 N=248320.
    if (M > 1)
      DIRECT_RUN(128, 1, 1, float, 0);
    DIRECT_RUN(32, 1, 4, float, 0);
  }

  if (M > 1)
    DIRECT_F16(128, 1, 1);

  if (K == 4096 && N == 6144) {
    if (integrated_gpu) {
      DIRECT_F16(32, 1, 1);
    } else {
      DIRECT_F16(32, 1, 4);
    }
  }
  if (K == 4096 && N == 4096)
    DIRECT_F16(32, 1, 8);
  if (K == 4096 && N == 12288)
    DIRECT_F16(32, 1, 2);
  if (K == 4096 && N == 24576) {
    if (integrated_gpu) {
      DIRECT_F16(256, 1, 1);
    } else {
      DIRECT_F16(32, 1, 4);
    }
  }
  if (K == 12288 && N == 4096) {
    if (integrated_gpu) {
      DIRECT_F16(32, 1, 2);
    } else {
      DIRECT_F16(32, 1, 8);
    }
  }
  // Generic tiers, mirroring the SYCL dispatch so an untuned model (the 27B)
  // still gets the direct launch instead of falling back per call.
  if (N <= 4096) {
    DIRECT_F16(32, 2, 4);
  } else if (N <= 8192) {
    DIRECT_F16(64, 1, 4);
  } else if (N <= 16384) {
    DIRECT_F16(64, 1, 2);
  }
  DIRECT_F16(128, 1, 1);
#undef DIRECT_F16
#undef DIRECT_RUN
  return false;
#else
  (void)list;
  (void)context_handle;
  (void)device_handle;
  (void)M;
  (void)N;
  (void)K;
  (void)A;
  (void)B;
  (void)C;
  (void)ScaleB;
  (void)postop;
  (void)other;
  (void)out_f32;
  return false;
#endif
}

}  // namespace xetla_int2
}  // namespace intel_gpu
}  // namespace ov
