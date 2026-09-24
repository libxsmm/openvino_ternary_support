# Bonsai 2 27B on the TernOCL int2 GPU path: work record

Date: 2026-09-23/24. Branch `feature_integrate_2bit_ocl_kernels`, forked from
`feature_integrate_2bit_xetla_kernels` at `824bf30f14`. Companion to
[ternocl_int2_build_and_run.md](ternocl_int2_build_and_run.md); this file
records what was changed relative to the XeTLA branch and what was measured.
The model conversion (GGUF -> IR, rotation insertion, sign folding) is
unchanged from the XeTLA branch; its record is in that branch's worklog.

## 1. Goal

Replace the SYCL/XeTLA int2 FullyConnected implementation by the standalone
TernOCL OpenCL kernels, reusing the rest of the integration (packing, node
cache, acceptance rules, epilogue folding, Hadamard fusion, tools), with no
SYCL or XeTLA left in the build.

## 2. Kernels (TernOCL)

* `int2_fp16_upcvt`: the XeTLA up-convert scheme (2-bit -> fp16 with the group
  scale folded in by integer ops, fp16 DPAS, fp32 accumulation) as a GEMV
  kernel (M <= 8, local k-slicing) and an M-tiled kernel (dequantized B reused
  across `MT_M/8` DPAS row blocks, 2D block I/O).
* `common/epilogue.clh`: POSTOP 0..4 and fp32 output, verified bit-identical
  to the XeTLA tile ops by TernOCL's parity tool (plugin kernel and OpenCL
  kernel on the same USM inputs), including `xetla_sigmoid`'s `x <= -10 -> 0`.
* `hadamard/hadamard_fwht.cl`: OpenCL port of the fused sign + FWHT with the
  same stage order as the SYCL kernel.
* Standalone, both against XeTLA's own tuned configurations, >= 2 GiB of
  rotating weights, B70: decode GEMV x1.01-1.06 on the 27B shapes, M = 12..48
  x1.0-3.5, M = 1024 x4.4-4.9.

## 3. Plugin changes

* `impls/ocl_v2/ternocl_int2/`: `TernoclInt2FCImplementationManager`
  (`impl_types::ocl`), same acceptance rules as the XeTLA manager plus
  `N % 16 == 0` (no N padding / staging: the kernels clip), and a single
  fused-chain -> POSTOP mapping shared by validation and execution, so an
  accepted node can never reach an epilogue the kernel lacks.
* Programs built with `clBuildProgram` on the engine's context once per option
  set, launched through `stream::set_arguments` / `enqueue_kernel` (plugin
  events, no host sync; the XeTLA impl waited on every input event on the host).
* Kernel sources embedded at configure time from the `thirdparty/TernOCL`
  submodule ([libxsmm/TernOCL](https://github.com/libxsmm/TernOCL), pinned
  commit); `TERNOCL_ROOT` overrides it with another checkout.
* Removed: the SYCL impl and its XeTLA kernels, the `OV_GPU_CREATE_INSTANCE_SYCL_OCL`
  registry macro, the default `sycl` engine, the XeTLA CMake requirement.
  `OV_XETLA_INT2_*` switches are `OV_TERNOCL_INT2_*`; rt_info keys
  `int2_hadamard_block` / `int2_hadamard_signs`; tools moved to `tools/int2`.

## 4. What it took to reach parity

First end-to-end run (B70, paged, 256 tokens): correct tokens, but 34.6 tok/s
and 620 ms TTFT against XeTLA's 43.1 / 131 ms. In order:

1. **Clones dropped the kernels.** OpenVINO clones cached impls on every shape
   update; the first version rebuilt its `cl_kernel` in each clone. Clones
   now share the handles like the stock OpenCL impls. (No measurable change
   by itself.)
2. **Host time was not it**: instrumented `execute_impl` at ~10 us per FC
   (resolve 6, set_arguments 1, enqueue 3), and the queue is in-order with no
   event sync.
3. **The fused epilogues were not running.** Under dynamic shapes
   `primitive_inst::is_valid_fusion()` only trusts fused eltwise ops on an
   `impl_types::ocl` FC whose kernel name is `fully_connected_gpu_bf_tiled` or
   `..._bfyx_ref`, and otherwise executes an unfused subgraph (FC + separate
   eltwise/activation primitives) on every token. The XeTLA impl is
   `impl_types::sycl` and never hit this. `ternocl_int2` is now accepted
   there; TTFT went 620 -> 94 ms and decode to parity. `BENCH_PROFILE` did not
   show the unfused subgraphs, so its per-token totals were misleading
   (12.1 ms of FC time before, 16.5 ms after: the "missing" work was the
   subgraph).
4. **Prompt-length tiles.** `sweep_midm.sh` at M = 12/20/32/48 over the 27B
   shapes; winners group by output width, giving a 3 (M band) x 3 (width
   class) table. TTFT 94 -> 86 ms.

## 5. Results (256 tokens, `BENCH_NO_EOS=1`, photosynthesis prompt, second run)

| | TernOCL decode | TernOCL TTFT | XeTLA decode | XeTLA TTFT |
|---|---|---|---|---|
| Arc Pro B70 (same GPU, alternating runs) | 42.6-42.8 tok/s | 86-91 ms | 43.1-43.2 tok/s | 125-129 ms |
| Arc 140V (Lunar Lake), LNL tile tables | 8.1-8.7 tok/s | 240 ms | 7.96 tok/s | 627 ms |

Across two different B70 cards of the same host the spread is ~3%, larger
than the TernOCL/XeTLA decode difference; compare on one card.

Output: token ids identical to the XeTLA branch for the first 135 tokens,
then the near-tie fork at token 136 that XeTLA's own variants already show;
decoded text is the expected essay in both. Both branches are deterministic.

## 5a. Harness check

lm-evaluation-harness GSM8K (1319, `gsm8k_cot_llama` 8-shot, thinking medium,
greedy) through `paged_serve_llm_27b`, batch 16, B70: **96.82%** (1277/1319),
identical to the XeTLA branch (96.8%, 1277/1319) and 0.1 above the vLLM
plugin; wall 50.3 min against XeTLA's 89.8 min (129 tok/s aggregate decode;
the 1200-1370-token 8-shot prefills and the M = 16 joint decode steps run on the
M-tiled kernels).

## 5b. MTP speculative decoding

Port of the vLLM/XeTLA MTP work (`xetla_vllm_plugin` `feature/bonsai2-mtp`) on
branch `feature_integrate_2bit_ocl_kernels_mtp` (details in build_and_run 7.4):

* Draft IR (`bonsai2_mtp_to_ir.py`): the ProCreations head grafted onto layer 3
  of the Bonsai 2 IR (dense fp16 or int8 weights, rotations removed), sharing
  the target's u2 `lm_head`; the Hadamard sign vector is folded into its norms.
* Rollback of the GDN/conv1d state with the paged ops' existing
  `cache_interval` (= 1): no kernel changes were needed.
* Verify-step (M = k+1) GEMV tiles swept per shape with TernOCL's bench on the
  B70 (M = 2..8) and Arc 140V (M = 2..4), all validated; the M = 1 tiles with
  a larger `SGM` ran at up to a quarter of the bandwidth.

| | plain | MTP k=3 | vLLM + XeTLA MTP k=3 |
|---|---|---|---|
| B70, batch 1 (tok/s) | 42.7 | **72.0** (x1.69) | 80.2 (x1.73 over 46.2) |
| Arc 140V, batch 1 (tok/s) | 11.4 | **21.4** (x1.88) | 11.06 |
| GSM8K 1319, B70 | 96.89% (batch 8) | **96.89%** (batch 8, 41.8 min) | 97.7% (300 examples) |

Per k=3 round on the B70: verify 25 ms GPU time (M = 4), 3 draft steps of
1.6 ms GPU time each (the int8 head at ~510 GB/s); the remaining ~11 ms are
per-inference host overhead of the ~1500-primitive graph, the main lever left.

## 6. Not done / open

* MTP: host overhead per inference (above); no adaptive k (MTP loses at
  batch 16); draft `lm_head` is a second copy of the target's.

* LNL tiles: prompt-length (M = 12..48) and M >= 64 (swept at M = 512) are
  tuned; LNL decode GEMV tiles exist only for the 27B shapes.
* Only the up-convert kernels are wired in. TernOCL's int2 x int8 DPAS kernels
  (upfront or fused activation quantization) are faster for long prefills and
  could back a `dpas_prefill` mode as in the XeTLA branch.
* Level Zero runtime: the impl needs the OpenCL runtime; a ZE build falls back
  to the stock FC kernels (and rejects rotated models).
* The 1.7B, 4B, 8B and Bonsai 1 27B models were not re-measured on this branch.
