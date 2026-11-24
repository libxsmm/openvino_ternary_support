# Bonsai 2 27B on Lunar Lake: OpenVINO (TernOCL) vs llama.cpp (TernSYCL)

How to build, run, and compare the best OpenVINO numbers for Ternary Bonsai 2 27B on a Lunar Lake
Arc 140V iGPU, with the PrismML llama.cpp fork as the reference. Includes the measurement practices
(BKMs) that make the two engines comparable.

Branch: `exp_lnl_int8_gemv_mtp` of [libxsmm/openvino_ternary_support](https://github.com/libxsmm/openvino_ternary_support).
It is the Bonsai 2 PR (openvinotoolkit/openvino#39260) plus the experiments below. The experiments are
not for upstream as they are.

## 1. Results (Arc 140V iGPU, Core Ultra 7 258V, 32 GB)

**Prefill (TTFT, ms).** OpenVINO: `genai_bench.py`. llama.cpp: `llama-bench` and the server's `prompt_ms`.

| Prompt tokens | OV, PR (fp16 prefill) | OV, this branch | llama.cpp (`llama-bench`) |
|---|---|---|---|
| 21 (photo) | 165 | 136-145 | 152 |
| 56 (math) | 188 | 176-178 | 177 |
| 135 (code) | 896 | 361-364 | 384 |
| 512 | 1321 | 937 (LNL tile) | 1107 |
| 2048 | 6000-6400 | 3753 (LNL tile) | 4490 |

**Decode (tok/s).** Photo / math / code, 256 tokens each. Tokens accepted per step are in parentheses.

| | OV, this branch | llama.cpp (`llama-server`) |
|---|---|---|
| plain | 13.5 / 13.5 / 13.5 | 13.1 / 13.1 / 13.0 |
| MTP k=2 | 24.5 / 26.2 / 23.6 | 24.5 / 25.2 / 23.5 |
| MTP k=3 (the best k for both engines) | **26.0 (2.91) / 28.3 (3.12) / 25.3 (2.84)** | 25.4 (2.91) / 25.7 (2.94) / 23.7 (2.72) |

- Both engines use the same MTP head (Q8_0, bit-identical weights). At equal acceptance (photo) the
  speed is equal. OV's lead on math and code comes from its texts accepting more drafts.
- The PR as it stands (fp16 prefill, fp16 GEMV for the verify rows) reaches 21.9 / 24.9 / 22.0 tok/s
  at k=3, and much less at k=4. The int8 GEMV closes that gap.

## 2. What the branch changes

All experiments are opt-in through environment variables. `run_lnl.sh` sets them.

| Variable | Effect |
|---|---|
| `TERNOCL_INT8_PREFILL=1` | prompts (M > 8) on the int2 x int8 DPAS GEMM, activations quantized to int8 per 128 inputs. The same as the release-internal property `GPU_TERNOCL_INT2_INT8_PREFILL`, which cannot be set from `compile_model` in a release build. 2.5x faster TTFT at 135 tokens |
| `TERNOCL_INT8_GEMV_MIN_M=2` | MTP verify steps (2-8 rows) on an int2 x int8 GEMV instead of the fp16 one. Same speed at 1 row, 3-18% faster at 2-8 |
| `TERNOCL_LNL_TILE=1` | int8 GEMM tile {8,128,16,1} on integrated GPUs for M >= 64: 6% faster at 512 tokens, ~10% slower at 135 |
| `TERNOCL_FWHT_QUANT=1` | FWHT fused with the activation quantization (TernOCL `exp_fwht_quant`). No measurable gain on LNL |
| `TERNOCL_INT8_GEMV_TILE=nsg,ls,u` | forces the int8 GEMV tile (for sweeps) |
| `TERNOCL_MAD=1` | puts `-cl-mad-enable` back (see below) |

One change is always on. The TernOCL kernels build without `-cl-mad-enable`, which OpenVINO's
`KernelGenerator` adds to every kernel. That flag made IGC schedule the int8 DPAS GEMM ~20% slower:
int8 prefill went from 459 to 364 ms at 135 tokens. It does not affect decode or the fp16 path,
and the generated tokens are identical. The same fix is in #39230 and #39260 (commit `5d519da487`).

## 3. Build from scratch

Tested with GCC 14.3, Python 3.12, and the Intel compute runtime with OpenCL. Build on any Linux host;
the install runs on LNL. The int2 path runs on the plugin's OpenCL runtime (`GPU_RT_TYPE=OCL`).

```bash
export WORK=$HOME/ov-lnl && mkdir -p $WORK && cd $WORK
git clone --single-branch -b exp_lnl_int8_gemv_mtp https://github.com/libxsmm/openvino_ternary_support.git openvino
git -C openvino submodule update --init --recursive    # TernOCL is pinned at exp_fwht_quant (c3a25c5)
git clone https://github.com/openvinotoolkit/openvino.genai.git openvino.genai
git -C openvino.genai checkout a1000dbd && git -C openvino.genai submodule update --init --recursive

python3 -m venv venv && ./venv/bin/pip install cmake ninja numpy
export PATH=$WORK/venv/bin:$PATH
cd openvino
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$WORK/install \
  -DOPENVINO_EXTRA_MODULES=$WORK/openvino.genai -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=$WORK/venv/bin/python \
  -DGPU_RT_TYPE=OCL -DENABLE_SYSTEM_OPENCL=OFF -DENABLE_INTEL_CPU=ON -DENABLE_INTEL_NPU=OFF \
  -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
  -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF \
  -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF \
  -DCMAKE_CXX_FLAGS=-Wno-error=dangling-reference      # GCC 14
cmake --build build -j $(nproc --all) && cmake --install build
source $WORK/install/setupvars.sh && python -c "import openvino_genai, openvino as ov; print(ov.get_version())"
```

Build time from scratch:

| Step | 120-core Xeon (`-j 120`) | Lunar Lake laptop, 8 cores (`-j 8`) |
|---|---|---|
| clone + submodules | 3 min | 5 min |
| configure | 27 s | 33 s |
| compile | 4.5 min | 45 min |
| total | about 8 min | about 51 min |

The compile is ~5.7 CPU-hours either way. Use `-j $(nproc --all)`: `nproc` follows `OMP_NUM_THREADS`
and can oversubscribe the cores.

The build tree takes 1.5 GB and the install 216 MB.

- **GenAI version:** `a1000dbd` is the GenAI master the numbers were taken with. Its MTP support for
  Qwen3.5 is required.
- **CPU plugin:** needed, because it runs the tokenizers.
- **`ENABLE_SYSTEM_OPENCL=OFF`:** avoids mixing system and bundled OpenCL C++ headers.
- **Unit tests:** add `-DENABLE_TESTS=ON` and build `ov_gpu_unit_tests`, then run
  `--gtest_filter='*ternocl*'` on a GPU host.

## 4. Models

You need a GenAI model directory with the ternary main model and the MTP draft
(`openvino_mtp_model.xml`).

Build the main model as in
[ternocl_int2_bonsai2_from_scratch.md](../../../docs/ternocl_int2_bonsai2_from_scratch.md), steps 3-4.
For the draft, use the combined GGUF that llama.cpp also runs, so both engines get the same head:

```bash
hf download ProCreations/Ternary-Bonsai-2-27B-MTP Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf --local-dir gguf
python $WORK/openvino/src/plugins/intel_gpu/tools/int2/bonsai2_mtp_to_ir.py \
    --gguf gguf/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf \
    --template bonsai27b-mtpx/openvino_mtp_model.xml \
    --gguf-py llama.cpp-prism/gguf-py --out-dir bonsai2-27b-genai
```

The `--template` is the optimum-intel export of the 27B with an MTP head (from-scratch guide, step 4.2).
The converter only reads the graph from it; every weight comes from the GGUF.

## 5. Run OpenVINO

```bash
PYTHON=$WORK/venv/bin/python ./run_lnl.sh $WORK/install <genai model dir>   # ~25 min; writes lnl_results/results.txt
```

It runs plain decode and MTP k=2 and k=3 on the three chat prompts, then the TTFT of 512- and
2048-token prompts with the LNL tile. Expected output:

```
##### default
plain k=0 photo   ttft   145.4 ms  decode   13.5 tok/s  tok/step 1.00  md5 5f3a2f8b35bd ref
plain k=0 math    ttft   185.1 ms  decode   13.5 tok/s  tok/step 1.00  md5 a7c1f3ac7b14 ref
plain k=0 code    ttft   368.7 ms  decode   13.5 tok/s  tok/step 1.00  md5 4748f383493b ref
mtp   k=2 photo   ttft   142.2 ms  decode   24.5 tok/s  tok/step 2.49  md5 5f3a2f8b35bd same
mtp   k=2 math    ttft   186.2 ms  decode   26.3 tok/s  tok/step 2.67  md5 a7c1f3ac7b14 same
mtp   k=2 code    ttft   371.9 ms  decode   23.6 tok/s  tok/step 2.39  md5 54247c083f0e DIFF@18
mtp   k=3 photo   ttft   143.7 ms  decode   26.0 tok/s  tok/step 2.91  md5 8e6ee7c9086d DIFF@4
mtp   k=3 math    ttft   187.4 ms  decode   28.3 tok/s  tok/step 3.12  md5 a7c1f3ac7b14 same
mtp   k=3 code    ttft   373.5 ms  decode   25.3 tok/s  tok/step 2.84  md5 2e2658eee196 DIFF@18
##### lnltile
tok512: 512 prompt tokens
tok2048: 2048 prompt tokens
plain k=0 tok512  ttft   936.9 ms
plain k=0 tok2048 ttft  3753.4 ms
```

To run one setting by hand: `TERNOCL_INT8_PREFILL=1 TERNOCL_INT8_GEMV_MIN_M=2 OV_CACHE_DIR=cache_x
taskset -c 4 python genai_bench.py <model dir> --modes mtp --k 3`.

## 6. Run llama.cpp (reference)

PrismML fork `d12ca13bb9` (branch `prism`), SYCL backend, oneAPI 2026.0:

```bash
git clone -b prism https://github.com/PrismML-Eng/llama.cpp llama.cpp-prism && cd llama.cpp-prism
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -G Ninja -DGGML_SYCL=ON -DGGML_NATIVE=OFF -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build-sycl -j --target llama-bench llama-server
<this dir>/run_llama.sh build-sycl/bin gguf/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf   # ~15 min
```

`GGML_SYCL_PTQ1_T2=all` puts all PQ2_0 weights on the TernSYCL int2 x int8 kernels. Without it,
llama.cpp prefill throughput is ~40% lower. Expected output:

```
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) | ... | pp135  |  352.80 ± 1.52 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) | ... | pp512  |  462.98 ± 0.63 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) | ... | pp2048 |  455.05 ± 0.81 |
| qwen35 27B PQ2_0 - 2.13 bpw (group 128) | ... | tg32   |   13.21 ± 0.02 |
plain    photo prompt   262.8 ms  decode    13.1 tok/s  tok/step 1.00  md5 dbb91e2def44 ref
plain    math  prompt   288.6 ms  decode    13.2 tok/s  tok/step 1.00  md5 57b5ee3d446a ref
plain    code  prompt   496.6 ms  decode    13.1 tok/s  tok/step 1.00  md5 9aaf344efc37 ref
mtp k=2  photo prompt   294.7 ms  decode    25.0 tok/s  tok/step 2.51  md5 dbb91e2def44 same
mtp k=2  math  prompt   328.8 ms  decode    25.3 tok/s  tok/step 2.56  md5 57b5ee3d446a same
mtp k=2  code  prompt   545.3 ms  decode    23.5 tok/s  tok/step 2.39  md5 9aaf344efc37 same
mtp k=3  photo prompt   298.3 ms  decode    25.5 tok/s  tok/step 2.91  md5 b23873cae32f DIFF
mtp k=3  math  prompt   333.5 ms  decode    25.7 tok/s  tok/step 2.94  md5 57b5ee3d446a same
mtp k=3  code  prompt   547.1 ms  decode    23.8 tok/s  tok/step 2.72  md5 9aaf344efc37 same
```

## 7. BKMs: making the numbers comparable

- **Use the same prompts in both engines.** Both benches send the same chat-templated text
  (`genai_bench.py` applies the template, and `llama_bench.py` embeds its output). Use EOS-ignored
  greedy decoding of 256 tokens, and no prefix or prompt cache.
- **Pin and pace.** Run the whole process with `taskset -c 4` (one P-core). Pinning individual
  threads after compilation slowed decode to 12.2 tok/s. Idle 10 s before each timed run
  (`--pace 10`): LNL throttles. When decode drops to ~11 tok/s, the TTFT of the same run is up
  to 15% high too.
- **Use one cache directory per variant.** The OpenVINO model cache (`CACHE_DIR`) does not key on
  the `TERNOCL_*` variables. A shared cache silently reuses kernels built under other settings.
- **Prefill chunking.**
  - `llama-bench` prefills in 512-token ubatches by default. With `-ub 2048` its throughput is
    ~30% lower on LNL.
  - `genai_bench.py` uses `--chunk 512` to match. GenAI's default `max_num_batched_tokens` is 256.
  - Compare 2048-token prompts only chunk-for-chunk.
- **TTFT definitions differ.** OV TTFT is the wall time of a 1-token generation. llama-server's
  `prompt_ms` is its internal prefill time; client-side, llama-server adds ~100 ms on short prompts.
  `llama-bench` `pp` is the cleanest kernel-level number.
- **Check the texts, not just the speed.** The benches print an md5 of the generated tokens and
  compare MTP output to plain decoding (`same` / `DIFF@<token>`).
  - int8 prefill changes the logits slightly, so texts diverge from fp16 prefill after a common
    prefix, and MTP acceptance moves with the text.
  - Compare speed only between runs with the same md5.
- **Profiling.**
  - OpenVINO's per-op profiler (`PERF_COUNT`) reports only the last stage of a TernOCL FC, and
    under-reports even that.
  - Use unitrace from [intel/pti-gpu](https://github.com/intel/pti-gpu) instead:
    `unitrace --opencl -d -h -t --chrome-device-logging <cmd>` for OpenVINO,
    `unitrace --chrome-kernel-logging -d <cmd>` for llama.cpp (Level Zero).
  - Sum the `gpu_op` events per kernel name.

## 8. Where the prefill gap came from

Before the fixes, OV int8 prefill was 1.35-1.4x slower than the same TernOCL GEMM run standalone.
llama.cpp uses the same tile, {8,128,8,2} for M > 32, so the tile was not the cause.

**Cause: `-cl-mad-enable`.** Toggling only that flag on the standalone driver reproduced the
in-plugin times:

| Kernel (M = 2048) | without flag | with flag | in plugin |
|---|---|---|---|
| gate_up | 18.3 ms | 25.7 ms | 25.9 ms |
| down | 9.85 ms | 12.8 ms | 12.5 ms |

The tile sweep at M = 2048 then found {8,128,16,1} best on LNL for all six 27B shapes.

**Remaining opportunities:**
- Quantize activations before the GEMM (`QMODE=0`, as llama.cpp does). It is 3-11% faster
  standalone.
- Fuse the small kernels around the FC (RMSNorm + FWHT + quantization, SwiGLU).
- Productize int8 prefill and int8 GEMV as real options, with B70 tiles and tests.
