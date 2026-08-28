# Building and Running the XeTLA int2 GPU Path

Instructions for building this fork and running a ternary (2-bit) model through
the XeTLA `FullyConnected` implementation on an Intel discrete GPU.

See [xetla_int2_architecture.md](xetla_int2_architecture.md) for how it works.

---

## 1. Prerequisites

| Component | Notes |
|---|---|
| GPU | Intel discrete GPU, Xe2 class |
| GPU runtime | Intel compute runtime (`intel_gpu_vars.sh` or distro packages) |
| Compiler | Intel oneAPI DPC++ (`icx` / `icpx`), 2026.0 or newer |
| Build tools | CMake >= 3.16, Ninja |
| Python | 3.10+ with `openvino` (model preparation only) |

The GPU plugin builds its SYCL context from its OpenCL context
(`sycl::make_context<backend::opencl>`), so the OpenCL backend must remain
visible to SYCL. Select it with `ONEAPI_DEVICE_SELECTOR=opencl:gpu`.

---

## 2. Sources

```bash
export WORK=$HOME/ov-int2
mkdir -p $WORK && cd $WORK

git clone https://github.com/libxsmm/openvino_ternary_support.git openvino
cd openvino
git checkout feature_integrate_2bit_xetla_kernels
git submodule update --init --recursive
cd ..

# XeTLA headers, pinned to the commit this integration builds against
git clone -b ov-int2-integration https://github.com/egeor/xetla.git xetla
```

XeTLA is header-only; nothing in it needs to be built.

---

## 3. Environment

```bash
source <intel-gpu-runtime>/intel_gpu_vars.sh
source <oneapi-install>/oneapi-vars.sh

# The plugin's SYCL context is created from its OpenCL context, so the OpenCL
# backend has to remain selectable.
export ONEAPI_DEVICE_SELECTOR=opencl:gpu

export OV_ROOT=$WORK/openvino
export XETLA_ROOT=$WORK/xetla
```

`XETLA_ROOT` must be set **at configure time**: the plugin's CMake checks for
`$XETLA_ROOT/include/xetla.hpp` and fails configuration if it is absent.

---

## 4. Build

```bash
cd $OV_ROOT

cmake -B build-sycl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGPU_RT_TYPE=OCL \
  -DENABLE_INTEL_CPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_PYTHON=OFF -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
  -DENABLE_ONEDNN_FOR_GPU=ON -DENABLE_CM_FOR_GPU=ON \
  -DENABLE_SYSTEM_OPENCL=OFF \
  -DTHREADING=TBB_ADAPTIVE

cmake --build build-sycl -j $(nproc)
```

Notes:

- `GPU_RT_TYPE=OCL` with SYCL available selects `ocl::sycl_engine`, which
  exposes a SYCL queue over the plugin's OpenCL objects. The XeTLA
  implementation enqueues onto that queue.
- `ENABLE_SYSTEM_OPENCL=OFF` uses the bundled OpenCL headers.
- Artifacts are written to `$OV_ROOT/bin/intel64/Release`. If you configure a
  second build tree from the same source, give it its own `OUTPUT_ROOT` so the
  two do not share that directory.

Confirm the path is compiled in:

```bash
strings bin/intel64/Release/libopenvino_intel_gpu_plugin.so | grep -c OV_XETLA_INT2   # non-zero
```

### Experimental native Level Zero path

The native Level Zero XeTLA path is an experimental follow-up to the OpenCL
runtime path above. Build it in a separate output directory with oneAPI 2025.3;
the 2026.0 Unified Runtime adapter is incompatible with the tested B70 driver.

```bash
source <intel-gpu-runtime>/intel_gpu_vars.sh
source <oneapi-2025.3>/oneapi-vars.sh

cmake -B build-ze-20253 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGPU_RT_TYPE=ZE -DOUTPUT_ROOT=$PWD/build-ze-20253 \
  -DENABLE_INTEL_CPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_PYTHON=OFF -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
  -DENABLE_ONEDNN_FOR_GPU=OFF -DENABLE_CM_FOR_GPU=ON \
  -DTHREADING=TBB_ADAPTIVE
cmake --build build-ze-20253 -j $(nproc)
```

The build may copy a newer `libze_loader.so` into its output directory than the
installed GPU driver supports. Remove that copied loader before running so the
plugin resolves the system loader supplied with the GPU runtime:

```bash
rm -f build-ze-20253/bin/intel64/Release/libze_loader.so*
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

For native ZE experiments on a shared environment, start with a clean runtime
environment to prevent unrelated Python-distribution libraries from shadowing
oneAPI:

```bash
srun --export=NONE -p <gpu-partition> /bin/bash -lc '...'
```

The experimental switches are intentionally opt-in:

| Variable | Effect |
|---|---|
| `OV_XETLA_INT2_ZE_DIRECT=1` | Append the supported int2 XeTLA GEMVs directly to the plugin's native ZE command list. `OV_XETLA_INT2_ZE_DIRECT_QKV=1` remains an alias. |
| `OV_ZE_REGULAR_LIST=1` | Use a mutable-capable regular ZE command list instead of an immediate list. |
| `OV_ZE_REPLAY_LIST=1` | After recording prefill and one decode iteration, replay the stable decode list. Requires `OV_ZE_REGULAR_LIST=1`. |
| `OV_XETLA_INT2_MERGE_MLP=1` | Merge parallel gate/up compressed FCs into a 2I FC followed by the existing SwiGLU primitive. |

These switches are validated for the listed Bonsai int2 benchmark only. They are
not a general dynamic-shape capture API: a production implementation must
invalidate or update a captured list when its layouts, state buffers, or input
contracts change.

For Lunar Lake integrated GPUs, the up-convert dispatcher selects separate
decode tiles rather than reusing the B70 table. The tuned Bonsai-8B choices are:

| Shape `(K, N)` | Operation | LNL tile `(WGN, KS, LS)` |
|---|---|---|
| `(4096, 6144)` | QKV | `(32, 1, 1)` |
| `(4096, 24576)` | merged gate/up | `(256, 1, 1)` |
| `(12288, 4096)` | down projection | `(32, 1, 2)` |
| `(4096, 151680)` | lm_head | `(128, 1, 2)` |

With `max_new_tokens=256` and `BENCH_NO_EOS=1`, this native ZE path produced
255 timed decode tokens at 35.7--35.9 tok/s on LNL. The 256 generated IDs for
the photosynthesis chat prompt matched the B70 OpenCL XeTLA result exactly.

---

## 5. Prepare a model

The implementation consumes ternary weights expressed as OpenVINO `u2` codes
`{0,1,2}` with a scalar zero point of 1 and per-group fp16 scales (group size
128). Convert an existing fp16 IR whose weights are already ternary:

```bash
python src/plugins/intel_gpu/tools/xetla_int2/quantize_ir_ternary.py \
  --in  <model-fp16>/openvino_model.xml \
  --out <model-u2>/openvino_model.xml
```

`--min-k` (default 1024) skips MatMuls too small to be worth compressing.

---

## 6. Run

Any OpenVINO application picks the implementation up automatically; no API
changes are required. To confirm it is selected:

```bash
OV_XETLA_INT2_DEBUG=1 <your-app> 2>&1 | grep "xetla-int2"
```

Accepted nodes are logged as `accepted <node id>`, and nodes that fall back
report the reason.

### Benchmark tool

`src/plugins/intel_gpu/tools/xetla_int2` contains a small greedy-decode
benchmark. It is not built by the main build; build it against the OpenVINO
build tree:

```bash
cd $OV_ROOT/src/plugins/intel_gpu/tools/xetla_int2
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DOpenVINO_DIR=$OV_ROOT/build-sycl
cmake --build build
```

```
bench_llm <model.xml> <device> [max_new_tokens] [comma-separated input ids]
```

```bash
./build/bench_llm <model-u2>/openvino_model.xml GPU 256 "<input ids>"
```

It reports prefill time, decode throughput, and the generated token ids:

```
TTFT (prefill) : ... ms
decode         : 255 tokens in ... s = ... tok/s
generated_ids  =...
```

The first run after a build compiles the kernels, so its reported prefill time
includes JIT and can be an order of magnitude higher than later runs. Re-run
once to get a representative number.

To compare against the stock path, set `OV_XETLA_INT2_DISABLE=1` and re-run.

---

## 7. Verifying correctness

Decode is deterministic run to run: two runs with the same prompt produce
identical `generated_ids`.

To validate numerically, run the same model and prompt on the CPU plugin in
fp32 and compare the token ids; they should match exactly over a few hundred
tokens when `XETLA_INT2_KERNEL=upcvt`. This requires an OpenVINO build or wheel
with the CPU plugin enabled, since the configure line above disables it.

`OV_XETLA_INT2_VERIFY=1` additionally checks the weight packing itself: at load
time it dequantizes the packed buffers and compares them against the IR
constant, which separates a packing error from a kernel error. Note it samples
only the first 4096 output channels of the first few nodes, so it is a cheap
smoke test rather than full coverage; an end-to-end comparison against fp32
remains the authoritative check.

---

## 8. Environment switches

Plugin:

| Variable | Effect |
|---|---|
| `OV_XETLA_INT2_DEBUG` | Log accepted/rejected nodes, shapes, scale layouts |
| `OV_XETLA_INT2_DISABLE` | Disable the implementation (falls back to OpenCL) |
| `OV_XETLA_INT2_ONLY_N=<list>` | Restrict to the listed output widths |
| `OV_XETLA_INT2_BARRIER=1` | Insert an explicit queue barrier before each GEMM |
| `OV_XETLA_INT2_DPAS_FOLD=1` | Fold post-ops into the int2 x int8 epilogue instead of the separate pass |
| `OV_XETLA_INT2_VERIFY=1` | At load time, dequantize the packed buffers and compare against the IR constant |

Kernel:

| Variable | Effect |
|---|---|
| `XETLA_INT2_CFG_DEBUG=1` | Print the tile/k-slicing config chosen per shape |
| `XETLA_INT2_DECODE_CFG=<n>` | Override the decode (M=1) configuration |
| `XETLA_INT2_OPROJ_CFG=<n>` | Override the o_proj configuration |
| `XETLA_INT2_KERNEL=upcvt` | Up-convert, fp16 DPAS everywhere (default) |
| `XETLA_INT2_KERNEL=dpas_prefill` | int2 x int8 DPAS for M>1, up-convert for decode |
| `XETLA_INT2_KERNEL=dpas` | int2 x int8 DPAS for every shape |
| `XETLA_SCRATCH_DEBUG=1` | Report k-slicing scratch growth |

Benchmark tool:

| Variable | Effect |
|---|---|
| `BENCH_PROFILE=1` | Per-op profile of one decode step, with call counts |
| `BENCH_NO_EOS=1` | Keep decoding past EOS so every run does equal work |
| `BENCH_DUMP_LOGITS=<step>:<path>` | Dump raw f32 logits at a decode step |

### Choosing a kernel variant

Both variants run on DPAS; they differ in the operand types fed to it. The
choice is made at runtime, per process, with `XETLA_INT2_KERNEL`.

The **up-convert** variant (default) dequantizes the 2-bit weights to fp16 in
registers and runs an fp16 DPAS GEMM against the native fp16 activations. It is
the accurate choice, and the fast choice for batch-1 decode.

The **int2 x int8** variant quantizes activations to int8 per (row, K-group) and
feeds the integer DPAS pipeline. It is faster on the large GEMMs of a long
prefill on discrete parts, but quantizing the activations costs roughly an order
of magnitude more relative error per GEMM, and it is slower at M=1 because it
must quantize activations on every step.

`dpas_prefill` combines the two: int2 x int8 for prefill (M>1), up-convert for
decode. On long prompts this can cut time-to-first-token substantially while
leaving decode throughput and accuracy unchanged. The trade-off is that prefill
writes a slightly different KV cache, so generated text may diverge from the
up-convert result; both remain coherent, but they are not token-identical. The
benefit scales with prompt length and is hardware-dependent, so measure it on
the target part rather than assuming it.
