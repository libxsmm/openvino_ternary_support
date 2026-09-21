# Building and Running the XeTLA int2 GPU Path

Instructions for building this fork and running a ternary (2-bit) model through
the XeTLA `FullyConnected` implementation on an Intel discrete GPU.

See [xetla_int2_architecture.md](xetla_int2_architecture.md) for how it works and
[xetla_int2_bonsai2_worklog.md](xetla_int2_bonsai2_worklog.md) for the record of
the Bonsai 2 (rotated basis) work. For Bonsai 2 27B specifically, the curated
from-scratch sequence (download, build, IR conversion, B70 and Lunar Lake runs)
is [xetla_int2_bonsai2_from_scratch.md](xetla_int2_bonsai2_from_scratch.md).

---

## 1. Prerequisites

| Component | Notes |
|---|---|
| GPU | Intel Xe2 GPU, discrete (Arc Pro B70) or integrated (Lunar Lake) |
| GPU runtime | Intel compute runtime (`intel_gpu_vars.sh` or distro packages) |
| Compiler | Intel oneAPI DPC++ (`icx` / `icpx`), 2026.0 or newer |
| Build tools | CMake >= 3.16, Ninja |
| Python | 3.10+ with `openvino` and `numpy` (model preparation only) |

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
| `OV_GPU_FUSE_RMS_ROPE=1` | Fold the per-head Q/K RMSNorm into the following RoPE kernel. `OV_GPU_DEBUG_RMS_ROPE=1` traces the match. |

These switches are validated for the listed Bonsai int2 benchmark only. They are
not a general dynamic-shape capture API: a production implementation must
invalidate or update a captured list when its layouts, state buffers, or input
contracts change.

### Fused Q/K RMSNorm and RoPE

Bonsai normalizes every attention head before the rotary embedding, so decode
runs 72 small RMSNorm kernels whose only consumer is a RoPE kernel. With
`OV_GPU_FUSE_RMS_ROPE=1` the RMSNorm is folded into the RoPE kernel: the
normalization weights become RoPE input 3, and each work group reduces one head
in local memory before rotating it, so the intermediate tensor is never written.

The transformation only fires for the shape it can prove: fp16, a rotate-half
RoPE covering the whole head, no gather or slice, and an RMSNorm with static
per-head weights whose sole consumer is that RoPE. It accepts both the form
where a view separates the two ops and the form where `RoPEFusionPreprocess` has
already absorbed the transpose. The kernel mirrors `rms_gpu_bfyx_opt.cl`
arithmetic, so generated tokens are identical to the unfused path.

Measured on B70 with the Bonsai-8B u2 model, 255 timed decode steps, native ZE
plus merged MLP, all runs token-identical to the unfused reference:

| Path | Unfused | Fused |
|---|---|---|
| stateful indirect SDPA | 143.27 tok/s | 146.53 tok/s |
| paged attention with list replay | 144.32 tok/s | 148.71 tok/s |

Per-token GPU kernel time for the region drops from about 506 us (363 us of
RMSNorm plus 143 us of RoPE) to about 303 us (146 us of the remaining
non-attention RMSNorms plus 157 us of fused RoPE).

On Lunar Lake with native ZE, the same indirect-SDPA configuration is
token-identical and improves from 35.88 to 36.09 tok/s. The transformed paged
attention harness is also token-identical there, but is currently slower on the
integrated GPU: 26.56 tok/s without replay and 22.24 tok/s with replay. Keep
the indirect path on LNL until its paged-attention and regular-list paths are
tuned for the integrated architecture.

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
128).

### 5.1 Export an fp16 IR

```bash
python -m venv venv && ./venv/bin/pip install openvino numpy

# Exporting from a checkpoint additionally needs optimum-intel. The 27B is a
# VLM, so it exports as several models and needs the image-text-to-text task;
# transformers 5.2.0 is the version its modelling code matches.
./venv/bin/pip install "optimum-intel[openvino]==2.1.0" "transformers==5.2.0"

./venv/bin/optimum-cli export openvino --model <8B-checkpoint>  \
    --task text-generation-with-past --weight-format fp16 bonsai8b-fp16
./venv/bin/optimum-cli export openvino --model <27B-checkpoint> \
    --task image-text-to-text --weight-format fp16 bonsai27b-fp16
```

The 8B export is a single `openvino_model.xml`. The 27B export produces
`openvino_language_model.xml` (the decoder, which is what gets quantized) plus
`openvino_text_embeddings_model.xml`, which the 27B benchmarks need as a
separate argument.

### 5.2 Quantize to ternary u2

```bash
./venv/bin/python src/plugins/intel_gpu/tools/xetla_int2/quantize_ir_ternary.py \
  --in  bonsai8b-fp16/openvino_model.xml \
  --out bonsai8b-u2/openvino_model.xml

./venv/bin/python src/plugins/intel_gpu/tools/xetla_int2/quantize_ir_ternary.py \
  --in  bonsai27b-fp16/openvino_language_model.xml \
  --out bonsai27b-u2/openvino_model.xml
```

`--min-k` (default 1024) skips MatMuls too small to be worth compressing.

Expected output, and a useful check that the weights were read correctly:

| Model | MatMuls rewritten | Weights |
|---|---|---|
| 8B | 253 | 14.09 GiB -> 1.87 GiB |
| 27B | 497 | 47.72 GiB -> 6.34 GiB |

A checkpoint exported as bf16 needs no special handling here: the tool decodes
bf16 constants explicitly. Reading them as fp16 instead leaves the ternary codes
intact but corrupts every group scale, which is silent -- the model still loads
and still generates fluent text.

---

### 5.3 Checkpoints with a tied output projection

Bonsai 1.7B and 4B set `tie_word_embeddings`, so `lm_head` aliases the embedding
and the export carries no separate weight for it. The head then stays dense and
is never compressed. Bonsai's embedding is itself ternary, so give the head its
own copy first and it packs like every other projection, matching the 8B and 27B
which ship untied:

```bash
./venv_export/bin/python src/plugins/intel_gpu/tools/xetla_int2/untie_embeddings.py \
  --in <checkpoint> --out <checkpoint>-untied
```

Export from the untied directory. `verify_ternary_u2.py` reports whether the
head ended up compressed.

### 5.4 Verifying the conversion

```bash
./venv/bin/python src/plugins/intel_gpu/tools/xetla_int2/verify_ternary_u2.py \
  --fp16 <model-fp16>/openvino_model.xml \
  --u2   <model-u2>/openvino_model.xml
```

It reconstructs every compressed weight and reports the error, which is exactly
zero when each 128-element group holds a single magnitude. Measured:

| Model | Compressed MatMuls | Head compressed | Max abs error | Groups with >1 magnitude |
|---|---|---|---|---|
| 1.7B | 197 | yes (after untying) | 9.77e-04 | 740 / 13.4M (0.0055%) |
| 4B | 253 | yes (after untying) | 9.77e-04 | 1643 / 31.4M (0.0052%) |
| 8B | 253 | yes | 0 | 0 |
| 27B | 497 | yes | 0 | 0 |

The 1.7B and 4B residual is not a group-size mismatch: their native scale group
is 128 as well, but a few thousand groups store two fp16 magnitudes a few ULPs
apart, so taking the maximum moves the other one. The relative error is 0.78% on
0.005% of groups.

## 6. Run

Any OpenVINO application picks the implementation up automatically; no API
changes are required. To confirm it is selected:

```bash
OV_XETLA_INT2_DEBUG=1 <your-app> 2>&1 | grep "xetla-int2"
```

Accepted nodes are logged as `accepted <node id>`, and nodes that fall back
report the reason.

### Benchmark tools

`src/plugins/intel_gpu/tools/xetla_int2` contains greedy-decode benchmarks. They
are not built by the main build; build them against the OpenVINO build tree:

```bash
cd $OV_ROOT/src/plugins/intel_gpu/tools/xetla_int2
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DOpenVINO_DIR=$OV_ROOT/build-sycl
cmake --build build
```

| Tool | Model | Attention path |
|---|---|---|
| `bench_llm` | 8B | stateful, indirect SDPA |
| `paged_bench_llm` | 8B | `SDPAToPagedAttention` |
| `bench_llm_27b` | 27B | stateful, indirect SDPA |
| `paged_bench_llm_27b` | 27B | `SDPAToPagedAttention` |

The 8B tools take one model. The 27B is a VLM decoder, so its tools take the
quantized decoder *and* the text-embeddings model, and consume rank-3 mrope
position ids:

```
bench_llm            <model.xml> <device> [max_new_tokens] [input ids]
paged_bench_llm      <model.xml> <device> [max_new_tokens] [input ids]
bench_llm_27b        <decoder.xml> <embeddings.xml> <device> [max_new_tokens] [input ids]
paged_bench_llm_27b  <decoder.xml> <embeddings.xml> <device> [max_new_tokens] [input ids]
```

`run_bonsai.sh <models-dir> [tokens]` runs all three reference configurations:

```bash
./run_bonsai.sh <models-dir> 256
```

The token ids of the chat prompt used for every number below are in
`prompt_photosynthesis_27b.txt`.

Useful environment variables:

| Variable | Effect |
|---|---|
| `BENCH_NO_EOS=1` | Generate the full token budget instead of stopping at EOS, so runs are comparable |
| `BENCH_MAX_LEN=<n>` | Upper bound on the sequence dimension. Over-reserving starves the ESIMD scratch and the enqueue fails with `CL_OUT_OF_RESOURCES`; 512 is a good default |
| `BENCH_STATIC_DECODE=1` | Pin the token dimension to 1 and feed the prompt one token at a time (27B tools) |
| `BENCH_PRECISION=f16` | Inference precision hint |

### Measured results

256 tokens, `BENCH_NO_EOS=1`, OpenCL runtime. Within a model the paths are
token-identical, and each model produces the same tokens on both GPUs:

| Model | Path | B70 | Lunar Lake |
|---|---|---|---|
| 1.7B | indirect | 296.2 | 114.4 |
| 1.7B | paged | 359.7 | 118.9 |
| 4B | indirect | 173.8 | 57.7 |
| 4B | paged | 180.1 | 58.4 |
| 8B | paged | 147.4 | 36.3 |
| 27B | paged | 45.0 | 7.4 |
| 27B | stateful + static decode | 41.3 | 6.6 |
| Bonsai 2 27B | paged | 44.7 | 7.0-7.8 |
| Bonsai 2 27B | stateful + static decode | 40.9 | |

Paged is ahead on every model on B70. On Lunar Lake it leads clearly for 1.7B
and 4B, while the two 27B paths sit within run-to-run variance of each other.

Compare paths only at an identical prompt: `bench_llm` defaults to a prompt with
the assistant and `<think>` suffix while `paged_bench_llm` does not, so leaving
the argument off makes the two disagree from the first token for reasons that
have nothing to do with the kernels.

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
once to get a representative number, or keep a persistent cache:

```bash
export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/syclcache_bonsai
```

---

## 7. Bonsai 2 27B (Hadamard-rotated basis)

[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
ships only as GGUF (`PQ2_0`), has the same Qwen3.5-27B architecture as Bonsai
27B, and stores its ternary weights in a *rotated basis*: the input of every
folded projection is sign-flipped and put through a blockwise (1024) normalised
Walsh-Hadamard transform, and the token embedding is stored rotated (whitepaper
A.2; the `prism.hadamard.*` GGUF metadata carries the contract).

### 7.1 Build the IR

The Bonsai 1 u2 IR from section 5 is the graph template; the tool swaps every
weight in from the GGUF and inserts the rotation:

```bash
# gguf-py of the PrismML llama.cpp fork (stock gguf does not know PQ2_0)
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp llama.cpp-prism
./venv/bin/pip install pyyaml

./venv/bin/python src/plugins/intel_gpu/tools/xetla_int2/bonsai2_gguf_to_ir.py \
  --gguf           <Ternary-Bonsai-2-27B-PQ2_0.gguf> \
  --template-dir   bonsai27b-u2 \
  --template-embed bonsai27b-fp16/openvino_text_embeddings_model.xml \
  --gguf-py        llama.cpp-prism/gguf-py \
  --out-dir        bonsai2-27b-u2
```

About 40 s. It writes `openvino_model.xml` (6.9 GB) and
`openvino_text_embeddings_model.xml` (2.5 GB, bf16, inverse rotation applied
offline). What it does:

* `PQ2_0` blocks are byte-for-byte the `u2` layout the plugin expects (codes
  `{0,1,2}`, zero point 1, fp16 scale per 128), so the 401 ternary matrices are
  copied, not re-quantized. llama.cpp's tiled GDN value-head order is undone the
  way the HF graph expects; the 96 GDN gate projections (`in_proj_a/b`) are
  dense bf16 in this release and replace the template's u2 subgraph.
* Norm weights, `A_log`, `dt_bias`, `conv1d` and the embedding come from the
  GGUF (norms are stored as `1+w`, which is what the exported graph holds).
* The rotation is inserted as `Reshape -> MatMul(H_1024) -> Reshape` in front
  of 257 projections (one shared 2 MB constant). Sign flips are folded into the
  preceding RMSNorm weight (129 norms; the dense `in_proj_a/b` are compensated),
  into the `up_proj` rows for `down_proj`'s input (64), and stay an explicit
  `Multiply` only for the 64 attention/GDN output projections, whose producer is
  not a per-channel weight.

`--verify` runs the same extraction on the *Bonsai 1* GGUF and compares it with
the Bonsai 1 IR: all 497 ternary matrices and the embedding are bit-identical,
the dense tensors match to fp16 rounding. `--explicit-signs` keeps every sign
flip as a `Multiply` (debug; same tokens).

### 7.2 Fused input transform

The GPU plugin folds the graph-level rotation back into the FullyConnected
(`FuseHadamardIntoFC`, registered after the horizontal FC fusion so a merged
gate/up projection absorbs the shared rotation once). The FC primitive carries
`hadamard_block` / `hadamard_signs`; the XeTLA int2 impl runs one fused
sign+FWHT SYCL kernel (`xetla/hadamard_fwht.cpp`, 128 items per 1024-block,
fp32 butterflies, launch-bound at decode) into a per-node scratch and points the
GEMV at it. Only the XeTLA impl honours the fields, so the impl manager throws
rather than silently falling back if such an FC is rejected.
`OV_XETLA_INT2_FUSE_HADAMARD=0` leaves the rotation in the graph;
`OV_XETLA_HADAMARD_DEBUG=1` traces the match.

### 7.3 Results (256 tokens, `BENCH_NO_EOS=1`, photosynthesis prompt)

| Path | Bonsai 27B | Bonsai 2 27B, rotation in graph | Bonsai 2 27B, fused |
|---|---|---|---|
| B70 paged | 44.9 tok/s, TTFT 107 ms | 39.7 tok/s | **44.7 tok/s, TTFT 107 ms** |
| B70 stateful + static decode | 41.3 tok/s | | **40.9 tok/s** |
| LNL paged | 8.0 tok/s, TTFT 453 ms | | **7.0-7.8 tok/s, TTFT 425-470 ms** |

Lunar Lake decode varies by +-6% between back-to-back runs (unified memory,
thermals); the B70 numbers repeat to three digits. Same command as on the B70,
no extra knobs.

The fused run is deterministic across runs. Its tokens match the graph-level
rotation for the first 135 tokens and then take a different (equally coherent)
continuation: the FWHT accumulates in fp32 where the graph MatMul used fp16.
The LNL run and the stateful run fork from the B70 paged run at the same
token 136 (a near-tie in the argmax). Decoded output starts *"The user wants a
concise explanation of photosynthesis in exactly or approximately 200
words..."*, then the essay.

### 7.4 Accuracy with lm-evaluation-harness (batched serving)

`tools/xetla_int2/paged_serve_llm_27b` serves a file of token-id requests with
continuous batching over the paged export (per-slot KV block ranges and
linear-attention state slots, a joint decode step, a `Gather` of the wanted
rows in front of `lm_head` so a prefill only emits the last token's logits).
`tools/xetla_int2/lm_eval_ov.py` is an lm-evaluation-harness LM that routes
every `generate_until` batch through it (chat template, thinking with
`reasoning_effort`, answer taken after `</think>`, greedy);
`run_lm_eval_ov.sh <jobid> <tag>` wraps both (`LIMIT`, `BATCH`, `THINK`,
`TASKS`).

Bonsai 2 27B, GSM8K test set (1319), `gsm8k_cot_llama` 8-shot, thinking
`medium`, greedy, B70, batch 16 -- against the same protocol on the vLLM
plugin (`xetla_vllm_plugin/scripts/eval_bonsai2_lm_eval.sh`):

| | OpenVINO + XeTLA int2 | vLLM + XeTLA int2 |
|---|---|---|
| exact match | **96.8%** (1277/1319) | 96.7% (1276/1319) |
| wall (1319 examples) | 89.8 min, 4.08 s/example | 90.5 min, 4.12 s/example |

38 items are wrong on both, 4 only here, 5 only on vLLM (fp-ordering noise
on near-tie items). The model card's math group is 96.57.

One bug was found on the way: the paged linear-attention ops always start
from the state in `la.block_indices[begin]` -- `past_len == 0` does *not*
imply a zero state (the op spec leaves zeroing to the user). A slot reused
without clearing its two state blocks inherits the previous request's GDN
state; accuracy then decayed from 96% on the first 130 requests to 87.4%
overall with runaway generations. The server clears the slot's state blocks
device-to-device on every refill; single-sequence benchmarks never hit this.

### 7.5 Prefill: M-tiled up-convert GEMM

For every `M > 1` the up-convert kernel now runs a real M tile on the 16-bit
DPAS (`WGM` 8 for M<=8, 16 for M<=16, else 32; `WGN` 64, `SGM` 8, `SGN` 16,
`SGK` 128), i.e. one pass over the weights per row tile, instead of the
per-row GEMV tier (M<128) or the earlier 128-wide tile (M>=128, ~8 ms/token
on the 27B: a 1260-token prompt went from 10.1 s to 3.2 s; `-2` keeps it). The result is bit-identical to the GEMV tier (same tokens
on the same binary). On the 21-token prompt: B70 TTFT 382 -> 107 ms, LNL
1833 -> ~440 ms; decode is untouched. `XETLA_INT2_PREFILL_CFG=-1` restores the
GEMV tier. The `M >= 128` path keeps its existing 128-wide GEMM tile. This
applies to every ternary model, not only Bonsai 2.

```bash
env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 OV_XETLA_INT2_MERGE_MLP=1 \
  ./build/paged_bench_llm_27b bonsai2-27b-u2/openvino_model.xml \
  bonsai2-27b-u2/openvino_text_embeddings_model.xml GPU 256 "$(cat prompt_photosynthesis_27b.txt)"
```

To compare against the stock path, set `OV_XETLA_INT2_DISABLE=1` and re-run.

---

## 8. Verifying correctness

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

## 9. Environment switches

Plugin:

| Variable | Effect |
|---|---|
| `OV_XETLA_INT2_DEBUG` | Log accepted/rejected nodes, shapes, scale layouts |
| `OV_XETLA_INT2_DISABLE` | Disable the implementation (falls back to OpenCL) |
| `OV_XETLA_INT2_ONLY_N=<list>` | Restrict to the listed output widths |
| `OV_XETLA_INT2_BARRIER=1` | Insert an explicit queue barrier before each GEMM |
| `OV_XETLA_INT2_DPAS_FOLD=1` | Fold post-ops into the int2 x int8 epilogue instead of the separate pass |
| `OV_XETLA_INT2_VERIFY=1` | At load time, dequantize the packed buffers and compare against the IR constant |
| `OV_XETLA_INT2_FUSE_HADAMARD=0` | Keep the Bonsai 2 input rotation as graph ops instead of fusing it into the FC (section 7) |
| `OV_XETLA_HADAMARD_DEBUG=1` | Trace the `FuseHadamardIntoFC` match per FullyConnected |

Kernel:

| Variable | Effect |
|---|---|
| `XETLA_INT2_CFG_DEBUG=1` | Print the tile/k-slicing config chosen per shape |
| `XETLA_INT2_DECODE_CFG=<n>` | Override the decode (M=1) configuration |
| `XETLA_INT2_PREFILL_CFG=-1` / `-2` | `-1`: per-row GEMV tier for M>1; `-2`: the old 128-wide tile for M>=128 (section 7.5) |
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
