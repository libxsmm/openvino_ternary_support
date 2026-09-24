# Building and Running the TernOCL int2 GPU Path

Instructions for building this fork and running a ternary (2-bit) model
through the TernOCL `FullyConnected` implementation (OpenCL kernels) on an Intel
Xe2 GPU.

See [ternocl_int2_architecture.md](ternocl_int2_architecture.md) for how it
works and [ternocl_int2_bonsai2_worklog.md](ternocl_int2_bonsai2_worklog.md) for
the record of the port from XeTLA. For Bonsai 2 27B specifically, the curated
from-scratch sequence (download, build, IR conversion, B70 and Lunar Lake runs)
is [ternocl_int2_bonsai2_from_scratch.md](ternocl_int2_bonsai2_from_scratch.md).

---

## 1. Prerequisites

| Component | Notes |
|---|---|
| GPU | Intel Xe2 GPU, discrete (Arc Pro B70) or integrated (Lunar Lake) |
| GPU runtime | Intel compute runtime (`intel_gpu_vars.sh` or distro packages) |
| Compiler | Intel oneAPI 2026.0 (`icx` / `icpx`), or any compiler OpenVINO supports |
| Build tools | CMake >= 3.16, Ninja |
| Python | 3.10+ with `openvino` and `numpy` (model preparation only) |
| Kernels | [TernOCL](https://github.com/libxsmm/TernOCL), pinned as the `thirdparty/TernOCL` submodule |

---

## 2. Sources

```bash
export WORK=$HOME/ov-int2
mkdir -p $WORK && cd $WORK

git clone https://github.com/libxsmm/openvino_ternary_support.git openvino
cd openvino
git checkout feature_integrate_2bit_ocl_kernels
git submodule update --init --recursive   # includes src/plugins/intel_gpu/thirdparty/TernOCL
cd ..
```

The configure step embeds three files of the TernOCL submodule into the
plugin: `int2_fp16_upcvt/int2_fp16_upcvt.cl`, `common/epilogue.clh` and
`hadamard/hadamard_fwht.cl`. To move to newer kernels, update the submodule
(`git -C src/plugins/intel_gpu/thirdparty/TernOCL pull` and commit the new
pointer); to try local kernel edits without committing, point `TERNOCL_ROOT`
at another TernOCL checkout.

---

## 3. Environment

```bash
unset LD_LIBRARY_PATH
source <intel-gpu-runtime>/intel_gpu_vars.sh
source <oneapi-install>/oneapi-vars.sh --force

export OV_ROOT=$WORK/openvino
# export TERNOCL_ROOT=<TernOCL checkout>  # optional: kernels from elsewhere than the submodule
```

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

- The TernOCL impl needs the OpenCL runtime (`GPU_RT_TYPE=OCL`); it builds its
  programs on the plugin's `cl_context`. No SYCL or XeTLA is involved.
- The kernel sources are read at configure time and re-read whenever they
  change (they are CMake configure dependencies), so editing a TernOCL kernel
  and rebuilding is enough.
- Artifacts are written to `$OV_ROOT/bin/intel64/Release`. If you configure a
  second build tree from the same source, give it its own `OUTPUT_ROOT`.

Confirm the path is compiled in:

```bash
strings bin/intel64/Release/libopenvino_intel_gpu_plugin.so | grep -c int2_fp16_upcvt_gemm_mt   # non-zero
```

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
./venv/bin/python src/plugins/intel_gpu/tools/int2/quantize_ir_ternary.py \
  --in  bonsai8b-fp16/openvino_model.xml \
  --out bonsai8b-u2/openvino_model.xml

./venv/bin/python src/plugins/intel_gpu/tools/int2/quantize_ir_ternary.py \
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
./venv_export/bin/python src/plugins/intel_gpu/tools/int2/untie_embeddings.py \
  --in <checkpoint> --out <checkpoint>-untied
```

Export from the untied directory. `verify_ternary_u2.py` reports whether the
head ended up compressed.

### 5.4 Verifying the conversion

```bash
./venv/bin/python src/plugins/intel_gpu/tools/int2/verify_ternary_u2.py \
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
OV_TERNOCL_INT2_DEBUG=1 <your-app> 2>&1 | grep "ternocl-int2"
```

Accepted nodes are logged as `accepted <node id>`, and nodes that fall back
report the reason.

The first run in a process builds the OpenCL programs it needs (one per
kernel option set, 16 for the Bonsai 2 27B), which shows up as a slow first
TTFT and first few decode steps. The GPU driver caches the binaries, so later
processes start faster; for benchmarks, discard the first run.

### Benchmark tools

`src/plugins/intel_gpu/tools/int2` contains greedy-decode benchmarks. They
are not built by the main build; build them against the OpenVINO build tree:

```bash
cd $OV_ROOT/src/plugins/intel_gpu/tools/int2
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
| `BENCH_MAX_LEN=<n>` | Upper bound on the sequence dimension. Over-reserving only wastes device memory; 512 is a good default |
| `BENCH_STATIC_DECODE=1` | Pin the token dimension to 1 and feed the prompt one token at a time (27B tools) |
| `BENCH_PRECISION=f16` | Inference precision hint |


### Measured results

Bonsai 2 27B, paged, 256 tokens, `BENCH_NO_EOS=1`, `OV_TERNOCL_INT2_MERGE_MLP=1`,
photosynthesis prompt (20 tokens), second run, OpenCL runtime. Both branches
measured in the same session on the same GPU:

| GPU | TernOCL decode | TernOCL TTFT | XeTLA decode | XeTLA TTFT |
|---|---|---|---|---|
| Arc Pro B70 | 42.8 tok/s | 86 ms | 43.1 tok/s | 127 ms |
| Arc 140V (Lunar Lake) | 8.1-8.7 tok/s | 240 ms | 8.0 tok/s | 627 ms |

On the B70 decode is at parity (the GEMVs are within a few percent of XeTLA's
and the rest of the step is identical); prefill is 1.5x faster from the tuned
M-tiled kernels. The other models listed in the XeTLA branch (1.7B, 4B, 8B,
Bonsai 1 27B) use the same shape families and run on this path, but have not
been re-measured here.

---

## 7. Bonsai 2 27B (Hadamard-rotated basis)

[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
ships only as GGUF (`PQ2_0`), has the same Qwen3.5-27B architecture as Bonsai
27B, and stores its ternary weights in a *rotated basis*: the input of every
folded projection is sign-flipped and put through a blockwise (1024) normalised
Walsh-Hadamard transform, and the token embedding is stored rotated.

### 7.1 Build the IR

The Bonsai 1 u2 IR from section 5 is the graph template; the tool swaps every
weight in from the GGUF and inserts the rotation:

```bash
# gguf-py of the PrismML llama.cpp fork (stock gguf does not know PQ2_0)
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp llama.cpp-prism
./venv/bin/pip install pyyaml

./venv/bin/python src/plugins/intel_gpu/tools/int2/bonsai2_gguf_to_ir.py \
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
`hadamard_block` / `hadamard_signs`; the TernOCL impl runs one fused sign+FWHT
OpenCL kernel (`thirdparty/TernOCL/hadamard/hadamard_fwht.cl`, 128 items per 1024-block,
fp32 butterflies) into a per-node scratch and points the GEMM at it. Only the
TernOCL impl honours the fields, so the impl manager throws rather than
silently falling back if such an FC is rejected.
`OV_TERNOCL_INT2_FUSE_HADAMARD=0` leaves the rotation in the graph;
`OV_TERNOCL_HADAMARD_DEBUG=1` traces the match.

### 7.3 Accuracy with lm-evaluation-harness (batched serving)

`tools/int2/paged_serve_llm_27b` serves a file of token-id requests with
continuous batching over the paged export (per-slot KV block ranges and
linear-attention state slots, a joint decode step, a `Gather` of the wanted
rows in front of `lm_head` so a prefill only emits the last token's logits).
`tools/int2/lm_eval_ov.py` is an lm-evaluation-harness LM that routes every
`generate_until` batch through it (chat template, thinking with
`reasoning_effort`, answer taken after `</think>`, greedy);
`run_lm_eval_ov.sh <jobid> <tag>` wraps both (`LIMIT`, `BATCH`, `THINK`,
`TASKS`, `OV_BIN`, `SERVE`, `GPU`).

The paged linear-attention ops always start from the state in
`la.block_indices[begin]` -- `past_len == 0` does *not* imply a zero state.
The server clears a slot's state blocks on every refill; single-sequence
benchmarks never hit this.

Bonsai 2 27B, GSM8K test set (1319), `gsm8k_cot_llama` 8-shot, thinking
`medium`, greedy, Arc Pro B70, batch 16, same protocol on all three:

| | exact match | wall (1319 examples) |
|---|---|---|
| OpenVINO + TernOCL int2 | **96.8%** (1277/1319) | 50.3 min |
| OpenVINO + XeTLA int2 | 96.8% (1277/1319) | 89.8 min |
| vLLM + XeTLA int2 | 96.7% (1276/1319) | 90.5 min |

The model card's math group is 96.57. The serving run is prefill-heavy (every
request is an 8-shot prompt of 1200-1370 tokens, and the joint decode step runs at
M = batch), which is where the M-tiled TernOCL kernels are fastest.

### 7.4 MTP speculative decoding

The ProCreations MTP head for Bonsai 2 27B
([ProCreations/Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP),
`model_mtp.safetensors`: one Qwen3.5 full-attention layer + `fc` + norms)
becomes a draft IR that reuses a full-attention layer of the target IR and the
target's u2 `lm_head`:

```bash
python tools/int2/bonsai2_mtp_to_ir.py --ir bonsai2-27b-u2/openvino_model.xml \
    --mtp Ternary-Bonsai-2-27B-MTP/model_mtp.safetensors \
    --gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --gguf-py llama.cpp-prism/gguf-py \
    --weights i8 --out bonsai2-27b-u2/openvino_mtp_i8_model.xml
```

The draft input is `[embed(t+1) | h_t]` with `h_t` the target's post-final-norm
hidden state (the input of its `lm_head` rotation, exposed as an extra output
at load time); the Hadamard sign vector folded into that norm is folded into
the draft's `pre_fc_norm_hidden` and `mtp.norm` (read from the GGUF).
`--weights i8` stores the head as int8 with per-128 fp16 scales.

Both `paged_bench_llm_27b` and `paged_serve_llm_27b` run the draft/verify loop
with `BENCH_MTP=<draft xml>` and `BENCH_MTP_K=<k>` (default 3): k draft steps,
one target step over the k+1 tokens, greedy acceptance, bonus token. The
linear-attention state is rolled back through the paged ops' `cache_interval`:
with `la.cache_interval = 1` the GDN and conv1d kernels write the state after
token t to `la.block_indices[begin + 1 + t]`, so each sequence keeps k+2 state
slots and continues from the slot of its last accepted token. Rejected
full-attention KV entries are simply overwritten. The verify steps run M = k+1
GEMVs, which have their own tiles (section 6 of the architecture doc).

Bonsai 2 27B, greedy, int8 draft (`BENCH_NO_EOS=1`, photosynthesis prompt):

| | plain | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| Arc Pro B70, 256 tokens (tok/s) | 42.7 | 62.9 | 66.8 | **72.0** | 67.3 |
| acceptance | | 88% | 76% | 66% | 55% |
| Arc 140V (LNL), 128 tokens (tok/s) | 11.4 | 14.4 | 16.8 | **21.4** | |

Generation is identical to plain decode (k=2, k=3 over all 256 tokens; k=1, k=4
take a near-tie fork at token 148). Serving 64 GSM8K prompts on the B70
(acceptance 83%, 3.5 tokens per round):

| batch | plain (tok/s) | MTP k=3 (tok/s) |
|---|---|---|
| 1 | 36.7 | 65.5 (x1.78) |
| 8 | 116.9 | 145.3 (x1.24) |
| 16 | 135.3 | 126.1 (x0.93) |

Full GSM8K with MTP k=3, batch 8 (`BENCH_MTP=... BENCH_MTP_K=3 BATCH=8
run_lm_eval_ov.sh ...`): **96.89%**, 41.8 min, 84.4% acceptance -- the same
score as without MTP. Above batch 8 the k+1-row verify steps cost more than
they save; use plain decode there.

---

## 8. Verifying correctness

Decode is deterministic run to run: two runs with the same prompt produce
identical `generated_ids`.

The kernels themselves are validated in TernOCL (`validate.sh`,
`validate_epilogues.sh`: every tile class, both dtypes, all epilogues, ragged
shapes, against an fp32 host reference), and its parity tool shows the fused
epilogues bit-identical to XeTLA's on the same inputs. End to end, the
generated ids match the XeTLA branch for the first 135 tokens of the reference
prompt, then fork at a near-tie argmax, the same fork XeTLA's own variants
show (fp32 vs fp16 accumulation order).

---

## 9. Environment switches

Plugin:

| Variable | Effect |
|---|---|
| `OV_TERNOCL_INT2_DEBUG=1` | Log accepted/rejected nodes and per-node packing |
| `OV_TERNOCL_INT2_DISABLE=1` | Disable the implementation (falls back to the stock OpenCL FC) |
| `OV_TERNOCL_INT2_ONLY_N=<list>` | Restrict to the listed output widths |
| `OV_TERNOCL_INT2_FOLD_GATES=bias\|sigmoid\|none` | Which GatedDeltaNet gate epilogues are folded (default both) |
| `OV_TERNOCL_INT2_MERGE_MLP=1` | Merge parallel gate/up compressed FCs into one 2I FC followed by the SwiGLU primitive |
| `OV_TERNOCL_INT2_FUSE_HADAMARD=0` | Keep the Bonsai 2 input rotation as graph ops (section 7) |
| `OV_TERNOCL_HADAMARD_DEBUG=1` | Trace the `FuseHadamardIntoFC` match per FullyConnected |
| `OV_GPU_FUSE_RMS_ROPE=1` | Fold the per-head Q/K RMSNorm into the following RoPE kernel |

Kernel selection (sweeps):

| Variable | Effect |
|---|---|
| `OV_TERNOCL_INT2_CFG_DEBUG=1` | Print every program built and the tile chosen per (shape, M class) |
| `OV_TERNOCL_INT2_GEMV=wgn,ls,u` | Override the GEMV tile (M <= 8) |
| `OV_TERNOCL_INT2_MID=mt_m,mt_n,wg_m,wg_n` | Override the M-tiled tile for 8 < M < 64 |
| `OV_TERNOCL_INT2_MT=mt_m,mt_n,wg_m,wg_n` | Override the M-tiled tile for M >= 64 |

Benchmark tool:

| Variable | Effect |
|---|---|
| `BENCH_PROFILE=1` | Per-op profile of one decode step, with call counts |
| `BENCH_NO_EOS=1` | Keep decoding past EOS so every run does equal work |
| `BENCH_DUMP_LOGITS=<step>:<path>` | Dump raw f32 logits at a decode step |
| `BENCH_MTP=<draft xml>` | MTP speculative decoding with that draft (bench and serve, section 7.4) |
| `BENCH_MTP_K=<k>` | Draft tokens per round (default 3) |
| `BENCH_MTP_DEBUG=1` | Per-round draft/accept trace (bench) |
