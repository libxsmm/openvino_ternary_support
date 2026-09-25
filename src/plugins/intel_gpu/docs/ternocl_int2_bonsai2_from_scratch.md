# Bonsai 2 27B on OpenVINO GPU with the TernOCL int2 kernels — from scratch

Curated, verified sequence (2026-09-24/25, branch `feature_integrate_2bit_ocl_kernels_mtp`,
which is `feature_integrate_2bit_ocl_kernels` plus MTP speculative decoding, section 5.4)
to go from an empty directory to Bonsai 2 27B decoding on an Arc Pro B70 and on
a Lunar Lake Arc 140V. For background see
[ternocl_int2_build_and_run.md](ternocl_int2_build_and_run.md) (full BKM, all
models) and [ternocl_int2_architecture.md](ternocl_int2_architecture.md).

Numbers you should reproduce (256 tokens, greedy, photosynthesis prompt, second run):

| GPU | decode | TTFT (20-token prompt) |
|---|---|---|
| Arc Pro B70 | 42.8 tok/s | 86 ms |
| Arc 140V (Lunar Lake) | 8.1–8.7 tok/s | 240 ms |

GSM8K (1319, 8-shot, thinking `medium`): 96.8% on the B70 in 50 min, the same
score as the XeTLA branch (96.8%, 90 min) and the vLLM plugin (96.7%).

---

## 0. What you need

| | |
|---|---|
| GPU | Intel Xe2: Arc Pro B70 (discrete) or Lunar Lake Arc 140V (integrated) |
| GPU runtime | Intel compute runtime (`intel_gpu_vars.sh`) — the plugin runs on its **OpenCL** runtime |
| Compiler | Intel oneAPI 2026.0 (`icx`/`icpx`) |
| Tools | CMake ≥ 3.16, Ninja, git, Python 3.10+ |
| Disk | ~150 GB: OpenVINO build ~25 GB, Bonsai 1 27B checkpoint 51 GB + fp16 IR 51 GB (template, see step 3), Bonsai 2 GGUF 7.2 GB, final IR 9.4 GB |
| RAM | ≥ 64 GB on the machine that runs the exports (step 3 loads the 27B in fp16) |

On this cluster the B70 hosts are the SLURM partitions `bmtxb70`
(`pcl-kini03/04`, 8 B70s each; pick one with `ZE_AFFINITY_MASK`) and `b70`;
Lunar Lake is `lnl` (`pcl-lnl01`). Every GPU step below is wrapped in
`srun --jobid=$JOB --overlap`. Get an allocation first:

```bash
JOB=$(sbatch -p bmtxb70 -w pcl-kini03 -t 12:00:00 --parsable --wrap "sleep 43200")   # or -p lnl -w pcl-lnl01
```

The environment every GPU command needs (put it in a file and `source` it):

```bash
# env.sh
unset LD_LIBRARY_PATH
source /swtools/intel-gpu/latest/intel_gpu_vars.sh
source /swtools/intel/2026.0/oneapi-vars.sh --force
export WORK=$HOME/ov-bonsai2
export OV_ROOT=$WORK/openvino
export LD_LIBRARY_PATH=$OV_ROOT/bin/intel64/Release:$LD_LIBRARY_PATH
export ZE_AFFINITY_MASK=0        # multi-GPU hosts: which B70 to use
```

---

## 1. Download the sources

```bash
mkdir -p $WORK && cd $WORK
git clone -b feature_integrate_2bit_ocl_kernels_mtp https://github.com/libxsmm/openvino_ternary_support.git openvino
cd openvino && git submodule update --init --recursive && cd ..      # ~5 min, 30 submodules incl. thirdparty/TernOCL

# gguf-py of the PrismML llama.cpp fork: stock gguf does not know the PQ2_0 tensor type
git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp llama.cpp-prism
```

The OpenCL kernels come from [TernOCL](https://github.com/libxsmm/TernOCL),
pinned as the submodule `src/plugins/intel_gpu/thirdparty/TernOCL`.

## 2. Build OpenVINO (GPU plugin + TernOCL int2 path) and the benchmark tools

The TernOCL sources are read from the submodule at configure time; set
`-DTERNOCL_ROOT=<dir>` only to build against another TernOCL checkout. No SYCL
or XeTLA is needed.

```bash
source env.sh && cd $OV_ROOT
cmake -B build-sycl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGPU_RT_TYPE=OCL \
  -DENABLE_INTEL_CPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_PYTHON=OFF -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF \
  -DENABLE_ONEDNN_FOR_GPU=ON -DENABLE_CM_FOR_GPU=ON \
  -DENABLE_SYSTEM_OPENCL=OFF \
  -DTHREADING=TBB_ADAPTIVE
cmake --build build-sycl -j $(nproc)          # 6 min on 24 cores with a warm ccache, 30-60 min cold

strings bin/intel64/Release/libopenvino_intel_gpu_plugin.so | grep -c int2_fp16_upcvt_gemm_mt   # non-zero: kernels embedded

cd src/plugins/intel_gpu/tools/int2
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DOpenVINO_DIR=$OV_ROOT/build-sycl && cmake --build build
ls build/   # bench_llm  paged_bench_llm  bench_llm_27b  paged_bench_llm_27b  paged_serve_llm_27b
```

The tools link with an RPATH to this build. To compare two OpenVINO builds
with one set of tools, configure them with `-DCMAKE_SKIP_RPATH=ON` and select
the build through `LD_LIBRARY_PATH` (OpenVINO loads the GPU plugin from next
to `libopenvino.so`).

Python side (model preparation only; no GPU, no OpenVINO build needed — it
uses the pip wheel):

```bash
cd $WORK
python3 -m venv venv && ./venv/bin/pip install openvino numpy pyyaml
./venv/bin/pip install "optimum-intel[openvino]==2.1.0" "transformers==5.2.0"   # step 3 only
```

## 3. Get the graph template: Bonsai 27B (1st generation) IR

Bonsai 2 27B ships only as GGUF and has *exactly* the Qwen3.5-27B architecture
of Bonsai 27B. Instead of rebuilding a 54 GB dense checkpoint and re-exporting,
the converter reuses the Bonsai 27B IR as the graph and swaps every weight in
from the GGUF (step 4). So the Bonsai 27B IR is a one-time prerequisite; if you
already have `bonsai27b-u2/` and `bonsai27b-fp16/openvino_text_embeddings_model.xml`
skip to step 4.

```bash
cd $WORK
hf download prism-ml/Ternary-Bonsai-27B-unpacked --local-dir bonsai27b-hf      # 51 GB, 12 safetensors shards

# fp16 export; the 27B is a VLM, so the task is image-text-to-text and it produces several models
./venv/bin/optimum-cli export openvino --model bonsai27b-hf \
    --task image-text-to-text --weight-format fp16 bonsai27b-fp16                # ~25 min, needs ~60 GB RAM

# rewrite the 497 ternary MatMuls as u2 codes + fp16 group scales (exact: the weights already are ternary)
./venv/bin/python $OV_ROOT/src/plugins/intel_gpu/tools/int2/quantize_ir_ternary.py \
    --in  bonsai27b-fp16/openvino_language_model.xml \
    --out bonsai27b-u2/openvino_model.xml
# expected: "rewriting 497 MatMul weights" ... "weights 47.72 GiB -> 6.34 GiB"
```

Only two artefacts of this step are used further: `bonsai27b-u2/openvino_model.xml`
(+`.bin`) and `bonsai27b-fp16/openvino_text_embeddings_model.xml` (+`.bin`,
2.5 GB). The rest of `bonsai27b-fp16` (51 GB) and `bonsai27b-hf` can be deleted.

## 4. Download Bonsai 2 27B and build its IR ("sidecar" step)

```bash
cd $WORK
hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir bonsai2-gguf   # 7.2 GB
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit tokenizer.json tokenizer_config.json chat_template.jinja \
    config.json generation_config.json --local-dir bonsai2-tok         # tokenizer + chat template (for the harness)

./venv/bin/python $OV_ROOT/src/plugins/intel_gpu/tools/int2/bonsai2_gguf_to_ir.py \
    --gguf           bonsai2-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --template-dir   bonsai27b-u2 \
    --template-embed bonsai27b-fp16/openvino_text_embeddings_model.xml \
    --gguf-py        llama.cpp-prism/gguf-py \
    --out-dir        bonsai2-27b-u2
```

~40 s, ~16 GB RAM. Expected log:

```
[ir] ...PQ2_0.gguf: 851 tensors, GDN nv=48 nk=16 hd=128
[ir] hadamard H1024: 401 folded weights, sign widths [5120, 6144, 17408]
[ir] 497 linear MatMuls in template
[ir] weights: 401 ternary, 96 dense, 64 up_proj row folds
[ir] hadamard: 257 rotations inserted, 64 explicit sign multiplies, 129 norm folds
[ir] embedding dequantised (248320, 5120)
[ir] embedding inverse-rotated
[ir] wrote bonsai2-27b-u2
```

Output: `bonsai2-27b-u2/openvino_model.xml` + `.bin` (6.9 GB; u2 weights, fp16
scales, the H_1024 rotation in front of every folded projection) and
`bonsai2-27b-u2/openvino_text_embeddings_model.xml` + `.bin` (2.5 GB bf16,
inverse-rotated offline). What the tool does and why is in section 7.1 of the
BKM. Self-check of the mapping (optional, 20 s): run it on the Bonsai 1 GGUF
with `--verify`; it must print `verify OK` (every ternary matrix bit-identical
to the template).

## 5. Run

### 5.1 B70: single-sequence benchmark (256 tokens, greedy)

```bash
source env.sh
TOOLS=$OV_ROOT/src/plugins/intel_gpu/tools/int2
IDS=$(cat $TOOLS/prompt_photosynthesis_27b.txt)      # chat-templated "Tell me about photosynthesis in 200 words"

srun --jobid=$JOB --overlap env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 OV_TERNOCL_INT2_MERGE_MLP=1 \
  $TOOLS/build/paged_bench_llm_27b $WORK/bonsai2-27b-u2/openvino_model.xml \
  $WORK/bonsai2-27b-u2/openvino_text_embeddings_model.xml GPU 256 "$IDS"
```

Run it twice; the first run builds the OpenCL programs (TTFT ~1.5 s, first
decode steps slower; the driver caches the binaries for later processes).
Second run:

```
TTFT (prefill) : 86 ms
decode         : 255 tokens in 5.96 s = 42.8 tok/s
generated_ids  =760,1156,6587,264,61446,15673,314,7022,71163,303,6681,466,12805,220,17,15,15,4105,13,...
```

which decodes (Bonsai 2 tokenizer) to *"The user wants a concise explanation of
photosynthesis in exactly or approximately 200 words. Let me craft a clear,
informative paragraph..."* followed by the essay. Runs are deterministic. The
ids are identical to the XeTLA branch for the first 135 tokens and then take
the same near-tie fork at token 136 that XeTLA's own variants (fused vs graph
rotation, paged vs stateful, B70 vs LNL) show.

For reference, the XeTLA branch on the same B70, same command, alternating
runs: 43.1 tok/s, TTFT 127 ms.

### 5.2 Lunar Lake / Arc 140V

Same binary, same command, `JOB` from the `lnl` partition. Expected (second run):

```
TTFT (prefill) : 240 ms
decode         : 255 tokens in 29-32 s = 8.1-8.7 tok/s
```

(XeTLA branch, same session: 8.0 tok/s, TTFT 627 ms.) LNL decode varies by a
few percent run to run (unified memory, thermals). A killed or crashed run can
leave device memory allocated until the SLURM job ends; recycle the allocation
before retrying.

### 5.3 Correctness with a standard harness (B70)

```bash
./venv/bin/pip install "lm_eval>=0.4.13" transformers
cd $WORK && mkdir -p eval && cd eval
srun --jobid=$JOB --overlap env OV_TERNOCL_INT2_MERGE_MLP=1 \
  $WORK/venv/bin/python $TOOLS/lm_eval_ov.py \
  --lm $WORK/bonsai2-27b-u2/openvino_model.xml --embed $WORK/bonsai2-27b-u2/openvino_text_embeddings_model.xml \
  --tokenizer $WORK/bonsai2-tok --serve $TOOLS/build/paged_serve_llm_27b \
  --tasks gsm8k_cot_llama --batch 16 --think medium --out .          # add --limit 100 for a 7-minute check
```

`run_lm_eval_ov.sh <jobid> <tag>` wraps the same call (`LIMIT`, `BATCH`,
`THINK`, `TASKS`, `OV_BIN`, `SERVE`, `GPU`). Full test set (1319 examples,
thinking, up to 4096 generated tokens): 50 min on the B70, `exact_match 0.968`
(1277/1319). First 100: ~0.96-0.98 depending on the slice.

### 5.4 MTP speculative decoding (optional)

```bash
hf download ProCreations/Ternary-Bonsai-2-27B-MTP model_mtp.safetensors --local-dir bonsai2-mtp
$WORK/venv/bin/pip install safetensors torch --extra-index-url https://download.pytorch.org/whl/cpu
$WORK/venv/bin/python $TOOLS/bonsai2_mtp_to_ir.py \
    --ir $WORK/bonsai2-27b-u2/openvino_model.xml --mtp bonsai2-mtp/model_mtp.safetensors \
    --gguf bonsai2-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf --gguf-py llama.cpp-prism/gguf-py \
    --weights i8 --out $WORK/bonsai2-27b-u2/openvino_mtp_i8_model.xml
# expected: "[mtp] draft: 513 ops ..." and a ~770 MB .bin

# bench / serve / run_lm_eval_ov.sh all take the draft from the environment
export BENCH_MTP=$WORK/bonsai2-27b-u2/openvino_mtp_i8_model.xml BENCH_MTP_K=3
```

Expected on the B70 with the 5.1 command: 72 tok/s at k=3 (plain 42.7), with the
same generated ids; on the Arc 140V with 128 tokens 21.4 tok/s (plain 11.4). Keep
batch <= 8 when serving with MTP.

## 6. Knobs that matter

| Variable | Default | Effect |
|---|---|---|
| `OV_TERNOCL_INT2_MERGE_MLP=1` | off | merge gate/up into one FC; used for every number above |
| `OV_TERNOCL_INT2_FUSE_HADAMARD=0` | fused | leave the rotation as graph ops (slower) |
| `OV_TERNOCL_INT2_DEBUG=1` | | which FCs the TernOCL impl accepted and why others were rejected |
| `OV_TERNOCL_INT2_CFG_DEBUG=1` | | every OpenCL program built and the tile chosen per (shape, M class) |
| `OV_TERNOCL_HADAMARD_DEBUG=1` | | trace the rotation fusion per FC (expect "fused 257 input rotations") |
| `OV_TERNOCL_INT2_DISABLE=1` | | fall back to the stock OpenVINO FC kernels |
| `BENCH_MAX_LEN` | 512 | context the bench reserves; prompt + new tokens must fit |

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot load library ... libsvml.so` | oneAPI not sourced in the process that runs the binary (`env.sh`) |
| `TernOCL kernel ... not found` at configure | the submodule is not checked out: `git submodule update --init src/plugins/intel_gpu/thirdparty/TernOCL` (or pass `-DTERNOCL_ROOT=<dir>`) |
| `ternocl int2: kernel build failed (...)` | the OpenCL compiler rejected the TernOCL source; the build log follows the message |
| `... carries a Hadamard input transform but the TernOCL impl rejected it` | an FC of the rotated model fell outside the impl's rules; the reason is in the message |
| `ModuleNotFoundError: yaml` from the converter | `pip install pyyaml` (gguf-py of the fork imports it) |
| converter: `need the PrismML llama.cpp fork's gguf-py` | pass `--gguf-py <fork>/gguf-py`; stock gguf lacks type id 142 (PQ2_0) |
| fluent nonsense output | wrong tokenizer for decoding (Bonsai 2 uses the Qwen3.5 vocabulary) — or a converter/mapping error: run `--verify` |
| decode ~35 tok/s and TTFT ~600 ms on the B70 | fused epilogues are not active (`ternocl_int2` missing from `primitive_inst::is_valid_fusion`) — an old build of this branch |
