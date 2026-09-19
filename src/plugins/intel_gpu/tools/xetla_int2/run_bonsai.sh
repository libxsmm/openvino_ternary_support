#!/usr/bin/env bash
# Reproduce the Bonsai 8B / 27B ternary GPU numbers.
#
#   run_bonsai.sh <models-dir> [tokens]
#
# Expects, under <models-dir>:
#   bonsai8b-u2/openvino_model.xml
#   bonsai27b-u2/openvino_model.xml
#   bonsai27b-fp16/openvino_text_embeddings_model.xml
# See xetla_int2_build_and_run.md for how to produce them.
set -uo pipefail

MODELS=${1:?usage: run_bonsai.sh <models-dir> [tokens]}
TOKENS=${2:-256}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$HERE/build

[ -x "$BENCH/paged_bench_llm_27b" ] || {
    echo "drivers not built; see the 'Benchmark tools' section of the doc" >&2
    exit 1
}

# The first run after a build JITs the kernels, so TTFT is an order of magnitude
# too high unless the cache persists across runs.
export SYCL_CACHE_PERSISTENT=1
export SYCL_CACHE_DIR=${SYCL_CACHE_DIR:-/tmp/syclcache_bonsai}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-opencl:gpu}

IDS=$(cat "$HERE/prompt_photosynthesis_27b.txt")

echo "### 8B, paged"
env OV_XETLA_INT2_MERGE_MLP=1 OV_GPU_FUSE_RMS_ROPE=1 \
    "$BENCH/paged_bench_llm" "$MODELS/bonsai8b-u2/openvino_model.xml" GPU "$TOKENS" \
    2>&1 | grep -aE "TTFT|decode |generated_ids"

echo "### 27B, paged"
env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 OV_XETLA_INT2_MERGE_MLP=1 \
    "$BENCH/paged_bench_llm_27b" "$MODELS/bonsai27b-u2/openvino_model.xml" \
    "$MODELS/bonsai27b-fp16/openvino_text_embeddings_model.xml" GPU "$TOKENS" "$IDS" \
    2>&1 | grep -aE "TTFT|decode |generated_ids"

echo "### 27B, stateful with a static decode shape"
env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_STATIC_DECODE=1 BENCH_NO_EOS=1 \
    OV_XETLA_INT2_MERGE_MLP=1 \
    "$BENCH/bench_llm_27b" "$MODELS/bonsai27b-u2/openvino_model.xml" \
    "$MODELS/bonsai27b-fp16/openvino_text_embeddings_model.xml" GPU "$TOKENS" "$IDS" \
    2>&1 | grep -aE "TTFT|decode |generated_ids"

if [ -f "$MODELS/bonsai2-27b-u2/openvino_model.xml" ]; then
    echo "### Bonsai 2 27B, paged (fused Hadamard input transform)"
    env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 OV_XETLA_INT2_MERGE_MLP=1 \
        "$BENCH/paged_bench_llm_27b" "$MODELS/bonsai2-27b-u2/openvino_model.xml" \
        "$MODELS/bonsai2-27b-u2/openvino_text_embeddings_model.xml" GPU "$TOKENS" "$IDS" \
        2>&1 | grep -aE "TTFT|decode |generated_ids"
fi
