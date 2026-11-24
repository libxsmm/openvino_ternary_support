#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# PrismML llama.cpp with TernSYCL on Lunar Lake, the reference for run_lnl.sh (README.md).
#   ./run_llama.sh <llama.cpp build-sycl/bin> <Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf>
set -euo pipefail
BIN=${1:?llama.cpp bin dir}
GGUF=${2:?combined PQ2_0 + MTP GGUF}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/llama_results}
mkdir -p "$OUT"
PIN=${PIN:-taskset -c 4}
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export GGML_SYCL_PTQ1_T2=all              # PQ2_0 weights on the TernSYCL int2 x int8 DPAS kernels
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost

echo "host $(hostname) $(date -Is)" | tee "$OUT/results.txt"
# 1. kernel-level prefill / decode (default ubatch 512)
$PIN "$BIN/llama-bench" -m "$GGUF" -ngl 99 -fa 1 -p 135,512,2048 -n 32 -r 3 2>&1 | grep -a "pp\|tg\|model" | tee -a "$OUT/results.txt"
# 2. server, chat prompts: plain and MTP (k = 3 is the best)
srv() { $PIN python3 "$HERE/llama_bench.py" --server "$BIN/llama-server" --pace 10 "$@" | tee -a "$OUT/results.txt"; }
srv --name "plain" --args "-m $GGUF -ngl 99 -fa on" --save "$OUT/ref.json" --log "$OUT/srv_plain.log"
for k in 2 3; do
  srv --name "mtp k=$k" --ref "$OUT/ref.json" --log "$OUT/srv_mtp$k.log" \
      --args "-m $GGUF -ngl 99 -fa on --spec-type draft-mtp --spec-draft-n-max $k --spec-draft-n-min 0 --spec-draft-p-min 0"
done
echo "done $(date -Is)" | tee -a "$OUT/results.txt"
