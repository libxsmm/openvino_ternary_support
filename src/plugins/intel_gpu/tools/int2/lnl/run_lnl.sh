#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce the best OpenVINO Bonsai 2 27B numbers on Lunar Lake (Arc 140V), see README.md.
#   ./run_lnl.sh <OpenVINO install dir> <GenAI model dir with openvino_mtp_model.xml>
set -euo pipefail
OV_INSTALL=${1:?OpenVINO install dir}
MODEL=${2:?GenAI model dir}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/lnl_results}
mkdir -p "$OUT"

set +u  # setupvars.sh reads unset variables
# shellcheck disable=SC1091
source "$OV_INSTALL/setupvars.sh" >/dev/null
set -u
PY=${PYTHON:-python3}
PIN=${PIN:-taskset -c 4}  # one P-core for the whole process (BKM: steadier than unpinned or per-thread pinning)

# The experiments of branch exp_lnl_int8_gemv_mtp (all opt-in):
export TERNOCL_INT8_PREFILL=1    # prompts (M > 8) on the int2 x int8 DPAS GEMM
export TERNOCL_INT8_GEMV_MIN_M=2 # MTP verify rows (2..8) on the int2 x int8 GEMV

# The model cache does not key on the TERNOCL_* variables: one cache dir per variant.
bench() {
  local tag=$1; shift
  echo "##### $tag  $(date +%T)" | tee -a "$OUT/results.txt"
  OV_CACHE_DIR=$OUT/cache_$tag $PIN "$PY" "$HERE/genai_bench.py" "$MODEL" "$@" 2>&1 \
    | grep --line-buffered -av "^\[transformers\]\|^Non default env value" | tee -a "$OUT/results.txt"
}

echo "host $(hostname) $(date -Is)" | tee "$OUT/results.txt"
# 1. chat prompts: TTFT, plain decode, MTP decode (k = 3 is the best for both engines)
bench default --modes plain,mtp --k 2,3
# 2. long prompts: TTFT only. The LNL int8 tile is faster from ~512 tokens (and ~10% slower at 135).
(export TERNOCL_LNL_TILE=1; bench lnltile --modes plain --prompts "" --long 512,2048 --n 1)
echo "done $(date -Is)" | tee -a "$OUT/results.txt"
