#!/bin/bash
# lm_eval (GSM8K by default) on the OpenVINO TernOCL int2 path, Bonsai 2 27B.
#   bash run_lm_eval_ov.sh <slurm-jobid> <TAG> [extra lm_eval_ov.py args...]
# Env: LIMIT, BATCH (8), THINK (medium), TASKS, PY (python with lm_eval + transformers),
#      OV_BIN (OpenVINO lib dir), SERVE (paged_serve_llm_27b), GPU (ZE_AFFINITY_MASK)
set -uo pipefail
J=${1:?jobid}; TAG=${2:?tag}; shift 2
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HERE=$(cd "$TOOLS/../../../../.." && pwd)   # OpenVINO root

M=/data/nfs_home/egeorgan/cpu_ternary_vllm
MD=${MD:-$M/ov_models/bonsai2-27b-u2}
TOK=${TOK:-$M/models/Ternary-Bonsai-2-27B-packed}
PY=${PY:-$M/xetla_vllm_plugin/.venv/bin/python}
OUT=$HERE/ov_logs/lm_eval_$TAG
OV_BIN=${OV_BIN:-$HERE/bin/intel64/Release}
SERVE=${SERVE:-$TOOLS/build/paged_serve_llm_27b}
mkdir -p "$OUT"
srun --jobid="$J" --overlap bash -lc "
unset LD_LIBRARY_PATH
source /swtools/intel-gpu/latest/intel_gpu_vars.sh >/dev/null 2>&1
source /swtools/intel/2026.0/oneapi-vars.sh --force >/dev/null 2>&1
export ZE_AFFINITY_MASK=${GPU:-0}
export LD_LIBRARY_PATH=$OV_BIN:\${LD_LIBRARY_PATH:-}
export OV_TERNOCL_INT2_MERGE_MLP=1 HF_DATASETS_OFFLINE=\${HF_DATASETS_OFFLINE:-0}
cd $OUT && $PY $TOOLS/lm_eval_ov.py --lm $MD/openvino_model.xml --embed $MD/openvino_text_embeddings_model.xml \
  --tokenizer $TOK --serve $SERVE --tasks ${TASKS:-gsm8k_cot_llama} \
  ${LIMIT:+--limit $LIMIT} --batch ${BATCH:-8} --think ${THINK:-medium} --out $OUT $* 2>&1 | tee $OUT/run.log
"
