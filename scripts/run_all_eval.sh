#!/bin/bash
# 按需激活你的环境,例如: source ~/miniconda3/etc/profile.d/conda.sh; conda activate <env>
[ -n "${CONDA_INIT:-}" ] && source "$CONDA_INIT" && conda activate "${CONDA_ENV:-base}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ADP="${SFT_ADAPTER:?请设置 SFT_ADAPTER 指向 SFT 的 checkpoint-1113}"
run(){ CUDA_VISIBLE_DEVICES=0,2,3,4 python "$ROOT/scripts/eval_fin.py" --tp 4 "$@"; }
# base(无 adapter)
for bs in "cflue 800" "finqa 0" "math500 0" "gpqa 0"; do set -- $bs; run --benchmark $1 --sample $2 --tag base; done
# SFT(checkpoint-1113)
for bs in "cflue 800" "finqa 0" "math500 0" "gpqa 0"; do set -- $bs; run --benchmark $1 --sample $2 --tag sft --adapter "$ADP"; done
echo "ALL EVAL DONE"
