#!/bin/bash
# 按需激活你的环境,例如: source ~/miniconda3/etc/profile.d/conda.sh; conda activate <env>
[ -n "${CONDA_INIT:-}" ] && source "$CONDA_INIT" && conda activate "${CONDA_ENV:-base}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
GRPO="${GRPO_ADAPTER:?请设置 GRPO_ADAPTER 指向 GRPO 的 checkpoint-1210}"
for bs in "cflue 800" "finqa 0" "math500 0" "gpqa 0"; do set -- $bs
  CUDA_VISIBLE_DEVICES=3,4,5,6 MKL_THREADING_LAYER=GNU python "$ROOT/scripts/eval_fin.py" --benchmark $1 --sample $2 --tp 4 --tag grpo --adapter "$GRPO"
done
echo "GRPO EVAL DONE"
