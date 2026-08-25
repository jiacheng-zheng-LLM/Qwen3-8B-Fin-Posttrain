#!/bin/bash
# 阶段一 · LoRA 监督微调（SFT）。
# 命令参数与 weights_archive/sft-lora-checkpoint-1113-FINAL/args.json 一一对应，
# 该 args.json 是定稿 run 的完整配置存档，可逐项核对。
#
# 关键点：
#   - liger kernel 融合交叉熵：化解 152k 大词表 LM 头的 logits 显存峰值，
#     使 max_length 能从 4096 提到 5120（24GB 卡上 6144 仍 OOM）。
#   - 不传 --train_type：ms-swift 4.4.2 无此开关（E-1），给了 --lora_rank 即隐式走 LoRA。
#   - truncation_strategy=delete：超长样本整条丢弃而非截断。截断会让 output 变空，
#     collator 随后撞 None（见 EXPERIMENT_LOG 的退化样本事故）。
#   - 纯 DDP（args.json 的 deepspeed=None）。ZeRO-2 只在探索阶段用过（见 EXPERIMENT_LOG E-2/E-3），
#     liger 解决显存后定稿不再挂 deepspeed；ZeRO-3 因无 NVLink 通信太贵（330 s/it）已否决。
set -u

# 按需激活你的环境,例如: source ~/miniconda3/etc/profile.d/conda.sh; conda activate <env>
[ -n "${CONDA_INIT:-}" ] && source "$CONDA_INIT" && conda activate "${CONDA_ENV:-base}"

# ---- 必填环境变量 ----
#   BASE_MODEL    基座 Qwen3-8B 路径
#   SFT_DATA_DIR  蒸馏产出的 SFT 数据目录（见 DISTILLATION.md）
#   REPO_ROOT     仓库根目录;默认按脚本位置推断
ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE="${BASE_MODEL:?请设置 BASE_MODEL 指向 Qwen3-8B}"
DATA_DIR="${SFT_DATA_DIR:-$ROOT/data/train}"
SW="${SWIFT_BIN:-swift}"
RUNDIR="${RUN_DIR:-$ROOT/runs}"; mkdir -p "$RUNDIR"
OUT="$RUNDIR/sft"

# 6 卡 DDP（定稿 run 用 6 卡：35584 条 / 有效 batch 96(bs1*ga16*6) = 371 步/epoch × 3 = 1113 步）
# 按你的机器改 CUDA_VISIBLE_DEVICES 与 NPROC_PER_NODE
GPUS="${SFT_GPUS:-0,2,3,4,5,6}"
NPROC="${SFT_NPROC:-6}"

env CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE="$NPROC" \
    MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=0 \
    NCCL_P2P_DISABLE=1 \
  "$SW" sft \
    --model "$BASE" --model_type qwen3 \
    --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
    --target_modules all-linear \
    --dataset "$DATA_DIR/cflue_mcq.f5120.json" \
              "$DATA_DIR/cflue_oe.f5120.json" \
              "$DATA_DIR/fin_qa.decontam.f5120.json" \
    --torch_dtype bfloat16 --attn_impl sdpa \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --weight_decay 0.1 --max_grad_norm 1.0 \
    --max_length 5120 --truncation_strategy delete --packing false \
    --use_liger_kernel true --gradient_checkpointing true \
    --save_strategy epoch --logging_steps 5 --seed 42 \
    --output_dir "$OUT" --report_to tensorboard \
  >> "$RUNDIR/sft_trainer.log" 2>&1

echo "SFT done → $OUT  (log: $RUNDIR/sft_trainer.log)"
