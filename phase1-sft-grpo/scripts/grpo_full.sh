#!/bin/bash
# 金融推理 GRPO smoke(分离部署,proven --adapters 方式):rollout(GPU5,6 TP2)+ 训练器(GPU2,3,4 DDP)。
set -u
# 按需激活你的环境,例如: source ~/miniconda3/etc/profile.d/conda.sh; conda activate <env>
[ -n "${CONDA_INIT:-}" ] && source "$CONDA_INIT" && conda activate "${CONDA_ENV:-base}"
# ---- 必填环境变量 ----
#   REPO_ROOT   本目录的上级(phase1-sft-grpo/);默认按脚本位置推断
#   BASE_MODEL  基座 Qwen3-8B 路径
#   SFT_ADAPTER SFT 产出的 LoRA adapter 目录(checkpoint-1113)
ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE="${BASE_MODEL:?请设置 BASE_MODEL 指向 Qwen3-8B}"
CKPT="${SFT_ADAPTER:?请设置 SFT_ADAPTER 指向 SFT 的 checkpoint-1113}"
DATA="$ROOT/data/grpo_hardcase.jsonl"; PORT=8137
SW="${SWIFT_BIN:-swift}"
RUNDIR="${RUN_DIR:-$ROOT/runs}"; mkdir -p "$RUNDIR"
SVLOG="$RUNDIR/rollout_server.log"; TRLOG="$RUNDIR/grpo_full_trainer.log"; OUT="$RUNDIR/grpo-full"
COMMON="MKL_THREADING_LAYER=GNU VLLM_USE_FLASHINFER_SAMPLER=0 NCCL_P2P_DISABLE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost"
rm -rf "$OUT"; : > "$SVLOG"
echo "[orch] rollout(GPU5,6 TP2, --adapters)..."
env CUDA_VISIBLE_DEVICES=5,6 $COMMON $SW rollout --model "$BASE" --adapters "$CKPT"   --vllm_tensor_parallel_size 2 --vllm_max_model_len 4096 --vllm_gpu_memory_utilization 0.8   --vllm_use_async_engine true --host 127.0.0.1 --port $PORT >> "$SVLOG" 2>&1 &
SVPID=$!
echo "[orch] PID=$SVPID,等 /health/=200(上限8min)..."
ready=0
for i in $(seq 1 96); do
  ! kill -0 "$SVPID" 2>/dev/null && { echo "[orch] ❌服务提前退出"; exit 1; }
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health/ 2>/dev/null)" = "200" ] && { ready=1; echo "[orch] ✅就绪 ~$((i*5))s"; break; }
  sleep 5
done
[ "$ready" != 1 ] && { echo "[orch] ❌未就绪"; kill -9 $SVPID 2>/dev/null; exit 1; }
echo "[orch] 训练器(GPU2,3,4 DDP)..."
env CUDA_VISIBLE_DEVICES=3,4 NPROC_PER_NODE=2 $COMMON $SW rlhf --rlhf_type grpo   --model "$BASE" --adapters "$CKPT" --ref_adapters "$CKPT"   --lora_rank 32 --lora_alpha 64 --target_modules all-linear   --external_plugins "$ROOT/plugin/fin_orm.py"   --reward_funcs fin_acc fin_format --reward_weights 1.0 0.1   --dataset "$DATA" --torch_dtype bfloat16 --attn_impl sdpa   --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port $PORT   --vllm_server_timeout 600 --vllm_server_pass_dataset true   --num_generations 4 --max_completion_length 1536  --per_device_train_batch_size 1 --gradient_accumulation_steps 4   --learning_rate 1e-6 --beta 0.04 --num_train_epochs 1 --max_length 4096   --gradient_checkpointing true --logging_steps 1 --save_strategy epoch   --output_dir "$OUT" --report_to tensorboard >> "$TRLOG" 2>&1
TR=$?; echo "[orch] 训练器退出=$TR,杀服务"; kill $SVPID 2>/dev/null; sleep 3; kill -9 $SVPID 2>/dev/null; pkill -9 -f "swift.*rollout" 2>/dev/null; pkill -9 -f "VLLM::Worker" 2>/dev/null; pkill -9 -f "swift/cli/rollout" 2>/dev/null
echo "[orch] GRPO smoke DONE exit=$TR"
