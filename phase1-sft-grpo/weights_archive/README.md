# 权重归档 · 合并前 LoRA adapter 备份

> 目的:保存所有训练产出的**合并前 LoRA 权重**独立一份,防止后续 merge 操作或误删丢失原始 adapter。
> 归档时间:2026-08-19。校验:每个 `adapter_model.safetensors` 的 sha256 与训练产出的原文件逐一一致(见 `SHA256SUMS.txt`)。

## 归档内容

| 目录 | 阶段 | 步数 | adapter 大小 | 用途 |
|---|---|---|---|---|
| `sft-lora-checkpoint-371` | SFT | 371 | 349MB | SFT 第1个 epoch 中间态(对照) |
| `sft-lora-checkpoint-742` | SFT | 742 | 349MB | SFT 第2个 epoch 中间态(对照) |
| `sft-lora-checkpoint-1113-FINAL` | SFT | 1113 | 349MB | **SFT 最终**(3 epoch 完成) |
| `grpo-lora-checkpoint-1210-FINAL` | GRPO | 1210 | 174MB | **SFT+GRPO 合体最终**(续训 SFT adapter) |

每个目录含:
- `adapter_model.safetensors` + `adapter_config.json` —— **合并/推理所需**(base + 此 adapter);
- `optimizer.pt` / `scheduler.pt` / `rng_state_*.pth` / `trainer_state.json` —— **续训所需**;
- `args.json` / `training_args.bin` —— 训练超参存档。

## 关系说明(重要)
- **GRPO 的 checkpoint-1210 不是"只有 GRPO 增量"**:GRPO 是用 `--adapters <SFT-1113>` 续训 SFT 适配器,故 **1210 = SFT+GRPO 合体权重**。合并它 = 得到最终交付模型。
- **SFT-1113 与 GRPO-1210 二选一即可作最终模型**:评测均值 SFT 47.6 vs GRPO 47.5(噪声内等价);GPQA 上 SFT(21.7)略优于 GRPO(19.2)。
- 合并后的全量 SFT 模型(16G)是**派生产物,未纳入本归档**——随时可由 `base + sft-lora-checkpoint-1113` 重新合出,不占备份优先级。

## 如何使用(合并前 / 合并后)

### A. 直接用 adapter 推理(不合并)
```bash
# vLLM 动态挂载 LoRA
python -m vllm.entrypoints.openai.api_server \
  --model <BASE:Qwen3-8B> --enable-lora \
  --lora-modules fin=weights_archive/grpo-lora-checkpoint-1210-FINAL
```

### B. 合并后用(生产推荐)
```bash
swift export --adapters weights_archive/grpo-lora-checkpoint-1210-FINAL \
  --merge_lora true --output_dir /srv/qwen3-fin-final
# 合并后原 adapter 仍完整保留在本归档,可随时回到"合并前"状态
```

### C. 续训(如需再跑一版 GRPO)
```bash
swift rlhf --adapters weights_archive/sft-lora-checkpoint-1113-FINAL \
  --ref_adapters weights_archive/sft-lora-checkpoint-1113-FINAL ...
```

## 完整性校验
```bash
cd weights_archive && sha256sum -c SHA256SUMS.txt
```

## Base 模型(合并时需要)
`/path/to/models/Qwen3-8B` —— adapter 必须配对此 base 才能合并/加载。
