# 数据说明

本仓库**不含任何模型权重、训练数据或上游数据集**。这里说明每一项去哪取、什么许可、
以及本仓库里保留了什么派生物。

---

## 1. SFT 训练数据的构成

SFT 数据由两部分组成，来源不同，须分开看：

| 组成 | 来源 | 许可 |
|---|---|---|
| **题干 / 选项 / 标准答案** | 真实金融考题（CFLUE 中文金融题库、FinQA 英文数值推理） | 遵守各自原始条款，见 §2 |
| **推理链（CoT）** | **本项目用 DeepSeek-V4-Pro-0813 重新蒸馏** | 属教师模型输出，见 [DISTILLATION.md](DISTILLATION.md) §5 |

题集的划分与规模按 DianJin-R1（[arXiv:2504.15716](https://arxiv.org/abs/2504.15716)）
论文 Table 1 组织：CFLUE MCQ 26 672 / CFLUE 开放题 5 045 / FinQA 4 851，
合计 36 568 条（论文另有专有子集 CCC 1 800 未开源，本项目未使用）。

> **本项目未使用上游发布的 `DianJin-R1-Data` 数据集本身**，只沿用了它的字段格式、
> 题集划分与数据规模。该数据集里的推理链是论文作者用 DeepSeek-R1 生成的，与本项目无关。

---

## 2. 需要自行获取的上游资源

| 资源 | 用途 | 体积 | 获取方式 | License |
|---|---|---|---|---|
| **Qwen3-8B** | 基座模型 | 16 G | HF [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 |
| **CFLUE** | 中文金融题源 + 评测集 | — | [aliyun/cflue](https://github.com/aliyun/cflue) | 遵守其原始条款 |
| **FinQA** | 英文数值题源 + 评测集 | — | HF [`dreamerdeo/finqa`](https://huggingface.co/datasets/dreamerdeo/finqa) | HF 匿名可读 |
| **MATH-500 / GPQA-diamond** | 通用能力评测集 | 8.8 M | HF 公开 | 各自原始许可 |

推理链需按 [DISTILLATION.md](DISTILLATION.md) 自行蒸馏——原始蒸馏脚本已遗失，
仓库内提供按论文方法复原的参考实现 `phase1-sft-grpo/scripts/distill_cot.py`。

---

## 3. 仓库内保留的派生数据

| 文件 | 条数 | 内容 | 说明 |
|---|---|---|---|
| `phase1-sft-grpo/data/grpo_hardcase.jsonl` | 2 421 | 难例 RL 集：**题干 + 标答，不含推理链** | 筛选逻辑是本项目产出；题干为第三方真实考题，使用时须遵守其原始条款 |
| `phase1-sft-grpo/data/grpo_hardcase_smoke.jsonl` | 200 | 上者的冒烟子集 | 同上 |

难例集是本项目用 `c/k ∈ (0,1)` 通过率筛出来的，不是现成文件的搬运。
若你偏好零再分发，可删掉这两个文件，用 `scripts/build_hardcase_rl.py` 从自己的数据重新筛。

---

## 4. 不在仓库内的内容

| 内容 | 体积 | 原因 |
|---|---|---|
| **SFT 训练数据（含蒸馏推理链）** | — | 推理链为商业 API 输出，再分发前须确认服务条款（见 [DISTILLATION.md](DISTILLATION.md) §5） |
| LoRA adapter 权重（`*.safetensors`） | ~850 M | `weights_archive/` 内保留超参、`trainer_state.json` 与 `SHA256SUMS.txt`，可验权重真伪 |
| 合并模型 / 部署模型 | 各 16 G | 派生物，可由 base + adapter 重新合并 |
| optimizer / scheduler / rng_state | ~2.7 G | 仅续训需要 |
| **全部运行记录** | 12 M | 原始 stdout 日志、`logging.jsonl`、tensorboard events 一律不发布 |

绘图所需的曲线与汇总数字已抽取到 `phase1-sft-grpo/artifacts/figure_inputs.json`
（含逐项出处）——`make_figures.py` 只读该文件与 `eval/*.json`，全部 8 张图可独立重绘。
评测结论的原始凭据是 `phase1-sft-grpo/eval/` 下的 15 个 JSON。

---

## 5. 去污染

训练前做过训练/测试重叠检查：

- CFLUE 训练 vs 测试：重叠 **0%**
- FinQA 训练 vs 测试：重叠 **37 条（0.76%）**，已从训练集剔除（4 814 条）

此后再按 ≤5120 token 过滤，实训 **35 584** 条（保留 97.41%）。
