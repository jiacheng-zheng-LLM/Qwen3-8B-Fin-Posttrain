# 推理链蒸馏说明

SFT 训练数据由两部分组成，来源不同：

- **题干、选项与标准答案** —— 第三方真实金融考题，本项目未做改动
- **推理链（CoT）** —— 由本项目用 **DeepSeek-R1** 蒸馏产出

本文说明这份数据是怎么造出来的。

---

## 1. 题集构成

| 子集 | 条数 | 内容 |
|---|---|---|
| CFLUE MCQ | 26 672 | 中文金融单项选择 |
| CFLUE 开放题 | 5 045 | 中文金融开放问答 |
| FinQA | 4 851 | 英文数值推理（含财报表格） |
| **合计** | **36 568** | |

题目来自 CFLUE（[aliyun/cflue](https://github.com/aliyun/cflue)）与 FinQA（[dreamerdeo/finqa](https://huggingface.co/datasets/dreamerdeo/finqa)）的真实考题，使用时须遵守各自的原始条款。

---

## 2. 蒸馏流程

对每道题，让教师模型生成 `<think>` 推理段 + `<answer>` 答案段（答案放在 `\boxed{}` 内），然后过两道校验；任一不通过就重采，**最多尝试 3 次**；三次都不通过的题**降级为non-reasoning 样本**（只保留题干与标答，不带推理链）。

两道校验判据：

1. **答案一致** —— 抽出的答案是否等于标准答案
2. **推理一致** —— 生成的推理是否与参考解释一致；答案蒙对但推理过程错误的样本**判为不通过**

第二道判据由 **GPT-4o** 执行，这是整个流程中唯一的 LLM 主观判断环节。

| 项 | 值 |
|---|---|
| 生成推理链的教师 | **DeepSeek-R1** |
| 一致性校验模型 | **GPT-4o** |
| 每题最多尝试次数 | **3** |
| 失败样本处理 | 降级为 non-reasoning 样本 |
| 输出格式 | `<think>…</think><answer>…\boxed{}…</answer>` |
| 调用方式 | 官方 API（教师与校验各自的官方接口） |
| 蒸馏日期 | **2026-08-15** |

---

## 3. 采样参数

采用**教师模型 DeepSeek-R1 的官方推荐值**：

| 项 | 值 | 出处 |
|---|---|---|
| **temperature** | **0.6** | 官方模型卡：「Set the temperature within the range of 0.5-0.7 (0.6 is recommended)」 |
| **top_p** | **0.95** | 官方模型卡的基准评测设置 |
| **max generation length** | **32 768 tokens** | 同上 |
| system prompt | **不加** | 官方模型卡：「Avoid adding a system prompt; all instructions should be contained within the user prompt」 |
| 起始标记 | 强制以 `<think>\n` 开头 | 官方建议，避免模型跳过推理段直接作答 |

来源：[DeepSeek-R1 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-R1)。

> 官方特别提示：temperature 低于 0.5 容易触发重复或不连贯输出，高于 0.7 则推理稳定性下降，
> 0.6 是推荐落点。数学类题目建议在 user prompt 内直接写明
> 「Please reason step by step, and put your final answer within `\boxed{}`」——
> 本项目的生成 prompt 正是这么写的（见 `scripts/distill_cot.py`）。

---

## 4. 训练前处理

蒸馏产物落盘后，训练前还做了两步（详见 `EXPERIMENT_LOG.md`）：

1. **去污染** —— FinQA 与测试集重叠 37 条（0.76%）剔除 → 4 814 条
2. **长度过滤** —— 剔除 >5120 token 的样本 → 实训 **35 584** 条（保留 97.41%）

CFLUE 训练集与测试集重叠为 **0%**，无需剔除。
