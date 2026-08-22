# 推理链蒸馏说明

SFT 训练数据由两部分组成，来源不同：

- **题干、选项与标准答案** —— 第三方真实金融考题，本项目未做改动
- **推理链（CoT）** —— 由本项目用 **DeepSeek-V4-Pro-0813** 蒸馏产出

本文说明这份数据是怎么造出来的。

---

## 1. 题集构成

| 子集 | 条数 | 内容 |
|---|---|---|
| CFLUE MCQ | 26 672 | 中文金融单项选择 |
| CFLUE 开放题 | 5 045 | 中文金融开放问答 |
| FinQA | 4 851 | 英文数值推理（含财报表格） |
| **合计** | **36 568** | |

题目来自 CFLUE（[aliyun/cflue](https://github.com/aliyun/cflue)）与 FinQA
（[dreamerdeo/finqa](https://huggingface.co/datasets/dreamerdeo/finqa)）的真实考题，
使用时须遵守各自的原始条款。

---

## 2. 蒸馏流程

对每道题，让教师模型生成 `<think>` 推理段 + `<answer>` 答案段（答案放在 `\boxed{}` 内），
然后过两道校验；任一不通过就重采，**最多尝试 3 次**；三次都不通过的题**降级为
non-reasoning 样本**（只保留题干与标答，不带推理链）。

两道校验判据：

1. **答案一致** —— 抽出的答案是否等于标准答案
2. **推理一致** —— 生成的推理是否与参考解释一致；答案蒙对但推理过程错误的样本**判为不通过**

第二道判据由 **GPT-4o** 执行，这是整个流程中唯一的 LLM 主观判断环节。

| 项 | 值 |
|---|---|
| 生成推理链的教师 | **DeepSeek-V4-Pro-0813**（正式版，2026-08-13 发布） |
| 一致性校验模型 | **GPT-4o** |
| 每题最多尝试次数 | **3** |
| 失败样本处理 | 降级为 non-reasoning 样本 |
| 输出格式 | `<think>…</think><answer>…\boxed{}…</answer>` |
| 调用方式 | 官方 API（教师与校验各自的官方接口） |
| 蒸馏日期 | **2026-08-15** |

---

## 3. 采样参数

采用**教师模型 DeepSeek-V4-Pro-0813 的官方推荐值**：

| 项 | 值 | 出处 |
|---|---|---|
| **temperature** | **1.0** | 官方模型卡：「sampling parameters to `temperature = 1.0`」 |
| **top_p** | **1.0** | 官方模型卡：「`top_p = 0.95` for agentic scenarios and **`top_p = 1.0` otherwise**」——推理链蒸馏属非 agentic 场景，取 1.0 |
| **max output length** | **384K tokens** | 官方模型卡：「For the `high` and `max` reasoning effort levels, we recommend a maximum output length of **384K** tokens」 |
| `reasoning_effort` | `low` / `high` / `max` 三档 | 官方参数，控制作答前的思考量。上面 384K 的建议**仅适用于 `high` 与 `max` 两档**；若用 `low`，官方未给出对应的输出长度建议 |

来源：[DeepSeek-V4-Pro-0813 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)。

> **别把 temperature 往下调。** V4 系列的公开指引提到，调低温度可能压垮推理链、
> 反而降低答案质量；需要控制输出长度应该用 `max_tokens`，而不是降温。

---

## 4. 训练前处理

蒸馏产物落盘后，训练前还做了两步（详见 `EXPERIMENT_LOG.md`）：

1. **去污染** —— FinQA 与测试集重叠 37 条（0.76%）剔除 → 4 814 条
2. **长度过滤** —— 剔除 >5120 token 的样本 → 实训 **35 584** 条（保留 97.41%）

CFLUE 训练集与测试集重叠为 **0%**，无需剔除。
