# 数据说明

本仓库**不含任何模型权重、训练数据或上游数据集**。这里说明每一项去哪取、什么许可、
以及本仓库里保留了什么派生物。

---

## 1. SFT 训练数据的构成

SFT 数据由两部分组成，来源不同，须分开看：

| 组成 | 来源 | 许可 |
|---|---|---|
| **题干 / 选项 / 标准答案** | 真实金融考题（CFLUE 中文金融题库、FinQA 英文数值推理） | 遵守各自原始条款，见 §2 |
| **推理链（CoT）** | **本项目用 DeepSeek-V4-Pro-0813 蒸馏产出** | 属教师模型输出，见 [DISTILLATION.md](DISTILLATION.md) |

题集规模：CFLUE MCQ 26 672 / CFLUE 开放题 5 045 / FinQA 4 851，合计 **36 568** 条。
造数据的完整流程、采样参数与校验方式见 [DISTILLATION.md](DISTILLATION.md)。

---

## 2. 需要自行获取的上游资源

| 资源 | 用途 | 体积 | 获取方式 | License |
|---|---|---|---|---|
| **Qwen3-8B** | 基座模型 | 16 G | HF [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 |
| **CFLUE** | 中文金融题源 + 评测集 | — | [aliyun/cflue](https://github.com/aliyun/cflue) | 遵守其原始条款 |
| **FinQA** | 英文数值题源 + 评测集 | — | HF [`dreamerdeo/finqa`](https://huggingface.co/datasets/dreamerdeo/finqa) | HF 匿名可读 |
| **MATH-500** | 通用数学评测集（500 题） | — | HF [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 原始许可 |
| **GPQA-diamond** | 通用科学评测集（198 题） | — | HF [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa)，取 `gpqa_diamond` 子集 | **gated**，需在 HF 上先同意条款 |

推理链需按 [DISTILLATION.md](DISTILLATION.md) 自行蒸馏——原始蒸馏脚本已遗失，
仓库内提供复原的参考实现 `scripts/distill_cot.py`。

---

## 3. 仓库内保留的派生数据

| 文件 | 条数 | 内容 | 说明 |
|---|---|---|---|
| `data/grpo_hardcase.jsonl` | 2 421 | 难例 RL 集：**题干 + 标答，不含推理链** | 筛选逻辑是本项目产出；题干为第三方真实考题，使用时须遵守其原始条款 |
| `data/grpo_hardcase_smoke.jsonl` | 200 | 上者的冒烟子集 | 同上 |

难例集是本项目用 `c/k ∈ (0,1)` 通过率筛出来的，不是现成文件的搬运。
若你偏好零再分发，可删掉这两个文件，用 `scripts/build_hardcase_rl.py` 从自己的数据重新筛。
