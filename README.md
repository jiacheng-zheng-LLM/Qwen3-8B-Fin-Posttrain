# qwen3-fin-posttrain

**Two-stage post-training (LoRA SFT → GRPO) of Qwen3-8B for financial reasoning, on 6×RTX4090 (24 GB, no NVLink).**
Includes the full experiment log, every evaluation artifact, and a documented **null result** for the RL stage.

> This is an independent reproduction of the recipe described in **DianJin-R1**
> ([arXiv:2504.15716](https://arxiv.org/abs/2504.15716)). It is **not affiliated with, endorsed by, or
> derived from the DianJin authors' models or code.** "DianJin-R1" appears here only as a citation.

---

## 结果速览

统一口径：同一 prompt、从 `\boxed{}` 抽答案，accuracy = correct / n。全部有 artifact 支撑（`phase1-sft-grpo/eval/`）。

| 基准 | n | base | SFT | SFT+GRPO | GRPO Δ |
|---|---|---|---|---|---|
| CFLUE（中文金融选择） | 800 | 48.00 | 53.87 | 54.75 | +0.88 |
| FinQA（英文数值推理） | 1127 | 52.17 | 60.51 | 61.76 | +1.24 |
| MATH-500（通用数学） | 500 | 40.40 | 54.20 | 54.20 | ±0.00 |
| GPQA-diamond（通用科学） | 198 | 11.11 | 21.72 | 19.19 | −2.53 |
| **均值** | — | **37.92** | **47.58** | **47.48** | **−0.10** |

**SFT 四基准全涨，均值 +9.66 pp**，且通用能力（MATH / GPQA）未退化反升。

**GRPO 净效应 ≈ 0。** 这是本项目最主要的结论，也是它值得读的原因。

---

## 为什么 GRPO 没涨——归因

训练本身跑通了：1210 步、约 5 小时、峰值 22.7 GB/卡、`exit=0`、无 OOM。但：

- **KL 全程 ≈ 0**（中位数 `8.4e-4`，全程最大 `7.2e-3`，β=0.04）。
  根因是 `--ref_adapters` 把参考模型锚在 SFT checkpoint 上 → 过度正则 → **策略几乎没离开起点**。
- **41.6% 的组内 reward 全同** → 组内优势恒为 0 → 该批次无梯度（`num_generations=4`，组太小）。
- reward 曲线在 100 步窗口下首末完全相等（`0.721 → 0.721`）。

**四个基准的 Δ 没有一个能与 0 区分开。** 以 GPQA 的 −2.53 pp 为例，它等于 198 题里少答对 5 题，
约 `0.62σ`；在 n=198、p≈0.20 下需要差 **7.95 pp（16 题）** 才达到 p<0.05。且 SFT / GRPO 两侧
准确率都**低于 4 选 1 随机线（25%）**，该分数本身不承载能力信号。所以正确的说法是
"GRPO 净效应为零"，而不是"GRPO 在 GPQA 上退化"。

**这个权衡有外部参照**：DianJin-R1 论文报告 GRPO 后 FinQA **−2.56 pp**（发生遗忘），
本项目 FinQA **+1.24 pp**。同一个过度正则，既挡住了论文里的遗忘，也挡住了论文里的增益。

> 教训：评估 GRPO 成败必须看 **KL 是否真的移动了策略**，而不是只看训练 reward 曲线——
> 后者是批次难度噪声，会误导。详见 `phase1-sft-grpo/EXPERIMENT_LOG.md`。

---

## 24 GB 卡上的工程约束

这部分可能比结果更有参考价值：

- **OOM 根因不是序列长度，是 152k 大词表的 LM 头 logits 峰值。** 引入 liger kernel 融合交叉熵后，
  可训 `max_length` 从 4096 顶到 5120，数据保留率 91.4% → **97.4%**。6144 仍 OOM。
- **无 NVLink 下 all-gather 慢约 30×** → 弃 ZeRO-3，改 DDP + ZeRO-2。
- **难例筛选**：GRPO 的优势来自组内 reward 方差，全对/全错的样本梯度为零。
  以采样通过率 `c/k ∈ (0,1)` 为唯一判据筛选，8000 题中全对 58.8% / 全错 11.0% ——
  **随机抽题会把近 7 成算力打在零梯度样本上**。浓缩出 2421 条有梯度样本。
- **生成长度**：均值 265 token，1210 步中仅 11 步触发截断（0.16%）→
  `max_completion_length=1536` 不是瓶颈，显存该花在 `num_generations` 上。

---

## 仓库结构

```
phase1-sft-grpo/            SFT → 难例筛选 → GRPO → 评测 → 部署
  EXPERIMENT_LOG.md         全程实验/事故日志（每次失败的现象-归因-修复）
  REPORT-sft.md             SFT 阶段三方对比交付报告
  scripts/                  数据构建、训练编排、评测、部署门禁与压测
    distill_cot.py          推理链蒸馏（按论文方法复原的参考实现）
    make_figures.py         一键从 artifact 重绘全部图表
  plugin/fin_orm.py         GRPO 双奖励插件（fin_acc + fin_format）
  eval/                     15 个评测结果 JSON（base / sft / grpo / gate 四档 + smoke）
  figures/                  16 张图表（中文 8 + en/ 英文 8）+ 逐图数据出处
  artifacts/                绘图所需的全部曲线与汇总数字（从运行记录抽出，含出处）
  weights_archive/          4 个 LoRA checkpoint 的超参、trainer_state、SHA256SUMS
  deploy/                   compose + Nginx 网关 + Prometheus + DEPLOY_LOG
  data/                     自建难例 RL 集

DATA.md                     上游数据去哪取、什么许可、仓库里保留了什么
DISTILLATION.md             SFT 推理链的蒸馏方法、教师模型与参数
```

---

## 复现

```bash
pip install -r requirements.txt
```

脚本全部用环境变量取路径，没有写死任何机器路径：

| 变量 | 用途 | 谁需要 |
|---|---|---|
| `BASE_MODEL` | Qwen3-8B 基座目录 | 训练 / 评测 |
| `SFT_ADAPTER` | SFT 产出的 LoRA adapter（checkpoint-1113） | GRPO / 评测 |
| `GRPO_ADAPTER` | GRPO 产出的 adapter（checkpoint-1210） | GRPO 评测 |
| `MERGED_MODEL` | 合并后的全量模型目录 | 上线门禁 |
| `SFT_DATA_DIR` | 蒸馏产出的 SFT 数据目录 | 难例筛选 |
| `EVAL_DATA_DIR` | 评测集根目录（见 [DATA.md](DATA.md)） | 评测 |
| `REPO_ROOT` | 仓库内 `phase1-sft-grpo/` 路径 | 可选，默认按脚本位置推断 |
| `SWIFT_BIN` · `CONDA_INIT` · `CONDA_ENV` · `RUN_DIR` | swift 可执行文件、conda 激活、运行产物落盘目录 | 可选 |


1. **准备上游依赖** —— 见 [DATA.md](DATA.md)。SFT 推理链需按 [DISTILLATION.md](DISTILLATION.md) 自行蒸馏。
2. **SFT** —— 超参见 `phase1-sft-grpo/weights_archive/sft-lora-checkpoint-1113-FINAL/args.json`
   （LoRA r32/α64 all-linear，lr 1e-4，3 ep，max_length 5120 + liger 融合 CE，DDP/ZeRO-2）。
3. **难例筛选** —— `scripts/build_hardcase_rl.py`：k=4 采样，只留 `0 < c/k < 1`。
4. **GRPO** —— `scripts/grpo_full.sh`，奖励插件 `plugin/fin_orm.py`，
   超参见 `weights_archive/grpo-lora-checkpoint-1210-FINAL/args.json`。
5. **评测** —— `scripts/eval_fin.py` + `scripts/run_all_eval.sh`。
6. **部署** —— `phase1-sft-grpo/deploy/README.md`：合并 → `deploy_gate.sh` 门禁 →
   compose 起 3 副本 + 网关 + Prometheus。

**重绘图表**（只依赖 matplotlib）：

```bash
python3 phase1-sft-grpo/scripts/make_figures.py [--lang en] [--only 3 6]
```

---

## 阅读须知

- **全部运行记录不随仓库发布**（stdout 日志、`logging.jsonl`、tensorboard events）。
  绘图所需的曲线与汇总数字已抽取到 `phase1-sft-grpo/artifacts/figure_inputs.json`，
  含逐项出处；`make_figures.py` 只读该文件与 `eval/*.json`，8 张图全部可独立重绘。
- **奖励函数在整理开源时改过名**：`dianjin_acc`/`dianjin_format` → `fin_acc`/`fin_format`。
  归档的 `trainer_state.json` / `args.json` 是当时运行的**原始记录**，未做改写。
- **SFT 的推理链是本项目自行蒸馏的**，不是上游数据集自带的那一份：题目集合与数据量
  与论文 Table 1 逐项一致（CFLUE MCQ 26 672 / 开放题 5 045 / FinQA 4 851；题干为 CFLUE
  与 FinQA 的真实考题，未使用上游 `DianJin-R1-Data` 数据集本身），重试策略
  T=3 与 GPT-4o 一致性校验也照论文，**唯一改动是生成推理链的教师模型换成 DeepSeek V4 Pro**。
  论文未披露 temperature / top_p / max_tokens，本项目采用教师模型
  **DeepSeek-V4-Pro-0813 的官方推荐值**（temperature 1.0 / top_p 1.0 / 输出上限 384K）。
  当时的蒸馏脚本与校验 prompt 已遗失，仓库内的 `scripts/distill_cot.py` 是按论文方法
  **复原**的参考实现，**不是**产出本项目数据的原始脚本。仓库内的 `scripts/distill_cot.py` 是按论文方法**复原**的
  参考实现，**不是**产出本项目数据的原始脚本。
  **评测判分与难例筛选环节不调用任何外部大模型。**
- **`EXPERIMENT_LOG.md` 内有 3 处后来复核发现写错的地方**，已就地加「2026-08-22 更正」注记并
  保留原文，而非抹掉。三处的更正依据都能用本仓库的 artifact 复算。

---

## License

代码采用 [Apache-2.0](LICENSE)。数据与上游依赖各自的许可见 [DATA.md](DATA.md)。
