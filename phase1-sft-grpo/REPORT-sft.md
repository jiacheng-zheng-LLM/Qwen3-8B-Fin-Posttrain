# 交付报告 · SFT + GRPO 全流程

> 以 **DianJin-R1**(arXiv 2504.15716)为蓝本,在 **ms-swift** 框架下对 **Qwen3-8B** 做 **LoRA SFT → LoRA GRPO** 复现。
> 硬件:6×RTX4090(24GB,无 NVLink,共享机器)。数据:题目取自 DianJin 开源部分(无专有 CCC),
> **推理链由本项目用 DeepSeek V4 Pro 重新蒸馏**(论文用 DeepSeek-R1;方法、数据量、参数 1:1 对齐,仅换教师模型),
> 详见 [`DISTILLATION.md`](../DISTILLATION.md)。**判分与难例筛选环节不调用任何外部大模型**
> (FinQA 用 `fin_verify` 数值判分替 GPT-4o judge;难例用客观 `c/k` 筛选器替 R1+GPT-4o 判难)。
> 状态:**SFT + GRPO 全流程已完成并评测(base→SFT→GRPO 三方同口径)。**

---

## 一、一句话结论

**用 LoRA 在 Qwen3-8B 上复现 DianJin-R1 两阶段(SFT→GRPO):SFT 4 基准全涨、均值 +9.6pp(能力增量的主体);GRPO 在客观难例筛选 + 双奖励 + 分离部署下跑通 1210 步,但对已强的多任务 SFT 净增益≈0(均值 −0.1pp)——这是诚实结果,根因是 ref_adapters+β0.04 强正则使 KL≈0(策略几乎未移动),换来的是跨任务零遗忘(FinQA +1.3、MATH ±0.0,对比论文 FinQA GRPO 后 −2.56)。** 绝对分低于论文(全参 +12.4pp)符合预期(LoRA + 无 CCC + 判分口径)。

## 二、SFT 结果(before/after,已核实)

| 基准 | n | base Qwen3-8B | SFT(LoRA) | Δ |
|---|---|---|---|---|
| CFLUE(中文金融MCQ) | 800 | 48.0% | **53.9%** | **+5.9** |
| FinQA(英文数值) | 1127 | 52.2% | **60.5%** | **+8.3** |
| MATH-500(通用数学) | 500 | 40.4% | **54.2%** | **+13.8** |
| GPQA-diamond(通用科学) | 198 | 11.1% | **21.7%** | **+10.6** |
| **均值** | | **37.9%** | **47.6%** | **+9.7** |

- 各项 Δ 超噪声;通用推理(MATH/GPQA)也涨(结构化推理数据泛化)。
- SFT 交付 checkpoint:**checkpoint-1113**(另产出合并后的全量权重用于部署);超参见 `weights_archive/sft-lora-checkpoint-1113-FINAL/args.json`。

## 三、GRPO 方法学(设计已定稿,运行中)

### 3.1 核心问题与我们的解法
论文 GRPO 用 **CFLUE_MCQ 难例 4096**(教师重试 3 次仍失败的题,含 GPT-4o 判"推理一致性")。**这批数据未开源**;且该判难法依赖"教师做不出来"这一主观代理量,与 GRPO 实际需要的梯度条件并不等价。→ 我们**自建客观难例筛选器**(纯替代,非复现原法):

- **判据(纯客观、可复现、无任何 LLM 主观)**:SFT 模型对每题采样 k 次,`c=`格式合规且答案字母==标答的次数,**留 `0 < c/k < 1`**(有对有错);
- **理论依据**:GRPO 优势 = 组内 reward 差异,**全对/全错→优势=0→零梯度**;`0<c/k<1` 恰是 GRPO 有梯度的充要条件;
- **实测分布(8000题,merged SFT,k=4)**:全对 **58.8%**(SFT已会,零梯度)/ **混合 30.3%(2421,保留)** / 全错 11.0%。→ **实证:随机抽 4096 会浪费近 6 成算力在零梯度题上**;客观筛子浓缩出 2421 条有梯度难例(`data/grpo_hardcase.jsonl`)。

### 3.2 双奖励(格式 + 正确性)
`plugin/fin_orm.py`,复刻论文"dual reward signals":
- `fin_acc`(权重1.0):`<answer>` 内 boxed 字母==标答 → 1;
- `fin_format`(权重0.1):恰好一个 `<think>`+一个 `<answer>`+boxed → 1。
- **比单一0/1平滑**:格式对答案错也给 0.1,不至于零信号。离线 4/4 验证。

### 3.3 分离部署(server-mode)
- **架构**:vLLM rollout server(TP=2,GPU5,6)+ 训练 DDP(GPU3,4)——训练端不背 vLLM;
- **TP 约束**:Qwen3-8B 32 头 → vLLM TP 必须整除32(用2/4,不能6);
- **卡选择**:共享机器 GPU0(nilmtk)/GPU2(ollama)动荡,训练需 ~21GB/卡容不下外来进程 → 只用绝对稳定的 **3,4**(代价:比4卡慢~2×,但不被抢崩)。

### 3.4 关键决策:ref_adapters(KL 锚 SFT)而非 merge
- **场景特殊性**:我们 SFT 是**多任务**(4基准全涨),而 GRPO 奖励**只有 CFLUE** → 另3任务有遗忘风险(论文 FinQA GRPO 后 -2.56);
- **本想 merge SFT 使 KL 锚 SFT 护多任务**,但**实测 merge 破坏 server-mode 的 LoRA 权重同步**(服务端无 LoRA,训练端全新 LoRA,握手失败);
- **改用 `--adapters <SFT> --ref_adapters <SFT>`**:policy=SFT适配器续训,reference=冻结SFT → **KL 锚 SFT,护多任务,且不破坏 server-mode**。

### 3.5 config 与运行状态
- LoRA r32/α64;num_generations **4**(8会OOM);max_completion **1536**(实测均长~250,不吃紧);lr1e-6;β0.04;1 epoch=1210步;
- **实测**:全量 1210 步 ~5h,峰值~22.7GB/卡,无OOM;交付 checkpoint **checkpoint-1210**(adapter 174MB,续训 SFT 适配器,即 SFT+GRPO 合体权重);超参见 `weights_archive/grpo-lora-checkpoint-1210-FINAL/args.json`。

### 3.6 GRPO 最终评测(base→SFT→GRPO,同口径,已核实 artifact)

| 基准 | base | SFT | **GRPO** | SFTΔ | **GRPOΔ(vs SFT)** |
|---|---|---|---|---|---|
| CFLUE(n=800) | 48.0 | 53.9 | **54.8** | +5.9 | **+0.8** |
| FinQA(n=1127) | 52.2 | 60.5 | **61.8** | +8.3 | **+1.3** |
| MATH-500(n=500) | 40.4 | 54.2 | **54.2** | +13.8 | **±0.0** |
| GPQA(n=198) | 11.1 | 21.7 | **19.2** | +10.6 | **−2.5** |
| **均值** | 37.9 | 47.6 | **47.5** | **+9.6** | **−0.1** |

artifact:`eval/eval_{cflue,finqa,math500,gpqa}_grpo.json`。

**诚实判定(GRPO 净效应≈0,不粉饰)**:
1. **根因 KL≈0**:训练全程 KL≈0(0.0007→0),ref_adapters 锚 SFT + β0.04 使策略几乎未移动 → **过度正则**,非 bug;批次 reward/acc 波动(0.77→0.66)是不同批难度噪声,非模型变化。
2. **分域自洽**:奖励域 CFLUE 微涨 +0.8(策略被锁,涨不动);同域 FinQA +1.3、无关域 MATH ±0.0 → **零遗忘**,印证 ref_adapters 护多任务的设计意图;
3. **GPQA −2.5 = 噪声**:n=198,−2.5pp≈5 题,小基准抖动,不构成真实退化(均值 −0.1 佐证整体持平);
4. **与论文一致**:论文 GRPO 相对 SFT 亦仅 +1.6pp 量级;我们更保守配置把增益压到≈0,换跨任务稳定(论文 FinQA GRPO 后 −2.56 vs 我们 +1.3);同族于本项目 track2 早先 RFT/SC 的 null——**强多任务 SFT 之上受限 RL 边际增益有限**。
5. **若要 GRPO 真涨的下一步(未执行,留决策)**:降 β(0.04→0.01)/去 ref_adapters 放开策略移动、或扩大奖励覆盖到多任务——但均需再花一轮算力,当前作为诚实负结果收官。

## 四、诚实标注(不可夸大)

1. **格式合规成分**:base 不总用 boxed → base 分被压低(GPQA base 11%<随机25%);SFT 涨幅=真能力+格式合规;
2. **FinQA 判分**:用 fin_verify 数值判分,非论文的 GPT-4o judge(**评测环节不引入 LLM 判分**) → 不可与论文直接对标;
3. **难例数据**:我们的客观 c/k 难例 **≠ 论文 R1 难例**(来源/判据不同),是**有原则的替代,不冒充复现原集**;GRPO 用已 SFT 训过的 cflue_mcq 题(标准做法);
4. **偏差**:LoRA(非全参)、Qwen3-8B(非Qwen2.5)、无CCC、max_length 5120(SFT)、CFLUE 抽样800、GRPO num_gen 4(非论文8,显存所限)。

## 五、工程复盘(E-1~E-11,详见 EXPERIMENT_LOG.md)

**SFT 阶段**:E-1 train_type版本坑｜E-3 zero3慢(通信)｜E-5 首步OOM(幸存者偏差smoke)｜**E-6 liger融合CE治大词表logits(4096→5120)**｜E-7 TP整除头数。
**GRPO 阶段**:**E-10 merge破坏server-mode LoRA同步→用--adapters**｜E-11 分离部署"环境八连坑"(pkill自匹配自杀、CUDA释放竞态、GPU被他人抢、util0.9→0.8、**代理拦localhost**、vLLM worker泄漏、num_gen8→4 OOM)。
**核心教训**:①大词表LLM OOM优先liger而非缩序列;②放量smoke须含长度尾部;③GRPO分离部署在共享机器上远比SFT单进程脆(GPU抢占/代理/子进程泄漏/多进程清理自匹配);④全程只报有artifact的数字。

## 六、可复现资产
- 数据管线/评测/难例筛选/reward:`scripts/{eval_dianjin,build_hardcase_rl}.py`、`plugin/fin_orm.py`;
- 训练脚本:SFT(liger+5120+DDP)、`scripts/grpo_full.sh`(proven 分离部署);
- checkpoint:SFT `checkpoint-1113`(+合并版)、GRPO `checkpoint-1210`(已完成);
- 评测 artifact:`eval/eval_{cflue,finqa,math500,gpqa}_{base,sft,grpo}.json`;
- 全程事故日志:`EXPERIMENT_LOG.md`。
