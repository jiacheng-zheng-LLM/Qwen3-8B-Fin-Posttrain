# 实验日志(ms-swift + Qwen3-8B · LoRA SFT→GRPO)

> 负责人守则:只记有 artifact(日志/文件/checkpoint)支撑的事实;问题实时登记(现象/根因/有效解法);不虚报。
> 环境:`qwen3fin`(ms-swift 4.4.2 / vLLM 0.26 / torch 2.11+cu130,无 flash-attn)。硬件:6×RTX4090(0,2,3,4,5,6),无 NVLink。

## 当前状态
- **[✅ SFT 完成·已核实] 全量 LoRA SFT v3**(2026-08-18):**liger + max_length 5120 + DDP + truncation_strategy delete**,数据 ≤5120 = 35584 条(保留97.4%),3 epoch,LoRA r32/α64,lr1e-4,bs1,ga16,sdpa,6卡,seed42。
  - **完成验收(非虚报)**:1113/1113 步、**4h29m**、14.52 s/it、峰值 20.47GB;loss **1.069→0.692**(最小0.646,223点,**0 真NaN**),稳降收敛;3 checkpoint 落地。
  - **交付 checkpoint**:`checkpoint-1113`(adapter 349MB,ep3 最终)。
- **[进行中] 评测阶段**:搭 scorer(CFLUE字母/FinQA fin_verify/MATH-500/GPQA)→ base Qwen3-8B 基线 + SFT(ckpt-1113)评测 → before/after(**评测环节全程不用 LLM 判分**)。
- 前序失败版本:v1/v2(zero2/4096 首步OOM=E-5)——已被 v3 定稿取代。

## 事故记录(smoke 阶段,全部在放量前拦下)

### E-1 [已解决] `swift sft` 不认 `--train_type lora`
- 现象:`ValueError: remaining_argv: ['--train_type', 'lora']`,训练直接不启动。
- 根因:`--train_type` 是较新版/5.x 写法;**ms-swift 4.4.2 靠"给了 `--lora_rank` 即隐式走 LoRA"**,无此开关。项目历史 SFT 命令也从未用 `--train_type`。
- 有效解法:**删掉 `--train_type lora`**,仅保留 `--lora_rank/--lora_alpha/--target_modules`。日志确认 LoRA 已激活(peft LORA,r=32)。

### E-2 [已解决] zero2 + max_length 长序列 OOM
- 现象:zero2 下 max_length 8192(4卡)、6144(6卡)均 `CUDA OutOfMemory`(GPU ~21-22GB 用满,还差 2-2.6GB)。
- 根因:**zero2 不分片参数,每卡各存整份 8B bf16 权重=16GB**,24GB 只剩 ~7GB 给激活;8192 激活 ~10GB、6144 也超 → 爆。
- 有效解法:**max_length 降到 4096**(激活 ~5-6GB,总 ~22GB 装得下,实测 22.08GB 无 OOM)+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。代价:截断 16% FinQA(见偏差登记)。

### E-3 [已解决·关键] zero3/8192 灾难性慢(330s/步 → 全量 ~105h)
- 现象:6卡 zero3 + 8192 能装下但 `train_speed=330 s/it`,0.27 samples/s,全量外推 4-5 天。
- 根因:**zero3 分片参数 → 每层每次前向/反向都要 all-gather 把整个 8B 权重拼回来;这些 4090 无 NVLink,gather 走慢速 PCIe/跨NUMA(0,2,3 与 4,5,6 分属两 NUMA)** → 通信主导。减卡不解(仍搬整模型);单卡消通信但丢并行(反慢 6×)。
- 有效解法:**改 zero2(权重整份在本地,零权重通信,仅同步极小 LoRA 梯度)+ 4096**。实测 `21.5 s/it`,快 15×,全量 ~6.8h。取舍本质:zero3 用"通信"换"显存",在无 NVLink 上通信太贵,不如 zero2+缩序列。

### E-4 [已放弃] packing 提速需 flash-attn(踩项目老雷)
- 现象:`--packing true` 报 `ValueError: The "packing" feature requires a flash attention implementation`。
- 根因:packing 把多条短样本拼进一条长序列,需 flash-attn 的变长注意力+position_ids 防跨样本串扰;sdpa 不支持。
- 处置:**放弃 packing**。因 flash-attn 在本机 torch2.11+cu130 无预编译 wheel、JIT 编译是历史 12h 事故雷区(OBS-4/INC-5),不在关键路径现装。→ 用 E-3 的 zero2/4096 达到可行速度,不依赖 packing。

### E-5 [处理中] 全量 SFT 第 1 步 backward OOM(smoke 幸存者偏差漏掉最长序列)
- 现象:全量 SFT(zero2/max_length4096/6卡)启动、模型加载、数据预处理均正常,**训练第 1 步 backward 时 GPU4 `CUDA OutOfMemory`**(需 2.12GB,仅剩 1.72GB,差 ~0.4GB),ETA 曾显示 5:29:47。后台任务误报 exit0,核实日志才发现是崩(未虚报)。
- 根因:**smoke 幸存者偏差**——我构造的 smoke 取每文件"前 60 条"(偏短,cflue_mcq p50=638),**没抽到会被截到整整 4096 的最长序列**;全量含大量长序列(fin_qa 15.9% >4096、cflue_mcq max 15613 tok),截到满 4096 后 backward 激活+大词表(~152k)logits/loss 峰值就爆。**正是项目反复踩的"小样本/短序列成功≠全量长序列安全"(INC-10/OBS-8 家族)**;我的过程失误 = smoke 未用最长样本压测。
- 有效解法(验证中):①**smoke 改用"每文件最长 30 条"(worst-case)压测**,不再幸存者偏差;②**降 max_length 4096→3584(或更低)**给峰值留余量,worst-case smoke 无 OOM 才放量。备选:再降 3072(代价:FinQA 截断率升)。
- 防线更新:**放量前的 smoke 必须包含数据长度分布的尾部(最长样本),而非随机/靠前样本**——否则内存/截断验证是假通过。已加为本实验铁律。
- 排查中又误判 1 次:worst-case 3584/3072 崩以为是 OOM,实为**退化样本**(instruction 单独超 max_length→截断后 output 空→swift truncation_strategy=delete 丢弃→collator 撞 None→`AttributeError: packed_length`);更正后靠"预过滤删超长样本"+"用近满长【有效】样本压测"分离(A)内存与(B)退化两问题。
- **最终解法(E-6 liger)**:见下。

### E-6 [已解决·关键] liger_kernel 融合交叉熵 → 解 4096 内存,兼容 transformers 5.8
- 背景:8B + 24GB + 大词表(~152k),真凶是 **LM 头 logits/loss 峰值**(4096 序列 logits ~1.25GB,CE upcast fp32 翻倍)。纯缩 max_length 到 3072 代价大(fin_qa 丢 31%),zero3 太慢(E-3),packing 需 flash-attn(E-4 地雷)。
- 参考:查原项目(同 6×4090)proven 配置 = **max_length 2560 / DDP(deepspeed None)/ truncation_strategy delete**(兜底安全网)。
- 有效解法:`pip install --no-deps liger_kernel`(0.8.1,Triton 实现非 nvcc,不动前沿栈)+ `--use_liger_kernel true`。**实测**:
  - liger @ **max_length 4096** DDP:**峰值 19.76GB(4GB 余量)、15.84 s/it、无 OOM、无兼容报错**、loss 0.52 ✓ → 采用;
  - liger @ **max_length 5120** DDP:**峰值 20.46GB、无 OOM**、loss 正常 ✓;
  - liger @ 6144:峰值冲 20.56GB 后 **OOM** → **5120 才是 liger 下的安全上限**。
- **定稿配置**:liger + **max_length 5120** + DDP + truncation_strategy delete + 数据预过滤 ≤5120(总 36531→35584,**保留 97.4%**)+ LoRA r32/α64 + lr1e-4 + 3ep + bs1 + ga16 + sdpa + 6卡 + expandable_segments。全量实测 ~4.9h。

> **2026-08-22 更正**:本条原写作「定稿 max_length 4096 / 4096 为 liger 安全上限」,
> 那是 4096 跑通、5120 还没试之前写下的,后续做了 5120 并采用,但本条没回改。
> 实证依据:`weights_archive/sft-lora-checkpoint-1113-FINAL/args.json` 中 `max_length: 5120`;
> 原 `logs/liger_test_5120.log` 无 OutOfMemory、峰值 20.46GB;原 `logs/prefilter5120.log` 保留 97.4%
> (4096 档为 95.2%)。文件头「全量 LoRA SFT v3」与 项目梳理记的 5120 是对的。
- 教训:大词表 LLM 微调 OOM,优先 liger 融合 CE(Triton,低风险),而非死缩序列或上 zero3。liger 与极前沿 transformers 5.8 兼容(apply_liger_kernel_to_qwen3 生效)。

### E-7 [已解决] 评测/推理 vLLM TP 必须整除注意力头数(32)
- 现象:`eval_fin.py --tp 6` 报 `Total number of attention heads (32) must be divisible by tensor parallel size (6)`。
- 根因:Qwen3-8B 有 32 头;vLLM 张量并行要求 TP 整除头数。**6 不整除 32**(合法 TP=1/2/4/8）。这是模型结构硬约束,与训练侧 DDP 无关(DDP 用 6 卡不受此限)。
- 有效解法:**评测/推理用 TP=4**(4卡,合法且够快)。**注意:后续 GRPO rollout 的 vLLM 也须 TP=2 或 4,不能 6**——分离部署 vLLM 独占卡数按此定。

### E-8 [已解决] swift export --merge_lora 崩 MKL 线程层(同 E-1 家族)
- 现象:合并 LoRA 时 `Error: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1`。
- 根因:export 命令未设 `MKL_THREADING_LAYER=GNU`(和多卡训练同一坑)。
- 解法:加 `MKL_THREADING_LAYER=GNU`(+`MKL_SERVICE_FORCE_INTEL=0`)重跑,合并成功 → 合并后全量模型(16G,4分片)。**凡 swift/torch 多进程命令一律带 GNU 线程层。**

### E-9 [流程铁律] 共享机器:每次 GPU 启动前必查空闲卡
- 现象:难例筛选 vLLM TP=4 在 GPU0 上 WorkerProc 起不来 → 根因 **GPU0 被别的用户占了 13.7GB**(`/root/.../nilmtk_new .../train.py`,非我们,不可杀)。
- 解法:改用空闲的 GPU 2,3,4,5 重跑。**新增铁律:每次 GPU 任务启动前 `nvidia-smi` 查空闲,避开他人占用的卡**(GPU0 现被占,可用池临时=2,3,4,5,6)。GRPO 分离部署据此定:vLLM 5,6 / 训练 2,3,4。

## GRPO 前置排查与准备(2026-08-18)

### 实测:SFT 模型的奖励可得性 + RL headroom(探针 60题 + 筛选 8000题)
- **奖励可得**(#1 排除):SFT 格式合规率 88.3%、reward>0 率 81.7% → GRPO 有非零信号,不会静默不学。
- **难例分布(8000 题 cflue_mcq,merged SFT,k=4,客观 c/k 判据)**:
  - **c=4 全对 58.8%(4702)**——SFT 已稳定会,GRPO **优势=0 零梯度,白训** → 排除;
  - **0<c<4 混合 30.3%(2421)**——**有梯度,RL 有信号** → **保留为 RL 数据**;
  - c=0 全错 11.0%(877)——无正样本可强化 → 排除。
- **#2 实证坐实**:若随机抽 4096 做 RL,~58.8% 是零梯度题,近 6 成算力浪费;客观筛子浓缩出 2421 条有梯度难例。**纯客观、可复现、判难环节零 LLM 调用**。

### GRPO 关键决策(经深度分析,部分与原项目相反)
- **合并 SFT LoRA(与原项目不同)**:原项目 GRPO 用 `--adapters`(不合并,KL锚base);**本场景是"多任务SFT(4基准全涨)+单任务窄奖励(仅CFLUE)"→ 遗忘风险高 → 合并使 KL 锚 SFT 保护另3任务增益**。`swift export --merge_lora` 产出合并后的全量模型。
- **RL 数据**:开源 cflue_mcq 题+标答,用 SFT 模型 c/k 自筛难例(2421)。**GRPO 数据不需要推理链**(模型自己生成+奖励判),故全程零 R1/GPT-4o。
- **双奖励**:`plugin/fin_orm.py`(fin_acc 权重1.0 + fin_format 权重0.1),比单一0/1平滑,离线4/4验证。
- **分离部署**:vLLM server TP=2(GPU5,6)+ 训练 DDP(GPU2,3,4),避开被占的 GPU0。

### E-10 [已解决·关键·修正合并决策] merge 破坏 GRPO server-mode 的 LoRA 权重同步
- 现象:GRPO smoke 用 merged 模型跑,rollout 服务正常(Uvicorn /health/=200),但训练器 `ConnectionError: Servers not reachable after 600s`。
- 根因:**server-mode GRPO 要求 rollout 服务和训练器 LoRA 结构对齐**(训练器每步把更新的 LoRA 权重同步给服务)。原项目 proven 做法:rollout **和** trainer **都 `--model BASE --adapters COLD`**(两边都有 LoRA)。我用 merged(服务端无 LoRA)+ trainer 全新 LoRA → **结构不匹配 → 权重同步握手失败 → 连不上**。
- 有效解法:**GRPO 回到 proven 的 `--adapters` 方式**——rollout 和 trainer 都 `--model BASE --adapters <SFT ckpt-1113> --lora_rank 32`,不用 merged 模型。**修正之前"合并"的决策**:merge 在 server-mode 下不可行。
- **SFT 保护改用 `--ref_adapters <SFT副本>`**(KL 锚 SFT)——达到合并想要的"护多任务 SFT"效果,又不破坏 server-mode(smoke 先用最简 ref=base 验通,full 再加 ref_adapters)。
- 补记:rollout 需 `--vllm_use_async_engine true`,trainer 需 `--vllm_server_pass_dataset true`(原项目 proven,我首版漏了)。
- 备注:合并后的全量模型不浪费——仍是 SFT 的完整权重交付物,可用于部署/评测;只是 GRPO 不用它。

### E-11 [部署坑合集] GRPO server-mode 分离部署"环境八连坑"(共享机器 + 代理 + 多进程)
GRPO smoke→full 一路踩的环境坑(非方法问题,均已定位修复,SFT 单进程时不暴露):
1. **merged 破坏 LoRA 同步**(E-10)→ 用 `--adapters`;
2. **pkill -f 自匹配自杀**:命令行含 "EngineCore"/"VLLM.Worker" 字样时 `pkill -f` 匹配到自己的 shell → 直接命令行清理会自杀。**修:清理用 PID,或把 pkill 放进脚本文件(bash 进程 cmdline 不含该字样,不自匹配)**;
3. **杀服务后同卡立启新服务 → CUDA 显存释放竞态 OOM**:改用干净空闲卡;
4. **共享机器 GPU 被他人抢**(GPU0 nilmtk / GPU2 ollama 陆续占用)→ **每次启动前 nvidia-smi 查空闲、动态选卡**(铁律 E-9);
5. **rollout vLLM `gpu_memory_utilization=0.9` 太高** → 自己占满没余量 OOM → 降 **0.8**;
6. **代理拦 localhost**:环境 `HTTP_PROXY=127.0.0.1:18080`(OBS-9)→ 训练器 requests 请求本地 rollout 服务被路由进代理 → `Servers not reachable`。**修:COMMON 加 `HTTP_PROXY= ... NO_PROXY=127.0.0.1,localhost`**(curl 默认绕 localhost 所以健康检查过了,requests 不绕);
7. **vLLM Worker 子进程泄漏**:orch 只杀 rollout 父进程,`VLLM::Worker` 子进程(cmdline 非 "swift rollout")存活占卡 → 新服务 OOM。**修:orch 清理补 `pkill -9 -f "VLLM::Worker"`(脚本内不自匹配)**;
8. **训练器 OOM(num_gen 8 × max_comp 1536 + ref_adapters)**:GPU0 主进程 21.77GB(base16G+ref参考模型+大词表logits over 8×1536)撑爆。**修:num_gen 8→4**(smoke 验证值,减半 logprob 内存,保留 1536+ref)。崩在 `logits`,同 SFT 大词表内存族。
- **总教训**:GRPO 分离部署在共享机器上远比 SFT 单进程脆——GPU 抢占、代理、子进程泄漏、多进程清理自匹配、rollout/训练双侧显存。每一项都要显式处理。

### ✅ 全量 GRPO proven 配置(2026-08-18,已稳定运行越过 step6 OOM 点)
- **部署**:rollout vLLM server TP=2(GPU **5,6**,util 0.8,async_engine)+ 训练 DDP(GPU **3,4** 两张最稳空闲卡,避开动荡的 GPU0/nilmtk、GPU2/ollama)。**注意 2卡≠省显存(DDP每卡都存整份base~21GB),纯为避开被占卡;代价=比4卡慢~2×**。
- **模型/数据**:`--model BASE --adapters <SFT ckpt-1113> --ref_adapters <同>`(KL锚SFT护多任务)+ RL 数据=客观难例 `grpo_hardcase.jsonl`(2421,c/k∈(0,1))。
- **超参**:LoRA r32/α64,num_generations **4**(8会OOM),max_completion **1536**(实测均长~250,不吃紧),bs1,ga4,lr1e-6,β0.04,1 epoch=**1210步**,双奖励(acc1.0+format0.1)。
- **env(必带)**:MKL_THREADING_LAYER=GNU、VLLM_USE_FLASHINFER_SAMPLER=0、NCCL_P2P_DISABLE=1、expandable_segments、**HTTP_PROXY= + NO_PROXY=127.0.0.1,localhost**(不然训练器连不上本地 rollout)。
- **实测**:每步 ~12-15s,全量 ~4.5-5h,峰值 ~22.5GB/卡;KL 有界,无 OOM。脚本 `scripts/grpo_full.sh`。

> **2026-08-22 更正**:本条原写作「reward 0.6→0.97、acc 0.5→0.875 **上升**」。
> 那两组数正是开跑后**头两步**的值(step1 reward 0.7125 / acc 0.625,step2 reward 0.975 / acc 0.875),
> 是单批噪声,不构成趋势,写成"上升"与收官段"全程持平、GRPO 净效应≈0"直接矛盾。
> 全程实际:reward 均值 0.731、首末 100 步 0.721→0.721(见 `figures/fig3_grpo_reward.png`)。
- **监管铁律**:每次 GPU 启动前 nvidia-smi 查空闲;清残留用 PID(pkill -f 会自匹配命令行自杀);盯训练卡被他人抢(一抢即 OOM)。
1. LoRA 替全参 SFT/GRPO(6×4090 算力);
2. 基座 Qwen3-8B 替 Qwen2.5-7B;
3. max_length 4096 替 16K → **截断 16% FinQA**(6144 只截 2.6% 但 zero2 装不下,zero3 太慢);
4. 无 CCC 子集(专有未发布);
5. **FinQA 评测用 fin_verify 数值判分,不用 LLM judge**(评测环节不引入 LLM 判分)→ 与采用 LLM judge 的实现不可直接对标;
6. GRPO `num_generations` 可能 <8(显存,colocate/分离待定);
7. GRPO 奖励域仅 CFLUE_MCQ。
> 保留的设计:两阶段 SFT→GRPO 逻辑、`<think></think><answer>\boxed{}</answer>` 格式、双奖励、先测 base 基线。

## 数据落地(已核验)
- 题目源:CFLUE 与 FinQA 的真实考题;落盘文件名为小写 `train/{cflue_mcq,cflue_oe,fin_qa}.json`。
- **推理链(CoT)由本项目用 DeepSeek-V4-Pro-0813 蒸馏产出**,教师生成 → 双判据校验 → 最多 3 次重试,
  失败样本降级为 non-reasoning。产物落在 `data/train/` 下。详见 [`DISTILLATION.md`](DISTILLATION.md)。
- 去污染:CFLUE test 重叠 0%;FinQA test 重叠 37 条(0.76%)已从训练剔除 → `fin_qa.decontam.json`(4814)。
- 格式:think/answer 各 100% 恰好一个;boxed 命中 cflue_mcq/fin_qa 100%、cflue_oe 6%(开放题无 boxed,仅 SFT)。
- S1 模板验证:Qwen3 默认模板 SFT 不产生双 think;仅误设 `enable_thinking=false` 才注入空 think → 全程严禁该参数。

---

## 收官:GRPO 完成 + 最终评测(2026-08-18,已核验 artifact)

**训练**:`checkpoint-1210`(adapter 174MB),1210 步 ~5h,峰值~22.7GB/卡,exit=0,无 OOM。KL 全程≈0(中位数 8.4e-4,全程最大 7.2e-3);批次 reward 首20 0.773→末20 **0.731**、acc 首20 0.675→末20 **0.631**(**不同批难度噪声,非模型退化**;KL≈0 说明策略几乎未移动)。放宽到 100 步窗口则 reward 0.721→0.721、acc 0.6225→0.6225,**完全持平**。

> **2026-08-22 更正**:本条原写作「末20 reward 0.659、acc 0.569」,复算对不上——
> 首20 的 0.773 / 0.675 与训练指标记录精确一致,末20 应为 0.731 / 0.631
> (读全部 1210 步复算)。原数字把降幅写大了约 7pp;
> 更正后降幅更小,而"策略几乎未移动"的结论不变、反而更稳——100 步窗口下首末完全相等。

**最终评测(base→SFT→GRPO,同口径 eval_fin.py,TP=4/GPU3-6)**:

| 基准 | base | SFT | GRPO | GRPOΔ(vs SFT) | artifact |
|---|---|---|---|---|---|
| CFLUE(800) | 48.0 | 53.9 | 54.8 | +0.8 | eval/eval_cflue_grpo.json |
| FinQA(1127) | 52.2 | 60.5 | 61.8 | +1.3 | eval/eval_finqa_grpo.json |
| MATH-500(500) | 40.4 | 54.2 | 54.2 | ±0.0 | eval/eval_math500_grpo.json |
| GPQA(198) | 11.1 | 21.7 | 19.2 | −2.5 | eval/eval_gpqa_grpo.json |
| 均值 | 37.9 | 47.6 | 47.5 | **−0.1** | |

**结论(诚实负结果,GRPO 净效应≈0)**:根因 KL≈0(ref_adapters+β0.04 过度正则,策略未移动)→ 奖励域 CFLUE 只微涨 +0.8;跨任务零遗忘(FinQA +1.3、MATH ±0.0);GPQA −2.5 为小基准噪声(≈5 题)。量级与同族 RFT/SC 的 null 一致:**强多任务 SFT 之上受限 RL 边际增益有限**。若求 GRPO 真涨需降 β/去 ref/扩奖励覆盖(未执行,留决策)。

**教训补记**:①GRPO 用 ref_adapters 锚 SFT 是"稳定性 vs 增益"的权衡——它护住非奖励任务不遗忘,代价是奖励任务也涨不动(KL 被压到 0);评估 GRPO 成败必须看 KL 是否真的移动了策略,而非只看训练 reward 曲线(reward 曲线是批次噪声,会误导)。②批次 acc(难例上、逐批不同题)≠ 评测 acc,严禁混为一谈对外汇报。
