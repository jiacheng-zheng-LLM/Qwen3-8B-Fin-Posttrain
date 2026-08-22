# 企业级部署事故日志(实时记档)

> 流程:合并→门禁→容器→编排→网关→监控。执行者:金融 LLM 微调专家(自主决策+全程监管)。
> 环境:8×RTX4090(生产按 6 卡用 0,2,3,4,5,6;避开卡1/402M、卡7/1642M 外来进程)。Docker 28.5.2 + 本地镜像 vllm/vllm-openai:v0.14.1。

## 环境探明(部署前)
- GPU:0/2/3/4/5/6 干净(<10MiB),卡1(402M)/卡7(1642M)有外来进程 → 生产 6 卡集用 0,2,3,4,5,6。
- Docker:28.5.2 可用,nvidia-container-runtime 就位;**registry 探测超时(代理坑 OBS-9 家族)→ 用本地已缓存镜像,不 docker pull**。
- 主机兜底:conda qwen3fin 内 vllm 0.26.0。
- 端口 8001-8003(副本)/4000(网关)/9090(prometheus)全空;磁盘 /data 4.1T 空闲。

## 阶段① 合并(merge)
- 目标:base + GRPO checkpoint-1210(SFT+GRPO 合体)→ 不可变全量模型 qwen3-fin-v1.0。
- **非破坏保证**:输出到新目录 deploy/models/qwen3-fin-v1.0,只读 base 与 adapter,weights_archive 不受影响。

### 阶段① 合并 —— ✅ 完成
- 产物:`deploy/models/qwen3-fin-v1.0`(16G,4 shards,swift 4.4.2 "Successfully merged")。
- **非破坏性已验证**:`sha256sum -c` 4 个归档 adapter 全 OK,合并只读不改原权重。
- 用时~2min(GPU0),项目 E-8 经验:export 加 `MKL_THREADING_LAYER=GNU`。

### 已发现问题与根因(部署阶段)

**D-1 · docker.io registry 不可达**
- 现象:`curl registry-1.docker.io/v2/` 超时。
- 根因:环境代理/网络限制(项目 OBS-9 代理坑家族)。
- 解法:①服务镜像 `vllm/vllm-openai:v0.14.1` 与网关 `nginx:alpine` 本地已缓存,直接用;②prometheus 走**国内镜像源 `docker.m.daocloud.io/prom/prometheus:v2.54.1`** 成功拉取(275MB)→ 打本地标签 `prom/prometheus:v2.54.1`。**部署全程零 docker pull(除 daocloud 源)。**

**D-2 · eval 脚本 BASE 硬编码,无法评测合并后全量模型**
- 现象:`eval_fin.py` 的 `BASE` 写死 Qwen3-8B,门禁需评测合并模型。
- 根因:原脚本只设计了 base+adapter 两种形态。
- 解法:加 `--model` 覆盖参数(`model=(a.model or BASE)`),门禁用 `--model <merged>` 直评全量模型,不与 `--adapter` 混用。

**D-3(观察,非故障)· PCIe-only GPU 无 NVLink → custom all-reduce 关闭**
- 现象:vLLM 日志 "Custom allreduce is disabled ... more than two PCIe-only GPUs",自动回退 PYNCCL。
- 根因:4090 无 NVLink(项目一贯前提)。TP 通信走 NCCL,功能正常,仅少量性能损失。
- 处置:无需干预;编排已设 `NCCL_P2P_DISABLE=1` 防挂起。

### 配置产物(声明式,版本化)
- `deploy/docker-compose.yml`:3×TP2 vLLM 副本(GPU 0,2 / 3,4 / 5,6)+ nginx 网关(4000)+ prometheus(9090);
- `deploy/gateway/nginx.conf`:least_conn 负载均衡 + 故障副本自动剔除 + 长请求超时 600s;**[后续修正]** 经 D-9~D-11 定案改为 `worker_processes 1` + 加权 round-robin(分发可断言),least_conn 未进最终配置;
- `deploy/monitoring/prometheus.yml`:直抓 3 副本原生 /metrics;
- `scripts/deploy_gate.sh`:质量门禁(合并模型 vs checkpoint,低于 3pp 拦截);
- `scripts/deploy_smoke_client.py`:端到端冒烟(验证 thinking 输出结构)。

### 阶段② 门禁(gate)—— 运行中
- 对合并模型在 CFLUE(800)/FinQA(全量)回归评测,参照 GRPO checkpoint 54.75/61.76,阈值 -3pp。

**D-4(观察,非故障)· `nginx -t` 孤立校验报 upstream host not found**
- 现象:独立 `docker run nginx:alpine nginx -t` 报 "host not found in upstream vllm-r1:8000"。
- 根因:孤立容器不在 compose 网络,服务名 DNS 无法解析;**真实 compose 网络中 depends_on 保证 vllm 副本先创建,内嵌 DNS(127.0.0.11)可解析**。
- 处置:非真实故障;`docker compose up` 后以网关 /health + 冒烟客户端实测为准。若 nginx 因启动时序 DNS 失败,回退方案:加 `resolver 127.0.0.11 valid=10s` + 变量式 proxy_pass 运行时解析。
- 校验:`docker compose config` 语法 OK,5 服务(vllm-r1/2/3 + gateway + prometheus)就绪。

### 阶段② 门禁 —— ✅ 通过
- CFLUE(800):合并 **54.50** vs 参照 54.75(-0.25,噪声内)→ PASS;
- FinQA(1127):合并 **61.76** vs 参照 61.76(**完全一致**)→ PASS;
- 结论:合并零损伤,`deploy_gate.sh` 退出码 0,允许上线。artifact:`eval/eval_{cflue,finqa}_gate.json`。

**D-5 · 起容器前 GPU0 被外来进程占 13GB(编排原计划用 0,2)**
- 现象:门禁跑完后 GPU0 突现 13172MiB/100%。
- 根因:他人 `nilmtk_new` python 进程(PID 1772173)占 GPU0;GPU1/GPU7 有 ollama(386M/1628M)。**共享机器 GPU 动荡(项目 E-9)**。
- 解法:**按铁律避开被占卡**,只用 5 张干净卡 2,3,4,5,6 → 另建 `docker-compose.run.yml`,布局 2×TP2(2,3 / 4,5)+ 1×TP1(6),保持 3 副本+负载均衡架构;`docker-compose.yml` 保留为生产 6 卡规范(0,2/3,4/5,6)。
- 教训:**部署前一刻仍须复查 GPU**——门禁的 40min 里外部占用发生了变化;生产环境应是专属 6 卡,不存在此问题。

### 阶段③④⑤⑥ 容器/编排/网关/监控 —— 拉起中
- `docker compose -f docker-compose.run.yml up -d`:5 容器创建。

**D-6 · prometheus 端口 9090 被占**
- 现象:up 时报 "Bind for :::9090 failed: port is already allocated"。
- 根因:他人容器 `experience-api-1` 占宿主 9090(部署前探测时尚空,起容器一刻被占——共享机器动态性)。
- 解法:prometheus 宿主端口改 9091(容器内仍 9090),两份 compose 同步;重启该服务成功。

**D-7 · vllm v0.14.1 镜像加载合并模型 tokenizer 崩溃(crashloop)** ★关键
- 现象:容器识别 Qwen3ForCausalLM 成功,但 `tokenization_qwen2_fast.py __init__: 'list' object has no attribute 'keys'`,反复重启。
- 根因:**镜像内 transformers 版本过旧**(镜像 6 个月前),合并模型的 tokenizer 由主机新版 transformers 保存,`added_tokens_decoder` 等字段格式新旧不兼容。**门禁用主机 vllm 0.26.0 加载同一 tokenizer 正常 → 确证是镜像版本滞后,非模型损坏。**
- 教训:**容器镜像的框架版本必须匹配模型工件的保存版本**;企业应把训练/合并用的框架版本固化进服务镜像(锁版本),而非用随手一个旧镜像。
- 解法:换新版 vllm 镜像(daocloud 源拉取,匹配主机 transformers)。
- 修复动作:daocloud 拉 `vllm/vllm-openai:v0.26.0`(匹配主机 vllm 0.26.0 + transformers 5.8.0),两份 compose 镜像引用同步升级;拉完打本地标签重起副本。
- 期间保持 gateway/prometheus 容器在线,仅重建 vllm 副本(最小化影响)。

**D-8 · v0.26.0 不识别 `--disable-log-requests`**
- 现象:新镜像启动报 "unrecognized arguments: --disable-log-requests"。
- 根因:vllm CLI 跨版本变化,该参数在 0.26 被移除/改名(请求日志默认关闭)。
- 解法:从 command 移除该 flag(两份 compose 同步)。教训:换镜像版本须同步核对 CLI 兼容性。

**D-9 · 网关请求全部集中到单副本 r1(r2/r3 为 0)**
- 现象:串行/并发请求经网关(:4000)全落 r1,r2/r3 零流量(直连 r2/r3 推理正常)。
- 初判:least_conn+keepalive 连接池复用 → 但改配置无效(见 D-10)。
- 真因:配置根本没生效(D-10)。最终配置改用 round-robin(每请求确定性轮换,分发可验证)。

**D-10 · bind-mount 的 nginx.conf 宿主编辑后容器内仍是旧配置** ★关键
- 现象:宿主改 nginx.conf 并 `nginx -s reload` 成功,但 `docker exec cat` 显示容器内仍是旧内容 → 路由不变。
- 根因:compose 以**单文件** bind-mount(`./gateway/nginx.conf:/etc/nginx/...:ro`);Edit/Write/sed 采用"写临时文件+原子 rename",**换了 inode**,而挂载绑定的是原 inode → 容器永远看旧文件。`reload` 重读的也是旧文件。
- 解法:**重建容器**(`docker compose up -d --force-recreate gateway`)重新按路径挂载当前宿主文件。生产更稳做法:挂载**目录**而非单文件,或把配置打进镜像/走 configmap。
- 教训:凡 bind-mount 单文件,改后必须重建容器(或挂目录),不能只 reload。

**D-11 · round-robin 配置正确但请求仍全集中 r1** ★根因
- 现象:容器内确认 round-robin、三副本健康可达、nginx 无错误,但逐请求差分显示每个请求都只增 r1。
- 根因:`worker_processes auto` 在多核机上起了大量 nginx worker,**每个 worker 维护独立的 round-robin 计数器且都从第一个 peer(r1)起**;低流量请求被内核分散到不同 worker,每个 worker 的"首个请求"都发 r1 → 全部集中 r1。
- 解法:网关 `worker_processes 1`(单一轮询计数器)。LLM 网关 I/O 密集、吞吐受限于后端推理,单 worker 足以支撑大量长连接。
- 验证:9 请求精确分发 r1+3/r2+3/r3+3 ✅。
- 教训:反向代理做负载均衡时,`worker_processes` 与均衡算法的**每-worker 状态**交互;小流量验证均衡必须用单 worker 或看 upstream_addr 日志,否则被 worker 分散假象误导。

**D-12 · 容器 healthcheck 误报 unhealthy(服务实为正常)**
- 现象:/health 返回 200、推理正常,但容器状态 (unhealthy);inspect 显示 "exec: python: executable file not found"。
- 根因:healthcheck 命令用 `python`,而 vllm v0.26.0 镜像只有 `python3`。
- 影响:K8s 等编排会因 unhealthy 反复重启健康的 Pod(生产严重故障)。
- 解法:healthcheck 改 `python3`(两份 compose),滚动重建副本。
- 教训:换镜像后必须核对 healthcheck 依赖的可执行文件在新镜像中存在。

---

## 收官:企业级部署全链路完成(2026-08-19,全部实测 artifact)

### 最终验证(全绿)
| 检查项 | 结果 |
|---|---|
| 合并非破坏 | ✅ 4 归档 adapter sha256 全 OK |
| 质量门禁 | ✅ CFLUE 54.50 / FinQA 61.76(≈checkpoint),exit 0 |
| 3 副本健康 | ✅ vllm-r1/r2/r3 全 healthy(python3 healthcheck) |
| 模型上卡 | ✅ GPU 2,3/4,5(TP2)+6(TP1),各~22GB |
| 网关负载均衡 | ✅ round-robin 精确分发(12 请求 r1+4/r2+4/r3+4) |
| 端到端 thinking | ✅ 冒烟 4/4 结构完整(think/answer/boxed) |
| 监控 | ✅ Prometheus 3/3 目标 up,指标流动 |
| 故障切换 | ✅ nginx proxy_next_upstream + max_fails(配置) |

### 架构(实跑版 docker-compose.run.yml)
Nginx 网关(:4000)→ vllm-r1(GPU2,3 TP2)/ r2(GPU4,5 TP2)/ r3(GPU6 TP1);Prometheus(:9091)抓 3 副本 /metrics。生产 6 卡规范见 docker-compose.yml(0,2/3,4/5,6 全 TP2)。

### 12 个真实问题复盘(D-1~D-12)
- **环境/网络**:D-1 registry 被墙(用本地镜像+daocloud 源)｜D-6 端口 9090 被占(改 9091)。
- **版本兼容(核心教训:服务镜像框架版本必须匹配模型工件)**:D-7 旧镜像 transformers 滞后→tokenizer 崩溃(换 v0.26.0)｜D-8 CLI flag 变更(去 --disable-log-requests)｜D-12 healthcheck 用 python(镜像仅 python3)。
- **负载均衡(层层深入)**:D-9 请求全集中 r1 → D-10 bind-mount 单文件被原子重命名编辑→容器看旧配置(须重建容器)→ D-11 真因 worker_processes auto 多 worker 各自轮询计数器都从 r1 起(改 worker_processes 1)。
- **共享机器**:D-5 起容器前 GPU0 被外来占用(避开,用干净 5 卡)｜D-3/D-4 无 NVLink custom allreduce 关闭 / nginx 孤立校验假阴性(非故障)。

### 可复用资产
- `deploy/docker-compose.{yml,run.yml}`、`gateway/nginx.conf`、`monitoring/prometheus.yml`;
- `scripts/{deploy_gate.sh,deploy_smoke_client.py}`;`deploy/README.md`(runbook);
- 模型工件 `deploy/models/qwen3-fin-v1.0`(合并前 adapter 在 `weights_archive/`,可回退)。

### 关键教训(可迁移)
1. **服务镜像的框架版本必须锁定=训练/合并版本**(D-7/D-8/D-12 同源);
2. **bind-mount 单文件改后必须重建容器或挂目录**,不能只 reload(D-10);
3. **反代做负载均衡,worker_processes 与均衡算法的每-worker 状态耦合**,小流量须单 worker 或看 upstream_addr 验证(D-11);
4. **共享机器每一步都要复查 GPU/端口**,占用是动态的(D-5/D-6);
5. **localhost 一律配 NO_PROXY**(OBS-9 全流程复现)。

---

## 加演A:并发压测(2026-08-19)
固定输出 256 token、temperature 0。
| 配置 | 并发 | 吞吐 req/s | 输出 tok/s | P50 | P95 | 成功 |
|---|---|---|---|---|---|---|
| 集群3副本 | 24/48/96 | 5.10/8.70/12.66 | 1305/2228/3242 | 4.1/4.9/6.7s | 5.6/6.9/11.0s | 100% |
| 单副本r1(TP2) | 96 | 12.79 | 3275 | 6.3s | 10.1s | 100% |
| r3 单独(TP1) | 24 | 4.55 | 1164 | 5.0s | — | 100% |

**发现1**:单 TP2 副本 ≈ 集群吞吐 → vLLM 连续批处理下单副本此负载未饱和,集群价值在容量余量+HA,非中负载吞吐;要显集群优势需压到单副本饱和(受同步客户端发压上限所限未达)。
**发现2**:异构队列 TP2+TP2+TP1 等权轮询,TP1 的 r3(4.55 vs TP2 更高)成 straggler,拖高集群 P95(11.0>10.1s)。修法:慢副本降权(见加演B);生产用同构 3×TP2(docker-compose.yml)则无此问题。

## 加演B:灰度发布(canary,nginx 加权)
- **灰度期**(r1/r2 weight=4,金丝雀 r3 weight=1):45 请求分发 44%/44%/11% ✅ 精确匹配权重;
- **金丝雀健康门禁**:查 r3 错误率=0% → 通过,允许提升;
- **提升全量**(r3 weight 1→4):30 请求均分 33%/33%/33% ✅ 金丝雀转正;
- **回滚路径**:若门禁不过,r3 weight 改 0(摘除)或降回 1,重建 gateway 即秒级回滚。
- 机制:每次改权重按 D-10 重建 gateway 生效;真实场景 r3 会是"新版本模型"副本,此处用同模型演示分流机制。

---

## 全流程总览(信息汇总索引)
本次从"训练完成"到"生产级部署"的完整链路,全部记录在本日志:
- **六阶段**:合并(阶段①)→ 门禁(②,CFLUE 54.50/FinQA 61.76 放行)→ 容器+编排+网关+监控(③④⑤⑥,3副本 healthy + round-robin + Prometheus);
- **12 个真实问题**:D-1~D-12(环境/网络、镜像版本匹配、负载均衡三连查、共享机器动态占用);
- **加演A 压测**:集群/单副本吞吐·延迟,两个诚实发现(单副本未饱和≈集群、TP1 straggler);
- **加演B 灰度**:金丝雀 11% 分流 → 健康门禁 → 全量提升 → 回滚路径。
- **配套资产**:docker-compose.{yml,run.yml}、gateway/nginx.conf、monitoring/prometheus.yml、README.md(runbook)、scripts/{deploy_gate.sh,deploy_smoke_client.py,deploy_loadtest.py}、models/qwen3-fin-v1.0(合并前 adapter 存 weights_archive/,可回退)。
- **五条可迁移教训**:①镜像框架版本=训练版本;②bind-mount 单文件改后必重建容器;③nginx 负载均衡设 worker_processes 1;④共享机器每步复查 GPU/端口;⑤localhost 一律 NO_PROXY。

## 停服释放 GPU(会话收尾)
- 动作:`docker compose -f docker-compose.run.yml down`(停全部容器,释放 GPU 2,3,4,5,6)。
- 模型工件与合并前 adapter 均落盘保留;下次一键 `docker compose -f docker-compose.run.yml up -d` 即可复起。
