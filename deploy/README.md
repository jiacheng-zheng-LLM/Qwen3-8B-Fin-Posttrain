# Qwen3-Fin 企业级部署 Runbook

生产模型:`qwen3-fin-v1.0`(Qwen3-8B + SFT + GRPO 合并的不可变工件)。
目标环境:6×RTX4090(生产卡集 0,2,3,4,5,6)。全程本地镜像,不依赖公网 registry。

## 架构
```
客户端 → Nginx 网关(:4000, OpenAI 兼容, 加权 round-robin 负载均衡, 故障剔除)
             ├─ vllm-r1  GPU 0,2  TP2  :8001
             ├─ vllm-r2  GPU 3,4  TP2  :8002
             └─ vllm-r3  GPU 5,6  TP2  :8003
        Prometheus(:9090)直抓 3 副本 /metrics
```
- 单副本:Qwen3-8B bf16(~16GB 权重分 2 卡)+ ~15GB/卡 KV,扛长推理链;
- 3 副本数据并行:任一副本故障,网关自动切换,服务不中断;
- TP=2 满足"整除 32 注意力头"约束(不可用 3/6)。

## 端到端流程(合并→门禁→容器→编排→网关→监控)

### 1. 合并(已完成,非破坏)
```bash
CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU \
  swift export --adapters <grpo-ckpt-1210> --merge_lora true \
  --output_dir deploy/models/qwen3-fin-v1.0
```
> 合并前 adapter 保留在 `weights_archive/`,sha256 可校验;随时可回到未合并态。

### 2. 质量门禁(上线前必过)
```bash
bash scripts/deploy_gate.sh          # 合并模型 vs GRPO checkpoint,低于 -3pp 则 exit 1
```
CI 中把它作为 deploy 前置 job;exit≠0 阻断发布。

### 3–5. 容器 / 编排 / 网关(一键起)
```bash
cd deploy
docker compose up -d                 # 3 副本 + 网关 + prometheus,全本地镜像
docker compose ps                    # 待 vllm-r* 变 healthy(首次加载数分钟)
```

### 6. 监控
- Prometheus UI:`http://<host>:9090`,关键指标见 `monitoring/prometheus.yml` 注释;
- 快速看板查询:
  - 并发:`sum(vllm:num_requests_running)`
  - 排队深度:`sum(vllm:num_requests_waiting)`
  - KV 占用(容量红线):`max(vllm:gpu_cache_usage_perc)`
  - 首 token 延迟 P99:`histogram_quantile(0.99, sum by(le)(rate(vllm:time_to_first_token_seconds_bucket[5m])))`

## 验证(端到端冒烟)
```bash
curl -s http://127.0.0.1:4000/health                       # 网关存活
python scripts/deploy_smoke_client.py --url http://127.0.0.1:4000/v1 --n 4   # thinking 输出结构
```
调用示例(OpenAI 兼容):
```bash
curl http://127.0.0.1:4000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"qwen3-fin",
  "messages":[{"role":"user","content":"某公司资产负债率如何计算?请把答案放到 \\boxed{} 中。"}],
  "max_tokens":1024}'
```
> 输出为 `<think>…</think><answer>…\boxed{…}</answer>`,后端按需折叠 think、从 boxed 抽结构化答案。**切勿设 enable_thinking=false**(会注入空 think 致输出崩坏)。

## 灰度与回滚
- **灰度**:新版本先起 1 副本接小流量(网关 upstream 加权),指标正常再扩到 3 副本;
- **回滚(合并态)**:`docker compose down` 后把 compose 的 model 指向上一版本 `qwen3-fin-vX`,`up -d`;
- **回滚到未合并态**:停容器,用 base + `weights_archive/grpo-lora-checkpoint-1210-FINAL` 挂 LoRA 服务(vLLM `--enable-lora`);合并工件为派生物,删了可从 base+adapter 重建。

## 停服 / 清理
```bash
cd deploy && docker compose down          # 停全部容器,释放 GPU
```

## 关键排障(本项目实测)
- **副本起不来/TP 卡死**:确认 compose 有 `ipc: host`(NCCL 共享内存)与 `NCCL_P2P_DISABLE=1`;
- **"Servers not reachable"/localhost 被拦**:容器内已设 `NO_PROXY=127.0.0.1,localhost`;宿主若有代理,勿传入容器;
- **GPU 抢占**:生产卡集固定 0,2,3,4,5,6,避开被外部进程占用的卡;
- **registry 超时**:用本地镜像;需新镜像走 `docker.m.daocloud.io` 国内源。
