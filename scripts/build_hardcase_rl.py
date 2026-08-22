# -*- coding: utf-8 -*-
"""build_hardcase_rl.py — 客观难例筛选(GRPO RL 数据)。
判据:SFT模型对每题采样 k 次,c=格式合规且答案字母==标答 的次数;留 0<c/k<1(GRPO 有梯度=有对有错)。
纯客观、可复现、无 R1/GPT-4o。输出 ms-swift GRPO 格式:{"messages":[{user}], "solution": 金标准字母}。
用法:CUDA_VISIBLE_DEVICES=0,2,3,4 python build_hardcase_rl.py --model <merged> --n 8000 --k 4
"""
import os, sys, json, re, glob, argparse
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent))
D = os.environ.get("SFT_DATA_DIR", ROOT + "/data/train")  # 蒸馏产出的 SFT 数据目录
TH = re.compile(r"<think>(.*?)</think>", re.S); AN = re.compile(r"<answer>(.*?)</answer>", re.S)
BOX = re.compile(r"\\boxed\{(.*?)\}", re.S)
def choices(t): return "".join(c for c in "ABCDEFG" if c in (t or ""))
def model_answer(s):  # 奖励口径:恰好1个think+1个answer+boxed
    if len(TH.findall(s)) != 1 or len(AN.findall(s)) != 1: return None
    bm = BOX.findall(AN.findall(s)[0])
    return choices(bm[-1]) if bm else None
def gold_of(x):
    m = BOX.findall(x["output"]); return choices(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)          # merged SFT 模型
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out", default=ROOT + "/data/grpo_hardcase.jsonl")
    a = ap.parse_args()
    data = json.load(open(f"{D}/cflue_mcq.json", encoding="utf-8"))[:a.n]
    items = [(x["instruction"], gold_of(x)) for x in data]
    items = [(q, g) for q, g in items if g]
    print(f"[hardcase] 扫 {len(items)} 题, k={a.k}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=True, tensor_parallel_size=a.tp,
              gpu_memory_utilization=0.9, max_model_len=6144, trust_remote_code=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
               for q, _ in items]
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1536, n=a.k)
    outs = llm.generate(prompts, sp)

    dist = {"c=0(全错,无正样本)": 0, "0<c<k(混合·GRPO有梯度)": 0, "c=k(全对,无梯度)": 0}
    hard = []
    for (q, gold), o in zip(items, outs):
        c = sum(1 for s in o.outputs if model_answer(s.text) == gold)
        if c == 0: dist["c=0(全错,无正样本)"] += 1
        elif c == a.k: dist["c=k(全对,无梯度)"] += 1
        else:
            dist["0<c<k(混合·GRPO有梯度)"] += 1
            hard.append({"messages": [{"role": "user", "content": q}], "solution": gold})
    with open(a.out, "w", encoding="utf-8") as f:
        for h in hard:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    n = len(items)
    print(json.dumps({"扫描题数": n, "分布": {k: f"{v}({100*v/n:.1f}%)" for k, v in dist.items()},
                      "难例(RL可用)": len(hard), "输出": a.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
