# -*- coding: utf-8 -*-
"""eval_fin.py — 统一评测(CFLUE/FinQA/MATH-500/GPQA)。
- 模型 thinking 开(Qwen3 原生);从 \boxed{} 抽答案。base 与 SFT(LoRA)同 prompt 公平对比。
- 判分:CFLUE/GPQA 字母匹配;MATH-500 math_verify(LaTeX 容差);FinQA fin_verify 数值(尺度容忍;评测环节不引入 LLM 判分)。
用法:CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 python eval_fin.py --benchmark cflue --adapter <ckpt或空> --sample 800 --tp 6
"""
import os, sys, json, re, glob, argparse, ast
from pathlib import Path
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(__file__)
PROJ = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, os.path.join(PROJ, "src"))

# fin_verify 是内部数值判分模块,不随本仓库发布;缺失时用下面的等价兜底。
try:
    from verifier.fin_verify import extract_answer as fin_extract, parse_number  # noqa
except Exception:
    def fin_extract(gen):
        """无 \boxed{} 时的答案抽取兜底:优先 ####,否则取正文最后一个数。"""
        if not gen:
            return None
        m = re.search(r"####\s*(.+)", gen)
        if m:
            return m.group(1).strip().split("\n")[0]
        nums = re.findall(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", gen)
        return nums[-1] if nums else None

    def parse_number(src):
        """→ (float | None, None)。与 fin_verify.parse_number 的返回形状一致。"""
        c = clean_num(src)
        if c is None:
            return None, None
        try:
            return float(c), None
        except ValueError:
            return None, None

# ---- 代码执行判分依赖外部沙箱模块,不在本仓库范围内(仅 --mode code 需要)----
try:
    from fin_score import num_ok, extract_ndigits, clean_num
    from sandbox_exec import run_isolated, extract_code
    _HAS_P2 = True
except Exception:                              # 缺失时 boxed 模式照常工作(本仓库全部已发布结果均由此模式产出)
    _HAS_P2 = False
    def extract_ndigits(q): return None
    def clean_num(s):                          # P-6 兜底：清洗 \%/$/逗号
        if s is None: return None
        s = str(s).replace("−", "-").strip()
        for a, b in (("\\%", ""), ("%", ""), ("\\$", ""), ("$", ""), ("\\,", ""),
                     (",", ""), ("~", ""), ("\\", "")):
            s = s.replace(a, b)
        m = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)
        return m.group(0) if m else None
    def num_ok(pred, gt, ndigits=None, allow_pct=True):   # 与 fin_score 同逻辑的兜底
        if isinstance(gt, bool): return isinstance(pred, bool) and pred == gt
        if isinstance(pred, bool): return False
        try: pv, gv = float(pred), float(gt)
        except Exception: return False
        if pv != pv or gv != gv: return False
        if ndigits is not None:
            try:
                n = int(ndigits)
                if round(pv, n) == round(gv, n): return True
                return abs(pv - gv) <= 0.5 * (10 ** (-n)) + 1e-9   # P-9 边界舍入
            except Exception: return False
        cands = (gv, gv*100.0, gv/100.0) if allow_pct else (gv,)
        return any(abs(pv - c) <= max(2e-3*abs(c), 1e-4) for c in cands)

def _is_json(ctx):
    try: json.loads(ctx); return True
    except Exception: return False

# PoT 系统提示(代码执行模式用;本仓库不发布该模式的训练数据)
POT_SYS = ("You are a financial analysis assistant. Read the question and the data, then write a "
           "self-contained Python program: define `def solution():` that returns the numeric answer. "
           "If a data table is given, it is preloaded as pandas DataFrame `df`. Put it in a ```python block.")

BASE = os.environ["BASE_MODEL"]  # 例:/path/to/models/Qwen3-8B
ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent))
EV = os.environ.get("EVAL_DATA_DIR", ROOT + "/data/eval_public")  # 评测集根目录,见 DATA.md
BOX = re.compile(r"\\boxed\{([^{}]*)\}")


def boxed(text):
    m = BOX.findall(text or "")
    return m[-1].strip() if m else None


def letters(s):
    return "".join(c for c in "ABCDEFG" if c in (s or "").upper())


# ---------- loaders: 返回 [{prompt, gold, kind}] ----------
def load_cflue(limit):
    d = json.load(open(f"{EV}/cflue/test.json"))
    d = d if isinstance(d, list) else d.get("data") or list(d.values())[0]
    out = []
    for r in d:
        ch = r["choices"]
        if isinstance(ch, str):
            ch = ast.literal_eval(ch)
        opts = "\n".join(f"{k}. {v}" for k, v in ch.items())
        p = (f"请回答下列金融选择题。\n\n{r['question']}\n{opts}\n\n"
             "请一步步思考，然后把最终答案选项（如 A）放到 \\boxed{} 中。")
        out.append({"prompt": p, "gold": letters(str(r["answer"])), "kind": "mcq"})
    return out[:limit] if limit else out


def load_finqa(limit):
    d = json.load(open(f"{EV}/finqa/test.json"))
    out = []
    for r in d:
        qa = r.get("qa", {})
        if not qa.get("question") or qa.get("exe_ans") is None:
            continue
        if isinstance(qa["exe_ans"], str):  # yes/no 跳过
            continue
        pre = " ".join(r.get("pre_text", []) or []); post = " ".join(r.get("post_text", []) or [])
        tbl = "\n".join(" | ".join(str(c) for c in row) for row in (r.get("table", []) or []))
        ctx = f"{pre}\n{tbl}\n{post}".strip()[:6000]
        p = (f"Answer the financial question based on the context.\n\nContext:\n{ctx}\n\n"
             f"Question: {qa['question']}\n\nThink step by step, then put the final numeric answer in \\boxed{{}}.")
        # code 模式：FinQA 上下文非 JSON → ctx_json="" (df=None,模型内联数值);题面给模型看
        out.append({"prompt": p, "gold": float(qa["exe_ans"]), "kind": "num",
                    "q": qa["question"], "ctx_prompt": ctx, "ctx_json": "",
                    "ndigits": extract_ndigits(qa["question"])})
    return out[:limit] if limit else out


def load_frheld(limit, manifest):
    """外部 held-out 数值评测集(不在本仓库范围内),读锁定的 manifest.records。"""
    man = json.load(open(manifest, encoding="utf-8"))
    out = []
    for lvl, d in man.get("levels", {}).items():
        for r in d.get("records", []):
            q, ctx = r["question"], str(r.get("context", ""))
            p = (f"Answer the financial question based on the data.\n\nData:\n{ctx[:6000]}\n\n"
                 f"Question: {q}\n\nThink step by step, then put the final numeric answer in \\boxed{{}}.")
            out.append({"prompt": p, "gold": r["ground_truth"], "kind": "num",
                        "q": q, "ctx_prompt": ctx[:6000], "ctx_json": ctx if _is_json(ctx) else "",
                        "ndigits": extract_ndigits(q), "level": lvl})
    return out[:limit] if limit else out


def load_math500(limit):
    m = [json.loads(l) for l in open(glob.glob(f"{EV}/math500/*.jsonl")[0])]
    out = []
    for r in m:
        p = (f"Solve the math problem.\n\n{r['problem']}\n\n"
             "Think step by step, then put the final answer in \\boxed{}.")
        out.append({"prompt": p, "gold": r["answer"], "kind": "math"})
    return out[:limit] if limit else out


def load_gpqa(limit):
    import csv
    g = list(csv.DictReader(open(glob.glob(f"{EV}/gpqa/gpqa_diamond.csv")[0])))
    out = []
    for i, r in enumerate(g):
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        perm = [(i + k) % 4 for k in range(4)]          # 确定性洗牌(禁 random):位置k展示opts[perm[k]]
        order = [opts[perm[k]] for k in range(4)]
        gold_letter = "ABCD"[perm.index(0)]             # 正确答案(opts[0])洗后所在字母
        body = "\n".join(f"{'ABCD'[k]}. {order[k]}" for k in range(4))
        p = (f"Answer the multiple-choice science question.\n\n{r['Question']}\n{body}\n\n"
             "Think step by step, then put the answer letter (e.g. A) in \\boxed{}.")
        out.append({"prompt": p, "gold": gold_letter, "kind": "mcq"})
    return out[:limit] if limit else out


LOADERS = {"cflue": load_cflue, "finqa": load_finqa, "math500": load_math500, "gpqa": load_gpqa}
CODE_OK = {"finqa", "frheld"}          # code 模式支持的数值基准(mcq/math 无意义)


# ---------- scorers（num 统一走共享 num_ok，boxed 与 code 同口径可比）----------
def score_boxed(it, gen):
    kind, gold = it["kind"], it["gold"]
    b = boxed(gen)
    if kind == "mcq":
        return b is not None and letters(b) == gold
    if kind == "num":
        src = b if b is not None else fin_extract(gen)
        if src is None:
            return False
        cs = clean_num(src)                       # P-6：先清洗 \%/$/逗号
        pv = None
        if cs is not None:
            try: pv = float(cs)
            except Exception: pv = None
        if pv is None:
            pv, _ = parse_number(src)             # 退回 fin_verify 解析
        if pv is None:
            return False
        return num_ok(pv, gold, ndigits=it.get("ndigits"))
    if kind == "math":
        if b is None:
            return False
        try:
            from math_verify import parse as mvparse, verify as mvverify
            return bool(mvverify(mvparse(str(gold)), mvparse(b)))
        except Exception:
            return b.replace(" ", "") == str(gold).replace(" ", "")
    return False


def score_code(it, gen):
    """代码执行模式:抽代码→沙箱执行→num_ok 判分。"""
    code = extract_code(gen)
    if code is None:
        return False
    pred, st = run_isolated(code, ctx_json=it.get("ctx_json", ""), timeout=8)
    if st != "ok" or pred is None:
        return False
    return num_ok(pred, it["gold"], ndigits=it.get("ndigits"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(LOADERS.keys()) + ["frheld"])
    ap.add_argument("--adapter", default="")
    ap.add_argument("--model", default="", help="覆盖 BASE:指向合并后的全量模型(门禁用),不与 --adapter 同用")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tp", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode", choices=["boxed", "code"], default="boxed",
                    help="boxed=文本+\\boxed(本仓库使用); code=写def solution()+执行(需外部沙箱模块)")
    ap.add_argument("--heldout", default=glob.glob(os.path.join(_P2[0], "..", "output", "heldout_manifest.json"))[0]
                    if _P2 and glob.glob(os.path.join(_P2[0], "..", "output", "heldout_manifest.json")) else "",
                    help="frheld 基准用的 held-out manifest")
    a = ap.parse_args()

    if a.mode == "code":
        assert _HAS_P2, ("code 模式需外部沙箱模块 fin_score.py / sandbox_exec.py;"
                     "本仓库只发布 boxed 模式——全部已公布结果均由 boxed 模式产出")
        assert a.benchmark in CODE_OK, f"code 模式只支持数值基准 {CODE_OK}（mcq/math 无意义）"
    if a.benchmark == "frheld":
        assert a.heldout and os.path.exists(a.heldout), "frheld 需 --heldout 指向 heldout_manifest.json"
        items = load_frheld(a.sample, a.heldout)
    else:
        items = LOADERS[a.benchmark](a.sample)
    tag = a.tag or (("base" if not a.adapter else "sft") + ("" if a.mode == "boxed" else "_code"))
    print(f"[{a.benchmark}/{tag}] mode={a.mode} {len(items)} 题 | adapter={a.adapter or '无(base)'}", flush=True)

    from vllm import LLM, SamplingParams
    tokp = dict(model=(a.model or BASE), dtype="bfloat16", enforce_eager=True, tensor_parallel_size=a.tp,
                gpu_memory_utilization=0.9, max_model_len=8192, trust_remote_code=True)
    lora_req = None
    if a.adapter:
        tokp.update(enable_lora=True, max_lora_rank=32)
    llm = LLM(**tokp)
    tok = llm.get_tokenizer()
    if a.adapter:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("sft", 1, a.adapter)

    if a.mode == "code":            # PoT：写 def solution()，thinking 默认开
        msgs = [[{"role": "user", "content": POT_SYS},
                 {"role": "user", "content": f"## Question\n{it['q']}\n## Data\n{it.get('ctx_prompt','')}"}] for it in items]
    else:
        msgs = [[{"role": "user", "content": it["prompt"]}] for it in items]
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    sp = SamplingParams(temperature=0.0, max_tokens=3072)
    outs = llm.generate(prompts, sp, lora_request=lora_req) if lora_req else llm.generate(prompts, sp)

    scorer = score_code if a.mode == "code" else score_boxed
    correct = sum(1 for it, o in zip(items, outs) if scorer(it, o.outputs[0].text))
    acc = correct / len(items)
    rep = {"benchmark": a.benchmark, "tag": tag, "mode": a.mode, "adapter": a.adapter or None,
           "n": len(items), "correct": correct, "accuracy": round(acc, 4)}
    print("\n=== 结果 ===", json.dumps(rep, ensure_ascii=False))
    out = f"{ROOT}/eval/eval_{a.benchmark}_{tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=2)
    print("报告 ->", out)


if __name__ == "__main__":
    main()
