#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""distill_cot.py — 推理链蒸馏（按 DianJin-R1 论文方法复原的参考实现）。

⚠️ 这是**复原件**，不是产出本项目 SFT 数据的原始脚本（原始脚本已遗失）。
   它依据论文 arXiv:2504.15716 公开描述的流程重写，供他人复现使用。
   用它重跑**不保证**得到与本项目完全相同的数据：当时的生成超参未留存记录，
   且教师模型本身带采样随机性。详见仓库根目录 DISTILLATION.md。

论文描述的流程（本脚本实现的部分）：
  1. 教师模型对每道题生成 <think> 推理 + <answer> 含 \\boxed{} 的答案；
  2. 校验两个判据 —— ① 抽出的答案 == 标准答案；② 推理与参考解释一致（LLM 判）；
  3. 最多尝试 T=3 次；三次都不通过 → 该题降级为 non-reasoning 样本（只留题面+标答）。

用法（OpenAI 兼容接口；教师与校验模型可指向不同服务）：
  export TEACHER_BASE_URL=https://api.deepseek.com/v1
  export TEACHER_API_KEY=...
  export TEACHER_MODEL=deepseek-v4-pro
  # 教师采样参数:V4-Pro-0813 官方推荐(非 agentic 场景)
  export GEN_TEMPERATURE=1.0
  export GEN_TOP_P=1.0
  export VERIFIER_BASE_URL=https://api.openai.com/v1
  export VERIFIER_API_KEY=...
  export VERIFIER_MODEL=gpt-4o
  python distill_cot.py --in data/cflue_mcq.raw.json --out data/cflue_mcq.json --task mcq

  # 不调 API，只跑内置纯函数自测：
  python distill_cot.py --selftest
"""
import argparse
import json
import os
import re
import sys
import time

BOX = re.compile(r"\\boxed\{(.*?)\}", re.S)
THINK = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)

MAX_ATTEMPTS = 3          # 论文 T=3
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------- prompts

GEN_PROMPT_MCQ = """假设你是一位金融行业专家，请回答下列问题。
注意：题目是单选题，只需要返回一个最合适的选项，若有多个合适的答案，只返回最准确的即可。

{question}

请一步步思考，并把答案选项放到\\boxed{{}}中，如\\boxed{{A}}。
"""

GEN_PROMPT_NUM = """You are a financial analysis expert. Answer the question using the data provided.

{question}

Please reason step by step, and put your final answer within \\boxed{{}}.
"""

# 复原版校验 prompt —— 按论文描述的两个判据构造。
# 原始 prompt 已遗失，措辞与当时使用的大概率不同。
VERIFY_PROMPT = """You are checking one distilled reasoning sample. Judge two things independently.

[Question]
{question}

[Gold answer]
{gold}

[Reference explanation]
{reference}

[Model reasoning]
{reasoning}

[Model answer]
{predicted}

Judge:
1. answer_match — does the model answer mean the same as the gold answer?
   Ignore formatting, units written out in words, and trailing zeros.
2. reasoning_consistent — is the model reasoning consistent with the reference
   explanation? It does not have to match step for step, but it must not rely on
   a wrong fact, a wrong figure, or a route that contradicts the reference.
   If the model reaches the right answer through clearly wrong reasoning, this is false.

Reply with JSON only, no other text:
{{"answer_match": true/false, "reasoning_consistent": true/false, "why": "<one short sentence>"}}
"""


# ------------------------------------------------------- 纯函数（可离线自测）

def extract_answer(text):
    """从模型输出里抽 <answer> 内最后一个 \\boxed{} 的内容。"""
    if not text:
        return None
    scope = ANSWER.findall(text)
    scope = scope[-1] if scope else text
    hits = BOX.findall(scope)
    return hits[-1].strip() if hits else None


def format_ok(text):
    """论文的结构约束:恰好一个 <think>、恰好一个 <answer>、且 answer 内有 boxed。"""
    return (len(THINK.findall(text or "")) == 1
            and len(ANSWER.findall(text or "")) == 1
            and extract_answer(text) is not None)


def norm_choice(s):
    """选择题答案归一:只留 A-G 字母,去重保序。"""
    seen, out = set(), []
    for c in (s or "").upper():
        if c in "ABCDEFG" and c not in seen:
            seen.add(c)
            out.append(c)
    return "".join(out)


def numeric_match(pred, gold, rel_tol=1e-3):
    """数值题比对:相对容差,并兜底百分比 ×100 / ÷100 的口径差。"""
    def first_num(s):
        m = NUM_RE.search((s or "").replace(",", "").replace("%", ""))
        return float(m.group()) if m else None

    p, g = first_num(pred), first_num(gold)
    if p is None or g is None:
        return False

    def close(a, b):
        return abs(a - b) <= rel_tol * max(1.0, abs(b))

    return close(p, g) or close(p * 100, g) or close(p / 100, g)


def answer_match(pred, gold, task):
    """任务相关的答案比对。task: mcq | numeric | open"""
    if pred is None:
        return False
    if task == "mcq":
        return norm_choice(pred) == norm_choice(gold) and norm_choice(pred) != ""
    if task == "numeric":
        return numeric_match(pred, gold)
    return pred.strip() == (gold or "").strip()


def build_sample(question, reasoning, answer_text, gold):
    """通过校验的样本 → SFT 训练格式(think/answer 各一)。"""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant",
             "content": f"<think>{reasoning}</think><answer>{answer_text}</answer>"},
        ],
        "solution": gold,
    }


def build_non_reasoning(question, gold):
    """论文:三次尝试都失败 → 降级保留为 non-reasoning 样本。"""
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<answer>\\boxed{{{gold}}}</answer>"},
        ],
        "solution": gold,
        "non_reasoning": True,
    }


# ------------------------------------------------------------------ API 层

def _client(prefix):
    from openai import OpenAI
    base = os.environ.get(f"{prefix}_BASE_URL")
    key = os.environ.get(f"{prefix}_API_KEY")
    model = os.environ.get(f"{prefix}_MODEL")
    if not (base and model):
        sys.exit(f"请设置 {prefix}_BASE_URL / {prefix}_MODEL（{prefix}_API_KEY 可选）")
    return OpenAI(base_url=base, api_key=key or "EMPTY"), model


def gen_once(client, model, question, task, gen_kwargs):
    prompt = (GEN_PROMPT_MCQ if task == "mcq" else GEN_PROMPT_NUM).format(question=question)
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **gen_kwargs,
    )
    return r.choices[0].message.content


def verify(client, model, question, gold, reference, reasoning, predicted):
    """论文的双判据校验。返回 (answer_match, reasoning_consistent)。"""
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(
            question=question, gold=gold, reference=reference or "(none provided)",
            reasoning=reasoning, predicted=predicted)}],
    )
    raw = r.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        d = json.loads(raw)
        return bool(d.get("answer_match")), bool(d.get("reasoning_consistent"))
    except json.JSONDecodeError:
        return False, False


# -------------------------------------------------------------------- 主流程

def run(args):
    teacher, t_model = _client("TEACHER")
    verifier, v_model = _client("VERIFIER")

    gen_kwargs = {}
    for k, env in (("temperature", "GEN_TEMPERATURE"),
                   ("top_p", "GEN_TOP_P"),
                   ("max_tokens", "GEN_MAX_TOKENS")):
        if os.environ.get(env):
            gen_kwargs[k] = float(os.environ[env]) if k != "max_tokens" else int(os.environ[env])
    _shown = gen_kwargs or ("(全部走服务端默认;V4-Pro-0813 官方推荐为 "
                            "temperature=1.0 / top_p=1.0,非 agentic 场景)")
    print(f"[gen kwargs] {_shown}")

    rows = json.load(open(args.inp, encoding="utf-8"))
    out, kept, degraded = [], 0, 0

    for i, item in enumerate(rows, 1):
        q = item[args.question_key]
        gold = str(item[args.answer_key])
        ref = item.get(args.reference_key)

        sample = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                text = gen_once(teacher, t_model, q, args.task, gen_kwargs)
            except Exception as e:                     # noqa: BLE001
                print(f"  [{i}] 第{attempt}次生成失败: {e}")
                time.sleep(2)
                continue

            if not format_ok(text):
                continue
            pred = extract_answer(text)
            if not answer_match(pred, gold, args.task):
                continue

            ok_a, ok_r = verify(verifier, v_model, q, gold, ref,
                                THINK.findall(text)[0], pred)
            if ok_a and ok_r:
                sample = build_sample(q, THINK.findall(text)[0],
                                      ANSWER.findall(text)[0], gold)
                break

        if sample is None:
            sample = build_non_reasoning(q, gold)
            degraded += 1
        else:
            kept += 1
        out.append(sample)

        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  通过 {kept} / 降级 {degraded}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n完成:{len(out)} 条 → {args.out}")
    print(f"  带推理链 {kept} ({kept/len(out)*100:.1f}%) / non-reasoning {degraded}")


# ------------------------------------------------------------ 离线自测纯函数

def selftest():
    cases = []
    good = "<think>先算净利润</think><answer>所以选\\boxed{D}</answer>"
    cases.append(("format_ok/good", format_ok(good) is True))
    cases.append(("format_ok/双think", format_ok("<think>a</think><think>b</think>"
                                                 "<answer>\\boxed{A}</answer>") is False))
    cases.append(("format_ok/无boxed", format_ok("<think>a</think><answer>A</answer>") is False))
    cases.append(("extract/取最后一个", extract_answer(
        "<think>\\boxed{X}</think><answer>\\boxed{A}\\boxed{B}</answer>") == "B"))
    cases.append(("norm_choice", norm_choice(" a, A ,B ") == "AB"))
    cases.append(("mcq/命中", answer_match("D", "D", "mcq") is True))
    cases.append(("mcq/不命中", answer_match("C", "D", "mcq") is False))
    cases.append(("mcq/空", answer_match("", "D", "mcq") is False))
    cases.append(("numeric/容差", numeric_match("1250.0004", "1250") is True))
    cases.append(("numeric/百分比×100", numeric_match("0.2755", "27.55") is True))
    cases.append(("numeric/百分比÷100", numeric_match("27.55", "0.2755") is True))
    cases.append(("numeric/千分位", numeric_match("1,250", "1250") is True))
    cases.append(("numeric/不命中", numeric_match("980", "1250") is False))
    cases.append(("build_sample", build_sample("q", "r", "a", "D")["solution"] == "D"))
    nr = build_non_reasoning("q", "D")
    cases.append(("non_reasoning/标记", nr["non_reasoning"] is True))
    cases.append(("non_reasoning/无think", "<think>" not in nr["messages"][1]["content"]))

    bad = [n for n, ok in cases if not ok]
    for n, ok in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n{len(cases) - len(bad)}/{len(cases)} 通过")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", help="输入题目 JSON（list）")
    ap.add_argument("--out", help="输出 JSON")
    ap.add_argument("--task", choices=["mcq", "numeric", "open"], default="mcq")
    ap.add_argument("--question-key", default="question")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--reference-key", default="explanation",
                    help="参考解释字段,供 GPT-4o 判推理一致性")
    ap.add_argument("--selftest", action="store_true", help="只跑纯函数自测,不调 API")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not (args.inp and args.out):
        ap.error("需要 --in 与 --out（或用 --selftest）")
    run(args)


if __name__ == "__main__":
    main()
