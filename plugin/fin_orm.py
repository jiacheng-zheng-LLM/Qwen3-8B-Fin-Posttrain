# -*- coding: utf-8 -*-
"""金融推理 GRPO 双奖励(格式 + 正确性),ms-swift 4.4.2 ORM。
- fin_format: 恰好一个 <think> + 一个 <answer> + boxed → 1 else 0(结构信号,小权重)。
- fin_acc:    <answer> 内 boxed 字母 == 标答 → 1 else 0(正确性,主权重)。
挂载:--external_plugins plugin/fin_orm.py --reward_funcs fin_acc fin_format --reward_weights 1.0 0.1
"""
import re
from swift.rewards import ORM, orms

TH = re.compile(r"<think>(.*?)</think>", re.S)
AN = re.compile(r"<answer>(.*?)</answer>", re.S)
BOX = re.compile(r"\\boxed\{(.*?)\}", re.S)


def _choices(t):
    return "".join(c for c in "ABCDEFG" if c in (t or ""))


def _fmt_ok(s):
    return len(TH.findall(s)) == 1 and len(AN.findall(s)) == 1 and bool(BOX.findall(s))


def _answer(s):
    if len(TH.findall(s)) != 1 or len(AN.findall(s)) != 1:
        return None
    bm = BOX.findall(AN.findall(s)[0])
    return _choices(bm[-1]) if bm else None


class FinFormat(ORM):
    def __call__(self, completions, **kwargs):
        return [1.0 if _fmt_ok(c) else 0.0 for c in completions]


class FinAcc(ORM):
    def __call__(self, completions, solution, **kwargs):
        out = []
        for c, gold in zip(completions, solution):
            a = _answer(c)
            out.append(1.0 if (a is not None and a == _choices(str(gold))) else 0.0)
        return out


orms['fin_format'] = FinFormat
orms['fin_acc'] = FinAcc
