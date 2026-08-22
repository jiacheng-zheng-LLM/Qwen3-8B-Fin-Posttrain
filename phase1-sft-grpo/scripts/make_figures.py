#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py —— 从交付包内的原始 artifact 重新绘制全部图表。

为什么需要它:ms-swift 自动产出的 output/*/images/*.png 只对 loss / grad_norm /
learning_rate 做了平滑,GRPO 的 reward、FinAcc、completions_length 等关键
曲线是**未平滑的原始噪声**,画出来是一整片实心色块,读不出任何信息。另外四张
最能说明问题的图(基准对比、难例分布、max_length 消融、部署压测)ms-swift 根本
不会生成。本脚本把这些图一次性重画/补齐,且全部可复现。

平滑方式:**居中滑动平均**(非 EMA)。EMA 在 loss 陡降段会滞后,把曲线画到原始
数据上方,看起来像"降得慢";居中窗口无相位滞后。窗口宽度写在图例里。

数据来源(全部在本交付包内):
  eval/eval_{bench}_{tag}.json                       → 图1 四基准对比
  artifacts/figure_inputs.json  → 图2/3/4/5/6/8 的全部曲线与汇总数字
  eval/*.json                   → 图1 四基准对比
  (原始运行记录不随仓库发布,绘图数据已在打包时抽取到 artifacts/)
  deploy/DEPLOY_LOG.md「加演A」表格                   → 图7 部署压测(见 LOADTEST 常量)

用法:
  python3 scripts/make_figures.py                # 中文标签 → figures/
  python3 scripts/make_figures.py --lang en      # 英文标签 → figures/en/
  python3 scripts/make_figures.py --only 3 6     # 只画指定编号
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent.parent  # phase1-sft-grpo/

# ---------------------------------------------------------------- 样式

C_BASE = "#9aa0a6"   # base 模型 / 中性
C_SFT = "#2b6cb0"    # SFT
C_GRPO = "#dd6b20"   # GRPO
C_OK = "#2f855a"     # 通过 / 有梯度
C_BAD = "#c53030"    # OOM / 失败 / 零梯度
C_RAW = "#f4c9a8"    # 原始噪声(浅)
C_GRID = "#dde1e6"
C_TEXT = "#2d3748"
C_MUTED = "#718096"


def pick_cjk_font() -> str | None:
    """挑一个装机可用的中文字体;没有就返回 None(自动退回英文标签)。"""
    prefer = [
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei", "Droid Sans Fallback", "AR PL UMing CN",
    ]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in have:
            return name
    return None


def setup_style(lang: str) -> str:
    """配置全局样式;返回实际生效的语言(中文字体缺失时降级为 en)。"""
    if lang == "zh":
        cjk = pick_cjk_font()
        if cjk is None:
            print("[warn] 未找到中文字体,自动改用英文标签", file=sys.stderr)
            lang = "en"
        else:
            plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
            # 多数开源中文字体只有 500 一档字重,声明 bold 会刷一屏 findfont 警告
            plt.rcParams["axes.titleweight"] = "normal"
    plt.rcParams.update({
        "axes.unicode_minus": False,          # 中文字体下负号会变方块
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.7,
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "axes.labelcolor": C_TEXT,
        "text.color": C_TEXT,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    })
    return lang


# ---------------------------------------------------------------- 文案

T = {
    "zh": {
        "step": "训练步 step",
        "acc": "准确率 (%)",
        "raw": "原始",
        "smooth": "滑动平均(窗口 {w})",

        "fig1_title": "四基准同口径评测:base → SFT → SFT+GRPO",
        "fig1_sub": "Qwen3-8B / LoRA / 统一 prompt,均从 \\boxed{} 抽答案;accuracy = correct / n",
        "fig1_mean": "均值",

        "fig2_title": "SFT 训练 loss(LoRA r32/α64,max_length 5120,3 epoch,1113 步)",
        "fig2_ep": "epoch 边界",

        "fig3_title": "GRPO 全程未移动:reward 持平 + 四成组内优势为 0",
        "fig3_reward": "总 reward",
        "fig3_acc": "FinAcc 分量",
        "fig3_zero": "组内 reward 方差为 0 的比例",
        "fig3_ylab_zero": "零优势组占比",
        "fig3_note": ("首/末 100 步均值 {a100:.3f} → {b100:.3f}(持平)\n"
                      "首/末 20 步 {a20:.3f} → {b20:.3f},属批次难度噪声"),
        "fig3_note2": "平均 {z:.1%} 的组 reward 全同 → 优势=0 → 该批无梯度",

        "fig4_title": "GRPO 的 KL 全程 ≈ 0 —— GRPO 净效应≈0 的直接归因",
        "fig4_ylab": "KL(policy ‖ ref = 冻结的 SFT)",
        "fig4_note": ("KL 中位数 {med:.1e},全程最大 {mx:.1e},β=0.04\n"
                      "--ref_adapters 把参考模型锚在 SFT → 过度正则\n"
                      "→ 策略几乎没从 SFT 起点移开"),

        "fig5_title": "客观难例筛选器:8000 题里只有 30.3% 对 GRPO 有梯度",
        "fig5_sub": "SFT 模型对每题采样 k=4,c = 格式合规且答案正确的次数;GRPO 优势 = 组内 reward 方差",
        "fig5_allright": "c=k 全对\n(优势=0,零梯度)",
        "fig5_mixed": "0<c<k 混合\n(有梯度,RL 可用)",
        "fig5_allwrong": "c=0 全错\n(优势=0,零梯度)",
        "fig5_note": "随机抽题 → 近 7 成算力打在零梯度样本上",

        "fig6_title": "max_length 消融:5120 是「装得下」与「留得住数据」的交点",
        "fig6_sub": "liger_kernel 融合交叉熵 + DDP/zero2 + 6×RTX4090(24GB),bs1 ga16",
        "fig6_mem": "峰值显存 (GiB/卡)",
        "fig6_keep": "预过滤后数据保留率",
        "fig6_limit": "单卡 24GB 物理上限",
        "fig6_oom": "OOM",
        "fig6_pass": "跑通",
        "fig6_final": "定稿",
        "fig6_foot": "OOM 档位柱高 {m:.2f} 是崩溃前最后一次记录,非可持续峰值",
        "fig6_panelA": "① 显存:能不能装下",
        "fig6_panelB": "② 数据:要丢多少",

        "fig7_title": "生产部署压测:3×TP2 副本 + Nginx 网关(固定输出 256 token,temperature 0)",
        "fig7_conc": "并发数",
        "fig7_tp": "吞吐 (req/s)",
        "fig7_lat": "延迟 (s)",
        "fig7_p50": "P50",
        "fig7_p95": "P95",
        "fig7_cluster": "集群 3 副本",
        "fig7_single": "单副本 r1(TP2)@96",
        "fig7_note": ("单 TP2 副本 12.79 req/s ≈ 集群 12.66\n"
                      "→ 此负载下单副本尚未饱和,\n集群价值在容量余量与 HA,而非中负载吞吐"),
        "fig7_panelA": "① 吞吐",
        "fig7_panelB": "② 延迟",

        "fig8_title": "GRPO 生成长度:均长 265 token,1536 的上限几乎从未触及",
        "fig8_mean": "批内均长",
        "fig8_max": "批内最长",
        "fig8_cap": "max_completion_length = 1536",
        "fig8_tok": "生成长度 (token)",
        "fig8_note": ("全程均长 {mean:.0f} token\n"
                      "1210 步中仅 {n} 步出现截断,平均截断率 {r:.2%}\n"
                      "→ 1536 不是瓶颈,显存该花在 num_generations 上"),
    },
    "en": {
        "step": "training step",
        "acc": "accuracy (%)",
        "raw": "raw",
        "smooth": "moving average (w={w})",

        "fig1_title": "Four benchmarks, identical protocol: base → SFT → SFT+GRPO",
        "fig1_sub": "Qwen3-8B / LoRA / same prompt, answer parsed from \\boxed{}; accuracy = correct / n",
        "fig1_mean": "mean",

        "fig2_title": "SFT training loss (LoRA r32/α64, max_length 5120, 3 epochs, 1113 steps)",
        "fig2_ep": "epoch boundary",

        "fig3_title": "GRPO never moved: flat reward + 40% of groups have zero advantage",
        "fig3_reward": "total reward",
        "fig3_acc": "FinAcc component",
        "fig3_zero": "fraction of groups with zero reward std",
        "fig3_ylab_zero": "zero-advantage groups",
        "fig3_note": ("first/last 100 steps {a100:.3f} → {b100:.3f} (flat)\n"
                      "first/last 20 steps {a20:.3f} → {b20:.3f}, batch-difficulty noise"),
        "fig3_note2": "on average {z:.1%} of groups share one reward → advantage=0 → no gradient",

        "fig4_title": "GRPO KL stays ≈ 0 — the direct cause of the ≈0 net effect",
        "fig4_ylab": "KL(policy ‖ ref = frozen SFT)",
        "fig4_note": ("median KL {med:.1e}, max {mx:.1e}, β=0.04\n"
                      "--ref_adapters anchors the reference at SFT → over-regularised\n"
                      "→ the policy barely left its SFT init"),

        "fig5_title": "Objective hard-case filter: only 30.3% of 8000 items carry GRPO gradient",
        "fig5_sub": "k=4 samples per item from the SFT model; c = #(format-valid AND correct); GRPO advantage = within-group reward variance",
        "fig5_allright": "c=k all correct\n(advantage=0, no gradient)",
        "fig5_mixed": "0<c<k mixed\n(has gradient, usable for RL)",
        "fig5_allwrong": "c=0 all wrong\n(advantage=0, no gradient)",
        "fig5_note": "random sampling → ~70% of compute spent on zero-gradient items",

        "fig6_title": "max_length ablation: 5120 is where 'fits in memory' meets 'keeps the data'",
        "fig6_sub": "liger_kernel fused CE + DDP/zero2 + 6×RTX4090 (24GB), bs1 ga16",
        "fig6_mem": "peak memory (GiB/GPU)",
        "fig6_keep": "data retention after pre-filter",
        "fig6_limit": "24GB physical limit per GPU",
        "fig6_oom": "OOM",
        "fig6_pass": "ran fine",
        "fig6_final": "chosen",
        "fig6_foot": "the OOM bar ({m:.2f}) is the last value logged before the crash, not a sustainable peak",
        "fig6_panelA": "(1) memory: does it fit",
        "fig6_panelB": "(2) data: how much is lost",

        "fig7_title": "Production load test: 3×TP2 replicas + Nginx gateway (fixed 256 output tokens, temperature 0)",
        "fig7_conc": "concurrency",
        "fig7_tp": "throughput (req/s)",
        "fig7_lat": "latency (s)",
        "fig7_p50": "P50",
        "fig7_p95": "P95",
        "fig7_cluster": "cluster, 3 replicas",
        "fig7_single": "single replica r1 (TP2) @96",
        "fig7_note": ("single TP2 replica 12.79 req/s ≈ cluster 12.66\n"
                      "→ the replica is not saturated at this load;\nthe cluster buys headroom and HA, not mid-load throughput"),
        "fig7_panelA": "(1) throughput",
        "fig7_panelB": "(2) latency",

        "fig8_title": "GRPO completion length: 265 tokens on average, the 1536 cap almost never hit",
        "fig8_mean": "mean length in batch",
        "fig8_max": "max length in batch",
        "fig8_cap": "max_completion_length = 1536",
        "fig8_tok": "completion length (tokens)",
        "fig8_note": ("overall mean {mean:.0f} tokens\n"
                      "only {n} of 1210 steps saw any clipping, mean clip rate {r:.2%}\n"
                      "→ 1536 is not the bottleneck; spend memory on num_generations"),
    },
}

BENCH_LABEL = {
    "zh": {"cflue": "CFLUE\n中文金融选择", "finqa": "FinQA\n英文数值推理",
           "math500": "MATH-500\n通用数学", "gpqa": "GPQA-d\n通用科学"},
    "en": {"cflue": "CFLUE\nZH finance MCQ", "finqa": "FinQA\nEN numerical",
           "math500": "MATH-500\ngeneral math", "gpqa": "GPQA-d\ngeneral science"},
}

# deploy/DEPLOY_LOG.md「加演A:并发压测(2026-08-19)」实测值
LOADTEST = {
    "concurrency": [24, 48, 96],
    "cluster_rps": [5.10, 8.70, 12.66],
    "cluster_p50": [4.1, 4.9, 6.7],
    "cluster_p95": [5.6, 6.9, 11.0],
    "single_rps_at96": 12.79,
    "single_p50_at96": 6.3,
    "single_p95_at96": 10.1,
}


# ---------------------------------------------------------------- 工具

def smooth(ys: list[float], frac: float = 0.035) -> tuple[list[float], int]:
    """居中滑动平均。返回 (平滑序列, 窗口宽度)。

    比 EMA 好在没有相位滞后 —— EMA 会把 loss 陡降段画到原始曲线上方。
    边缘按可用点收缩窗口,不做 padding(不凭空造数)。
    """
    n = len(ys)
    w = max(3, int(n * frac) | 1)          # 取奇数
    half = w // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out, w


def curve(group: str) -> dict:
    """从 artifacts/figure_inputs.json 取一组曲线。

    原始 logging.jsonl 不随仓库发布(见 DATA.md);打包时已把绘图用到的
    序列抽到 artifacts/ 下,字段出处见该文件的 _provenance。
    """
    return _artifacts()[group]


def win_mean(vals: list[float], n: int, tail: bool = False) -> float:
    """首 n 个(或末 n 个)取值的均值。

    用于对比 GRPO reward 的起止水平:窗口越宽越能滤掉批次难度噪声。
    首/末 20 步会显出下降,放宽到 100 步则完全持平——后者才是策略是否移动的证据。
    """
    if not vals:
        return float("nan")
    w = vals[-n:] if tail else vals[:n]
    return sum(w) / len(w)


def series(group: dict, key: str) -> tuple[list[int], list[float]]:
    """(steps, values);缺列时返回空列表,由调用方决定是否降级。"""
    ys = group.get(key)
    return (list(group["steps"]), list(ys)) if ys else ([], [])


def load_evals() -> dict[tuple[str, str], dict]:
    out = {}
    for p in sorted((ROOT / "eval").glob("eval_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[(d["benchmark"], d["tag"])] = d
    return out


def _artifacts() -> dict:
    """artifacts/figure_inputs.json:从原始日志抽出的汇总事实集。

    原始 stdout 日志不随仓库发布(含机器路径/主机名),故绘图所需的少量
    汇总数字在打包时抽取到此文件。字段出处见其中的 _provenance。
    """
    return json.loads((ROOT / "artifacts" / "figure_inputs.json")
                      .read_text(encoding="utf-8"))


def load_hardcase() -> dict:
    """难例筛选分布(原 logs/hardcase.log 尾部汇总 JSON)。"""
    d = _artifacts()["hardcase_distribution"]
    parsed = {"_total": int(d["total"])}
    for k, v in d["buckets"].items():
        parsed[k] = (int(v["n"]), float(v["pct"]))
    return parsed


def load_retention() -> dict[int, float]:
    """{max_length: 保留率%}(原 logs/prefilter*.log)。"""
    return {int(k): float(v)
            for k, v in _artifacts()["length_retention_pct"].items()}


def load_liger_runs() -> list[dict]:
    """各 max_length 档的峰值显存与 OOM 判定(原 output/liger_test*/ + logs/)。"""
    a = _artifacts()
    oom = a["liger_oom"]
    return [{**r, "oom": bool(oom.get(str(r["max_length"]), False))}
            for r in a["liger_runs"]]


def annotate(ax, text, xy=(0.98, 0.05), ha="right", va="bottom", fontsize=9):
    ax.annotate(text, xy=xy, xycoords="axes fraction", ha=ha, va=va,
                fontsize=fontsize, color=C_TEXT,
                bbox=dict(boxstyle="round,pad=0.45", fc="#f7fafc",
                          ec="#cbd5e0", lw=0.8))


def header(ax, title, sub, legend_ncols=0):
    """统一的「标题 / 副标题 / 图例」三层排版,避免互相压字。"""
    pad = 52 if legend_ncols else 30
    ax.set_title(title, pad=pad, loc="center")
    ax.text(0.0, 1.105 if legend_ncols else 1.02, sub, transform=ax.transAxes,
            fontsize=9, color=C_MUTED, va="bottom")
    if legend_ncols:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005),
                  ncols=legend_ncols)


# ---------------------------------------------------------------- 各图

def fig1_benchmarks(t, lang, outdir):
    ev = load_evals()
    benches = ["cflue", "finqa", "math500", "gpqa"]
    tags = [("base", C_BASE, "base"), ("sft", C_SFT, "SFT"), ("grpo", C_GRPO, "SFT+GRPO")]

    vals = {tag: [ev[(b, tag)]["accuracy"] * 100 for b in benches] for tag, _, _ in tags}
    for tag, _, _ in tags:
        vals[tag].append(sum(vals[tag]) / len(benches))     # 追加均值组
    labels = [BENCH_LABEL[lang][b] for b in benches] + [t["fig1_mean"]]

    fig, ax = plt.subplots(figsize=(10, 5.4))
    w, n = 0.26, len(labels)
    xs = list(range(n))
    for i, (tag, color, disp) in enumerate(tags):
        pos = [x + (i - 1) * w for x in xs]
        bars = ax.bar(pos, vals[tag], w, label=disp, color=color,
                      edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, vals[tag]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color=C_TEXT)

    d_sft = vals["sft"][-1] - vals["base"][-1]
    d_grpo = vals["grpo"][-1] - vals["sft"][-1]
    ax.annotate(f"SFT {d_sft:+.2f}pp", xy=(n - 1, vals["sft"][-1] + 4.2),
                ha="center", fontsize=10, color=C_SFT)
    ax.annotate(f"GRPO {d_grpo:+.2f}pp", xy=(n - 1 + w, vals["grpo"][-1] + 8.0),
                ha="center", fontsize=10, color=C_GRPO)

    ax.axvline(n - 1.5, color="#a0aec0", lw=0.9, ls=":")
    ax.set_xticks(xs, labels)
    ax.set_ylabel(t["acc"])
    ax.set_ylim(0, 70)
    ax.grid(axis="x", visible=False)
    header(ax, t["fig1_title"], t["fig1_sub"], legend_ncols=3)

    ns = " · ".join(f"{b} n={ev[(b, 'base')]['n']}" for b in benches)
    ax.text(1.0, -0.17, ns, transform=ax.transAxes, ha="right",
            fontsize=8.5, color="#a0aec0")
    save(fig, outdir / "fig1_benchmark_comparison.png")


def fig2_sft_loss(t, lang, outdir):
    g = curve("sft_loss")
    xs, ys = list(g["steps"]), list(g["loss"])
    sm, w = smooth(ys, 0.05)
    max_steps = g["max_steps"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(xs, ys, color=C_RAW, lw=1.0, label=t["raw"])
    ax.plot(xs, sm, color=C_SFT, lw=2.2, label=t["smooth"].format(w=w))
    for e in (1, 2):
        ax.axvline(max_steps * e / 3, color="#a0aec0", lw=0.9, ls="--",
                   label=t["fig2_ep"] if e == 1 else None)
    ax.set_xlabel(t["step"])
    ax.set_ylabel("loss")
    ax.set_title(t["fig2_title"], pad=14)
    ax.legend(loc="upper right")
    annotate(ax, f"{ys[0]:.3f} → {ys[-1]:.3f}", xy=(0.98, 0.55))
    save(fig, outdir / "fig2_sft_loss.png")


def fig3_grpo_reward(t, lang, outdir):
    g = curve("grpo")
    xs, rw = series(g, "reward")
    _, acc = series(g, "acc")
    _, zero = series(g, "frac_reward_zero_std")

    sm_rw, w = smooth(rw)
    sm_acc, _ = smooth(acc)
    sm_zero, _ = smooth(zero)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 6.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.13))

    ax.plot(xs, rw, color=C_RAW, lw=0.5, alpha=0.7)
    ax.plot(xs, sm_rw, color=C_GRPO, lw=2.2, label=t["fig3_reward"])
    ax.plot(xs, sm_acc, color=C_SFT, lw=1.9, ls="--", label=t["fig3_acc"])
    ax.axhline(win_mean(rw, 100), color="#a0aec0", lw=0.9, ls=":")
    ax.set_ylabel("reward")
    ax.set_ylim(0, 1.28)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=2)
    ax.set_title(t["fig3_title"], pad=34, loc="center")
    annotate(ax, t["fig3_note"].format(
        a100=win_mean(rw, 100), b100=win_mean(rw, 100, tail=True),
        a20=win_mean(rw, 20), b20=win_mean(rw, 20, tail=True)))

    ax2.plot(xs, zero, color=C_RAW, lw=0.4, alpha=0.35)
    ax2.plot(xs, sm_zero, color=C_BAD, lw=2.0, label=t["fig3_zero"])
    zbar = sum(zero) / len(zero)
    ax2.axhline(zbar, color=C_BAD, lw=0.9, ls=":")
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax2.set_ylabel(t["fig3_ylab_zero"])
    ax2.set_xlabel(t["step"])
    ax2.legend(loc="upper left", fontsize=9)
    annotate(ax2, t["fig3_note2"].format(z=zbar), xy=(0.98, 0.08), fontsize=8.5)
    save(fig, outdir / "fig3_grpo_reward.png")


def fig4_grpo_kl(t, lang, outdir):
    g = curve("grpo")
    xs, kl = series(g, "kl")
    sm, w = smooth(kl)
    med = sorted(kl)[len(kl) // 2]

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(xs, kl, color=C_RAW, lw=0.8, label=t["raw"])
    ax.plot(xs, sm, color=C_BAD, lw=2.2, label=t["smooth"].format(w=w))
    ax.axhline(med, color="#a0aec0", lw=0.9, ls=":")
    ax.set_xlabel(t["step"])
    ax.set_ylabel(t["fig4_ylab"])
    ax.set_title(t["fig4_title"], pad=32, loc="center")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=2)
    annotate(ax, t["fig4_note"].format(med=med, mx=max(kl)), xy=(0.98, 0.5))
    save(fig, outdir / "fig4_grpo_kl.png")


def fig5_hardcase(t, lang, outdir):
    hc = load_hardcase()
    order = [
        ("c=k(全对,无梯度)", t["fig5_allright"], C_BASE),
        ("0<c<k(混合·GRPO有梯度)", t["fig5_mixed"], C_OK),
        ("c=0(全错,无正样本)", t["fig5_allwrong"], C_BAD),
    ]

    fig, ax = plt.subplots(figsize=(10, 3.0))
    left = 0.0
    for key, label, color in order:
        cnt, pct = hc[key]
        ax.barh(0, pct, left=left, height=0.5, color=color, label=label,
                edgecolor="white", linewidth=1.4)
        txt = f"{pct:.1f}%\nn={cnt}"
        if pct >= 18:                      # 段够宽,文字放段内
            ax.text(left + pct / 2, 0, txt, ha="center", va="center",
                    color="white", fontsize=10)
        else:                              # 窄段(11%)放段外,免得压边或截断
            ax.text(left + pct / 2, 0.33, txt, ha="center", va="bottom",
                    color=color, fontsize=9.5)
        left += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 0.82)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.grid(visible=False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(t["fig5_title"], pad=28, loc="center")
    ax.text(0.5, 1.06, t["fig5_sub"], transform=ax.transAxes, ha="center",
            fontsize=8.5, color=C_MUTED, va="bottom")
    # 分类说明交给图例:英文标签比中文长得多,内联放不下
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=3,
              fontsize=9, handlelength=1.2, columnspacing=2.4)
    ax.text(0.5, -0.44, t["fig5_note"], transform=ax.transAxes, ha="center",
            va="top", fontsize=9.5, color=C_BAD)
    save(fig, outdir / "fig5_hardcase_distribution.png")


def fig6_maxlen(t, lang, outdir):
    runs = load_liger_runs()
    keep = load_retention()

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.6, 5.0), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.24))

    # ---- panel A:峰值显存 ----
    xs = list(range(len(runs)))
    chosen_len = max((r["max_length"] for r in runs if not r["oom"]), default=None)
    for i, r in enumerate(runs):
        is_chosen = (r["max_length"] == chosen_len)
        color = C_BAD if r["oom"] else (C_SFT if is_chosen else C_OK)
        axA.bar(i, r["peak_mem"], 0.52, color=color,
                hatch="//" if r["oom"] else None,
                edgecolor="white", linewidth=1.0)
        axA.text(i, r["peak_mem"] + 0.3, f"{r['peak_mem']:.2f}",
                 ha="center", va="bottom", fontsize=10, color=C_TEXT)
        tag = t["fig6_oom"].format(m=r["peak_mem"]) if r["oom"] else t["fig6_pass"]
        axA.text(i, 1.0, tag, ha="center", va="bottom", fontsize=9, color="white")
        if is_chosen:
            axA.text(i, r["peak_mem"] / 2, t["fig6_final"], ha="center",
                     va="center", fontsize=11, color="white")
    axA.axhline(24, color=C_BAD, lw=1.4, ls="--")
    axA.text(len(runs) - 0.45, 24.3, t["fig6_limit"], ha="right", va="bottom",
             fontsize=9, color=C_BAD)
    axA.set_xticks(xs, [str(r["max_length"]) for r in runs])
    axA.set_xlabel("max_length")
    axA.set_ylabel(t["fig6_mem"])
    axA.set_ylim(0, 27)
    axA.set_title(t["fig6_panelA"], fontsize=11, pad=10)
    axA.grid(axis="x", visible=False)
    oomed = [r for r in runs if r["oom"]]
    if oomed:
        axA.text(0.0, -0.185, t["fig6_foot"].format(m=oomed[0]["peak_mem"]),
                 transform=axA.transAxes, fontsize=8.5, color=C_MUTED)

    # ---- panel B:数据保留率 ----
    lens = sorted(keep)
    chosen = max((r["max_length"] for r in runs if not r["oom"]), default=None)
    for i, L in enumerate(lens):
        is_chosen = (L == chosen)
        axB.bar(i, keep[L], 0.52, color=C_SFT if is_chosen else C_BASE,
                edgecolor="white", linewidth=1.0)
        axB.text(i, keep[L] + 0.5, f"{keep[L]:.1f}%", ha="center", va="bottom",
                 fontsize=10, color=C_TEXT)
        if is_chosen:
            axB.text(i, keep[L] / 2, t["fig6_final"], ha="center", va="center",
                     fontsize=11, color="white")
    axB.set_xticks(range(len(lens)), [str(L) for L in lens])
    axB.set_xlabel("max_length")
    axB.set_ylabel(t["fig6_keep"])
    axB.set_ylim(0, 108)
    axB.yaxis.set_major_formatter(PercentFormatter())
    axB.set_title(t["fig6_panelB"], fontsize=11, pad=10)
    axB.grid(axis="x", visible=False)

    fig.text(0.5, 1.10, t["fig6_title"], ha="center", va="bottom", fontsize=13)
    fig.text(0.5, 1.035, t["fig6_sub"], ha="center", va="bottom",
             fontsize=9, color=C_MUTED)
    save(fig, outdir / "fig6_maxlen_ablation.png")


def fig7_loadtest(t, lang, outdir):
    d = LOADTEST
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(9.2, 6.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1.15, 1.0], hspace=0.14))

    axA.plot(d["concurrency"], d["cluster_rps"], "o-", color=C_SFT, lw=2.2,
             ms=6, label=t["fig7_cluster"])
    for x, y in zip(d["concurrency"], d["cluster_rps"]):
        axA.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9, color=C_SFT)
    axA.scatter([96], [d["single_rps_at96"]], marker="D", s=75, color=C_GRPO,
                zorder=5, label=t["fig7_single"])
    axA.annotate(f"{d['single_rps_at96']:.2f}", (96, d["single_rps_at96"]),
                 textcoords="offset points", xytext=(-4, -17), ha="right",
                 fontsize=9, color=C_GRPO)
    axA.set_ylabel(t["fig7_tp"])
    axA.set_ylim(0, 16)
    axA.legend(loc="upper left", ncols=2)
    axA.set_title(t["fig7_title"], pad=16, loc="center")
    annotate(axA, t["fig7_note"])

    axB.plot(d["concurrency"], d["cluster_p50"], "s-", color=C_MUTED, lw=1.9,
             ms=5.5, label=t["fig7_p50"])
    axB.plot(d["concurrency"], d["cluster_p95"], "^-", color=C_BAD, lw=1.9,
             ms=5.5, label=t["fig7_p95"])
    axB.scatter([96, 96], [d["single_p50_at96"], d["single_p95_at96"]],
                marker="D", s=55, color=C_GRPO, zorder=5,
                label=t["fig7_single"])
    for x, y in zip(d["concurrency"], d["cluster_p50"]):
        axB.annotate(f"{y:.1f}s", (x, y), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=8.5, color=C_MUTED)
    for x, y in zip(d["concurrency"], d["cluster_p95"]):
        axB.annotate(f"{y:.1f}s", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8.5, color=C_BAD)
    axB.set_ylabel(t["fig7_lat"])
    axB.set_xlabel(t["fig7_conc"])
    axB.set_xticks(d["concurrency"])
    axB.set_ylim(0, 14)
    axB.legend(loc="upper left", ncols=3)
    save(fig, outdir / "fig7_deploy_loadtest.png")


def fig8_completion_len(t, lang, outdir):
    g = curve("grpo")
    xs, mean_len = series(g, "completions_mean_length")
    _, max_len = series(g, "completions_max_length")
    _, clipped = series(g, "completions_clipped_ratio")

    sm_mean, w = smooth(mean_len)
    sm_max, _ = smooth(max_len)

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.plot(xs, mean_len, color=C_RAW, lw=0.5, alpha=0.7)
    ax.plot(xs, sm_mean, color=C_GRPO, lw=2.2, label=t["fig8_mean"])
    ax.plot(xs, sm_max, color=C_SFT, lw=1.8, ls="--", label=t["fig8_max"])
    ax.axhline(1536, color=C_BAD, lw=1.3, ls="--")
    ax.text(xs[-1], 1556, t["fig8_cap"], ha="right", va="bottom",
            fontsize=9, color=C_BAD)
    ax.set_xlabel(t["step"])
    ax.set_ylabel(t["fig8_tok"])
    ax.set_ylim(0, 1780)
    ax.set_title(t["fig8_title"], pad=32, loc="center")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=2)
    annotate(ax, t["fig8_note"].format(
        mean=sum(mean_len) / len(mean_len),
        n=sum(1 for c in clipped if c > 0),
        r=sum(clipped) / len(clipped)), xy=(0.98, 0.52))
    save(fig, outdir / "fig8_completion_length.png")


# ---------------------------------------------------------------- 主流程

FIGS = {
    1: fig1_benchmarks,
    2: fig2_sft_loss,
    3: fig3_grpo_reward,
    4: fig4_grpo_kl,
    5: fig5_hardcase,
    6: fig6_maxlen,
    7: fig7_loadtest,
    8: fig8_completion_len,
}


def save(fig, path: Path):
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")


def main():
    ap = argparse.ArgumentParser(description="重绘本项目的全部图表")
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--only", nargs="*", type=int, choices=sorted(FIGS),
                    help="只画指定编号,默认全画")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    lang = setup_style(args.lang)
    outdir = args.outdir or (ROOT / "figures" / ("" if lang == "zh" else "en"))
    outdir.mkdir(parents=True, exist_ok=True)

    todo = args.only or sorted(FIGS)
    print(f"[make_figures] lang={lang}  outdir={outdir.relative_to(ROOT)}")
    for i in todo:
        FIGS[i](T[lang], lang, outdir)
    print(f"[make_figures] 完成 {len(todo)} 张")


if __name__ == "__main__":
    main()
