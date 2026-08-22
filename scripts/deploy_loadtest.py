# -*- coding: utf-8 -*-
"""并发压测:测网关(或单副本)的吞吐与延迟。
用法:python deploy_loadtest.py --url http://127.0.0.1:4000/v1 --concurrency 24 --total 96 --max-tokens 256
指标:吞吐 req/s、输出 tokens/s、端到端延迟 P50/P95/P99、成功率。temperature=0 定长输出便于比较。
"""
import argparse, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("分析:某公司营业收入8000万元,营业成本5200万元,期间费用1200万元,所得税率25%,"
          "请一步步计算净利润和净利率,并把净利率放进 \\boxed{}")


def one(url, max_tokens):
    body = json.dumps({"model": "qwen3-fin", "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        dt = time.time() - t0
        out_tok = d.get("usage", {}).get("completion_tokens", 0)
        return (True, dt, out_tok)
    except Exception as e:
        return (False, time.time() - t0, 0)


def pct(xs, p):
    if not xs: return 0.0
    xs = sorted(xs); k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:4000/v1")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--total", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(one, a.url, a.max_tokens) for _ in range(a.total)]
        for f in futs:
            results.append(f.result())
    wall = time.time() - t0

    ok = [r for r in results if r[0]]
    lat = [r[1] for r in ok]
    tot_out = sum(r[2] for r in ok)
    print(f"  并发={a.concurrency:3d} 总请求={a.total:4d} 成功={len(ok):4d}/{a.total} "
          f"| 墙钟={wall:6.1f}s 吞吐={len(ok)/wall:6.2f}req/s 输出={tot_out/wall:7.1f}tok/s "
          f"| 延迟 P50={pct(lat,50):5.2f}s P95={pct(lat,95):5.2f}s P99={pct(lat,99):5.2f}s 均值={sum(lat)/len(lat):5.2f}s")


if __name__ == "__main__":
    main()
