# -*- coding: utf-8 -*-
"""部署端到端冒烟:打网关(OpenAI 兼容)→ 验证 Qwen3 thinking 输出可解析。
用法:python deploy_smoke_client.py --url http://127.0.0.1:4000/v1 --n 4
检查:①网关连通;②<think>/<answer>/\\boxed{} 结构完整;③金融题能出答案。
"""
import argparse, json, re, sys, urllib.request

TH = re.compile(r"<think>(.*?)</think>", re.S)
AN = re.compile(r"<answer>(.*?)</answer>", re.S)
BOX = re.compile(r"\\boxed\{([^{}]*)\}")

QS = [
    "某公司总资产100万元,总负债40万元,则资产负债率是多少?\nA. 20%\nB. 40%\nC. 60%\nD. 80%\n请一步步思考,然后把答案选项放到 \\boxed{} 中。",
    "一只股票年初价格50元,年末价格60元,期间分红2元,则该股票的持有期收益率约为?\nA. 20%\nB. 24%\nC. 16%\nD. 12%\n请一步步思考,然后把答案选项放到 \\boxed{} 中。",
    "银行一年期存款利率3%,通胀率5%,则实际利率约为?\nA. 8%\nB. 2%\nC. -2%\nD. 0%\n请一步步思考,然后把答案选项放到 \\boxed{} 中。",
    "某债券面值1000元,票面利率5%,期限1年,若市场利率上升到8%,债券价格大致?\nA. 上升\nB. 不变\nC. 下降\nD. 无法判断\n请一步步思考,然后把答案选项放到 \\boxed{} 中。",
]


def call(url, q, model="qwen3-fin"):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": q}],
        "temperature": 0.6, "max_tokens": 2048,
    }).encode()
    req = urllib.request.Request(url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:4000/v1")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    ok = 0
    for i, q in enumerate(QS[:a.n]):
        try:
            txt = call(a.url, q)
        except Exception as e:
            print(f"[{i+1}] ❌ 请求失败:{e}")
            continue
        nth, nan = len(TH.findall(txt)), len(AN.findall(txt))
        box = BOX.findall(txt)
        good = (nth == 1 and nan == 1 and len(box) >= 1)
        ok += good
        ans = box[-1].strip() if box else "?"
        print(f"[{i+1}] {'✅' if good else '⚠'} think={nth} answer={nan} boxed={ans!r} 长度={len(txt)}")
        if not good:
            print(f"      片段:{txt[:200]!r}")
    print(f"\n冒烟结果:{ok}/{a.n} 结构完整")
    sys.exit(0 if ok == a.n else 1)


if __name__ == "__main__":
    main()
