#!/bin/bash
# 企业上线质量门禁:合并后全量模型必须复现 GRPO checkpoint 分数(容忍 bf16+采样噪声)。
# 不达标 exit 1 拦截上线。artifact:eval/eval_{cflue,finqa}_gate.json
set -u
# 按需激活你的环境,例如: source ~/miniconda3/etc/profile.d/conda.sh; conda activate <env>
[ -n "${CONDA_INIT:-}" ] && source "$CONDA_INIT" && conda activate "${CONDA_ENV:-base}"
ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MODEL="${MERGED_MODEL:?请设置 MERGED_MODEL 指向合并后的全量模型目录}"
GATELOG="${RUN_DIR:-$ROOT/runs}/gate.log"; mkdir -p "$(dirname "$GATELOG")"
# 参照(GRPO checkpoint-1210 已核实分数)与阈值(不得低于参照 3pp)
declare -A REF=( ["cflue"]=54.75 ["finqa"]=61.76 )
declare -A N=( ["cflue"]=800 ["finqa"]=0 )
TOL=3.0
export MKL_THREADING_LAYER=GNU NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

echo "[gate] 评测合并模型:$MODEL" | tee "$GATELOG"
for bm in cflue finqa; do
  echo "[gate] --- $bm (参照 ${REF[$bm]}) ---" | tee -a "$GATELOG"
  CUDA_VISIBLE_DEVICES=2,3,4,5 python "$ROOT/scripts/eval_fin.py" \
    --benchmark "$bm" --sample "${N[$bm]}" --tp 4 --tag gate \
    --model "$MODEL" 2>>"$GATELOG" | tee -a "$GATELOG"
done

echo "[gate] ===== 门禁判定 =====" | tee -a "$GATELOG"
python - "$ROOT" "$TOL" "${REF[cflue]}" "${REF[finqa]}" <<'PY' | tee -a "$GATELOG"
import json, sys
ROOT, tol, ref_c, ref_f = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
ref = {"cflue": ref_c, "finqa": ref_f}
ok = True
for bm in ["cflue", "finqa"]:
    d = json.load(open(f"{ROOT}/eval/eval_{bm}_gate.json"))
    got = d["accuracy"] * 100
    lo = ref[bm] - tol
    passed = got >= lo
    ok = ok and passed
    print(f"  {bm:6s}: 合并={got:.2f}  参照={ref[bm]:.2f}  下限={lo:.2f}  {'✅PASS' if passed else '❌FAIL'} (n={d['n']})")
print("门禁结论:", "✅ 通过,允许上线" if ok else "❌ 未通过,拦截上线")
sys.exit(0 if ok else 1)
PY
GATE_RC=${PIPESTATUS[0]}
echo "[gate] 退出码=$GATE_RC"
exit $GATE_RC
