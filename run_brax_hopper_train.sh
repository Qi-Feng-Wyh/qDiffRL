#!/bin/bash
# ======================================================================
# brax hopper 训练脚本（只训练；评测渲染用 run_pg_cartpole_eval.sh
# 改 ENVS 或直接命令行 evaluate.py）
#
#   任务 : hopper（brax 版，有接触）
#   算法 : ppo / rpo / gippo / ivwh
#   种子 : 1, 2, 3
#   预算 : 每个 run 1000 万步，num_envs=256，horizon=32（10M 是实测甜区）
#          （梯度算法带 --clip-grad-a 1.0；rpo 带 --lambda-ent 0.05 防冻结）
#   注意：RPO 在 hopper 上有已知的冻结风险（默认 lambda_ent=0.25 会导致
#         熵膨胀+梯度截断正反馈），脚本已内置 0.05 的修正值。
#
# 用法：nohup bash run_brax_hopper_train.sh > train_hopper.log 2>&1 &
# 已有 checkpoint 的 run 会自动跳过（断点续跑）。
# 注意：RPO 在 hopper 上有已知的冻结风险（lambda_ent 膨胀 + 梯度截断），
#       若日志里 grad_a_norm 崩到 ~0.01 且 kl≈0，考虑给 RPO 单独加
#       --lambda-ent 0.05 再试。
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

ENVS=(hopper)
ALGOS=(ppo rpo gippo ivwh)
SEEDS=(1 2 3)
NUM_ENVS=256
HORIZON=32
TOTAL_STEPS=10000000

mkdir -p logs runs

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            tag="${algo}_${env}_seed${seed}"
            if [[ -f "runs/${tag}.pkl" ]]; then
                echo "[skip] ${tag} 已有 checkpoint，跳过"
                continue
            fi
            echo "================================================================"
            echo "[train] ${tag}  (开始于 $(date '+%F %T'))"
            echo "================================================================"
            extra=()
            if [[ "${algo}" == "rpo" ]]; then
                extra+=(--clip-grad-a 1.0 --lambda-ent 0.05)   # 防冻结
            elif [[ "${algo}" != "ppo" ]]; then
                extra+=(--clip-grad-a 1.0)
            fi
            ${PY} train.py --algo "${algo}" --env "${env}" --seed "${seed}" \
                --total-steps "${TOTAL_STEPS}" --num-envs "${NUM_ENVS}" \
                --horizon "${HORIZON}" --log-interval 10 "${extra[@]}" \
                2>&1 | tee "logs/train_${tag}.log"
            echo "[train] ${tag} 完成 ( $(date '+%F %T') )"
        done
    done
done

echo "================================================================"
echo "训练全部完成 ( $(date '+%F %T') )"
echo "================================================================"
