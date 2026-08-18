#!/bin/bash
# ======================================================================
# 手写版 cartpole（MJXCartpole，swing-up）训练脚本
# （评测用 run_cartpole_eval.sh）
#
#   任务 : cartpole（手写版摆起：θ=π 下垂出发、gear=10、50Hz、500 步、
#          滑轨 ±20m 实质无墙、奖励 cosθ − 0.1x² − 0.001a²）
#   算法 : ppo / rpo / gippo / ivwh
#   种子 : 1, 2, 3
#   预算 : 每个 run 200 万步，num_envs=64，horizon=32
#          （梯度算法带 --clip-grad-a 1.0；ppo 带 --ent-coef 0.01）
#
# 用法：nohup bash run_cartpole_train.sh > train_cartpole.log 2>&1 &
# 说明：RPO/PPO 在 1M 步即达 ~+190（满分 ~+200），2M 留有富余；
#       更长预算有过训回落风险。已有 checkpoint 自动跳过。
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh       # 共享 GPU 独立 MPS 通道
export XLA_PYTHON_CLIENT_PREALLOCATE=false        # 关闭 JAX 显存预分配
export PYTHONUNBUFFERED=1

ENVS=(cartpole)
ALGOS=(ppo rpo gippo ivwh)
SEEDS=(1 2 3)
NUM_ENVS=64
TOTAL_STEPS=2000000
HORIZON=32

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
            if [[ "${algo}" == "ppo" ]]; then
                extra+=(--ent-coef 0.01)      # 防后期熵塌缩振荡
            else
                extra+=(--clip-grad-a 1.0)    # 解析梯度算法：逐样本梯度裁剪
            fi
            ${PY} train.py --algo "${algo}" --env "${env}" --seed "${seed}" \
                --total-steps "${TOTAL_STEPS}" --num-envs "${NUM_ENVS}" \
                --horizon "${HORIZON}" \
                --log-interval 5 "${extra[@]}" \
                2>&1 | tee "logs/train_${tag}.log"
            echo "[train] ${tag} 完成 ( $(date '+%F %T') )"
        done
    done
done

echo "================================================================"
echo "训练全部完成 ( $(date '+%F %T') )，接下来运行评测："
echo "  bash run_cartpole_eval.sh"
echo "================================================================"
