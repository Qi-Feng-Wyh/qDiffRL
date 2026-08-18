#!/bin/bash
# ======================================================================
# pg:CartpoleSwingup 训练脚本（只训练，评测用 run_pg_cartpole_eval.sh）
#
#   算法 : ppo / rpo / gippo / ivwh
#   种子 : 1, 2, 3
#   预算 : 每个 run 200 万步，num_envs=64，horizon=32
#          （梯度算法带 --clip-grad-a 1.0；ppo 带 --ent-coef 0.01 防后期塌缩）
#
# 用法：nohup bash run_pg_cartpole_train.sh > train_cartpole.log 2>&1 &
# 说明：该配置是实验验证的甜区（RPO 1M 步即达 ~800 分），更长预算/更大
#       窗口实测会过训（后期振荡回落）。
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh       # 共享 GPU 独立 MPS 通道
export XLA_PYTHON_CLIENT_PREALLOCATE=false        # 关闭 JAX 显存预分配
export PYTHONUNBUFFERED=1

ENVS=(pg:CartpoleSwingup)
ALGOS=(ppo rpo gippo ivwh)
SEEDS=(1 2 3)
NUM_ENVS=64
TOTAL_STEPS=2000000
HORIZON=32

mkdir -p logs runs

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            tag="${algo}_${env//:/}_seed${seed}"
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
echo "  bash run_pg_cartpole_eval.sh"
echo "================================================================"
