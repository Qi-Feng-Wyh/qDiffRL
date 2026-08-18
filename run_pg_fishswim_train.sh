#!/bin/bash
# ======================================================================
# pg:FishSwim 训练脚本（只训练；评测渲染用 run_pg_cartpole_eval.sh
# 改 ENVS 或直接命令行 evaluate.py）
#
#   任务 : pg:FishSwim（无接触，流体推进）
#   算法 : ppo / rpo / gippo / ivwh
#   种子 : 1, 2, 3
#   预算 : 每个 run 500 万步，num_envs=64，horizon=32（RPO 默认参数 5M
#          即达 ~745 分并进入平台期，加长只有过训风险）
#          （rpo/gippo/ivwh 带 --clip-grad-a 1.0）
#
# 用法：nohup bash run_pg_fishswim_train.sh > train_fishswim.log 2>&1 &
# 已有 checkpoint 的 run 会自动跳过（断点续跑）。
# 注意：PPO 在该任务上更新过猛会震荡（clip_frac 长期 >0.8），
#       如需改善可给 ppo 单独加 --ppo-epochs 5 --actor-lr 1e-4。
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

ENVS=(pg:FishSwim)
ALGOS=(ppo rpo gippo ivwh)
SEEDS=(1 2 3)
NUM_ENVS=64
HORIZON=32
TOTAL_STEPS=5000000

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
            if [[ "${algo}" != "ppo" ]]; then
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
