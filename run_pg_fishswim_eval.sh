#!/bin/bash
# ======================================================================
# pg:FishSwim 评测脚本（评测/渲染/画图；与 run_pg_fishswim_train.sh 配套）
#
# 用法：bash run_pg_fishswim_eval.sh
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

ENVS=(pg:FishSwim)
ALGOS=(ppo rpo gippo ivwh)
SEEDS=(1 2 3)
EVAL_EPISODES=128
VIDEO_STEPS=1000         # fish 游速慢，给满一个 episode

mkdir -p eval figures

# ====== 1. 批量数值评测 ======
for env in "${ENVS[@]}"; do
    echo "================================================================"
    echo "[eval] ${env} 全部 checkpoint，各 ${EVAL_EPISODES} episodes"
    echo "================================================================"
    ${PY} evaluate.py --ckpt "runs/*_${env//:/}_*.pkl" \
        --episodes "${EVAL_EPISODES}" --no-video
done

# ====== 2. 每个 算法×种子 渲染一条轨迹视频 ======
for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            ckpt="runs/${algo}_${env//:/}_seed${seed}.pkl"
            [[ -f "${ckpt}" ]] || continue
            ${PY} evaluate.py --ckpt "${ckpt}" --episodes 8 \
                --video mp4 --video-steps "${VIDEO_STEPS}" || true
        done
    done
done

# ====== 3. 画图 ======
for env in "${ENVS[@]}"; do
    echo "================================================================"
    echo "[plot] ${env}"
    echo "================================================================"
    ${PY} plot.py --runs "runs/*_${env//:/}_*.json" --diagnostics
done

echo "================================================================"
echo "全部完成。产物：eval/（图+视频）、figures/（曲线+诊断图）"
echo "================================================================"
