#!/bin/bash
# ======================================================================
# pg:CartpoleSwingup 评测脚本（只评测/渲染/画图，训练用
# run_pg_cartpole_train.sh；可重复运行，与训练互不依赖）
#
# 用法：bash run_pg_cartpole_eval.sh
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl        # 无显示器渲染 mp4

ENVS=(pg:CartpoleSwingup)
ALGOS=(ppo rpo gippo ivwh)
SEEDS=(1 2 3)
EVAL_EPISODES=1000
VIDEO_STEPS=1000

mkdir -p eval figures

# ====== 1. 批量数值评测 ======
for env in "${ENVS[@]}"; do
    echo "================================================================"
    echo "[eval] ${env} 全部 checkpoint，各 ${EVAL_EPISODES} episodes"
    echo "================================================================"
    ${PY} evaluate.py --ckpt "runs/*_${env//:/}_*.pkl" \
        --episodes "${EVAL_EPISODES}" --no-video
done

# ====== 2. 每个 任务×算法×种子 渲染一条轨迹视频 ======
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

# ====== 3. 画图（学习曲线 + 诊断图，多种子自动聚合）======
for env in "${ENVS[@]}"; do
    echo "================================================================"
    echo "[plot] ${env}"
    echo "================================================================"
    ${PY} plot.py --runs "runs/*_${env//:/}_*.json" --diagnostics
done

echo "================================================================"
echo "全部完成。产物位置："
echo "  评测图表   eval/eval_*.png  /  轨迹视频 eval/rollout_*.mp4"
echo "  对比图     figures/learning_curves_*.png, figures/diagnostics_*.png"
echo "================================================================"
