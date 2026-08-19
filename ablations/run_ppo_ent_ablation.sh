#!/bin/bash
# ======================================================================
# 消融实验：PPO 熵系数 ent_coef ∈ {0, 0.01, 0.05}（固定种子 1）
# 任务：pg:CartpoleSwingup，5M 步 / 64 envs / horizon 128
# 产物命名为 ppo_pgCartpoleSwingupEnt{0,001,005}_seed1.*，不覆盖主实验；
# 主实验 seed1 文件会先备份到 backup/，结束后恢复。
# 用法：nohup bash ablations/run_ppo_ent_ablation.sh > abl_ppo_ent.log 2>&1 &
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

cp -a runs/ppo_pgCartpoleSwingup_seed1.json runs/ppo_pgCartpoleSwingup_seed1.pkl backup/

for cfg in "0 Ent0" "0.01 Ent001" "0.05 Ent005"; do
    set -- $cfg
    echo "=== ent_coef=$1 开始 $(date '+%T') ==="
    ${PY} train.py --algo ppo --env pg:CartpoleSwingup --seed 1 \
        --total-steps 5000000 --num-envs 64 --horizon 128 \
        --ent-coef $1 --log-interval 20 > logs/abl_ppo_${2}_s1.log 2>&1
    mv runs/ppo_pgCartpoleSwingup_seed1.json runs/ppo_pgCartpoleSwingup${2}_seed1.json
    mv runs/ppo_pgCartpoleSwingup_seed1.pkl runs/ppo_pgCartpoleSwingup${2}_seed1.pkl
    echo "=== ent_coef=$1 完成 $(date '+%T') ==="
done

cp -a backup/ppo_pgCartpoleSwingup_seed1.json backup/ppo_pgCartpoleSwingup_seed1.pkl runs/

${PY} evaluate.py --ckpt runs/ppo_pgCartpoleSwingup_seed1.pkl \
    "runs/ppo_pgCartpoleSwingupEnt*_seed1.pkl" \
    --episodes 128 --video mp4 --video-steps 1000
${PY} plot.py --runs runs/ppo_pgCartpoleSwingup_seed1.json \
    "runs/ppo_pgCartpoleSwingupEnt*_seed1.json"
