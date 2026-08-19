#!/bin/bash
# ======================================================================
# 消融实验：RPO 复用轮数 M ∈ {1, 2, 10}（固定种子 1，M=5 为主实验基线）
# 任务：pg:CartpoleSwingup，5M 步 / 64 envs / horizon 128
# 产物命名为 rpo_pgCartpoleSwingupM{1,2,10}_seed1.*，不覆盖主实验；
# 主实验 seed1 文件会先备份到 backup/，结束后恢复。
# 用法：nohup bash ablations/run_rpo_M_ablation.sh > abl_rpo_M.log 2>&1 &
# ======================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

cp -a runs/rpo_pgCartpoleSwingup_seed1.json runs/rpo_pgCartpoleSwingup_seed1.pkl backup/

for M in 1 2 10; do
    echo "=== M=$M 开始 $(date '+%T') ==="
    ${PY} train.py --algo rpo --env pg:CartpoleSwingup --seed 1 \
        --total-steps 5000000 --num-envs 64 --horizon 128 \
        --rpo-epochs $M --clip-grad-a 1.0 --lambda-ent 0.1 \
        --log-interval 20 > logs/abl_rpo_M${M}_s1.log 2>&1
    mv runs/rpo_pgCartpoleSwingup_seed1.json runs/rpo_pgCartpoleSwingupM${M}_seed1.json
    mv runs/rpo_pgCartpoleSwingup_seed1.pkl runs/rpo_pgCartpoleSwingupM${M}_seed1.pkl
    echo "=== M=$M 完成 $(date '+%T') ==="
done

cp -a backup/rpo_pgCartpoleSwingup_seed1.json backup/rpo_pgCartpoleSwingup_seed1.pkl runs/

${PY} evaluate.py --ckpt runs/rpo_pgCartpoleSwingup_seed1.pkl \
    "runs/rpo_pgCartpoleSwingupM*_seed1.pkl" \
    --episodes 128 --video mp4 --video-steps 1000
${PY} plot.py --runs runs/rpo_pgCartpoleSwingup_seed1.json \
    "runs/rpo_pgCartpoleSwingupM*_seed1.json"
