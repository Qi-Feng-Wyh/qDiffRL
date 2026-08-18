"""
统一训练入口：GI-PPO / RPO / PPO on MJX。

用法：
    python train.py --algo gippo --env cartpole
    python train.py --algo rpo   --env ant --seed 1
    python train.py --algo ppo   --env hopper

常用消融：
    python train.py --algo gippo --env ant --delta-oorr 1.0      # GI-PPO 附录 7.3.4
    python train.py --algo rpo   --env ant --rpo-epochs 1        # RPO 无样本复用
    python train.py --algo rpo   --env ant --lambda-kl 0.0       # RPO 无 KL 正则
    python train.py --algo rpo   --env ant --c-low 1e9 --c-high 1e9   # RPO 无裁剪
"""

import argparse
import json
import os
import time

from algos import get_config, make_agent
from algos.common import save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="gippo",
                   choices=["gippo", "rpo", "ppo", "ddcg", "aobg", "ivwh"])
    p.add_argument("--env", default="cartpole",
                   choices=["cartpole", "hopper", "ant", "ball_wall",
                            "pg:PendulumSwingup", "pg:AcrobotSwingup",
                            "pg:CartpoleSwingup", "pg:PointMass",
                            "pg:ReacherEasy", "pg:SwimmerSwimmer6",
                            "pg:FishSwim", "pg:HopperHop", "pg:HopperStand"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total-steps", dest="total_steps", type=int, default=None)
    p.add_argument("--num-envs", dest="num_envs", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--actor-lr", dest="actor_lr", type=float, default=None)
    p.add_argument("--critic-lr", dest="critic_lr", type=float, default=None)
    p.add_argument("--clip-grad-a", dest="clip_grad_a", type=float, default=None)
    p.add_argument("--log-interval", dest="log_interval", type=int, default=None)
    # GI-PPO
    p.add_argument("--alpha0", type=float, default=None)
    p.add_argument("--alpha-max", dest="alpha_max", type=float, default=None)
    p.add_argument("--alpha-lr", dest="alpha_lr", type=float, default=None)
    p.add_argument("--alpha-beta", dest="alpha_beta", type=float, default=None)
    p.add_argument("--delta-det", dest="delta_det", type=float, default=None)
    p.add_argument("--delta-oorr", dest="delta_oorr", type=float, default=None)
    # RPO
    p.add_argument("--rpo-epochs", dest="rpo_epochs", type=int, default=None)
    p.add_argument("--rpo-batch", dest="rpo_batch", type=int, default=None)
    p.add_argument("--c-low", dest="c_low", type=float, default=None)
    p.add_argument("--c-high", dest="c_high", type=float, default=None)
    p.add_argument("--lambda-kl", dest="lambda_kl", type=float, default=None)
    p.add_argument("--lambda-ent", dest="lambda_ent", type=float, default=None)
    # DDCG
    p.add_argument("--ddcg-samples", dest="ddcg_samples", type=int, default=None)
    p.add_argument("--ddcg-envs", dest="ddcg_envs", type=int, default=None)
    p.add_argument("--ddcg-sigma", dest="ddcg_sigma", type=float, default=None)
    p.add_argument("--ddcg-c", dest="ddcg_c", type=float, default=None)
    # AoBG
    p.add_argument("--aobg-gamma", dest="aobg_gamma", type=float, default=None)
    p.add_argument("--aobg-delta", dest="aobg_delta", type=float, default=None)
    # PPO
    p.add_argument("--ppo-epochs", dest="ppo_epochs", type=int, default=None)
    p.add_argument("--clip-eps", dest="clip_eps", type=float, default=None)
    p.add_argument("--ent-coef", dest="ent_coef", type=float, default=None)
    p.add_argument("--out", default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("algo", "env", "out") and v is not None}
    cfg = get_config(args.algo, args.env, **overrides)
    print(json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2))

    agent = make_agent(cfg)
    t0 = time.time()
    ts, history = agent.train()
    print(f"训练结束，用时 {time.time() - t0:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    # 文件名标签：pg: 前缀的冒号不适合文件名/正则解析，去掉
    tag = f"{cfg.algo}_{cfg.env_name.replace(':', '')}_seed{cfg.seed}"
    path = os.path.join(args.out, tag + ".json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    ckpt = save_checkpoint(os.path.join(args.out, tag + ".pkl"), ts, cfg)
    print(f"训练曲线已保存到 {path}")
    print(f"模型已保存到 {ckpt}（用 evaluate.py 评测与渲染）")


if __name__ == "__main__":
    main()
