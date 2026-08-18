"""
三个算法（GI-PPO / RPO / PPO）在三个 MJX 任务（cartpole / hopper / ant）上的配置。

超参数来源：
* GI-PPO : "Gradient Informed Proximal Policy Optimization" 附录 7.4.2
* RPO    : "Reparameterization Proximal Policy Optimization" Table 2 / 4
* PPO    : 常见约定的 PPO-Clip 实现（论文中作为基线，此处按通用做法设置）

最终配置 = 环境默认值 (ENV_DEFAULTS) <- 算法默认值 (ALGO_DEFAULTS)
           <- 算法×环境覆盖 (ALGO_ENV) <- 命令行覆盖
"""

from dataclasses import dataclass, replace
from typing import Sequence, Tuple


@dataclass
class Config:
    # ---------------- 环境 ----------------
    algo: str = "gippo"                 # gippo / rpo / ppo
    env_name: str = "cartpole"
    backend: str = "mjx"
    seed: int = 0
    num_envs: int = 64
    horizon: int = 32                   # 采样窗口，同时是 BPTT 截断长度
    episode_length: int = 1000
    total_steps: int = 1_000_000
    # cartpole 默认使用手写的纯 MJX Cartpole（MJXCartpole）：杆初始垂直向下，
    # 目标是摆起（swing-up）+ 稳在顶端，全程无提前终止（保持可微）。
    # 置 False 则退回 brax inverted_pendulum（起始接近竖直的静态平衡任务）。
    use_pure_mjx_cartpole: bool = True

    # ---------------- 折扣 / GAE ----------------
    gamma: float = 0.99
    lam: float = 0.95

    # ---------------- 网络 ----------------
    actor_hidden: Sequence[int] = (64, 64)
    critic_hidden: Sequence[int] = (64, 64)
    activation: str = "elu"             # elu / silu / relu / tanh
    layer_norm: bool = False
    num_critics: int = 1                # RPO 使用 double critic
    init_log_std: float = -1.0
    normalize_obs: bool = True
    normalize_adv: bool = True

    # ---------------- 优化器 ----------------
    # tanh 压缩策略动作（SB3 风格的 squashed Gaussian，带 logp 修正）。
    # 仅 PPO 开启：RPO / GI-PPO 的数学依赖无界高斯重参数化 a = mu + sigma*eps
    # （可逆性是两篇论文的硬性要求），必须保持 False。
    squash_actions: bool = False

    optimizer: str = "adam"             # adam / adamw
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_schedule: str = "constant"       # constant / linear / exponential
    lr_min: float = 1e-5
    clip_grad_a: float = -1.0           # >0 时对解析 action-gradient 做逐样本范数裁剪

    # ---------------- Actor / Critic ----------------
    actor_lr: float = 3e-4              # GI-PPO 的 PPO 更新 / RPO 与 PPO 的 actor 学习率
    critic_lr: float = 1e-3
    critic_epochs: int = 16
    critic_minibatches: int = 4
    use_target_critic: bool = False     # RPO / SHAC 使用 target critic
    critic_tau: float = 0.2             # target <- (1-tau)*target + tau*online

    # ---------------- GI-PPO 专属 ----------------
    alpha_lr: float = 1e-2              # Step 1（alpha-policy 回归）学习率
    alpha_epochs: int = 16
    alpha_batch: int = 2048
    alpha0: float = 5e-1
    alpha_max: float = 1.0
    alpha_beta: float = 1.02
    delta_det: float = 0.4              # 方差准则阈值
    delta_oorr: float = 0.75            # out-of-range-ratio 上限
    use_pi_h: bool = True               # PPO 更新用虚拟策略 pi_h 作重要性采样分母

    # ---------------- PPO 专属（GI-PPO 的 Step 3 复用）----------------
    ppo_epochs: int = 5
    ppo_batch: int = 2048
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    adaptive_lr: bool = False           # 按 KL 自适应调整 actor 学习率
    target_kl: float = 0.008

    # ---------------- DDCG 专属（参数空间复合梯度，无 critic）----------------
    ddcg_samples: int = 32      # N：每次迭代的参数扰动样本数
    ddcg_envs: int = 8          # M：每个扰动下平均回报的并行环境数
    ddcg_sigma: float = 0.02    # 参数平滑 std（论文 Tennis 0.01 / 玩具 0.1）
    ddcg_c: float = 0.3         # Eq.14 检测的松弛系数（论文固定 0.3，[0.1,0.9] 鲁棒）
    # AoBG 专属（Suh et al. 2022）
    aobg_gamma: float = 1.0     # 精度阈值 gamma，逐任务调（ALGO_ENV 有论文参考值）
    aobg_delta: float = 0.05    # Bernstein 置信度

    # ---------------- RPO 专属 ----------------
    rpo_epochs: int = 5                 # 论文中的 M（第 1 轮 on-policy，其余 off-policy）
    rpo_batch: int = 0                  # <=0 表示全批量更新
    c_low: float = 0.8                  # 重要性比下界 1 - c_low
    c_high: float = 1.0                 # 重要性比上界 1 + c_high
    lambda_surr: float = 1.0
    lambda_kl: float = 0.2
    lambda_ent: float = 0.25

    # ---------------- 日志 ----------------
    log_interval: int = 10


# ----------------------------------------------------------------------
ENV_DEFAULTS = {
    # swing-up 版 cartpole：50Hz 决策（action_repeat=2），500 决策步 = 10s 物理时长
    "cartpole": dict(env_name="cartpole", num_envs=64, episode_length=500,
                     total_steps=1_000_000),
    "ant": dict(env_name="ant", num_envs=64, episode_length=1000,
                total_steps=4_000_000),
    "hopper": dict(env_name="hopper", num_envs=256, episode_length=1000,
                   total_steps=15_000_000),
    # Ball with Wall：单决策 episode（1 步 = 1 次抛出），无接触式 RL 的
    # 不连续地形任务（Suh et al. 2022 / DDCG Part I）
    "ball_wall": dict(env_name="ball_wall", num_envs=64, episode_length=1,
                      total_steps=200_000),
}

# mujoco_playground dm_control 套件（env_name 用 "pg:" 前缀）。
# 均为无接触/光滑任务，作为解析梯度方法的主场任务集。
for _pg in ("PendulumSwingup", "AcrobotSwingup", "CartpoleSwingup",
            "PointMass", "ReacherEasy", "SwimmerSwimmer6", "FishSwim",
            "HopperHop", "HopperStand"):   # dm_control 版 hopper（有接触）
    ENV_DEFAULTS["pg:" + _pg] = dict(env_name="pg:" + _pg, num_envs=64,
                                     episode_length=1000,
                                     total_steps=1_000_000)

ENV_NET = {
    "ball_wall": dict(actor_hidden=(64, 64), critic_hidden=(64, 64)),
    **{"pg:" + n: dict(actor_hidden=(64, 64), critic_hidden=(64, 64))
       for n in ("PendulumSwingup", "AcrobotSwingup", "CartpoleSwingup",
                 "PointMass", "ReacherEasy", "SwimmerSwimmer6", "FishSwim",
                 "HopperHop", "HopperStand")},
    "cartpole": dict(actor_hidden=(64, 64), critic_hidden=(64, 64)),
    "ant": dict(actor_hidden=(128, 64, 32), critic_hidden=(64, 64)),
    "hopper": dict(actor_hidden=(128, 64, 32), critic_hidden=(64, 64)),
}

# RPO 论文使用统一的 (400,200,100) + LayerNorm + SiLU + double critic
RPO_NET = dict(actor_hidden=(400, 200, 100), critic_hidden=(400, 200, 100),
               activation="silu", layer_norm=True, num_critics=2)

ALGO_DEFAULTS = {
    # ---- GI-PPO：ELU、单 critic、无 target critic ----
    "gippo": dict(
        activation="elu", layer_norm=False, num_critics=1,
        optimizer="adam", adam_betas=(0.9, 0.999), max_grad_norm=1.0,
        lr_schedule="constant", use_target_critic=False,
        critic_epochs=16, critic_minibatches=4,
        alpha_epochs=16, alpha_batch=2048, alpha_beta=1.02,
        alpha0=5e-1, alpha_max=1.0, delta_det=0.4,
        ppo_epochs=5, ppo_batch=2048, clip_eps=0.2, ent_coef=0.0,
    ),
    # ---- RPO：SiLU+LayerNorm、double critic、AdamW、指数衰减 ----
    "rpo": dict(
        **RPO_NET,
        optimizer="adamw", adam_betas=(0.7, 0.95), weight_decay=0.0,
        max_grad_norm=1.0, lr_schedule="exponential", lr_min=1e-5,
        actor_lr=5e-4, critic_lr=5e-4,
        critic_epochs=16, critic_minibatches=4,
        use_target_critic=True, critic_tau=0.2,
        rpo_epochs=5, rpo_batch=0,
        c_low=0.8, c_high=1.0, lambda_surr=1.0,
    ),
    # ---- IVW-H：SHAC 风格的单次更新 + target critic（附录 E Alg.1，
    #      训练超参沿用 GIPPO/SHAC 设置）----
    "ivwh": dict(
        **RPO_NET,
        optimizer="adamw", adam_betas=(0.7, 0.95), weight_decay=0.0,
        max_grad_norm=1.0, lr_schedule="exponential", lr_min=1e-5,
        actor_lr=5e-4, critic_lr=5e-4,
        critic_epochs=16, critic_minibatches=4,
        use_target_critic=True, critic_tau=0.2,
    ),
    # ---- AoBG：与 DDCG 同骨架，权重选择换成置信区间约束（Eq.4-5）----
    "aobg": dict(
        activation="elu", layer_norm=False, num_critics=1,
        optimizer="adam", adam_betas=(0.9, 0.999), max_grad_norm=1.0,
        lr_schedule="constant", actor_lr=1e-3,
        use_target_critic=False,
    ),
    # ---- DDCG：无 critic 的直接梯度上升；网络用轻量 MLP，恒定 lr ----
    "ddcg": dict(
        activation="elu", layer_norm=False, num_critics=1,
        optimizer="adam", adam_betas=(0.9, 0.999), max_grad_norm=1.0,
        lr_schedule="constant", actor_lr=1e-3,
        use_target_critic=False,
    ),
    # ---- PPO：SB3 风格的常规实现 ----
    # tanh squash + 恒定 lr + 10 epochs × minibatch 64（SB3 默认约定）；
    # 网络仍与 RPO 同骨架，便于公平对比；无解析梯度。
    "ppo": dict(
        **RPO_NET,
        squash_actions=True,
        optimizer="adam", adam_betas=(0.9, 0.999), max_grad_norm=0.5,
        lr_schedule="constant", use_target_critic=False,
        actor_lr=3e-4, critic_lr=5e-4,
        critic_epochs=10, critic_minibatches=4,
        ppo_epochs=10, ppo_batch=64, clip_eps=0.2, ent_coef=1e-3,
        adaptive_lr=False, target_kl=0.008,
    ),
}

# AoBG 的 gamma 逐任务参考值（DDCG 论文附录 K / §5.3；注意量纲依赖实现，需扫参校准）
_AOBG_GAMMA = {"cartpole": 1.0, "hopper": 1e5, "ant": 1e6}

ALGO_ENV = {
    # GI-PPO：论文附录 7.4.2 的可微物理任务超参数
    ("gippo", "cartpole"): dict(critic_lr=1e-3, alpha_lr=1e-2, alpha_batch=2048,
                                alpha0=5e-1, delta_oorr=0.75, actor_lr=3e-4,
                                adaptive_lr=True, target_kl=0.008),
    ("gippo", "ant"): dict(critic_lr=2e-3, alpha_lr=5e-4, alpha_batch=2048,
                           alpha0=5e-1, delta_oorr=0.5, actor_lr=1e-4),
    ("gippo", "hopper"): dict(critic_lr=2e-4, alpha_lr=5e-3, alpha_batch=8192,
                              alpha0=5e-3, delta_oorr=0.75, actor_lr=1e-4),
    # RPO：论文 Table 4 的 KL / 熵系数（cartpole 论文未涉及，取保守值）
    ("rpo", "cartpole"): dict(lambda_kl=0.2, lambda_ent=0.1),
    ("rpo", "ant"): dict(lambda_kl=0.25, lambda_ent=0.2, num_envs=128),
    ("rpo", "hopper"): dict(lambda_kl=0.2, lambda_ent=0.25, num_envs=512),
    # AoBG：预置论文 gamma 参考值（可用 --aobg-gamma 覆盖）
    ("aobg", "ball_wall"): dict(aobg_gamma=0.014),   # 论文 Table 3 优化设置
    ("aobg", "cartpole"): dict(aobg_gamma=_AOBG_GAMMA["cartpole"]),
    ("aobg", "hopper"): dict(aobg_gamma=_AOBG_GAMMA["hopper"]),
    ("aobg", "ant"): dict(aobg_gamma=_AOBG_GAMMA["ant"]),
    # PPO：RPG 类方法之外的无梯度基线，样本效率低，步数放大
    ("ppo", "cartpole"): dict(total_steps=2_000_000),
    ("ppo", "ant"): dict(total_steps=20_000_000),
    ("ppo", "hopper"): dict(total_steps=50_000_000),
}


def get_config(algo: str, env_name: str, **overrides) -> Config:
    if algo not in ALGO_DEFAULTS:
        raise KeyError(f"未知算法 {algo}，可选：{list(ALGO_DEFAULTS)}")
    if env_name not in ENV_DEFAULTS:
        raise KeyError(f"未知环境 {env_name}，可选：{list(ENV_DEFAULTS)}")

    kwargs = dict(algo=algo)
    kwargs.update(ENV_DEFAULTS[env_name])
    kwargs.update(ENV_NET[env_name])
    kwargs.update(ALGO_DEFAULTS[algo])
    kwargs.update(ALGO_ENV.get((algo, env_name), {}))
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return replace(Config(), **kwargs)
