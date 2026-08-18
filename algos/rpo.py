"""
RPO (Reparameterization Proximal Policy Optimization)，对应 RPO 论文 Algorithm 1。

一次迭代：
  1. 用当前策略 pi_{theta_old} 采集 N 条短轨迹（SHAC 式短窗口 + 终端价值自举）；
  2. 一次 BPTT 反向，计算并缓存每个时间步的 action-gradient  \nabla_{a_k} R(tau)
     （它是 gamma^k \nabla_a Q^{pi_old}(s_k, a_k) 的无偏 Monte Carlo 估计）；
  3. 做 M 轮策略更新。第 1 轮是 on-policy（rho = 1、KL = 0，此时完全等价于 SHAC
     的 RPG 更新），后续轮次是 off-policy：通过“动作再生成”重建从 theta 到动作的
     计算图，把缓存的 action-gradient 重新反传，并用重要性比加权；
  4. 用 TD(lambda) 目标训练 double critic，并对 target critic 做 Polyak 更新。

三个关键机制
------------
(a) 动作再生成（Equation 10）。off-policy 时缓冲区里的动作 a 不再等于当前策略的
    采样结果，需要求出让当前策略能复现该动作的噪声
        eps_reg = f_theta^{-1}(a; s) = (a - mu_theta(s)) / sigma_theta,
    再令 a = f_theta(eps_reg; s)。代码里把 eps_reg 做 stop_gradient，于是
        a_regen := mu_theta(s) + sigma_theta * sg(eps_reg)
    数值上等于 a，但对 theta 可导，且 \nabla_theta a_regen = \nabla_theta f_theta。
    把线性替代目标  sum_i <w_i * \nabla_a R_i , a_regen_i>  对 theta 求导，
    得到的正是 Equation 11 要的  w * \nabla_theta a * \nabla_a R(tau)。

(b) 为 RPG 定制的梯度裁剪（Equation 11）。与 PPO 不同，它是**非对称**的、且
    **不依赖优势函数的符号**：只有当 rho 落在 [1-c_low, 1+c_high] 内时该样本才
    贡献梯度，并以 rho 加权；否则该样本梯度直接置零。这样做是因为 RPG 并不像
    REINFORCE 那样显式抬高/压低某个动作的 log 概率。

(c) 显式 KL 正则（Equation 12）与熵奖励（Equation 13）。论文实验表明仅靠裁剪不足
    以稳定训练；KL 项在第 1 轮 on-policy 更新时恒为 0，只在样本复用时起作用。
"""

import jax
import jax.numpy as jnp

from .common import BaseAgent, TrainState, run_epochs, schedule_lr, set_lr
from .networks import gaussian_entropy, gaussian_kl, gaussian_logprob


class RPO(BaseAgent):

    objective = "shac"                 # 需要 dR/da（SHAC 式短窗口回报的动作梯度）
    log_keys = ("episode_return", "kl", "eff_sample_ratio", "max_iw",
                "entropy", "actor_lr", "grad_a_norm")

    # ------------------------------------------------------------------
    def _rpo_loss(self, params, batch):
        cfg = self.cfg
        mu, log_std = self.actor.apply(params, batch["obs"])
        sigma = jnp.exp(log_std)

        # ---- (a) 动作再生成：数值等于 a，但重建了 theta -> a 的计算图 ----
        eps_reg = jax.lax.stop_gradient((batch["act"] - mu) / sigma)
        a_regen = mu + sigma * eps_reg

        # ---- (b) 重要性比与 RPG 专用的非对称裁剪 ----
        logp = gaussian_logprob(batch["act"], mu, log_std)
        rho = jnp.exp(jnp.clip(logp - batch["logp_old"], -20.0, 20.0))
        mask = ((rho >= 1.0 - cfg.c_low) & (rho <= 1.0 + cfg.c_high))
        w = jax.lax.stop_gradient(rho * mask.astype(jnp.float32))

        # ---- 线性替代目标：其 theta 梯度 = w * \nabla_theta a * \nabla_a R ----
        surr = jnp.mean(jnp.sum(w[:, None] * batch["grad_a"] * a_regen, axis=-1))

        # ---- (c) KL 正则 + 熵奖励 ----
        kl = jnp.mean(gaussian_kl(batch["mu_old"], batch["log_std_old"],
                                  mu, log_std))
        ent = jnp.mean(gaussian_entropy(log_std))

        # Equation 14：最大化 lam_surr*L_surr - lam_kl*L_KL + lam_ent*L_ent
        loss = -(cfg.lambda_surr * surr - cfg.lambda_kl * kl
                 + cfg.lambda_ent * ent)
        return loss, {}

    # ------------------------------------------------------------------
    def _update(self, ts: TrainState, env_state, key):
        cfg = self.cfg
        key, k_roll, k_actor, k_critic = jax.random.split(key, 4)

        # ===== 1-2. 采样短轨迹 + 一次 BPTT 缓存 action-gradient =====
        # 注意：这里传入 target critic 参数，终端价值自举与其梯度都走 target 网络
        batch, env_state, metrics = self.rollout_fn(
            ts.actor_params, self.value_params(ts), ts.rms, env_state, k_roll)

        # ===== 3. M 轮策略更新（第 1 轮 on-policy，其余复用缓存梯度）=====
        lr = schedule_lr(cfg, cfg.actor_lr, self.progress(ts))
        actor_opt = set_lr(ts.actor_opt, lr)
        data = dict(obs=batch["obs"], act=batch["act"], grad_a=batch["grad_a"],
                    logp_old=batch["logp_old"], mu_old=batch["mu_old"],
                    log_std_old=batch["log_std_old"])
        actor_params, actor_opt, actor_loss = run_epochs(
            self._rpo_loss, self.actor_tx, ts.actor_params, actor_opt,
            data, k_actor, cfg.rpo_epochs, cfg.rpo_batch)

        # 诊断：更新后整体的 KL、重要性比、有效样本率（论文附录 F）
        mu_n, ls_n = self.actor.apply(actor_params, batch["obs"])
        logp_n = gaussian_logprob(batch["act"], mu_n, ls_n)
        kl = jnp.mean(gaussian_kl(batch["mu_old"], batch["log_std_old"],
                                  mu_n, ls_n))
        rho = jnp.exp(jnp.clip(logp_n - batch["logp_old"], -20.0, 20.0))
        eff = jnp.mean(((rho >= 1.0 - cfg.c_low)
                        & (rho <= 1.0 + cfg.c_high)).astype(jnp.float32))

        # ===== 4. Critic（TD(lambda) 目标）+ target critic Polyak =====
        critic_params, critic_opt, critic_target, critic_loss = \
            self.update_critic(ts, batch, k_critic)

        ts = ts.replace(
            actor_params=actor_params, critic_params=critic_params,
            critic_target_params=critic_target, actor_opt=actor_opt,
            critic_opt=critic_opt, rms=self.update_rms(ts, batch),
            actor_lr=lr, step=ts.step + cfg.num_envs * cfg.horizon,
        )
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss, kl=kl,
                       eff_sample_ratio=eff, max_iw=jnp.max(rho),
                       entropy=jnp.mean(gaussian_entropy(ls_n)), actor_lr=lr)
        return ts, env_state, metrics, key
