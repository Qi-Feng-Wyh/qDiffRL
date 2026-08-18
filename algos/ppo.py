"""
PPO (Proximal Policy Optimization, clip 版本) —— 无梯度基线。

RPO / GI-PPO 两篇论文都把 PPO 作为对照，但都没有给出实现细节，这里按通用约定实现：

  L(theta) = E[ min( rho * A, clip(rho, 1-eps, 1+eps) * A ) ] + c_ent * H(pi_theta)
  rho = pi_theta(a|s) / pi_{theta_old}(a|s)

按 SB3 的常规约定：策略为 tanh 压缩高斯（squashed Gaussian），logp 带
Jacobian 修正（cfg.squash_actions=True，见 rollout.py）；恒定学习率、
10 epochs × minibatch 64。RPO / GI-PPO 不使用 squash（数学上要求无界高斯）。

与另外两个算法的唯一结构差异是：rollout 使用 objective="none"，
即**完全不穿过模拟器做反向传播**，因此单步 wall-clock 更便宜，
但样本效率显著低于基于解析梯度的方法（RPO 论文附录 G.2）。
优势估计、观测归一化、网络结构、critic 训练流程与 RPO 保持一致，保证公平对比。
"""

import jax
import jax.numpy as jnp

from .common import BaseAgent, TrainState, run_epochs, schedule_lr, set_lr
from .networks import (gaussian_entropy, gaussian_kl, gaussian_logprob,
                       tanh_squash_logp_correction)


class PPO(BaseAgent):

    objective = "none"                 # 不需要解析梯度
    log_keys = ("episode_return", "kl", "clip_frac", "entropy", "actor_lr")

    # ------------------------------------------------------------------
    def _ppo_loss(self, params, batch):
        cfg = self.cfg
        mu, log_std = self.actor.apply(params, batch["obs"])
        logp = gaussian_logprob(batch["act"], mu, log_std)
        if cfg.squash_actions:   # 与 rollout 采样侧一致的 tanh 修正
            logp = logp - tanh_squash_logp_correction(batch["act"])
        ratio = jnp.exp(jnp.clip(logp - batch["logp_old"], -20.0, 20.0))
        adv = batch["adv"]
        s1 = ratio * adv
        s2 = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
        pg_loss = -jnp.mean(jnp.minimum(s1, s2))
        ent = jnp.mean(gaussian_entropy(log_std))
        return pg_loss - cfg.ent_coef * ent, {}

    # ------------------------------------------------------------------
    def _update(self, ts: TrainState, env_state, key):
        cfg = self.cfg
        key, k_roll, k_actor, k_critic = jax.random.split(key, 4)

        # 采样（无反向传播）
        batch, env_state, metrics = self.rollout_fn(
            ts.actor_params, self.value_params(ts), ts.rms, env_state, k_roll)
        adv_n, _ = self.normalized_adv(batch["adv"])

        lr = ts.actor_lr if cfg.adaptive_lr else \
            schedule_lr(cfg, cfg.actor_lr, self.progress(ts))
        actor_opt = set_lr(ts.actor_opt, lr)
        actor_params, actor_opt, actor_loss = run_epochs(
            self._ppo_loss, self.actor_tx, ts.actor_params, actor_opt,
            dict(obs=batch["obs"], act=batch["act"], adv=adv_n,
                 logp_old=batch["logp_old"]),
            k_actor, cfg.ppo_epochs, cfg.ppo_batch)

        # 诊断 + 可选的 KL 自适应学习率
        mu_n, ls_n = self.actor.apply(actor_params, batch["obs"])
        logp_n = gaussian_logprob(batch["act"], mu_n, ls_n)
        if cfg.squash_actions:
            logp_n = logp_n - tanh_squash_logp_correction(batch["act"])
        kl = jnp.mean(gaussian_kl(batch["mu_old"], batch["log_std_old"],
                                  mu_n, ls_n))
        ratio = jnp.exp(jnp.clip(logp_n - batch["logp_old"], -20.0, 20.0))
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > cfg.clip_eps)
                             .astype(jnp.float32))
        if cfg.adaptive_lr:
            lr = jnp.where(kl > 2.0 * cfg.target_kl, lr / 1.5,
                           jnp.where(kl < 0.5 * cfg.target_kl, lr * 1.5, lr))
            lr = jnp.clip(lr, 1e-6, 1e-2)

        critic_params, critic_opt, critic_target, critic_loss = \
            self.update_critic(ts, batch, k_critic)

        ts = ts.replace(
            actor_params=actor_params, critic_params=critic_params,
            critic_target_params=critic_target, actor_opt=actor_opt,
            critic_opt=critic_opt, rms=self.update_rms(ts, batch),
            actor_lr=lr, step=ts.step + cfg.num_envs * cfg.horizon,
        )
        metrics.update(actor_loss=actor_loss, critic_loss=critic_loss, kl=kl,
                       clip_frac=clip_frac,
                       entropy=jnp.mean(gaussian_entropy(ls_n)), actor_lr=lr)
        return ts, env_state, metrics, key
