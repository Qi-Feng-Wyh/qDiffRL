"""
GI-PPO (Gradient Informed PPO)，对应论文 Algorithm 1。

一次迭代的三个步骤：
  Step 1  用解析梯度构造 alpha-policy 的目标动作 \tilde a = a + alpha * \nabla_a A，
          通过最小化 Equation 10 的回归损失把 pi_theta 拉向 pi_alpha；
  Step 2  用三条准则（方差 / 偏差 / out-of-range-ratio）自适应调整 alpha；
  Step 3  以虚拟策略 pi_h = 0.5 (pi_{\bar theta} + pi_alpha) 为重要性采样分母，
          做标准的 PPO clip 更新（Equation 18），作为“安全网”。

Step 2 中行列式的估计（Lemma 4.4 + Lemma 3.2 + Proposition 4.5）：
    det(I + alpha \nabla_a^2 A) = det(dg_alpha/deps) / det(dg_{\bar theta}/deps)
    Step 1 之后 pi_theta \approx pi_alpha，于是可直接用概率比估计
        psi_i = pi_{\bar theta}(s_i, a_i) / pi_theta(s_i, \tilde a_i)
    完全不需要显式计算任何 Jacobian / Hessian。
"""

import numpy as np
import jax
import jax.numpy as jnp

from .common import (BaseAgent, TrainState, make_optimizer, run_epochs,
                     schedule_lr, set_lr)
from .networks import gaussian_kl, gaussian_logprob

LOG2 = float(np.log(2.0))


class GIPPO(BaseAgent):

    objective = "gae"                      # 需要 dA/da
    log_keys = ("episode_return", "alpha", "psi_min", "psi_max", "R_alpha",
                "oorr", "kl", "actor_lr", "grad_a_norm")

    def __init__(self, cfg):
        self.alpha_tx = make_optimizer(cfg.alpha_lr, cfg)
        super().__init__(cfg)

    def _extra_init(self, actor_params, critic_params):
        return dict(alpha=jnp.asarray(self.cfg.alpha0, jnp.float32),
                    alpha_opt=self.alpha_tx.init(actor_params))

    # ------------------------------------------------------------------
    # 损失函数
    # ------------------------------------------------------------------
    def _alpha_loss(self, params, batch):
        """Equation 10：|| g_theta(s,eps) - g_alpha(s,eps) ||^2。

        g_alpha(s,eps) = a + alpha * \nabla_a A（batch['target']，外部算好）。
        """
        mu, log_std = self.actor.apply(params, batch["obs"])
        a_pred = mu + jnp.exp(log_std) * batch["eps"]
        return jnp.mean(jnp.sum(jnp.square(a_pred - batch["target"]), -1)), {}

    def _ppo_loss(self, params, batch):
        """Equation 18：以 pi_h 为分母的 clip 目标。"""
        mu, log_std = self.actor.apply(params, batch["obs"])
        logp = gaussian_logprob(batch["act"], mu, log_std)
        ratio = jnp.exp(logp - batch["logp_h"])
        adv = batch["adv"]
        s1 = ratio * adv
        s2 = jnp.clip(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
        return -jnp.mean(jnp.minimum(s1, s2)), {}

    # ------------------------------------------------------------------
    def _update(self, ts: TrainState, env_state, key):
        cfg = self.cfg
        key, k_roll, k_alpha, k_ppo, k_critic = jax.random.split(key, 5)
        alpha, alpha_opt = ts.extra["alpha"], ts.extra["alpha_opt"]

        # ========== 采样 + 解析梯度 dA/da ==========
        batch, env_state, metrics = self.rollout_fn(
            ts.actor_params, self.value_params(ts), ts.rms, env_state, k_roll)

        adv_n, adv_std = self.normalized_adv(batch["adv"])
        dA = batch["grad_a"] / adv_std      # 优势缩放时梯度同步缩放，保持量纲一致

        # ========== Step 1：把 pi_theta 更新到 alpha-policy ==========
        target_act = batch["act"] + alpha * dA            # Equation 8
        actor_params, alpha_opt, alpha_loss = run_epochs(
            self._alpha_loss, self.alpha_tx, ts.actor_params, alpha_opt,
            dict(obs=batch["obs"], eps=batch["eps"], target=target_act),
            k_alpha, cfg.alpha_epochs, cfg.alpha_batch)

        # ========== Step 2：三条准则 + alpha 自适应 ==========
        mu_a, ls_a = self.actor.apply(actor_params, batch["obs"])
        logp_a = gaussian_logprob(batch["act"], mu_a, ls_a)      # log pi_alpha(s,a)
        logp_at = gaussian_logprob(target_act, mu_a, ls_a)       # log pi_alpha(s,\tilde a)

        # (a) 方差准则：psi ~ det(I + alpha \nabla_a^2 A)，应落在 [1-δ, 1+δ]
        psi = jnp.exp(jnp.clip(batch["logp_old"] - logp_at, -20.0, 20.0))
        psi = jnp.nan_to_num(psi, nan=1e8, posinf=1e8, neginf=0.0)
        psi_min, psi_max = jnp.min(psi), jnp.max(psi)

        # (b) 偏差准则：Equation 16 的额外回报应为正（Proposition 4.3）
        ratio_a = jnp.exp(logp_a - batch["logp_old"])
        R_alpha = jnp.mean(ratio_a * adv_n)

        # (c) out-of-range-ratio（Equation 12）：给 PPO 留出更新空间
        R_oorr = jnp.mean((jnp.abs(ratio_a - 1.0) > cfg.clip_eps).astype(jnp.float32))

        bad = ((psi_min < 1.0 - cfg.delta_det) | (psi_max > 1.0 + cfg.delta_det)
               | (R_alpha < 0.0) | (R_oorr > cfg.delta_oorr))
        new_alpha = jnp.where(bad, alpha / cfg.alpha_beta, alpha * cfg.alpha_beta)
        new_alpha = jnp.clip(new_alpha, 1e-12, cfg.alpha_max)

        # ========== Step 3：PPO 更新（分母换成 pi_h）==========
        # pi_h = 0.5 (pi_{\bar theta} + pi_alpha)，logaddexp 保证数值稳定
        logp_h = jnp.logaddexp(batch["logp_old"], logp_a) - LOG2 \
            if cfg.use_pi_h else batch["logp_old"]

        lr = schedule_lr(cfg, cfg.actor_lr, self.progress(ts)) \
            if not cfg.adaptive_lr else ts.actor_lr
        actor_opt = set_lr(ts.actor_opt, lr)
        actor_params, actor_opt, ppo_loss = run_epochs(
            self._ppo_loss, self.actor_tx, actor_params, actor_opt,
            dict(obs=batch["obs"], act=batch["act"], adv=adv_n, logp_h=logp_h),
            k_ppo, cfg.ppo_epochs, cfg.ppo_batch)

        mu_n, ls_n = self.actor.apply(actor_params, batch["obs"])
        kl = jnp.mean(gaussian_kl(batch["mu_old"], batch["log_std_old"], mu_n, ls_n))
        if cfg.adaptive_lr:                # Cartpole 使用 KL 自适应学习率
            lr = jnp.where(kl > 2.0 * cfg.target_kl, lr / 1.5,
                           jnp.where(kl < 0.5 * cfg.target_kl, lr * 1.5, lr))
            lr = jnp.clip(lr, 1e-6, 1e-2)

        # ========== Critic / 归一化统计量 ==========
        critic_params, critic_opt, critic_target, critic_loss = \
            self.update_critic(ts, batch, k_critic)

        ts = ts.replace(
            actor_params=actor_params, critic_params=critic_params,
            critic_target_params=critic_target, actor_opt=actor_opt,
            critic_opt=critic_opt, rms=self.update_rms(ts, batch),
            actor_lr=lr, step=ts.step + cfg.num_envs * cfg.horizon,
            extra=dict(alpha=new_alpha, alpha_opt=alpha_opt),
        )
        metrics.update(alpha=new_alpha, alpha_loss=alpha_loss, ppo_loss=ppo_loss,
                       critic_loss=critic_loss, psi_min=psi_min, psi_max=psi_max,
                       psi_mean=jnp.mean(psi), R_alpha=R_alpha, oorr=R_oorr,
                       kl=kl, actor_lr=lr,
                       alpha_decreased=bad.astype(jnp.float32))
        return ts, env_state, metrics, key
