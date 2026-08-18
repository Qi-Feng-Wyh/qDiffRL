"""
IVW-H（stepwise Inverse-Variance Weighting）—— 论文附录 E Algorithm 1 的实现。

论文：Onoda et al., ICLR 2026,
"Does 'Do Differentiable Simulators Give Better Policy Gradients?' Give Better Policy Gradients?"

核心思想：在每个时间步 t、每个动作维 a、每类策略输出 phi ∈ {mu, log_std} 上，
分别用 1 阶（路径/BPTT）与 0 阶（似然比）两个梯度估计量，按它们在**并行环境
维度上的经验方差**做元素级逆方差加权（Eq. 16-18），合成目标梯度 G 后通过
vector-Jacobian product 回传到网络参数。不做任何显式不连续检测（论文 Part II
的立场：连续控制任务上方差控制已足够）。

与本框架已有组件的对应关系：
* 1 阶梯度：rollout(objective="shac") 缓存的 grad_a = dJ_shac/du_t（在 u=0 处），
  等于 RP 损失对每步 mu 的梯度（u 是加在动作上的扰动，du = dmu）。
  对 log_std 的路径为 grad_a * sigma * eps（链式法则）。
* 0 阶梯度：L_lr = -sum gamma^{c_t} * A_t * logpi(a_t|s_t) 对 mu / log_std 的
  逐样本梯度，即 -w*A*eps/sigma 与 -w*A*(1-eps^2)。w = gamma^{c_t}（rollout 新增
  的 w_gam 字段）保证与 1 阶梯度量纲一致（J_shac 含 gamma^t 因子）。
* 方差：把 (H*N) 批重塑回 (H, N)，沿 N（actor）轴逐 (t, a) 估计（Eq. 16）。
* 融合与回传：G = alpha*g1 + (1-alpha)*g0（Eq. 17-18），用 jax.vjp 把 G 当作
  策略输出的余切向量回传到 theta，单次 Adam 更新（SHAC 风格，无多轮复用）。
* Critic：沿用 RPO 的设置（double critic + target，TD(lambda) 目标，
  对应伪代码第 9 步 Fit V by MSE to A + V）。

与论文的已知差异：论文策略的 sigma 是状态依赖的，本框架是状态无关参数，
sigma 路径的梯度先按 (t,n,a) 逐样本构造、经 VJP 求和到全局 log_std 参数上。
"""

import jax
import jax.numpy as jnp
import optax

from .common import BaseAgent, TrainState, schedule_lr, set_lr
from .networks import gaussian_entropy, gaussian_kl


class IVWH(BaseAgent):

    objective = "shac"             # 与 RPO 共用可微 rollout，缓存 dR/da
    log_keys = ("episode_return", "alpha_mu", "alpha_std", "kl", "entropy",
                "actor_lr", "grad_a_norm")

    # ------------------------------------------------------------------
    def _update(self, ts: TrainState, env_state, key):
        cfg = self.cfg
        key, k_roll, k_critic = jax.random.split(key, 3)
        n_env, horizon = cfg.num_envs, cfg.horizon

        # ====== 1. 采样 + BPTT 缓存 1 阶动作梯度（含 gamma^t 因子）======
        batch, env_state, metrics = self.rollout_fn(
            ts.actor_params, self.value_params(ts), ts.rms, env_state, k_roll)

        act_dim = self.env.action_size
        shp = (horizon, n_env, act_dim)

        def unflat(x):
            return x.reshape(shp)

        act = unflat(batch["act"])
        eps = unflat(batch["eps"])
        mu_old = unflat(batch["mu_old"])
        ls_old = unflat(batch["log_std_old"])
        adv = batch["adv"].reshape((horizon, n_env))
        w = batch["w_gam"].reshape((horizon, n_env))      # gamma^{c_t}

        # ====== 2. 两类梯度估计量（均为负目标的梯度，即下降方向）======
        # 1 阶（RP / 路径梯度）：L_rp = -J 对 mu 的梯度 = -grad_a
        g1_mu = -unflat(batch["grad_a"])
        g1_ls = -unflat(batch["grad_a"]) * jnp.exp(ls_old) * eps

        # 0 阶（LR / 似然比）：L_lr = -sum w*A*logp 的逐样本梯度
        wa = (w * adv)[..., None]                          # (H, N, 1)
        g0_mu = -wa * eps * jnp.exp(-ls_old)               # -w*A*(a-mu)/sigma^2
        g0_ls = -wa * (1.0 - eps ** 2)                     # -w*A*dlogp/dlog_std

        # ====== 3. 跨 actor 的逐步方差（Eq. 16）与 IVW 融合（Eq. 17-18）======
        def ivw(g0, g1):
            v0 = jnp.var(g0, axis=1, ddof=1)               # 沿 actor 轴 -> (H, A)
            v1 = jnp.var(g1, axis=1, ddof=1)
            alpha = v0 / (v0 + v1 + 1e-12)
            return alpha, v0, v1

        alpha_mu, v0_mu, v1_mu = ivw(g0_mu, g1_mu)
        alpha_ls, v0_ls, v1_ls = ivw(g0_ls, g1_ls)

        G_mu = alpha_mu[:, None] * g1_mu + (1.0 - alpha_mu[:, None]) * g0_mu
        G_ls = alpha_ls[:, None] * g1_ls + (1.0 - alpha_ls[:, None]) * g0_ls

        # ====== 4. VJP 回传到网络参数，单次 Adam 更新（SHAC 风格）======
        obs = batch["obs"]

        def apply_fn(params):
            return self.actor.apply(params, obs)

        _, vjp_fn = jax.vjp(apply_fn, ts.actor_params)
        grads = vjp_fn((G_mu.reshape((-1, act_dim)),
                        G_ls.reshape((-1, act_dim))))[0]   # dL/dtheta

        lr = schedule_lr(cfg, cfg.actor_lr, self.progress(ts))
        actor_opt = set_lr(ts.actor_opt, lr)
        updates, actor_opt = self.actor_tx.update(grads, actor_opt,
                                                  ts.actor_params)
        actor_params = optax.apply_updates(ts.actor_params, updates)

        # 诊断：更新后 KL / 熵
        mu_n, ls_n = self.actor.apply(actor_params, obs)
        kl = jnp.mean(gaussian_kl(mu_old.reshape((-1, act_dim)),
                                  ls_old.reshape((-1, act_dim)), mu_n, ls_n))

        # ====== 5. Critic（TD(lambda) + target Polyak，伪代码第 9 步）======
        critic_params, critic_opt, critic_target, critic_loss = \
            self.update_critic(ts, batch, k_critic)

        ts = ts.replace(
            actor_params=actor_params, critic_params=critic_params,
            critic_target_params=critic_target, actor_opt=actor_opt,
            critic_opt=critic_opt, rms=self.update_rms(ts, batch),
            actor_lr=lr, step=ts.step + n_env * horizon)

        metrics.update(
            critic_loss=critic_loss, kl=kl, actor_lr=lr,
            alpha_mu=jnp.mean(alpha_mu), alpha_std=jnp.mean(alpha_ls),
            v0_mean=jnp.mean(v0_mu), v1_mean=jnp.mean(v1_mu),
            entropy=jnp.mean(gaussian_entropy(ls_n)),
        )
        return ts, env_state, metrics, key
