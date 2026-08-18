"""
DDCG (Discontinuity Detection Composite Gradient) —— 论文 Algorithm 的独立实现。

论文：Onoda et al., ICLR 2026,
"Does 'Do Differentiable Simulators Give Better Policy Gradients?' Give Better Policy Gradients?"

与框架内其他算法（PPO / RPO / GI-PPO 这类 actor-critic PPO 风格）不同，
DDCG 是**参数空间随机平滑 + 复合梯度估计**的直接梯度上升，无 critic：

  1. 随机平滑：zeta_i = theta + sigma * eps_i,  eps_i ~ N(0, I)，共 N 个（论文 Eq. 2）
  2. 每个 zeta_i 用确定性策略（均值动作）从当前环境状态 rollout 一个窗口，
     得回报 R_i = f(zeta_i)，并用 BPTT 穿过 MJX 求 1 阶梯度 grad R_i（Eq. 6）
  3. 0 阶估计（REINFORCE / score function，Eq. 4）：
         g0 = mean_i [ (R_i - b) * eps_i / sigma ]，  b = f(theta) 为均值参数回报基线
  4. 不连续检测（Eq. 14，主算法省略 eps_v 项）：
         v1 >= 2(1-c) * Var[R] / sigma^2 - 2 ||g1||^2
     其中 v1 为 1 阶梯度样本的逐维方差之和（Eq. 11），c = 0.3 全局固定
  5. 通过则 alpha = alpha_opt = v0/(v0+v1)（IVW，Eq. 8）；否则 alpha = 0 退回纯 0 阶
  6. theta <- theta + lr * (alpha * g1 + (1-alpha) * g0)（Adam 上升）

实现要点 / 与论文的对应与取舍：
* 论文 Tennis（策略优化）设定为线性策略、N=1000；本框架用 MLP 高斯策略，
  N 太大会让 BPTT 显存爆炸，默认 N=32、每个扰动用 M=8 个并行环境平均回报，
  均可通过 --ddcg-samples / --ddcg-envs 调整。
* 策略的 log_std 参数不影响确定性 rollout，扰动它对 R 无影响却会给 0 阶估计
  注入纯噪声，因此扰动时对 log_std 叶子置零。
* 论文 R 为折扣未明的累计回报；这里按 SHAC/RL 惯例使用 gamma 折扣
  （沿用 cfg.gamma），如需严格不折扣可用 --gamma 1.0（无命令行入口时需改配置）。
* 环境步数诚实计费：每次迭代模拟 (N*M + num_envs) * horizon 步，
  ts.step 按此累计，与其他算法对比样本效率时口径一致。
"""

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from .common import BaseAgent, TrainState, schedule_lr, set_lr
from .networks import rms_normalize, rms_update


class DDCG(BaseAgent):

    objective = "none"       # 不使用框架的 rollout_fn（DDCG 自带参数扰动 rollout）
    log_keys = ("episode_return", "alpha", "test_pass", "v0", "v1",
                "ret_mean", "actor_lr")

    # ------------------------------------------------------------------
    def _traj_return(self, actor_params, rms, state):
        """确定性策略（均值动作）rollout 一个窗口。

        返回 (每环境折扣回报 (n_env,), 末状态, 原始观测序列 (H,n_env,obs))。
        整段对 actor_params 可微（穿过 MJX 的 BPTT）。
        """
        cfg = self.cfg
        gamma = cfg.gamma

        def body(state, t):
            obs = rms_normalize(state.obs, rms) if cfg.normalize_obs else state.obs
            mu, _ = self.actor.apply(actor_params, obs)
            act = jnp.tanh(mu) if cfg.squash_actions else jnp.clip(mu, -1.0, 1.0)
            nstate = self.env.step(state, act)
            # 段内折扣权重（done 后由环境自动重置切断，折扣计数同样清零）
            return nstate, (nstate.reward, state.obs, nstate.done)

        final, (rews, raw_obs, _) = jax.lax.scan(
            body, state, jnp.arange(cfg.horizon))
        # 窗口内折扣累计回报（32 步窗口内一般不跨 episode 边界，
        # 边界处的梯度截断由环境的 where 自动重置保证）
        t_idx = jnp.arange(cfg.horizon)[(...,) + (None,) * (rews.ndim - 1)]
        rets = jnp.sum(rews * gamma ** t_idx, axis=0)
        return rets, final, raw_obs

    # ------------------------------------------------------------------
    @staticmethod
    def _mask_log_std(params, eps_tree):
        """把 log_std 叶子的扰动置零（不影响确定性 rollout，只会污染 0 阶估计）。"""
        paths_leaves = list(jax.tree_util.tree_flatten_with_path(params)[0])
        masked = []
        for (path, _), e in zip(paths_leaves,
                                jax.tree_util.tree_leaves(eps_tree)):
            is_log_std = any(getattr(k, "key", None) == "log_std" for k in path)
            masked.append(jnp.zeros_like(e) if is_log_std else e)
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(params), masked)

    # ------------------------------------------------------------------
    def _update(self, ts: TrainState, env_state, key):
        cfg = self.cfg
        key, k_eps = jax.random.split(key)
        params = ts.actor_params
        N, M, sigma = cfg.ddcg_samples, cfg.ddcg_envs, cfg.ddcg_sigma
        n_env, horizon = cfg.num_envs, cfg.horizon

        # ====== 1. 均值参数（baseline）rollout：提供 b、推进环境、更新归一化 ======
        b_ret, final_state, raw_obs = self._traj_return(params, ts.rms, env_state)
        b = jnp.mean(b_ret)                       # 论文基线 b = f(theta)

        # ====== 2. 参数空间随机平滑采样 ======
        flat0, unravel = ravel_pytree(params)
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(k_eps, len(leaves))
        eps_tree = jax.tree_util.tree_unflatten(
            treedef, [jax.random.normal(k, (N,) + l.shape, l.dtype)
                      for k, l in zip(keys, leaves)])
        eps_tree = self._mask_log_std(params, eps_tree)
        zeta_tree = jax.tree_util.tree_map(
            lambda p, e: p[None] + sigma * e, params, eps_tree)   # (N, ...)

        # ====== 3. 每个扰动的回报与 1 阶解析梯度（BPTT 穿过 MJX）======
        sub_state = jax.tree_util.tree_map(lambda x: x[:M], env_state)

        def ret_of(zeta):
            r, _, _ = self._traj_return(zeta, ts.rms, sub_state)
            return jnp.mean(r)

        R, G1 = jax.vmap(jax.value_and_grad(ret_of))(zeta_tree)   # (N,), (N,...)

        # ====== 4. 两个基本估计量与经验方差 ======
        G1f = jax.vmap(lambda t: ravel_pytree(t)[0])(G1)          # (N, D)
        Eps = jax.vmap(lambda t: ravel_pytree(t)[0])(eps_tree)    # (N, D)

        g1 = jnp.mean(G1f, axis=0)                                # 1 阶估计
        v1 = jnp.sum(jnp.var(G1f, axis=0, ddof=1))                # Eq. 11

        G0f = ((R - b) / sigma)[:, None] * Eps                    # Eq. 4 逐样本
        g0 = jnp.mean(G0f, axis=0)                                # 0 阶估计
        v0 = jnp.sum(jnp.var(G0f, axis=0, ddof=1))

        var_f = jnp.var(R, ddof=1)                                # V[f(x)]

        # ====== 5. 权重选择（子类钩子：DDCG=Eq.14 门控；AoBG=置信区间约束）======
        stats = dict(g0=g0, g1=g1, v0=v0, v1=v1, var_f=var_f,
                     G0f=G0f, G1f=G1f, N=N, d=flat0.size, sigma=sigma)
        alpha, g, sel_metrics = self._select_gradient(stats)

        # ====== 6. Adam 上升（对 -R 下降）======
        lr = schedule_lr(cfg, cfg.actor_lr, self.progress(ts))
        actor_opt = set_lr(ts.actor_opt, lr)
        updates, actor_opt = self.actor_tx.update(
            unravel(-g), actor_opt, params)
        actor_params = optax.apply_updates(params, updates)

        ts = ts.replace(
            actor_params=actor_params, actor_opt=actor_opt,
            rms=rms_update(ts.rms, raw_obs.reshape((-1,) + raw_obs.shape[2:]))
            if cfg.normalize_obs else ts.rms,
            actor_lr=lr,
            step=ts.step + (N * M + n_env) * horizon)             # 诚实计费

        metrics = dict(
            episode_return=jnp.mean(final_state.info["last_ep_ret"]),
            ret_mean=jnp.mean(R), ret_var=var_f, baseline=b,
            alpha=alpha, v0=v0, v1=v1,
            g0_norm=jnp.linalg.norm(g0), g1_norm=jnp.linalg.norm(g1),
            actor_lr=lr,
        )
        metrics.update(sel_metrics)
        return ts, final_state, metrics, key

    # ------------------------------------------------------------------
    def _select_gradient(self, st):
        """DDCG 的权重选择：Eq. 14 不连续检测 + IVW（Eq. 15 / Eq. 8）。

        返回 (alpha, 复合梯度 g, 诊断 metrics)。AoBG 子类覆盖此方法。
        """
        g0, g1, v0, v1 = st["g0"], st["g1"], st["v0"], st["v1"]
        rhs = 2.0 * (1.0 - self.cfg.ddcg_c) * st["var_f"] / st["sigma"] ** 2 \
            - 2.0 * jnp.dot(g1, g1)
        test_pass = v1 >= rhs
        alpha_opt = v0 / (v0 + v1 + 1e-12)
        alpha = jnp.where(test_pass, alpha_opt, 0.0)
        g = alpha * g1 + (1.0 - alpha) * g0
        return alpha, g, dict(alpha_opt=alpha_opt, test_rhs=rhs,
                              test_pass=test_pass.astype(jnp.float32))
