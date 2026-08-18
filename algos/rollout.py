"""
可微 rollout：一次前向（+ 可选一次反向）同时得到经验缓冲、GAE 优势与解析梯度。

三种梯度目标（由 cfg / objective 选择），对应三种算法：

objective = "gae"   (GI-PPO)
    J_gae = sum_t w_t * delta_t,   w_t = (gamma*lambda)^{c_t}
    => dA_t/da_t = (1/w_t) * dJ_gae/du_t        （GI-PPO 论文 Equation 15）

objective = "shac"  (RPO，同时也是 SHAC / SAPO 的目标)
    J_shac = sum_t g_t * r_t + (段末) gamma * g_t * V(s_{t+1}),  g_t = gamma^{c_t}
    => dR/da_t = dJ_shac/du_t 就是 RPO 论文中要缓存的 action-gradient
       （它天然带有 gamma^t 因子，正是 gamma^k \nabla_a Q^{pi_old}(s_k,a_k) 的无偏估计）

objective = "none"  (PPO)
    不做反向传播，只跑前向采样，省下穿过 MJX 的昂贵反传。

两个通用技巧
------------
1) 用“动作扰动量” u_t 作为求导变量：a_t = mu_theta(s_t) + sigma * eps_t + u_t，
   在 u = 0 处对 u 求梯度。反向传播会自动沿 d r_k/d a_t、d s_k/d a_t 链式展开
   （含未来时刻策略动作这条路径），无需二次模拟。
2) c_t 是“自本条 episode 段起点算起”的步数（每次 done 清零）。auto-reset 会在
   episode 边界切断梯度，用段内计数才能保证上面两个恒等式在边界处仍然成立。
"""

import jax
import jax.numpy as jnp

from .networks import (Critic, GaussianActor, critic_mean, logprob_from_eps,
                       rms_normalize, tanh_squash_logp_correction)


def make_rollout_fn(env, actor: GaussianActor, critic: Critic, cfg,
                    objective: str = "gae"):
    assert objective in ("gae", "shac", "none")
    gamma, lam = cfg.gamma, cfg.lam
    gl = gamma * lam
    n_env, horizon = cfg.num_envs, cfg.horizon

    # ------------------------------------------------------------------
    # 被求导的标量目标 J(u)
    # ------------------------------------------------------------------
    def _surrogate(u_seq, eps_seq, actor_params, critic_params, rms, state0):

        def body(carry, xs):
            state, cnt = carry
            eps_t, u_t = xs

            raw_obs = state.obs
            obs = rms_normalize(raw_obs, rms) if cfg.normalize_obs else raw_obs

            mu, log_std = actor.apply(actor_params, obs)
            act = mu + jnp.exp(log_std) * eps_t + u_t     # g_theta(s,eps) + u
            # PPO（squash_actions=True）走 SB3 风格的 tanh 压缩；
            # RPO / GI-PPO 保持硬 clip（其数学要求无界高斯重参数化）
            act_env = jnp.tanh(act) if cfg.squash_actions \
                else jnp.clip(act, -1.0, 1.0)

            nstate = env.step(state, act_env)             # 可微的 MJX 步进
            done = nstate.done
            trunc = nstate.info.get("truncation", jnp.zeros_like(done))
            next_raw = nstate.info["obs_before_reset"]    # 重置前真实的 s_{t+1}
            next_obs = rms_normalize(next_raw, rms) if cfg.normalize_obs else next_raw

            v = critic_mean(critic.apply(critic_params, obs))
            v_next = critic_mean(critic.apply(critic_params, next_obs))

            terminated = done * (1.0 - trunc)             # 真终止不自举，超时截断要自举
            delta = nstate.reward + gamma * (1.0 - terminated) * v_next - v

            w_gae = gl ** cnt          # GAE 权重
            w_gam = gamma ** cnt       # SHAC 折扣权重
            cnt_next = jnp.where(done > 0, jnp.zeros_like(cnt), cnt + 1.0)

            logp = logprob_from_eps(eps_t, log_std)
            if cfg.squash_actions:   # tanh 变量代换的 Jacobian 修正
                logp = logp - tanh_squash_logp_correction(act)
            out = dict(
                obs=obs, raw_obs=raw_obs, act=act, eps=eps_t, logp=logp,
                mu=mu, log_std=log_std, delta=delta, done=done, trunc=trunc,
                v=v, w_gae=w_gae, w_gam=w_gam, reward=nstate.reward,
                boot=gamma * w_gam * v_next,   # 段末自举项 gamma^{c+1} V(s_{t+1})
                disc_r=w_gam * nstate.reward,  # 折扣即时奖励 gamma^{c} r_t
            )
            return (nstate, cnt_next), out

        cnt0 = jnp.zeros((n_env,))
        (final_state, _), outs = jax.lax.scan(
            body, (state0, cnt0), (eps_seq, u_seq))

        if objective == "gae":
            J = jnp.sum(outs["w_gae"] * outs["delta"])
        elif objective == "shac":
            # 段内折扣奖励之和
            J = jnp.sum(outs["disc_r"])
            # 段在窗口内结束：真终止不自举，超时截断才加自举价值
            terminated = outs["done"] * (1.0 - outs["trunc"])
            truncated = outs["done"] - terminated
            J = J + jnp.sum(outs["boot"] * truncated)
            # 窗口末尾仍未结束的段 -> 用 V(s_H) 自举（SHAC 的短窗口价值自举）
            J = J + jnp.sum(outs["boot"][-1] * (1.0 - outs["done"][-1]))
        else:
            J = jnp.zeros(())
        return J, (outs, final_state)

    # ------------------------------------------------------------------
    # 对外的 rollout 接口
    # ------------------------------------------------------------------
    def rollout(actor_params, critic_params, rms, state0, key):
        """critic_params：用于计算 V 的参数。若算法使用 target critic，
        由调用方直接把 target 参数传进来即可（RPO / SHAC 的做法）。"""
        eps = jax.random.normal(key, (horizon, n_env, env.action_size))
        u0 = jnp.zeros_like(eps)

        if objective == "none":
            # PPO：不需要穿过模拟器的反向传播
            _, (outs, final_state) = _surrogate(
                u0, eps, actor_params, critic_params, rms, state0)
            grad_u = jnp.zeros_like(u0)
        else:
            grad_fn = jax.value_and_grad(_surrogate, argnums=0, has_aux=True)
            (_, (outs, final_state)), grad_u = grad_fn(
                u0, eps, actor_params, critic_params, rms, state0)

        if objective == "gae":
            # Equation 15：dA_t/da_t = (1/w_t) * dJ/du_t
            dgrad = grad_u / (outs["w_gae"][..., None] + 1e-12)
        else:
            # SHAC / RPO：dJ/du_t 本身就是要缓存的 action-gradient（含 gamma^t）
            dgrad = grad_u
        dgrad = jnp.nan_to_num(dgrad, nan=0.0, posinf=0.0, neginf=0.0)
        if cfg.clip_grad_a > 0:      # 可选：逐样本范数裁剪，抑制解析梯度爆炸
            nrm = jnp.linalg.norm(dgrad, axis=-1, keepdims=True)
            dgrad = dgrad * jnp.minimum(1.0, cfg.clip_grad_a / (nrm + 1e-8))

        # GAE：A_t = delta_t + gamma*lambda*(1-done_t)*A_{t+1}
        # 它同时也是 RPO / SHAC critic 训练所需的 TD(lambda) 目标（ret = A + V）
        def gae_body(adv, x):
            delta, done = x
            adv = delta + gl * (1.0 - done) * adv
            return adv, adv

        _, adv = jax.lax.scan(gae_body, jnp.zeros((n_env,)),
                              (outs["delta"], outs["done"]), reverse=True)
        ret = adv + outs["v"]

        flat = lambda x: x.reshape((horizon * n_env,) + x.shape[2:])
        batch = dict(
            obs=flat(outs["obs"]),
            raw_obs=flat(outs["raw_obs"]),
            act=flat(outs["act"]),
            eps=flat(outs["eps"]),
            logp_old=flat(outs["logp"]),
            mu_old=flat(outs["mu"]),
            log_std_old=flat(outs["log_std"]),
            adv=flat(adv),
            ret=flat(ret),
            v=flat(outs["v"]),
            w_gam=flat(outs["w_gam"]),   # IVW-H 的 0 阶梯度量纲对齐用
            grad_a=flat(dgrad),        # GI-PPO 的 dA/da 或 RPO 的 dR/da
        )
        batch = jax.tree_util.tree_map(jax.lax.stop_gradient, batch)

        n_done = jnp.sum(outs["done"])
        metrics = dict(
            mean_reward=jnp.mean(outs["reward"]),
            # 完整回合回报：由环境 info 跨窗口累计（见 envs.py 的 ep_ret/last_ep_ret），
            # 不再有“窗口内部分和、无 done 轮次记 0”的锯齿问题
            episode_return=jnp.mean(final_state.info["last_ep_ret"]),
            num_episodes=n_done,
            grad_a_norm=jnp.mean(jnp.linalg.norm(batch["grad_a"], axis=-1)),
            # 解析梯度的样本方差，可作为其“质量”的直接指标（GI-PPO 论文 7.3.1）
            grad_a_var=jnp.mean(jnp.var(batch["grad_a"], axis=0)),
        )
        return batch, final_state, metrics

    return rollout
