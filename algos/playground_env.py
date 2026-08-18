"""
mujoco_playground（dm_control 套件）环境到本框架统一接口的封装。

用法：make_env(cfg) 中 cfg.env_name 以 "pg:" 前缀触发，如 pg:PendulumSwingup。
可用环境名见 mujoco_playground.registry.dm_control_suite.ALL_ENVS。

两个兼容性处理（ playground==0.1.0 + mujoco==3.11 ）：
1. playground 的 mjx_env.make_data 仍使用旧版 mujoco 的 nconmax 参数名，
   mujoco 3.11 已改名为 naconmax——导入期对 mujoco.mjx.make_data 打别名补丁；
2. playground 的 State.info 不含框架需要的字段，这里统一补齐：
   truncation / obs_before_reset（GAE 自举）、ep_ret / last_ep_ret（回合统计）。
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import struct


def _patch_playground_mjx():
    """给 mujoco 3.11 的 mjx.make_data 加 nconmax -> naconmax 别名。

    playground 0.1.0 按旧 API 传 nconmax/njmax；3.11 已改名。
    只在本模块导入时应用，不影响其他路径。
    """
    from mujoco import mjx

    if getattr(mjx.make_data, "_pg_patched", False):
        return
    orig = mjx.make_data

    def make_data_compat(*args, **kwargs):
        if "nconmax" in kwargs:
            kwargs["naconmax"] = kwargs.pop("nconmax")
        return orig(*args, **kwargs)

    make_data_compat._pg_patched = True
    mjx.make_data = make_data_compat


_patch_playground_mjx()

from mujoco_playground import registry  # noqa: E402  依赖上面的补丁，顺序敏感


@struct.dataclass
class PlaygroundState:
    data: Any                     # playground 的 mjx State（其 .data 是 mjx.Data）
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: Dict[str, Any] = struct.field(pytree_node=True, default_factory=dict)


class PlaygroundEnv:
    """playground dm_control 环境 + vmap 并行 + 可微自动重置（where 截断梯度）。

    与 MJXCartpole / BraxMJXEnv 同接口：reset(key) / step(state, act)。
    """

    def __init__(self, env_name: str, num_envs: int):
        self.pg_env = registry.load(env_name)
        self.num_envs = num_envs
        self.obs_size = int(self.pg_env.observation_size)
        self.action_size = int(self.pg_env.action_size)
        self.episode_length = int(getattr(self.pg_env, "episode_length", 1000))
        self.dt = float(self.pg_env.dt)
        self.mj_model = getattr(self.pg_env, "mj_model", None)  # 渲染用

    # ---------------- 内部工具 ----------------
    def _obs_flat(self, obs):
        """playground 的 obs 可能是 dict，统一展平为向量。"""
        if isinstance(obs, dict):
            return jnp.concatenate(
                [jnp.atleast_1d(obs[k]).reshape(-1) for k in sorted(obs)], axis=-1)
        return obs

    def _reset_one(self, key):
        return self.pg_env.reset(key)          # playground State

    # ---------------- 对外接口 ----------------
    def reset(self, key):
        keys = jax.random.split(key, self.num_envs)
        st = jax.vmap(self._reset_one)(keys)   # vmap 后的 playground State
        obs = jax.vmap(self._obs_flat)(st.obs)
        zeros = jnp.zeros((self.num_envs,))
        info = {
            "steps": zeros,
            "truncation": zeros,
            "obs_before_reset": obs,
            "ep_ret": zeros,
            "last_ep_ret": zeros,
            "first_data": st.data,             # 重置状态缓存（供自动重置复用）
            "first_obs": obs,
        }
        return PlaygroundState(data=st, obs=obs, reward=zeros, done=zeros,
                               info=info)

    def step(self, state: PlaygroundState, action):
        st = jax.vmap(self.pg_env.step)(state.data, action)
        obs = jax.vmap(self._obs_flat)(st.obs)
        reward = st.reward
        done = st.done

        # playground 原生 env 只在真终止时置 done（摆起类任务永不真终止），
        # 超时截断必须由封装层自己判定——否则回合永不结束、回报恒为 0
        steps = state.info["steps"] + 1.0
        terminated = st.done                                   # 真终止（如摔倒）
        timeout = (steps >= self.episode_length).astype(jnp.float32)
        truncation = timeout * (1.0 - terminated)              # 超时截断（需自举）
        done = jnp.maximum(terminated, timeout)                # 任一即回合结束

        obs_before_reset = obs                  # 真实的 s_{t+1}

        # ---- 自动重置：done 处用缓存的首帧状态替换，where 切断梯度 ----
        def where_done(x, y):
            d = jnp.reshape(done, [done.shape[0]] + [1] * (x.ndim - 1))
            return jnp.where(d > 0, x, y)

        new_data = jax.tree_util.tree_map(
            where_done, state.info["first_data"], st.data)
        new_obs = where_done(state.info["first_obs"], obs)
        # playground State 是 flax struct，用 replace 换 data 字段
        new_st = st.replace(data=new_data)

        ep_ret = state.info["ep_ret"] + reward
        info = dict(state.info)
        info.update(
            steps=jnp.where(done > 0, 0.0, steps),
            truncation=truncation,
            obs_before_reset=obs_before_reset,
            ep_ret=jnp.where(done > 0, 0.0, ep_ret),
            last_ep_ret=jnp.where(done > 0, ep_ret,
                                  state.info["last_ep_ret"]),
        )
        return PlaygroundState(data=new_st, obs=new_obs, reward=reward,
                               done=done, info=info)
