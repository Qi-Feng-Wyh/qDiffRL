"""
MuJoCo-XLA (MJX) 可微环境封装。

GI-PPO 需要环境提供 Equation 2 中的解析梯度
    d s_{t+1+k} / d a_t ,  d r_{t+k} / d a_t
在 MJX 中这一点是自动满足的：mjx.step 完全由 JAX 写成，因此可以直接被
jax.grad / jax.vjp 反向传播。本文件提供两种构建方式：

1. `BraxMJXEnv`：包装 brax 自带的 mjx 后端环境（ant / hopper / inverted_pendulum），
   直接复用其 XML、观测与奖励定义，工程上最省事。
2. `MJXCartpole`：一个完全用 mujoco.mjx 手写的 Cartpole 例子（含内嵌 XML），
   用来说明“不依赖 brax、直接在 MJX 上写可微环境”的写法。

两者对外暴露统一接口：
    env.reset(key) -> state
    env.step(state, action) -> state
    state.obs / state.reward / state.done
    state.info['truncation']        : 1 表示因超时结束（需要 bootstrap V(s')）
    state.info['obs_before_reset']  : 自动重置之前的下一帧观测（用于 GAE 自举）
    env.obs_size / env.action_size / env.num_envs

关于梯度的两点说明：
* 自动重置使用 `jnp.where(done, reset_state, next_state)` 实现，因此在 episode
  边界上梯度会被自然截断（where 只把梯度传给被选中的分支），符合论文需要。
* Ant / Hopper 存在接触与足底碰撞，MJX 的接触求解是分段光滑的，解析梯度会带偏差与
  高方差——这正是 GI-PPO 通过 alpha 自适应要处理的情形（论文 4.3.1 节）。
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import struct

def _patch_mjx_solver():
    """给 mujoco 3.x 的 MJX 约束求解器打补丁，使其支持反向传播。

    mujoco 3.11 的 `mjx._src.solver.solve` 主循环使用 `jax.lax.while_loop`，
    反向模式微分（GI-PPO / RPO 的解析梯度都依赖它）会直接报错：
        ValueError: Reverse-mode differentiation does not work for
        lax.while_loop or lax.fori_loop with dynamic start/stop values.
    好在库自带 `_while_loop_scan`：语义等价的 lax.scan 实现（cond 不满足后
    自动退化为 no-op），文档注明 "reverse-mode autodiff ok"。这里仅把主循环
    替换为它，其余逻辑与原版逐行一致；前向数值结果不变（提前收敛时多余的
    迭代是 no-op），代价是始终跑满 `opt.iterations` 次。

    说明：`mjx._src.forward` 在调用点按属性查找 `solver.solve`，因此在导入期
    替换模块属性即可生效。以下私有符号均来自锁定版本 mujoco==3.11.0，若升级
    依赖导致 ImportError / AttributeError，需要重新核对本函数。
    """
    try:
        import mujoco
        from jax import numpy as jp
        from mujoco.mjx._src import math as mjx_math
        from mujoco.mjx._src import solver as mjx_solver
        from mujoco.mjx._src.types import DisableBit, OptionJAX, SolverType
    except ImportError:
        return

    Context = mjx_solver.Context
    _rescale = mjx_solver._rescale
    _linesearch = mjx_solver._linesearch
    _update_constraint = mjx_solver._update_constraint
    _update_gradient = mjx_solver._update_gradient
    _while_loop_scan = mjx_solver._while_loop_scan

    def solve(m, d):
        """逐行复制自 mujoco 3.11 的 solver.solve，仅主循环换成 _while_loop_scan。"""
        if not isinstance(m.opt._impl, OptionJAX):
            raise ValueError('solve requires JAX backend implementation.')

        def cond(ctx):
            improvement = _rescale(m, ctx.prev_cost - ctx.cost)
            gradient = _rescale(m, mjx_math.norm(ctx.grad))
            done = ctx.solver_niter >= m.opt.iterations
            done |= improvement < m.opt.tolerance
            done |= gradient < m.opt.tolerance
            return ~done

        def body(ctx):
            ctx = _linesearch(m, d, ctx)
            prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad
            ctx = _update_constraint(m, d, ctx)
            ctx = _update_gradient(m, d, ctx)
            if m.opt.solver == SolverType.NEWTON:
                search = -ctx.Mgrad
            else:
                # polak-ribiere:
                beta = jp.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
                beta = beta / jp.maximum(mujoco.mjMINVAL,
                                         jp.dot(prev_grad, prev_Mgrad))
                beta = jp.maximum(0, beta)
                search = -ctx.Mgrad + beta * ctx.search
            return ctx.replace(search=search,
                               solver_niter=ctx.solver_niter + 1)

        # warmstart:
        qacc = d.qacc_smooth
        if not m.opt.disableflags & DisableBit.WARMSTART:
            warm = Context.create(m, d.replace(qacc=d.qacc_warmstart),
                                  grad=False)
            smth = Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
            qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart,
                            d.qacc_smooth)
        d = d.replace(qacc=qacc)

        ctx = Context.create(m, d)
        if m.opt.iterations == 1:
            ctx = body(ctx)
        else:
            # 唯一改动：lax.while_loop -> 可微的 scan 实现
            ctx = _while_loop_scan(cond, body, ctx, m.opt.iterations)

        return d.tree_replace({
            'qfrc_constraint': ctx.qfrc_constraint,
            'qacc': ctx.qacc,
            '_impl.efc_force': ctx.efc_force,
        })

    mjx_solver.solve = solve


_patch_mjx_solver()

# brax 的 mjx 后端 = MuJoCo MJX
from brax import envs as brax_envs
from brax.envs.base import Wrapper
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

# brax 环境名映射：论文中的 Cartpole 对应 brax 的 inverted_pendulum
_BRAX_NAME = {
    "cartpole": "inverted_pendulum",
    "inverted_pendulum": "inverted_pendulum",
    "hopper": "hopper",
    "ant": "ant",
}


class DiffAutoResetWrapper(Wrapper):
    """自动重置，并额外保存重置前的观测 `obs_before_reset`。

    brax 自带的 AutoResetWrapper 会把 done 时的 obs 直接替换成新 episode 的首帧
    观测，导致我们拿不到 s_{T}，无法正确地对截断（truncation）做价值自举。
    这里重写一份并把重置前的观测存进 info。
    """

    def reset(self, rng: jnp.ndarray):
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["obs_before_reset"] = state.obs
        # 回合回报统计（跨 rollout 窗口保留，修复 episode_return 只统计窗口内部分和的 bug）
        state.info["ep_ret"] = jnp.zeros_like(state.reward)
        state.info["last_ep_ret"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state, action: jnp.ndarray):
        # 上一帧若已结束，先把 done / steps 清零，再推进一步
        if "steps" in state.info:
            steps = jnp.where(state.done > 0, jnp.zeros_like(state.info["steps"]),
                              state.info["steps"])
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        # ---- 回合回报统计：done 时把当前回合总回报存入 last_ep_ret 并清零 ----
        ep_ret = state.info["ep_ret"] + state.reward
        state.info["last_ep_ret"] = jnp.where(state.done > 0, ep_ret,
                                              state.info["last_ep_ret"])
        state.info["ep_ret"] = jnp.where(state.done > 0, jnp.zeros_like(ep_ret),
                                         ep_ret)

        obs_before_reset = state.obs  # 关键：保留真实的 s_{t+1}

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done > 0, x, y)  # done 时取重置状态 -> 梯度自然被切断

        pipeline_state = jax.tree_util.tree_map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state)
        obs = where_done(state.info["first_obs"], state.obs)
        state.info["obs_before_reset"] = obs_before_reset
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class BraxMJXEnv:
    """brax(mjx 后端) 环境 + EpisodeWrapper + VmapWrapper + 可微自动重置。"""

    def __init__(self, env_name: str, num_envs: int, episode_length: int,
                 action_repeat: int = 1, backend: str = "mjx"):
        base = brax_envs.get_environment(_BRAX_NAME[env_name], backend=backend)
        env = EpisodeWrapper(base, episode_length, action_repeat)  # 提供 truncation / steps
        env = VmapWrapper(env, batch_size=None)                    # 批并行
        env = DiffAutoResetWrapper(env)
        self.env = env
        self.unwrapped = base            # 未包装的原始环境（评测/渲染用）
        self.sys = base.sys              # brax System（HTML 渲染需要）
        self.dt = base.dt
        self.mj_model = getattr(base.sys, "mj_model", None)  # 可能为 None
        self.num_envs = num_envs
        self.obs_size = base.observation_size
        self.action_size = base.action_size

    def reset(self, key):
        keys = jax.random.split(key, self.num_envs)
        return self.env.reset(keys)

    def step(self, state, action):
        return self.env.step(state, action)


# ======================================================================
#  纯 MJX 实现的 Cartpole（不依赖 brax）
# ======================================================================
# 积分器用默认半欧拉，用 RK4 多四倍显存
_CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01"/>
  <default>
    <joint damping="0.05"/>
    <geom rgba="0.7 0.7 0.7 1"/>
  </default>
  <worldbody>
    <!-- 固定近景相机：视野约 ±2m（奖励的 x² 惩罚会把小车约束在中段），
         保留小车位移的视觉信息 -->
    <camera name="fixed" pos="0 -5.0 0.7" xyaxes="1 0 0 0 0 1"/>
    <!-- 装饰滑轨：必须 contype=conaffinity=0，否则它穿过小车/杆产生嵌入接触，
         会把整个机构“焊死”（曾导致摆起任务完全学不动，2026-08 修复） -->
    <!-- 轨道有效范围 ±20m：实质上“无墙”，撞边借力被移出策略空间，
         居中约束完全交给奖励的 -0.1x^2 项 -->
    <geom name="rail" type="capsule" pos="0 0 0" quat="0.707 0 0.707 0"
          size="0.02 10.0" rgba="0.3 0.3 0.7 1" contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-10 10" damping="0.1"/>
      <geom name="cart" type="box" size="0.1 0.05 0.05" mass="1.0"/>
      <body name="pole" pos="0 0 0">
        <!-- 摆阻尼要小（0.001~0.01），大了能量泵不起来（参考 cartpole_swingup_guide.md） -->
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.005"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.6" size="0.045" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <!-- gear=10：必须小到不能一次拉起摆杆，逼出“来回摆荡蓄能”行为；
         之前的 gear=50 会诱发蛮力直推/撞墙借力的钻空子策略 -->
    <motor joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@struct.dataclass
class MJXState:
    data: Any                     # mjx.Data
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: Dict[str, Any] = struct.field(pytree_node=True, default_factory=dict)


class MJXCartpole:
    """手写的 MJX Cartpole（摆起+平衡），奖励完全光滑，适合作为解析梯度的“干净”基准。

    观测: [x, xdot, sin(theta), cos(theta), thetadot]，thetadot clip 到 ±10
    动作: 1 维，范围 [-1, 1]（gear=10N）
    奖励: cos(theta) - 0.1*x^2 - 0.001*a^2
        注意不惩罚 thetadot^2——无条件惩罚角速度会抑制蓄能摆荡，
        是 swing-up 奖励设计最常见的错误（见 cartpole_swingup_guide.md）
    终止: 仅有超时截断（保持动力学全程可微）
    控制频率: action_repeat=2，即 50Hz 决策 / 100Hz 物理
    """

    def __init__(self, num_envs: int, episode_length: int = 500,
                 action_repeat: int = 2):
        import mujoco
        from mujoco import mjx

        self._mjx = mjx
        mj_model = mujoco.MjModel.from_xml_string(_CARTPOLE_XML)
        self.mj_model = mj_model         # 渲染用的 CPU 侧模型
        self.model = mjx.put_model(mj_model)
        self.action_repeat = action_repeat
        self.dt = float(mj_model.opt.timestep) * action_repeat  # 决策间隔
        # 渲染时使用 XML 内置的固定近景相机（见 _CARTPOLE_XML 的 camera）
        self.use_model_camera = True
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.obs_size = 5
        self.action_size = 1

    # ---------------- 内部工具 ----------------
    def _obs(self, data):
        x, th = data.qpos[0], data.qpos[1]
        dx, dth = data.qvel[0], data.qvel[1]
        # dth clip 到 ±10，防早期随机策略产生极端观测值
        return jnp.stack([x, dx, jnp.sin(th), jnp.cos(th),
                          jnp.clip(dth, -10.0, 10.0)])

    def _reward(self, data, act):
        x, th = data.qpos[0], data.qpos[1]
        return (jnp.cos(th) - 0.1 * x ** 2 - 0.001 * jnp.sum(act ** 2))

    def _reset_one(self, key):
        mjx = self._mjx
        k1, k2 = jax.random.split(key)
        data = mjx.make_data(self.model)
        # theta=0 为竖直向上（XML 中杆沿 +z），theta=pi 为竖直向下：
        # 从自然下垂位置出发 -> 摆起（swing-up）任务。
        # 初始角扰动 ±0.3（参考指南：加大初始随机化是能否学出来的分水岭）
        qpos = jnp.array([0.0, jnp.pi]) + jax.random.uniform(
            k1, (2,), minval=jnp.array([-0.1, -0.3]),
            maxval=jnp.array([0.1, 0.3]))
        qvel = jax.random.uniform(k2, (2,), minval=-0.1, maxval=0.1)
        data = data.replace(qpos=qpos, qvel=qvel)
        return mjx.forward(self.model, data)

    # ---------------- 对外接口 ----------------
    def reset(self, key):
        keys = jax.random.split(key, self.num_envs)
        data = jax.vmap(self._reset_one)(keys)
        obs = jax.vmap(self._obs)(data)
        zeros = jnp.zeros((self.num_envs,))
        info = {
            "rng": jax.random.split(jax.random.fold_in(key, 1), self.num_envs),
            "steps": zeros,
            "truncation": zeros,
            "obs_before_reset": obs,
            "ep_ret": zeros,          # 当前回合累计回报（跨 rollout 窗口保留）
            "last_ep_ret": zeros,     # 上一个完整回合的总回报
            # 自动重置的状态池：step 中仅在确有 done 时才刷新（见 step 的 lax.cond）
            "reset_data": data,
            "reset_obs": obs,
        }
        return MJXState(data=data, obs=obs, reward=zeros, done=zeros, info=info)

    def step(self, state: MJXState, action):
        mjx = self._mjx

        def one_step(data, act):
            data = data.replace(ctrl=jnp.clip(act, -1.0, 1.0))
            # action_repeat：每个决策步推进多次物理步（50Hz 决策 / 100Hz 物理）
            data, _ = jax.lax.scan(
                lambda d, _: (mjx.step(self.model, d), None),
                data, None, length=self.action_repeat)
            return data, self._obs(data), self._reward(data, act)

        data, obs, reward = jax.vmap(one_step)(state.data, action)

        steps = state.info["steps"] + 1.0
        truncation = (steps >= self.episode_length).astype(jnp.float32)
        done = truncation                                # 本环境只有超时终止

        # ---- 自动重置：where 会在 done 处切断梯度 ----
        splitted = jax.vmap(lambda k: jax.random.split(k, 2))(state.info["rng"])
        rng, sub = jnp.moveaxis(splitted, 1, 0)   # (num_envs,2,2) -> (2,num_envs,2)
        # 按需生成重置状态（此前每步无条件 vmap 一次 _reset_one，内含 mjx.forward，
        # 前向开销翻倍且撑大 BPTT 图；lax.cond 只执行被选中的分支）
        need_reset = jnp.any(done > 0)

        def _fresh(_):
            d = jax.vmap(self._reset_one)(sub)
            return d, jax.vmap(self._obs)(d)

        def _reuse(_):
            return state.info["reset_data"], state.info["reset_obs"]

        reset_data, reset_obs = jax.lax.cond(need_reset, _fresh, _reuse,
                                             operand=None)

        def where_done(x, y):
            d = jnp.reshape(done, [done.shape[0]] + [1] * (x.ndim - 1))
            return jnp.where(d > 0, x, y)

        new_data = jax.tree_util.tree_map(where_done, reset_data, data)
        new_obs = where_done(reset_obs, obs)

        # 回合回报统计：done 时结算并存档
        ep_ret = state.info["ep_ret"] + reward
        info = dict(state.info)
        info.update(rng=rng,
                    steps=jnp.where(done > 0, 0.0, steps),
                    truncation=truncation,
                    obs_before_reset=obs,
                    ep_ret=jnp.where(done > 0, 0.0, ep_ret),
                    last_ep_ret=jnp.where(done > 0, ep_ret,
                                          state.info["last_ep_ret"]),
                    reset_data=reset_data,
                    reset_obs=reset_obs)
        return MJXState(data=new_data, obs=new_obs, reward=reward,
                        done=done, info=info)


def make_env(cfg):
    """按配置构建环境。`use_pure_mjx_cartpole=True` 时用手写纯 MJX Cartpole。"""
    if cfg.env_name.startswith("pg:"):    # mujoco_playground dm_control 套件
        from .playground_env import PlaygroundEnv
        return PlaygroundEnv(cfg.env_name[3:], cfg.num_envs)
    if cfg.env_name == "ball_wall":
        from .ball_wall import BallWallEnv
        return BallWallEnv(cfg.num_envs)
    if getattr(cfg, "use_pure_mjx_cartpole", False) and cfg.env_name == "cartpole":
        return MJXCartpole(cfg.num_envs, cfg.episode_length)
    return BraxMJXEnv(cfg.env_name, cfg.num_envs, cfg.episode_length,
                      backend=cfg.backend)
