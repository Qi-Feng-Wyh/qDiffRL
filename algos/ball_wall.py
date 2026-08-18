"""
Ball with Wall 环境（Suh et al., ICML 2022 / DDCG 论文 Part I 的标志性任务）。

任务：从小车般的抛出点以固定初速抛球，前方有一堵墙；撞墙为法向非弹性碰撞
（水平速度清零，沿墙滑落）。回报 = 落点的水平距离。擦墙而过与撞墙之间
回报发生跳变——这是论文用来展示"1 阶梯度经验偏差"的最小不连续系统。

映射到框架的 RL 接口（单决策 episode）：
* 观测: [x0, y0]（抛出点，y0 带 ±0.05 噪声）
* 动作: 1 维，映射为抛出角 theta = 0.7 + 0.5 * clip(a, -1, 1)（弧度）
* step(): 用 lax.scan 微步模拟整个飞行过程（dt=0.005, 400 步），全程可微；
  每步即一个完整 episode（done=1，真终止无自举）
* 回报: 落点水平距离（约 4~10m）

由于观测接近常量，策略的有效自由度就是抛出角——与论文"直接优化 theta"
的设定等价：DDCG/AoBG 的参数扰动、PPO/RPO 的动作空间估计都直接对应
论文里的估计器语义。

物理参数为自建（原文参数未公开）：起点 (0, ~1)、v0=10 m/s、
墙 x=4m 高 2m。45° 弹道在 x=4 处高约 3.4m（越墙），约 27° 以下撞墙——
不连续崖壁在 theta ≈ 27° 附近。
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class BallWallState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    info: Dict[str, Any] = struct.field(pytree_node=True, default_factory=dict)


class BallWallEnv:
    """单决策 Ball with Wall：一次动作 = 一次抛出，回报 = 落点距离。"""

    V0 = 10.0           # 初速度 m/s
    G = 9.8             # 重力
    DT = 0.005          # 微步长
    T_STEPS = 400       # 每次抛出的微步数（2s）
    WALL_X = 4.0        # 墙位置
    WALL_H = 2.0        # 墙高
    THETA_BASE = 0.7    # 动作 -> 角度映射：theta = BASE + SCALE * a
    THETA_SCALE = 0.5

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.obs_size = 2
        self.action_size = 1
        self.episode_length = 1        # 单决策 episode
        self.mj_model = None           # 无 MuJoCo 模型（视频渲染不可用，评测数值正常）
        self.dt = 0.02                 # 仅用于渲染 fps 换算的占位

    # ---------------- 物理：可微飞行模拟 ----------------
    def _flight_distance(self, theta, y0):
        """抛出角 theta、初始高度 y0 -> 落点水平距离（可微）。"""
        vx0 = self.V0 * jnp.cos(theta)
        vy0 = self.V0 * jnp.sin(theta)

        def f(s, _):
            x, y, vx, vy, landed, x_land = s
            vy = vy - self.G * self.DT
            x2 = x + vx * self.DT
            y2 = y + vy * self.DT
            # 撞墙：跨越墙面且高度低于墙顶 -> 法向非弹性（vx 清零，贴在墙上）
            hit = (~landed) & (x < self.WALL_X) & (x2 >= self.WALL_X) \
                & (y2 < self.WALL_H)
            x2 = jnp.where(hit, self.WALL_X, x2)
            vx = jnp.where(hit, 0.0, vx)
            # 落地：y 触底 -> 停止，记录落点
            land = (~landed) & (y2 <= 0.0)
            x_land = jnp.where(land, x2, x_land)
            y2 = jnp.where(land, 0.0, y2)
            vx = jnp.where(land, 0.0, vx)
            vy = jnp.where(land, 0.0, vy)
            landed = landed | land
            return (x2, y2, vx, vy, landed, x_land), None

        s0 = (0.0, y0, vx0, vy0, jnp.array(False), 0.0)
        (xf, _, _, _, landed, x_land), _ = jax.lax.scan(
            f, s0, None, length=self.T_STEPS)
        return jnp.where(landed, x_land, xf)

    # ---------------- 对外接口 ----------------
    def _obs_of(self, y0):
        return jnp.stack([jnp.zeros_like(y0), y0], axis=-1)

    def reset(self, key):
        k1, k2 = jax.random.split(key)
        y0 = 1.0 + 0.05 * jax.random.uniform(
            k1, (self.num_envs,), minval=-1.0, maxval=1.0)
        obs = self._obs_of(y0)
        zeros = jnp.zeros((self.num_envs,))
        info = {
            "rng": jax.random.split(k2, self.num_envs),
            "steps": zeros,
            "truncation": zeros,
            "obs_before_reset": obs,
            "ep_ret": zeros,
            "last_ep_ret": zeros,
        }
        return BallWallState(obs=obs, reward=zeros, done=zeros, info=info)

    def step(self, state: BallWallState, action):
        theta = self.THETA_BASE + self.THETA_SCALE * jnp.clip(action, -1.0, 1.0)
        dist = jax.vmap(self._flight_distance)(theta.squeeze(-1),
                                               state.obs[:, 1])

        # 下一帧观测（等价于 auto-reset 的新 episode 首帧）
        splitted = jax.vmap(lambda k: jax.random.split(k, 2))(state.info["rng"])
        rng, sub = jnp.moveaxis(splitted, 1, 0)
        y0_new = 1.0 + 0.05 * jax.vmap(
            lambda k: jax.random.uniform(k, (), minval=-1.0, maxval=1.0))(sub)
        new_obs = self._obs_of(y0_new)

        # 批量维度从输入状态推导（DDCG/AoBG 会把状态切成前 M 个环境做扰动
        # rollout，不能写死成 self.num_envs）
        ones = jnp.ones_like(state.done)
        zeros = jnp.zeros_like(state.done)
        ep_ret = state.info["ep_ret"] + dist
        info = dict(state.info)
        info.update(
            rng=rng,
            steps=zeros,                             # 单步 episode，步数恒回零
            truncation=zeros,                        # 真终止，不需自举
            obs_before_reset=new_obs,                # 不被使用（truncation=0）
            ep_ret=zeros,                            # 每步结算
            last_ep_ret=ep_ret,
        )
        return BallWallState(obs=new_obs, reward=dist, done=ones, info=info)
