"""
策略网络 / 价值网络（支持多 critic 集成）/ 高斯分布工具 / 观测归一化。

三个算法共用同一套网络定义：
* GI-PPO  : ELU + 无 LayerNorm + 单 critic（GI-PPO 论文附录 7.4.2）
* RPO     : SiLU + LayerNorm + double critic（RPO 论文 Table 2，沿用 SAPO 设置）
* PPO     : 与 RPO 相同骨架，便于公平对比

策略统一写成重参数化形式
    a = f_theta(eps; s) = mu_theta(s) + sigma_theta * eps,   eps ~ N(0, I)
采用状态无关的对角标准差（sigma 为可学习参数向量），于是
    d f_theta / d eps = diag(sigma) > 0，
既满足 GI-PPO 论文 Equation 3 的可逆性要求，也满足 RPO 论文中
f_theta 可逆（动作再生成 eps_reg = f_theta^{-1}(a; s)，Equation 10）的要求。
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax import struct

LOG2PI = float(np.log(2.0 * np.pi))
HALF_LOG2PIE = float(0.5 * np.log(2.0 * np.pi * np.e))

ACTIVATIONS = {
    "elu": nn.elu,
    "silu": nn.silu,
    "relu": nn.relu,
    "tanh": nn.tanh,
}


class MLP(nn.Module):
    hidden: Sequence[int]
    out_dim: int
    out_scale: float = 1.0
    activation: str = "elu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        for h in self.hidden:
            x = nn.Dense(h, kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)))(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act(x)
        return nn.Dense(self.out_dim,
                        kernel_init=nn.initializers.orthogonal(self.out_scale))(x)


class GaussianActor(nn.Module):
    """输出对角高斯策略的均值与 log_std。"""

    hidden: Sequence[int]
    act_dim: int
    init_log_std: float = -1.0
    activation: str = "elu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, obs):
        mu = MLP(self.hidden, self.act_dim, 0.01, self.activation,
                 self.layer_norm)(obs)
        log_std = self.param(
            "log_std",
            lambda key: jnp.full((self.act_dim,), self.init_log_std, jnp.float32),
        )
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mu, jnp.broadcast_to(log_std, mu.shape)


class Critic(nn.Module):
    """价值网络，输出形状 (..., num_critics)。

    RPO 遵循 SAPO 使用 double critic，并用两者均值作为 TD 目标。
    """

    hidden: Sequence[int]
    num_critics: int = 1
    activation: str = "elu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, obs):
        outs = [MLP(self.hidden, 1, 1.0, self.activation, self.layer_norm)(obs)
                for _ in range(self.num_critics)]
        return jnp.concatenate(outs, axis=-1)


def critic_mean(v):
    """把 (..., C) 的集成输出取均值，得到标量价值估计。"""
    return jnp.mean(v, axis=-1)


# ----------------------------------------------------------------------
# 对角高斯分布工具
# ----------------------------------------------------------------------
def gaussian_logprob(x, mu, log_std):
    z = (x - mu) * jnp.exp(-log_std)
    return -0.5 * jnp.sum(z ** 2 + 2.0 * log_std + LOG2PI, axis=-1)


def logprob_from_eps(eps, log_std):
    """采样时可直接由噪声求 log 概率：log q(eps) - sum(log sigma)。"""
    return -0.5 * jnp.sum(eps ** 2 + 2.0 * log_std + LOG2PI, axis=-1)


def tanh_squash_logp_correction(u):
    """tanh 压缩（squashed Gaussian）的 logp 修正项 Σ log(1 - tanh(u)^2)。

    squash 后动作的 logp = 原高斯 logp - 该修正项（变量代换的 Jacobian）。
    用数值稳定形式 2*(log2 - u - softplus(-2u))（与 SB3 相同），仅在
    cfg.squash_actions=True（PPO）时使用。
    """
    return jnp.sum(2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)), axis=-1)


def gaussian_kl(mu_old, log_std_old, mu_new, log_std_new):
    """KL( N_old || N_new )。RPO 的 KL 正则与 PPO 的自适应学习率都用它。"""
    var_old = jnp.exp(2.0 * log_std_old)
    var_new = jnp.exp(2.0 * log_std_new)
    return jnp.sum(
        log_std_new - log_std_old
        + (var_old + (mu_old - mu_new) ** 2) / (2.0 * var_new)
        - 0.5,
        axis=-1,
    )


def gaussian_entropy(log_std):
    """对角高斯的微分熵 H = sum_i (log sigma_i + 0.5 log(2 pi e))。"""
    return jnp.sum(log_std + HALF_LOG2PIE, axis=-1)


# ----------------------------------------------------------------------
# 观测归一化（running mean / std）
# ----------------------------------------------------------------------
@struct.dataclass
class RunningMeanStd:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    @classmethod
    def create(cls, dim: int):
        return cls(jnp.zeros((dim,)), jnp.ones((dim,)), jnp.array(1e-4))


def rms_update(rms: RunningMeanStd, x: jnp.ndarray) -> RunningMeanStd:
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = x.shape[0]
    delta = batch_mean - rms.mean
    tot = rms.count + batch_count
    new_mean = rms.mean + delta * batch_count / tot
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * rms.count * batch_count / tot
    return RunningMeanStd(new_mean, m2 / tot, tot)


def rms_normalize(x, rms: RunningMeanStd, clip: float = 10.0):
    """归一化是可微的线性变换，解析梯度可以正常穿过。"""
    return jnp.clip((x - rms.mean) / jnp.sqrt(rms.var + 1e-8), -clip, clip)
