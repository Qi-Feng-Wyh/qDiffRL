"""
三个算法共享的基础设施：训练状态、优化器、minibatch 训练循环、Agent 基类。
"""

import os
import pickle
from dataclasses import asdict, fields
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
from flax import struct

from .config import Config
from .envs import make_env
from .networks import (Critic, GaussianActor, RunningMeanStd, rms_update)
from .rollout import make_rollout_fn


# ----------------------------------------------------------------------
# 训练状态：算法专属的额外量（如 GI-PPO 的 alpha）放在 extra 里
# ----------------------------------------------------------------------
@struct.dataclass
class TrainState:
    actor_params: Any
    critic_params: Any
    critic_target_params: Any     # 不用 target critic 时与 critic_params 相同
    actor_opt: Any
    critic_opt: Any
    rms: RunningMeanStd
    actor_lr: jnp.ndarray
    step: jnp.ndarray
    extra: Dict[str, Any]


# ----------------------------------------------------------------------
# Checkpoint：只保存评测需要的东西（网络参数 + 观测归一化统计量 + 配置）
# ----------------------------------------------------------------------
def save_checkpoint(path: str, ts: "TrainState", cfg: Config):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rms = jax.device_get(ts.rms)
    payload = dict(
        cfg=asdict(cfg),
        actor_params=jax.device_get(ts.actor_params),
        critic_params=jax.device_get(ts.critic_params),
        rms=dict(mean=rms.mean, var=rms.var, count=rms.count),
        step=int(ts.step),
    )
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_checkpoint(path: str):
    """返回 (cfg, payload)。cfg 中的 Sequence 字段会被还原成 tuple。"""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    valid = {f.name for f in fields(Config)}
    kwargs = {}
    for k, v in payload["cfg"].items():
        if k not in valid:
            continue
        kwargs[k] = tuple(v) if isinstance(v, list) else v
    return Config(**kwargs), payload


def make_optimizer(lr: float, cfg: Config):
    """用 inject_hyperparams 包一层，方便在 jit 内部动态改学习率。"""
    b1, b2 = cfg.adam_betas

    def _make(learning_rate):
        if cfg.optimizer == "adamw":
            core = optax.adamw(learning_rate, b1=b1, b2=b2,
                               weight_decay=cfg.weight_decay)
        else:
            core = optax.adam(learning_rate, b1=b1, b2=b2)
        return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), core)

    return optax.inject_hyperparams(_make)(learning_rate=lr)


def set_lr(opt_state, lr):
    return opt_state._replace(hyperparams={**opt_state.hyperparams,
                                           "learning_rate": lr})


def schedule_lr(cfg: Config, lr0: float, progress):
    """progress in [0,1]。adaptive 由各算法自行按 KL 调整，这里原样返回。"""
    lr0 = jnp.asarray(lr0, jnp.float32)
    p = jnp.clip(progress, 0.0, 1.0)
    if cfg.lr_schedule == "linear":
        return jnp.maximum(lr0 * (1.0 - p), cfg.lr_min)
    if cfg.lr_schedule == "exponential":
        return lr0 * (cfg.lr_min / lr0) ** p
    return lr0


def polyak(target, online, tau):
    """target <- (1-tau)*target + tau*online。"""
    return jax.tree_util.tree_map(lambda t, o: (1.0 - tau) * t + tau * o,
                                  target, online)


def run_epochs(loss_fn, tx, params, opt_state, data, key, epochs, batch_size):
    """在 data 上跑 epochs 轮 minibatch SGD，全程 lax.scan 以便整体 jit。

    返回 (params, opt_state, mean_loss)。batch_size <= 0 表示全批量。
    """
    n = jax.tree_util.tree_leaves(data)[0].shape[0]
    bs = n if batch_size <= 0 else batch_size
    nmb = max(n // bs, 1)
    bs = n // nmb
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def epoch_step(carry, k):
        params, opt_state = carry
        perm = jax.random.permutation(k, n)
        shuffled = jax.tree_util.tree_map(
            lambda x: x[perm][: nmb * bs].reshape((nmb, bs) + x.shape[1:]), data)

        def mb_step(carry, batch):
            params, opt_state = carry
            (loss, aux), grads = grad_fn(params, batch)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        (params, opt_state), losses = jax.lax.scan(
            mb_step, (params, opt_state), shuffled)
        return (params, opt_state), jnp.mean(losses)

    (params, opt_state), losses = jax.lax.scan(
        epoch_step, (params, opt_state), jax.random.split(key, epochs))
    return params, opt_state, jnp.mean(losses)


# ----------------------------------------------------------------------
# Agent 基类
# ----------------------------------------------------------------------
class BaseAgent:
    """负责环境/网络/优化器的构建、初始化与训练主循环。

    子类需要实现 `_update(ts, env_state, key)`，并可覆盖：
      * `objective`  : 传给 rollout 的梯度目标（"gae" / "shac" / "none"）
      * `_extra_init`: 返回算法专属的额外状态
      * `log_keys`   : 日志中打印的指标
    """

    objective = "none"
    log_keys = ("episode_return", "actor_lr")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env = make_env(cfg)
        self.actor = GaussianActor(tuple(cfg.actor_hidden), self.env.action_size,
                                   cfg.init_log_std, cfg.activation, cfg.layer_norm)
        self.critic = Critic(tuple(cfg.critic_hidden), cfg.num_critics,
                             cfg.activation, cfg.layer_norm)
        self.actor_tx = make_optimizer(cfg.actor_lr, cfg)
        self.critic_tx = make_optimizer(cfg.critic_lr, cfg)
        self.rollout_fn = make_rollout_fn(self.env, self.actor, self.critic,
                                          cfg, self.objective)
        self._jit_update = jax.jit(self._update)

    # ---------------- 初始化 ----------------
    def _extra_init(self, actor_params, critic_params):
        return {}

    def init(self, key):
        cfg = self.cfg
        k_a, k_c, k_e = jax.random.split(key, 3)
        dummy = jnp.zeros((1, self.env.obs_size))
        actor_params = self.actor.init(k_a, dummy)
        critic_params = self.critic.init(k_c, dummy)
        ts = TrainState(
            actor_params=actor_params,
            critic_params=critic_params,
            critic_target_params=critic_params,
            actor_opt=self.actor_tx.init(actor_params),
            critic_opt=self.critic_tx.init(critic_params),
            rms=RunningMeanStd.create(self.env.obs_size),
            actor_lr=jnp.asarray(cfg.actor_lr, jnp.float32),
            step=jnp.asarray(0, jnp.int32),
            extra=self._extra_init(actor_params, critic_params),
        )
        return ts, self.env.reset(k_e)

    # ---------------- 公共小工具 ----------------
    def value_params(self, ts: TrainState):
        """rollout 中用于计算 V 的参数（是否使用 target critic）。"""
        return ts.critic_target_params if self.cfg.use_target_critic \
            else ts.critic_params

    def progress(self, ts: TrainState):
        return ts.step / float(self.cfg.total_steps)

    def normalized_adv(self, adv):
        if not self.cfg.normalize_adv:
            return adv, 1.0
        std = jnp.std(adv) + 1e-8
        return (adv - jnp.mean(adv)) / std, std

    def critic_loss(self, params, batch):
        """所有 critic head 一起回归到 TD(lambda) 目标。"""
        v = self.critic.apply(params, batch["obs"])          # (B, C)
        return jnp.mean(jnp.square(v - batch["ret"][:, None])), {}

    def update_critic(self, ts, batch, key):
        cfg = self.cfg
        n = batch["obs"].shape[0]
        bs = max(n // max(cfg.critic_minibatches, 1), 1)
        critic_params, critic_opt, loss = run_epochs(
            self.critic_loss, self.critic_tx, ts.critic_params, ts.critic_opt,
            dict(obs=batch["obs"], ret=batch["ret"]), key,
            cfg.critic_epochs, bs)
        target = polyak(ts.critic_target_params, critic_params, cfg.critic_tau) \
            if cfg.use_target_critic else critic_params
        return critic_params, critic_opt, target, loss

    def update_rms(self, ts, batch):
        return rms_update(ts.rms, batch["raw_obs"]) if self.cfg.normalize_obs \
            else ts.rms

    # ---------------- 训练主循环 ----------------
    def _update(self, ts, env_state, key):
        raise NotImplementedError

    def train(self, log_fn=None):
        cfg = self.cfg
        key = jax.random.PRNGKey(cfg.seed)
        ts, env_state = self.init(key)
        key = jax.random.fold_in(key, 1)

        steps_per_iter = cfg.num_envs * cfg.horizon
        num_iters = cfg.total_steps // steps_per_iter
        history = []
        for it in range(1, num_iters + 1):
            ts, env_state, metrics, key = self._jit_update(ts, env_state, key)
            if it % cfg.log_interval == 0 or it == 1:
                m = {k: float(v) for k, v in jax.device_get(metrics).items()}
                m["iter"], m["env_steps"] = it, it * steps_per_iter
                history.append(m)
                if log_fn is not None:
                    log_fn(m)
                else:
                    body = " ".join(f"{k}={m[k]:.4g}" for k in self.log_keys
                                    if k in m)
                    print(f"[{cfg.algo}|{it:5d}] steps={m['env_steps']:>9d} {body}")
        return ts, history
