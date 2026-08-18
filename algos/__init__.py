# ====== JAX 持久化编译缓存 ======
# 训练/评测的整段 jit（含 MJX 反传）在每个新进程里都要重新编译，耗时数分钟。
# 开启编译缓存后，相同配置（算法 / 环境 / 网络形状等决定计算图的量）的重复运行
# 直接命中缓存，跳过编译。注意：改动网络结构、num_envs、horizon 或升级 jax
# 会使缓存失效并自动重编译。
import os as _os

import jax as _jax

_jax.config.update(
    "jax_compilation_cache_dir",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                  ".jax_cache"))
# 默认只缓存编译耗时 >1s 的模块；本项目每次编译都是分钟级，放宽阈值无妨
_jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

from .config import Config, get_config, ALGO_DEFAULTS, ENV_DEFAULTS
from .common import BaseAgent, TrainState
from .gippo import GIPPO
from .rpo import RPO
from .ppo import PPO
from .ddcg import DDCG
from .aobg import AoBG
from .ivwh import IVWH
from .envs import make_env, BraxMJXEnv, MJXCartpole

AGENTS = {"gippo": GIPPO, "rpo": RPO, "ppo": PPO, "ddcg": DDCG,
          "aobg": AoBG, "ivwh": IVWH}


def make_agent(cfg):
    """按 cfg.algo 构建对应的 Agent。"""
    return AGENTS[cfg.algo](cfg)


__all__ = ["Config", "get_config", "ALGO_DEFAULTS", "ENV_DEFAULTS",
           "BaseAgent", "TrainState", "GIPPO", "RPO", "PPO", "DDCG", "IVWH",
           "AoBG", "AGENTS", "make_agent", "make_env", "BraxMJXEnv",
           "MJXCartpole"]
