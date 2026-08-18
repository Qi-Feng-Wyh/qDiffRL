"""
策略评测 + 可视化脚本。

功能：
  1. 加载 train.py 保存的 checkpoint，在 MJX 环境里跑 N 条完整 episode；
  2. 同时报告 **随机评测**（从策略分布采样动作）与 **确定性评测**（直接用均值动作），
     这与 RPO 论文 Table 1 / Table 6 的两套评测协议一致；
  3. 画出回报分布直方图 + 单条 episode 的逐步奖励曲线；
  4. 渲染一条轨迹：优先输出 mp4（需要 mujoco.Renderer 与离屏 GL），
     否则回退到 brax 的交互式 HTML（纯 CPU、无需 GL，服务器上最稳）。

用法：
    python evaluate.py --ckpt runs/rpo_ant_seed0.pkl --episodes 128 --video mp4
    python evaluate.py --ckpt runs/gippo_cartpole_seed0.pkl --video html
    python evaluate.py --ckpt runs/*.pkl --episodes 64 --no-video     # 批量对比
"""

import argparse
import glob
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from algos import make_agent
from algos.common import load_checkpoint
from algos.networks import RunningMeanStd, rms_normalize


# ----------------------------------------------------------------------
# 评测 rollout（整段 jit，一次跑满 num_envs 条 episode）
# ----------------------------------------------------------------------
def _to_env_action(cfg, act):
    """与训练时 rollout 一致的动作压缩：PPO 用 tanh，其余硬 clip。"""
    return jnp.tanh(act) if cfg.squash_actions else jnp.clip(act, -1.0, 1.0)


def make_eval_fn(agent, deterministic: bool, num_steps: int):
    cfg, env, actor = agent.cfg, agent.env, agent.actor

    def eval_fn(actor_params, rms, key):
        k_reset, k_act = jax.random.split(key)
        state0 = env.reset(k_reset)

        def body(carry, k):
            state, ep_ret, ep_len = carry
            obs = rms_normalize(state.obs, rms) if cfg.normalize_obs else state.obs
            mu, log_std = actor.apply(actor_params, obs)
            if deterministic:
                act = mu                                   # 确定性评测：直接用均值
            else:
                eps = jax.random.normal(k, mu.shape)        # 随机评测：从分布采样
                act = mu + jnp.exp(log_std) * eps
            nstate = env.step(state, _to_env_action(cfg, act))
            done = nstate.done
            ep_ret_new = ep_ret + nstate.reward
            ep_len_new = ep_len + 1.0
            out = dict(done=done,
                       ep_ret=ep_ret_new * done,           # 只在结束时记录
                       ep_len=ep_len_new * done,
                       reward=nstate.reward)
            zero = jnp.zeros_like(ep_ret_new)
            return (nstate,
                    jnp.where(done > 0, zero, ep_ret_new),
                    jnp.where(done > 0, zero, ep_len_new)), out

        n_env = cfg.num_envs
        init = (state0, jnp.zeros((n_env,)), jnp.zeros((n_env,)))
        _, outs = jax.lax.scan(body, init, jax.random.split(k_act, num_steps))
        return outs

    return jax.jit(eval_fn)


def run_eval(agent, actor_params, rms, key, episodes, deterministic):
    """跑到收集够 `episodes` 条完整 episode 为止。"""
    cfg = agent.cfg
    eval_fn = make_eval_fn(agent, deterministic, cfg.episode_length)
    returns, lengths, reward_traces = [], [], None
    while len(returns) < episodes:
        key, sub = jax.random.split(key)
        outs = jax.device_get(eval_fn(actor_params, rms, sub))
        mask = outs["done"] > 0
        returns.extend(outs["ep_ret"][mask].tolist())
        lengths.extend(outs["ep_len"][mask].tolist())
        if reward_traces is None:
            reward_traces = outs["reward"][:, 0]     # 第 0 号环境的逐步奖励
    return (np.array(returns[:episodes]), np.array(lengths[:episodes]),
            reward_traces, key)


# ----------------------------------------------------------------------
# 轨迹渲染
# ----------------------------------------------------------------------
def record_trajectory(agent, actor_params, rms, key, num_steps):
    """录制第 0 号环境的物理状态序列（确定性策略）。"""
    cfg, env, actor = agent.cfg, agent.env, agent.actor
    take0 = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    @jax.jit
    def rollout(key):
        state0 = env.reset(key)
        # brax 环境的物理状态在 pipeline_state，纯 MJX 环境在 data
        field = "pipeline_state" if hasattr(state0, "pipeline_state") else "data"

        def body(state, _):
            obs = rms_normalize(state.obs, rms) if cfg.normalize_obs else state.obs
            mu, _ = actor.apply(actor_params, obs)
            nstate = env.step(state, _to_env_action(cfg, mu))
            return nstate, take0(getattr(nstate, field))

        _, traj = jax.lax.scan(body, state0, None, length=num_steps)
        return traj

    return jax.device_get(rollout(key))


def save_mp4(agent, traj, path, fps=30, width=640, height=480):
    """用 mujoco.Renderer 逐帧渲染。需要 EGL/OSMesa 等离屏 GL 后端。"""
    import mujoco
    import imageio

    mj_model = agent.env.mj_model
    if mj_model is None:
        raise RuntimeError("该环境没有暴露 mj_model，无法用 MuJoCo 渲染器出图")
    data = mujoco.MjData(mj_model)
    # brax 路径 traj 直接有 qpos/qvel；playground 路径嵌套在 .data（mjx.Data）里
    traj_data = traj if hasattr(traj, "qpos") else traj.data
    qpos = np.asarray(traj_data.qpos)
    qvel = np.asarray(traj_data.qvel)
    n = qpos.shape[0]
    stride = max(int(round(1.0 / (fps * agent.env.dt))), 1)

    # 仅当环境显式声明使用自带相机（如手写 cartpole 的固定近景相机）时
    # 才用模型相机；其余环境（brax / playground）一律用默认自由相机
    use_model_cam = getattr(agent.env, "use_model_camera", False)
    cam = 0 if (mj_model.ncam > 0 and use_model_cam) else -1
    frames = []
    with mujoco.Renderer(mj_model, height, width) as renderer:
        for i in range(0, n, stride):
            data.qpos[:] = qpos[i]
            data.qvel[:] = qvel[i]
            mujoco.mj_forward(mj_model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    imageio.mimsave(path, frames, fps=fps)
    return path


def save_html(agent, traj, path):
    """brax 的交互式 HTML 播放器，纯 CPU，无需 GL。"""
    from brax.io import html

    n = jax.tree_util.tree_leaves(traj)[0].shape[0]
    states = [jax.tree_util.tree_map(lambda x: x[i], traj) for i in range(n)]
    saver = getattr(html, "save", None) or getattr(html, "save_html")
    saver(path, agent.env.sys, states)
    return path


# ----------------------------------------------------------------------
# 画图
# ----------------------------------------------------------------------
def plot_eval(tag, stoch, deter, reward_trace, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(stoch, bins=25, alpha=0.7, label="stochastic", color="tab:blue")
    axes[0].hist(deter, bins=25, alpha=0.7, label="deterministic", color="tab:orange")
    axes[0].set_xlabel("Episode Return")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{tag}\nreturn distribution")
    axes[0].legend()

    axes[1].boxplot([stoch, deter], tick_labels=["stochastic", "deterministic"])
    axes[1].set_ylabel("Episode Return")
    axes[1].set_title("evaluation protocols")

    axes[2].plot(reward_trace, lw=0.8, color="tab:green")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("per-step reward (env #0)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ----------------------------------------------------------------------
def evaluate_one(ckpt_path, args):
    cfg, payload = load_checkpoint(ckpt_path)
    # 评测时用独立的并行环境数，且不影响训练配置
    cfg = replace(cfg, num_envs=args.eval_envs, seed=args.seed)
    agent = make_agent(cfg)

    actor_params = jax.device_put(payload["actor_params"])
    r = payload["rms"]
    rms = RunningMeanStd(jnp.asarray(r["mean"]), jnp.asarray(r["var"]),
                         jnp.asarray(r["count"]))

    key = jax.random.PRNGKey(args.seed)
    stoch, len_s, trace, key = run_eval(agent, actor_params, rms, key,
                                        args.episodes, deterministic=False)
    deter, len_d, _, key = run_eval(agent, actor_params, rms, key,
                                    args.episodes, deterministic=True)

    tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    print(f"\n=== {tag} (训练步数 {payload['step']:,}) ===")
    print(f"  随机评测   : {stoch.mean():10.2f} ± {stoch.std():.2f}  "
          f"(n={len(stoch)}, 平均长度 {len_s.mean():.0f})")
    print(f"  确定性评测 : {deter.mean():10.2f} ± {deter.std():.2f}  "
          f"(n={len(deter)}, 平均长度 {len_d.mean():.0f})")

    os.makedirs(args.out, exist_ok=True)
    png = plot_eval(tag, stoch, deter, trace,
                    os.path.join(args.out, f"eval_{tag}.png"))
    print(f"  评测图表   : {png}")

    if args.video != "none":
        key, k_rec = jax.random.split(key)
        traj = record_trajectory(agent, actor_params, rms, k_rec,
                                 min(cfg.episode_length, args.video_steps))
        base = os.path.join(args.out, f"rollout_{tag}")
        try:
            if args.video == "mp4":
                print(f"  轨迹视频   : {save_mp4(agent, traj, base + '.mp4')}")
            else:
                print(f"  轨迹动画   : {save_html(agent, traj, base + '.html')}")
        except Exception as e:      # 无 GL / brax 版本差异时回退
            print(f"  [警告] {args.video} 渲染失败（{type(e).__name__}: {e}）")
            try:
                print(f"  已回退到 HTML: {save_html(agent, traj, base + '.html')}")
            except Exception as e2:
                print(f"  [警告] HTML 渲染同样失败：{e2}")

    return dict(tag=tag, stochastic=stoch, deterministic=deter)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="train.py 保存的 .pkl，支持通配符与多个文件")
    p.add_argument("--episodes", type=int, default=128,
                   help="每种评测协议采集的完整 episode 数")
    p.add_argument("--eval-envs", dest="eval_envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", default="html", choices=["mp4", "html", "none"])
    p.add_argument("--no-video", dest="video", action="store_const", const="none")
    p.add_argument("--video-steps", dest="video_steps", type=int, default=600)
    p.add_argument("--out", default="eval")
    args = p.parse_args()

    paths = []
    for pattern in args.ckpt:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    results = [evaluate_one(path, args) for path in paths]

    if len(results) > 1:            # 多个 checkpoint 时额外画一张横向对比图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.6 * len(results) + 3, 4.5))
        ax.boxplot([r["stochastic"] for r in results],
                   tick_labels=[r["tag"] for r in results])
        ax.set_ylabel("Episode Return (stochastic)")
        ax.set_title("Final performance comparison")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        out = os.path.join(args.out, "comparison.png")
        fig.savefig(out, dpi=130)
        print(f"\n横向对比图：{out}")


if __name__ == "__main__":
    main()
