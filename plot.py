"""
训练曲线可视化：把 runs/ 下的 json 日志画成学习曲线与算法诊断图。

用法：
    python plot.py --runs "runs/*_ant_*.json"                 # 同任务三算法对比
    python plot.py --runs "runs/*.json" --group-by env        # 按任务分子图
    python plot.py --runs "runs/gippo_ant_*.json" --diagnostics

生成两类图（文件名自动携带算法/环境/种子，如
learning_curves_gippo_cartpole_seed0+rpo_cartpole_seed0.png，避免互相覆盖）：
  1. learning_curves_<tag>.png —— 每个 (算法, 任务) 的平均 episode 回报
     （多 seed 取均值，阴影为标准差），横轴是环境交互步数；
  2. diagnostics_<tag>.png   —— 各算法专属的内部指标随训练的变化：
       GI-PPO : alpha、psi 的极值、R_alpha、out-of-range-ratio
       RPO    : 有效样本率、最大重要性比、KL、熵
       PPO    : clip_frac、KL、熵
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"gippo": "tab:blue", "rpo": "tab:red", "ppo": "tab:green",
          "ddcg": "tab:purple", "ivwh": "tab:orange", "aobg": "tab:brown"}

# (日志字段, 图标题, 是否用对数纵轴)。标题用英文以避免缺字体时显示成方块。
DIAGNOSTICS = {
    "gippo": [("alpha", "alpha  (analytic-gradient strength)", True),
              ("psi_min", "psi min  (variance criterion)", False),
              ("psi_max", "psi max  (variance criterion)", False),
              ("R_alpha", "R_alpha  (bias criterion)", False),
              ("oorr", "out-of-range-ratio", False),
              ("grad_a_norm", "|dA/da|", True)],
    "rpo": [("eff_sample_ratio", "effective sample ratio", False),
            ("max_iw", "max importance weight", True),
            ("kl", "KL(old || new)", True),
            ("entropy", "policy entropy", False),
            ("actor_lr", "actor learning rate", True),
            ("grad_a_norm", "|dR/da|", True)],
    "ppo": [("clip_frac", "clip fraction", False),
            ("kl", "KL(old || new)", True),
            ("entropy", "policy entropy", False),
            ("actor_lr", "actor learning rate", True)],
    "ivwh": [("alpha_mu", "alpha mu (1st-order weight)", False),
             ("alpha_std", "alpha std (1st-order weight)", False),
             ("v0_mean", "0th-order grad variance", True),
             ("v1_mean", "1st-order grad variance", True),
             ("kl", "KL(old || new)", True),
             ("entropy", "policy entropy", False)],
    "aobg": [("alpha", "alpha (1st-order weight)", False),
             ("feasible", "Bernstein feasible (eps<=gamma)", False),
             ("B", "bias surrogate ||g1-g0||", True),
             ("eps_conf", "confidence radius eps", True),
             ("ret_mean", "return under perturbation", False),
             ("actor_lr", "actor learning rate", True)],
    "ddcg": [("alpha", "alpha (1st-order weight)", False),
             ("test_pass", "discontinuity test pass", False),
             ("v0", "0th-order grad variance", True),
             ("v1", "1st-order grad variance", True),
             ("ret_mean", "return under perturbation", False),
             ("actor_lr", "actor learning rate", True)],
}


def setup_font():
    """若系统装有中文字体则启用，否则保持默认（图中标签本身都是英文）。"""
    from matplotlib import font_manager
    plt.rcParams["axes.unicode_minus"] = False   # 用 ASCII 减号，避免缺字形
    for name in ("Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                 "SimHei", "PingFang SC", "Microsoft YaHei"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            return name
    return None

FNAME = re.compile(
    r"(?P<algo>gippo|rpo|ppo|ddcg|aobg|ivwh)_(?P<env>\w+?)_seed(?P<seed>\d+)")


def load_runs(patterns):
    """返回 {(algo, env): [(seed, history), ...]}。"""
    runs = defaultdict(list)
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    for path in paths:
        m = FNAME.search(os.path.basename(path))
        if m is None:
            print(f"[跳过] 文件名不符合 <algo>_<env>_seed<n>.json：{path}")
            continue
        with open(path) as f:
            runs[(m["algo"], m["env"])].append((int(m["seed"]), json.load(f)))
    return runs


def seed_tag(seeds):
    """种子列表 -> 文件名片段：单种子 'seed0'，多种子 'seed0-1-2'。"""
    return "seed" + "-".join(str(s) for s in sorted({int(s) for s in seeds}))


def run_tag(algo, env, histories):
    """某个 (algo, env) 组合的文件名片段，如 rpo_cartpole_seed0-1。"""
    return f"{algo}_{env}_{seed_tag(s for s, _ in histories)}"


def series(history, key):
    xs = [h["env_steps"] for h in history if key in h]
    ys = [h[key] for h in history if key in h]
    return np.array(xs), np.array(ys)


def aggregate(histories, key):
    """多 seed 对齐到最短长度后求均值与标准差。"""
    curves = [series(h, key) for h in histories]
    curves = [(x, y) for x, y in curves if len(y)]
    if not curves:
        return None, None, None
    n = min(len(y) for _, y in curves)
    x = curves[0][0][:n]
    ys = np.stack([y[:n] for _, y in curves])
    return x, ys.mean(0), ys.std(0)


def smooth(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_learning_curves(runs, out_dir, window, group_by):
    envs = sorted({env for _, env in runs})
    algos = sorted({algo for algo, _ in runs})
    panels = envs if group_by == "env" else algos
    ncol = min(len(panels), 3)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4 * nrow),
                             squeeze=False)

    for i, panel in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        for (algo, env), histories in sorted(runs.items()):
            if (group_by == "env" and env != panel) or \
               (group_by == "algo" and algo != panel):
                continue
            x, mean, std = aggregate([h for _, h in histories], "episode_return")
            if x is None:
                continue
            k = len(x) - len(smooth(mean, window))
            x_s = x[k:]
            m_s, s_s = smooth(mean, window), smooth(std, window)
            label = f"{algo} ({len(histories)} seeds)" if group_by == "env" \
                else f"{env} ({len(histories)} seeds)"
            ax.plot(x_s, m_s, label=label, color=COLORS.get(algo), lw=1.8)
            ax.fill_between(x_s, m_s - s_s, m_s + s_s, alpha=0.2,
                            color=COLORS.get(algo))
        ax.set_title(panel)
        ax.set_xlabel("Environment Step")
        ax.set_ylabel("Episode Return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    # 文件名带上各组合的算法/环境/种子，避免不同批次的结果互相覆盖
    tag = "+".join(run_tag(a, e, hs) for (a, e), hs in sorted(runs.items()))
    out = os.path.join(out_dir, f"learning_curves_{tag}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_diagnostics(runs, out_dir, window):
    outs = []
    for (algo, env), histories in sorted(runs.items()):
        specs = DIAGNOSTICS.get(algo, [])
        specs = [s for s in specs if any(s[0] in h for h in histories[0][1])]
        if not specs:
            continue
        ncol = min(len(specs), 3)
        nrow = int(np.ceil(len(specs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow),
                                 squeeze=False)
        for i, (key, title, logy) in enumerate(specs):
            ax = axes[i // ncol][i % ncol]
            x, mean, std = aggregate([h for _, h in histories], key)
            if x is None:
                ax.axis("off")
                continue
            k = len(x) - len(smooth(mean, window))
            m_s, s_s = smooth(mean, window), smooth(std, window)
            ax.plot(x[k:], m_s, color=COLORS.get(algo), lw=1.5)
            ax.fill_between(x[k:], m_s - s_s, m_s + s_s, alpha=0.2,
                            color=COLORS.get(algo))
            if logy:
                ax.set_yscale("log")
            # 给几个有明确参考线的指标画出阈值
            if key.startswith("psi"):
                ax.axhline(1.0, ls="--", c="gray", lw=0.8)
            if key in ("R_alpha",):
                ax.axhline(0.0, ls="--", c="gray", lw=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Environment Step")
            ax.grid(alpha=0.3)
        for j in range(len(specs), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{algo} on {env}", fontsize=12)
        fig.tight_layout()
        out = os.path.join(out_dir, f"diagnostics_{run_tag(algo, env, histories)}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outs.append(out)
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/*.json"])
    p.add_argument("--out", default="figures")
    p.add_argument("--window", type=int, default=5, help="滑动平均窗口")
    p.add_argument("--group-by", dest="group_by", default="env",
                   choices=["env", "algo"], help="子图按任务分还是按算法分")
    p.add_argument("--diagnostics", action="store_true",
                   help="额外输出各算法的内部诊断指标图")
    args = p.parse_args()

    setup_font()
    runs = load_runs(args.runs)
    if not runs:
        print("没有找到任何日志文件。")
        return
    os.makedirs(args.out, exist_ok=True)
    for (algo, env), hs in sorted(runs.items()):
        print(f"  {algo:6s} {env:10s} {len(hs)} seeds")

    print(f"\n学习曲线：{plot_learning_curves(runs, args.out, args.window, args.group_by)}")
    if args.diagnostics:
        for out in plot_diagnostics(runs, args.out, args.window):
            print(f"诊断图：{out}")


if __name__ == "__main__":
    main()
