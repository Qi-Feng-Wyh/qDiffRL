# GI-PPO / RPO / PPO on MuJoCo-XLA (MJX)

三个算法共享同一套 MJX 环境封装与可微 rollout，可在 **Cartpole、Hopper、Ant** 上直接对比：

| 算法 | 论文 | 解析梯度用法 | 是否穿过模拟器反传 |
| --- | --- | --- | --- |
| **GI-PPO** | Son et al., NeurIPS 2023 | 构造 \(\alpha\)-policy 作为回归目标，再用 REINFORCE 式 PPO 更新兜底 | 是（1 次/迭代） |
| **RPO** | Zhong et al., 2026 | 缓存 action-gradient，通过动作再生成在 M 轮更新中反复复用 | 是（1 次/迭代） |
| **PPO** | 通用 PPO-Clip 约定 | 不使用 | 否 |
| **DDCG** | Onoda et al., ICLR 2026 | 参数空间扰动，Eq.14 不连续检测门控后 IVW 混合 | 是（每扰动 1 次） |
| **AoBG** | Suh et al., ICML 2022 | 参数空间扰动，置信区间约束的 IVW | 是（每扰动 1 次） |
| **IVW-H** | Onoda et al., ICLR 2026 | 逐步逐动作维 IVW 混合 1 阶/0 阶梯度，VJP 回传 | 是（1 次/迭代） |

```
qDiffRL/
├── train.py            # 统一训练入口（--algo gippo|rpo|ppo|ddcg|aobg|ivwh）
├── evaluate.py         # 策略评测 + 轨迹渲染 + 回报分布图
├── plot.py             # 训练曲线与算法诊断指标可视化
└── algos/
    ├── config.py       # 环境 × 算法的超参数矩阵
    ├── networks.py     # 高斯策略 / 集成 critic / 分布工具 / 观测归一化
    ├── envs.py         # MJX 环境封装（Brax-MJX + 纯 MJX Cartpole）
    ├── ball_wall.py    # Ball with Wall（单决策、撞墙不连续任务）
    ├── playground_env.py  # mujoco_playground dm_control 套件封装
    ├── rollout.py      # 可微 rollout：三种梯度目标（gae / shac / none）
    ├── common.py       # 训练状态、优化器、minibatch 循环、Agent 基类
    ├── gippo.py        # GI-PPO
    ├── rpo.py          # RPO
    ├── ppo.py          # PPO
    ├── ddcg.py         # DDCG
    ├── aobg.py         # AoBG
    └── ivwh.py         # IVW-H
```

## 1. 安装与运行

```bash
pip install "jax[cuda12]" flax optax
pip install mujoco mujoco-mjx brax
pip install matplotlib imageio imageio-ffmpeg        # 可视化用

# 训练：结束时会在 runs/ 下同时写出 <algo>_<env>_seed<n>.json 与 .pkl
python train.py --algo gippo --env cartpole
python train.py --algo rpo   --env ant
python train.py --algo ppo   --env hopper

# 评测 + 渲染一条轨迹
python evaluate.py --ckpt runs/rpo_ant_seed0.pkl --episodes 128 --video mp4

# 画学习曲线与内部诊断指标
python plot.py --runs "runs/*_ant_*.json" --diagnostics
```

MJX 需要 GPU 才有实用速度。GI-PPO 与 RPO 都要反向传播 32 步物理模拟，显存约为纯前向的
3～5 倍；8GB 显卡建议把 `--num-envs` 减半。PPO 不反传，可以用更大的 `--num-envs`。

## 2. 一个 rollout，三种梯度目标

三个算法的差别集中在 `rollout.py` 里被求导的那个标量上。共同的技巧是：在动作上加一个
恒为零的扰动量 \(u_t\)，令 \(a_t=\mu_\theta(s_t)+\sigma\odot\epsilon_t+u_t\)，在 \(u=0\) 处对 \(u\) 求梯度；
以及用「自当前 episode 段起点算起」的步数 \(c_t\) 作为折扣指数，使恒等式在 auto-reset 切断
梯度的边界处依然成立。

**`objective="gae"`（GI-PPO）**

\[
J_{\text{gae}}=\sum_t w_t\,\delta_t,\quad w_t=(\gamma\lambda)^{c_t}
\;\Longrightarrow\;
\frac{\partial A_t}{\partial a_t}=\frac{1}{w_t}\frac{\partial J_{\text{gae}}}{\partial u_t}
\]

即 GI-PPO 论文的 Eq. 15，一次 reverse-mode AD 拿到所有时刻的 \(\nabla_aA\)。

**`objective="shac"`（RPO，同时也是 SHAC / SAPO 的目标）**

\[
J_{\text{shac}}=\sum_t \gamma^{c_t} r_t \;+\; \gamma^{c+1}V(s_{\text{段末}})
\;\Longrightarrow\;
\nabla_{a_k}R(\tau)=\frac{\partial J_{\text{shac}}}{\partial u_k}
\]

它天然带有 \(\gamma^k\) 因子，正是 \(\gamma^k\nabla_aQ^{\pi_{old}}(s_k,a_k)\) 的无偏估计，即 RPO 要缓存的 action-gradient。

**`objective="none"`（PPO）** 跳过 `value_and_grad`，完全不穿过 MJX 反传。

## 3. RPO 的三个关键实现点

**动作再生成（Eq. 10）。** off-policy 更新时缓冲区里的 \(a\) 不再是当前策略的采样结果，需要
求出能复现它的噪声 \(\epsilon_{reg}=f_\theta^{-1}(a;s)=(a-\mu_\theta(s))/\sigma_\theta\)。代码里对 \(\epsilon_{reg}\) 做
`stop_gradient`，于是

\[
a_{regen}:=\mu_\theta(s)+\sigma_\theta\odot \mathrm{sg}(\epsilon_{reg})
\]

数值上等于 \(a\)，但对 \(\theta\) 可导。再把线性替代目标 \(\sum_i \langle w_i\nabla_aR_i,\ a_{regen,i}\rangle\)
对 \(\theta\) 求导，得到的正是 Eq. 11 要的 \(\rho\,\nabla_\theta a\,\nabla_a R(\tau)\)——这就是「复用缓存梯度」的
全部实现。

**非对称裁剪（Eq. 11）。** 与 PPO 不同，它不看优势的符号，只在 \(\rho\in[1-c_{low},\,1+c_{high}]\)
时保留该样本的梯度并以 \(\rho\) 加权，否则整条置零。原因是 RPG 并不像 REINFORCE 那样显式
抬高/压低某个动作的 log 概率，PPO 的对称 clip 语义在这里不成立。日志里的
`eff_sample_ratio` 就是论文附录 F 的 effective sample ratio。

**第一轮即 SHAC。** 默认 `rpo_batch=0`（全批量），因此第 1 轮更新时 \(\rho\equiv1\)、\(\mathrm{KL}\equiv0\)，
RPO 精确退化为 SHAC 的 RPG 更新；\(M-1\) 轮样本复用是纯增量。设 `--rpo-epochs 1` 即可复现
论文 Figure 4(b) 的 no-sample-reuse 消融。

## 4. 与论文的对应关系

| 论文条目 | 代码位置 |
| --- | --- |
| GI-PPO Eq. 14/15 GAE 及其梯度 | `rollout.py`（`objective="gae"`） |
| GI-PPO Def. 4.1 / Eq. 8,10 \(\alpha\)-policy 与回归损失 | `gippo.py: _alpha_loss` |
| GI-PPO Lemma 4.4 行列式估计 | `gippo.py: psi` |
| GI-PPO Eq. 12/16/18 三条准则与 \(\pi_h\) 损失 | `gippo.py: R_oorr / R_alpha / _ppo_loss` |
| RPO Eq. 7 替代目标 | `rollout.py` + `rpo.py: surr` |
| RPO Eq. 10 动作再生成 | `rpo.py: eps_reg / a_regen` |
| RPO Eq. 11 RPG 专用裁剪 | `rpo.py: mask / w` |
| RPO Eq. 12/13/14 KL 与熵正则 | `rpo.py: _rpo_loss` |
| RPO Eq. 15 + double critic + target 网络 | `common.py: critic_loss / update_critic / polyak` |
| PPO-Clip | `ppo.py: _ppo_loss` |

GI-PPO 的行列式估计无需 Hessian：由 Lemma 3.2 与 Prop. 4.5，Step 1 后 \(\pi_\theta\approx\pi_\alpha\)，故

\[
\det\!\big(I+\alpha\nabla_a^2A\big)\approx\exp\big(\log\pi_{\bar\theta}(s,a)-\log\pi_\theta(s,\tilde a)\big),
\qquad \tilde a=a+\alpha\nabla_aA .
\]

## 5. 常用实验命令

```bash
# 三算法同任务对比
for A in gippo rpo ppo; do python train.py --algo $A --env ant --seed 0; done

# GI-PPO 附录 7.3.4：放开 out-of-range-ratio 上限
python train.py --algo gippo --env ant --delta-oorr 1.0

# RPO 消融（对应论文 Figure 4）
python train.py --algo rpo --env ant --rpo-epochs 1                 # 无样本复用
python train.py --algo rpo --env ant --lambda-kl 0.0                # 无 KL 正则
python train.py --algo rpo --env ant --c-low 1e9 --c-high 1e9       # 无裁剪
```

## 5.1 评测与可视化

`evaluate.py` 加载 checkpoint 后同时跑两套协议——**随机评测**（从策略分布采样）与
**确定性评测**（直接用均值动作），对应 RPO 论文 Table 1 与 Table 6；输出回报均值±标准差、
回报分布直方图/箱线图、单条 episode 的逐步奖励曲线，并渲染一条轨迹。渲染优先出 mp4
（需要 `mujoco.Renderer` 与 EGL 等离屏 GL 后端），失败时自动回退到 brax 的交互式 HTML
播放器（纯 CPU，服务器上最稳）。给多个 `--ckpt` 时还会额外生成横向对比箱线图。

```bash
python evaluate.py --ckpt "runs/*_ant_*.pkl" --episodes 128 --no-video   # 批量对比
python evaluate.py --ckpt runs/gippo_cartpole_seed0.pkl --video html
```

`plot.py` 从 json 日志出图：`learning_curves.png` 是多 seed 均值±标准差的学习曲线；
`--diagnostics` 会按算法额外输出内部指标——GI-PPO 的 `alpha / psi / R_alpha / oorr`
（psi 图上标出 1.0 参考线、R_alpha 图上标出 0 参考线，便于判断哪条准则在压制 alpha），
RPO 的有效样本率 / 最大重要性比 / KL / 熵，PPO 的 clip 比例 / KL / 熵。

日志字段：`alpha / psi / R_alpha / oorr` 属于 GI-PPO；`eff_sample_ratio / max_iw / kl`
属于 RPO；`clip_frac / kl` 属于 PPO；`grad_a_norm` 是解析 action-gradient 的平均范数。

## 6. 与论文的已知差异

- **标准差是状态无关的可学习向量**（三算法一致）。RPO 论文沿用 SAPO 的状态相关 std 与
  自动温度调节的熵正则（把熵加到奖励上、目标熵 \(-\dim(\mathcal A)/2\)）；这里按 RPO 正文
  Eq. 13 实现为策略损失里的显式熵项，更简洁但探索行为会略有不同。
- **并行环境数**。RPO 论文在 DFlex 上用 Hopper 1024 / Ant 128；MJX 反传显存更紧，
  这里默认 Hopper 512 / Ant 128，可用 `--num-envs` 调整。
- **模拟器不同**。两篇论文分别使用 DFlex 与 Rewarped，本实现是 MJX，接触模型与奖励定义
  都不同，绝对分数不可直接与论文数字对齐，只应做同一实现内部的横向比较。
- **接触梯度有偏**。Ant / Hopper 的 MJX 接触求解分段光滑，解析梯度带偏差与高方差。
  这正是两个算法各自的设计目标：GI-PPO 会自动压小 \(\alpha\) 退回 PPO；RPO 靠裁剪与 KL 正则
  约束更新幅度。若 `grad_a_norm` 频繁爆炸，可开启 `--clip-grad-a 1.0` 或缩短 `--horizon`。
