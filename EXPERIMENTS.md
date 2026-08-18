# 实验设置与复现指南

> 本文档描述实验平台（qDiffRL）的构成、任务集、算法、超参数与评测协议，
> 作为论文实验部分的配套材料。算法推导与论文公式对应关系见 `report.md`，
> 工程细节与已知问题见 `README.md`, `AGENTS.md` / `BUGFIX_REPORT.md`。

## 1. 平台概览

本项目在 **MuJoCo-XLA（MJX）可微物理仿真**上统一实现了六个策略梯度算法，
所有算法共享同一套环境封装、网络定义与可微 rollout——环境步进完全由 JAX
写成，因此可以用 `jax.grad` 直接对物理模拟反向传播获得解析梯度
（`dA/da` 或 `dR/da`），算法之间的差异只集中在梯度估计与损失函数上。
这保证了跨算法对比**不存在实现差异带来的混淆**。

## 2. 安装

> 注：GitHub 地址待定，以下为本地安装方式。

```bash
# Python 3.10（受 playground 0.1.0 约束），GPU 需 CUDA 12
pip install "jax[cuda12]==0.6.2" flax==0.10.7 optax==0.2.8 \
    mujoco==3.11.0 mujoco-mjx==3.11.0 brax==0.14.1 \
    playground==0.1.0 matplotlib imageio imageio-ffmpeg
```

或 clone 后 `pip install -e .`（依赖锁定见 `pyproject.toml`；MJX 求解器补丁
依赖 mujoco 3.11 的内部 API，版本升级需重新核对 `algos/envs.py` 的
`_patch_mjx_solver`）。

## 3. 任务集

按接触复杂度排列（解析梯度方法的核心自变量）：

| 任务 | 命令名 | 观测/动作 | 奖励 | episode | 接触 |
| --- | --- | --- | --- | --- | --- |
| 摆起倒立摆（手写） | `cartpole` | 5 / 1 | `cosθ − 0.1x² − 0.001a²` | 500 决策步（50Hz） | 无 |
| 摆起倒立摆（官方） | `pg:CartpoleSwingup` | 5 / 1 | dm_control 容忍函数 | 1000 | 无 |
| 双摆甩起 | `pg:AcrobotSwingup` | 6 / 1 | dm_control 容忍函数 | 1000 | 无（欠驱动） |
| 鱼形游动 | `pg:FishSwim` | 24 / 5 | 前进速度 | 1000 | 无（流体） |
| 抛球撞墙 | `ball_wall` | 2 / 1 | 落点距离 | 1（单决策） | **人工不连续** |
| 单足跳跃 | `hopper` / `pg:HopperHop` | 11~15 / 3~4 | 存活+速度 / 乘性容忍 | 1000 | 有 |
| 四足蚂蚁 | `ant` | 27 / 8 | 存活+前进 | 1000 | 有（最复杂） |

## 4. 算法清单

| 算法 | 出处 | 梯度用法 | 一句概括 |
| --- | --- | --- | --- |
| **PPO** | 通用基线（SB3 风格 squash） | 不用解析梯度 | 标准 PPO-Clip |
| **GI-PPO** | Son et al., NeurIPS 2023 | α-policy 回归 + PPO 兜底 | 三准则自适应混合系数 α |
| **RPO** | Zhong et al., 2026 | 缓存 action-gradient，M 轮复用 | 动作再生成 + 重要性比裁剪 |
| **DDCG** | Onoda et al., ICLR 2026 | 参数空间扰动 + Eq.14 检测门控 | 检测不连续则退回 0 阶 |
| **AoBG** | Suh et al., ICML 2022 | 参数空间扰动 + 置信区间约束 | 精度阈值 γ 需逐任务调 |
| **IVW-H** | Onoda et al., ICLR 2026 | 逐步逐动作维逆方差加权 | 无检测，纯方差控制 |

## 5. 运行命令

```bash
# 训练
python train.py --algo {ppo,rpo,gippo,ivwh,ddcg,aobg} \
    --env {cartpole,hopper,ant,ball_wall,pg:*} \
    --seed 1 --total-steps 5000000 --log-interval 5

# 评测（随机 + 确定性两种协议，各 N 条完整 episode）+ 轨迹渲染
python evaluate.py --ckpt runs/<tag>.pkl --episodes 128 --video mp4

# 学习曲线与诊断图（多种子自动聚合 均值±std）
python plot.py --runs "runs/*_<env>_*.json" --diagnostics
```

批量实验：`run_*_train.sh` / `run_*_eval.sh` 脚本（断点续跑、批量评测、
自动渲染与画图）。

### 5.1 各任务推荐训练配置（实验验证的甜区）

| 任务 | total_steps | num_envs | horizon | 额外参数 | 依据 |
| --- | --- | --- | --- | --- | --- |
| pg:CartpoleSwingup | 2M | 64 | 32 | 梯度算法 `--clip-grad-a 1.0`；ppo 加 `--ent-coef 0.01` | RPO 1M 步即达 ~800 分；更大预算/更长窗口实测过训回落 |
| hopper（brax） | 10M | 256 | 32 | 同上；rpo 加 `--lambda-ent 0.05` | 10M 为甜区；rpo 默认 λ_ent=0.25 会冻结（见 BUGFIX 记录） |
| pg:FishSwim | 5M | 64 | 32 | 梯度算法 `--clip-grad-a 1.0` | RPO 默认配置 5M 步达 ~745 并进平台期 |

防过训原则：先用小预算试跑观察曲线，预算定在平台起点附近；entropy 掉到
负值后若回报高位震荡即为过训前兆（目前只在训练结束保存 checkpoint，
训过头没有中间点可回退）。

## 6. 超参数及含义

### 6.1 通用

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `num_envs` | 64（hopper 256） | 并行环境数，每迭代样本量 = num_envs × horizon |
| `horizon` | 32 | 采样窗口 = BPTT 截断长度 |
| `gamma / lam` | 0.99 / 0.95 | 折扣因子 / GAE λ |
| `actor_lr / critic_lr` | 3e-4 / 1e-3 | 学习率（算法有专属覆盖） |
| `clip_grad_a` | 关 | >0 时对解析梯度做逐样本范数裁剪（接触任务建议 1.0） |

### 6.2 算法专属

| 算法 | 参数 | 含义 |
| --- | --- | --- |
| PPO | `clip_eps=0.2`, `ppo_epochs`, `ent_coef` | clip 范围 / 更新轮数 / 熵正则 |
| GI-PPO | `alpha0`, `alpha_max`, `alpha_lr`, `delta_det`, `delta_oorr` | 初始/上限 α、Step1 回归 lr、方差准则阈值、oorr 上限 |
| RPO | `rpo_epochs`(M), `c_low/c_high`, `lambda_kl`, `lambda_ent` | 复用轮数 / 重要性比区间 / KL 与熵系数 |
| DDCG | `ddcg_samples`(N), `ddcg_envs`(M), `ddcg_sigma`(σ), `ddcg_c` | 扰动样本数 / 每扰动环境数 / 平滑 std / Eq.14 松弛系数（0.3） |
| AoBG | `aobg_gamma`, `aobg_delta` | 精度阈值（逐任务调）/ Bernstein 置信度 0.05 |
| IVW-H | 无专属（沿用 SHAC 式超参） | — |

## 7. 训练日志指标含义

### 通用字段

`steps`（累计环境步数）、`episode_return`（最近完成回合回报均值）、
`kl`（更新前后策略 KL）、`actor_lr`、`grad_a_norm`（解析梯度范数，PPO 为 0）、
`mean_reward`（窗口内每步平均奖励）。

### 算法专属

| 算法 | 字段 | 含义（健康参考） |
| --- | --- | --- |
| GI-PPO | `alpha` | 解析梯度混合系数；持续衰减≈0 说明梯度被判不可信 |
| | `psi_min/psi_max` | 方差准则，应在 [1−δ_det, 1+δ_det]（默认 [0.6,1.4]） |
| | `R_alpha` | 偏差准则，<0 触发 α 下调 |
| | `oorr` | 超窗样本占比，上限 δ_oorr |
| RPO | `eff_sample_ratio` | 重要性比在界内样本占比（≈1 健康） |
| | `max_iw` | 最大重要性权重（接近 1 最好） |
| | `entropy` | 策略熵（过快塌缩 = 过早收敛） |
| PPO | `clip_frac` | 被 clip 样本占比（0.05~0.2 健康） |
| DDCG | `test_pass` / `alpha` | Eq.14 是否通过 / 1 阶权重 |
| | `v0 / v1` | 0 阶 / 1 阶梯度经验方差 |
| AoBG | `B` / `eps_conf` / `feasible` | 偏差替代度量 / 置信半径 / 可行性 |
| IVW-H | `alpha_mu / alpha_std` | μ / log_std 通道的 1 阶权重均值 |

## 8. 评测协议

- 每个 checkpoint 报告两种协议：**随机评测**（从策略分布采样动作）与
  **确定性评测**（均值动作），各采集 128 条完整 episode；
- 汇报确定性评测的均值 ± 标准差为主指标；episode 平均长度辅助判断终止率；
- 初始状态由环境随机化（种子 0），多算法对比使用同一评测种子。

## 9. 评测图表解读

### 9.1 评测图（eval/eval_\<tag\>.png，三联面板）

| 面板 | 内容 | 读法 |
| --- | --- | --- |
| 左：return distribution | 128 条 episode 回报的直方图，蓝色=随机协议、橙色=确定性协议 | 峰越靠右越好；**双峰** = 策略只对部分初始状态有效（如 ant 起步摔 vs 会走）；蓝橙分离程度反映对探索噪声的鲁棒性 |
| 中：evaluation protocols | 两种协议的箱线图 | 看中位数与离群点；确定性应 ≥ 随机，**倒挂**（随机反而好）说明策略均值陷在坏模式里、靠噪声续命——未收敛好的信号 |
| 右：per-step reward | 渲染轨迹（0 号环境）的逐步奖励 | 定位失败时段：曲线跳水点对应视频里摔倒/失控的瞬间；摆起任务应看到“-1 爬升到 +0.9 并保持”的形状 |

判读流程：先看终端的确定性均值 → 再看左图橙色是否单峰靠右 → 有双峰/长尾则看右图和视频定位失败时段。

### 9.2 学习曲线（figures/learning_curves_\<tag\>.png）

横轴环境步数、纵轴回合回报（多种子取均值，阴影为标准差）。注意两点：
训练日志的回报是**带探索噪声的行为策略**口径，通常低于确定性评测；
跨算法比较以评测数字为准，曲线只看趋势与样本效率（多快到达平台）。

### 9.3 诊断图（figures/diagnostics_\<tag\>.png）

各算法内部指标随训练的变化（GI-PPO 的 α/ψ/R_alpha/oorr、RPO 的
有效样本率/最大重要性比/KL/熵、PPO 的 clip_frac 等，字段含义见第 7 节）。
典型用法：性能异常时先查崩溃先兆链——`grad_a_norm` 跳变 →
`psi_max`/`max_iw` 尖峰 → `kl` 尖峰 → return 下跌。

## 10. 可复现性说明

- **硬件**：NVIDIA RTX 3090（24GB），CUDA 12 驱动；
- **版本锁定**：jax 0.6.2 / mujoco(-mjx) 3.11.0 / brax 0.14.1 /
  playground 0.1.0 / flax 0.10.7 / optax 0.2.8；
- **随机种子**：训练 seed ∈ {1, 2, 3}（另见各实验说明），评测 seed 0；
- **已知实现差异**（与论文原版的区别）：状态无关 log_std、PPO 采用
  tanh squash、并行环境数不同、模拟器不同（绝对分数不可跨论文比较）——
  详见 README 第 6 节与 BUGFIX_REPORT.md；
- checkpoint 为 pickle（含 actor/critic 参数、观测归一化统计量、完整配置）。
