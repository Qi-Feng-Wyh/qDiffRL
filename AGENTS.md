# AGENTS.md

> 本文件面向 AI 编码代理，介绍本项目的结构、运行方式与开发约定。
> 更详细的算法推导与论文对应关系见 `README.md`（中文）。

## 1. 项目概览

本项目在 **MuJoCo-XLA (MJX) 可微物理仿真**上实现了四个策略梯度算法，并可在
Cartpole / Hopper / Ant 三个任务上直接横向对比：

| 算法 | 模块 | 说明 |
| --- | --- | --- |
| **GI-PPO** | `algos/gippo.py` | Son et al., NeurIPS 2023。用解析梯度构造 α-policy 回归目标 + PPO 更新兜底 |
| **RPO** | `algos/rpo.py` | 缓存 action-gradient，通过"动作再生成"在 M 轮更新中复用 |
| **PPO** | `algos/ppo.py` | SB3 风格 PPO-Clip 基线（tanh squash + logp 修正），不穿过模拟器反传 |
| **DDCG** | `algos/ddcg.py` | Onoda et al., ICLR 2026（Part I）。参数空间随机平滑 + 0/1 阶梯度 IVW 混合 + Eq.14 不连续检测门控；无 critic 的直接梯度上升（自带参数扰动 rollout，不用 rollout.py）。注意 env_steps 日志未计入扰动模拟（实际 ×(N·M+num_envs)/num_envs） |
| **IVW-H** | `algos/ivwh.py` | Onoda et al., ICLR 2026（Part II，附录 E Alg.1）。逐步逐动作维 IVW 混合 1 阶（rollout 缓存的 dR/da）与 0 阶（似然比）梯度，VJP 回传，单次 Adam 更新 + target critic；无额外模拟开销 |
| **AoBG** | `algos/aobg.py` | Suh et al., ICML 2022。与 DDCG 共用参数扰动骨架（继承 DDCG 类），权重选择换成带精度约束的方差最小化（Eq.4-5：B=\|g1−g0\|、Bernstein 置信半径 ε、阈值 γ）。**γ 量纲依赖实现，论文参考值（1 / 1e5 / 1e6）在本实现下会恒不可行（α≡0），需用 `--aobg-gamma` 按 v0 量级（1e6+）校准** |

核心思想：MJX 的环境步进完全由 JAX 写成，因此可以用 `jax.grad` 直接对物理模拟
反向传播，拿到解析的动作梯度（`dA/da` 或 `dR/da`）。三个算法共享同一套环境封装、
网络定义与可微 rollout，差别集中在损失函数上。

## 2. 技术栈

- **语言**：Python 3.10（项目根目录下有已配置好的 `.venv`，Python 3.10.12）
- **核心依赖**：`jax` 0.6.2（含 `jax-cuda12` GPU 插件）、`flax` 0.10.7（linen 网络）、
  `optax` 0.2.8（优化器）、`mujoco` / `mujoco-mjx` 3.11.0、`brax` 0.14.1
- **可视化**：`matplotlib`、`imageio` / `imageio-ffmpeg`（渲染 mp4）
- **没有** `pyproject.toml` / `setup.py` / `requirements.txt`；依赖以 README 中的
  pip 命令为准，已安装在 `.venv` 中。运行代码请使用 `.venv/bin/python` 或先激活 venv。
- **硬件要求**：MJX 需要 GPU（CUDA 12）才有实用速度。GI-PPO / RPO 每迭代要反传
  32 步物理模拟，显存约为纯前向的 3～5 倍；8GB 显卡建议把 `--num-envs` 减半。
  注意 JAX 默认**预分配约 75% 显存**（nvidia-smi 看到的高占用是预留而非实际使用）。
  **本机为共享 GPU，运行训练/评测命令时必须设
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`**（按需分配，避免挤占其他用户）。
- **JIT 编译缓存**：`algos/__init__.py` 导入时开启持久化编译缓存（目录 `.jax_cache/`）。
  相同计算图配置重复运行跳过编译；改网络结构 / `num_envs` / `horizon` / 升级 jax
  会使缓存失效并自动重编译。

## 3. 目录结构与模块划分

```
DiffRL/
├── train.py            # 统一训练入口（--algo gippo|rpo|ppo），结果写入 runs/
├── evaluate.py         # 加载 checkpoint 评测 + 轨迹渲染（mp4 / HTML）
├── plot.py             # 从 runs/*.json 画学习曲线与算法诊断图
├── README.md           # 算法推导、论文公式与代码的对应表（先读这个）
└── algos/              # 核心包（__init__.py 导出 get_config / make_agent / AGENTS）
    ├── config.py       # Config dataclass + 环境×算法超参数矩阵，get_config() 合并
    ├── networks.py     # MLP / GaussianActor / Critic（集成）/ 高斯工具 / 观测归一化
    ├── envs.py         # BraxMJXEnv（brax mjx 后端）与纯 MJX 手写 Cartpole
    ├── ball_wall.py    # Ball with Wall 环境（单决策任务，可微飞行模拟 + 撞墙不连续）
    ├── playground_env.py  # mujoco_playground dm_control 套件封装（env_name 用 "pg:" 前缀）
    ├── rollout.py      # 可微 rollout：objective ∈ {gae, shac, none}，一次反传拿到解析梯度
    ├── common.py       # TrainState、优化器、minibatch 循环 run_epochs、BaseAgent 基类
    ├── gippo.py        # GI-PPO（objective="gae"）
    ├── rpo.py          # RPO（objective="shac"）
    ├── ppo.py          # PPO（objective="none"）
    ├── ddcg.py         # DDCG（独立参数扰动 rollout；含 _select_gradient 子类钩子）
    ├── aobg.py         # AoBG（继承 DDCG，仅覆盖权重选择规则）
    └── ivwh.py         # IVW-H（objective="shac"，与 RPO 共用 rollout）
```

架构要点：

- **BaseAgent（common.py）** 负责环境/网络/优化器构建与训练主循环；每个算法子类
  只需实现 `_update(ts, env_state, key)`（整段被 `jax.jit`），并设置类属性
  `objective` 与 `log_keys`。新增算法应继承 `BaseAgent`。
- **rollout.py** 是唯一的可微采样入口：在动作上加恒为零的扰动量 `u_t` 并在
  `u=0` 处求导；`objective` 决定被求导的标量（GAE 加权和 / SHAC 折扣回报 / 不求导）。
  三个算法的"解析梯度"都来自这一个函数。
- **配置合并顺序**（config.py）：`ENV_DEFAULTS` ← `ENV_NET` ← `ALGO_DEFAULTS` ←
  `ALGO_ENV[(algo, env)]` ← 命令行覆盖。改超参数优先改对应字典，不要硬编码到算法里。
- **envs.py** 的统一接口：`env.reset(key) / env.step(state, act)`，且
  `state.info` 必须包含 `truncation`、`obs_before_reset`（GAE 自举依赖；
  自动重置用 `jnp.where` 实现以在 episode 边界切断梯度）以及 `ep_ret` /
  `last_ep_ret`（跨 rollout 窗口的回合回报统计，训练日志的 `episode_return`
  取自后者）。
- **cartpole 默认为摆起（swing-up）任务（2026-08 起）**：
  `Config.use_pure_mjx_cartpole=True` 时使用手写的 `MJXCartpole`
  （杆初始垂直向下 θ=π，奖励以 cos θ 为核心，无提前终止、只有超时截断）；
  置 False 退回 brax inverted_pendulum（起始近竖直的静态平衡任务，带提前终止）。
  注意：`MJXCartpole` 观测为 5 维（brax 版为 4 维），切换后需重新编译与重训；
  渲染只支持 `--video mp4`（HTML 回退依赖 brax System，纯 MJX 环境没有）。
  2026-08 按 `cartpole_swingup_guide.md` 调参后的关键参数：**gear=10**（原 50，
  防止蛮力直推/撞墙借力）、摆阻尼 0.005、奖励 `cosθ − 0.1x² − 0.001a²`
  （**不惩罚 θ̇²**——无条件惩罚角速度会抑制蓄能摆荡）、初始 θ₀~U(π±0.3)、
  action_repeat=2（50Hz 决策/100Hz 物理）、episode_length=500（10s 物理时长）、
  观测 θ̇ clip ±10；滑轨限位 ±20m（实质无墙，居中完全靠 −0.1x² 惩罚约束，
  备份 `backup/envs.py.20260811_prerail20`）。
  指南改动前备份：`backup/2026-08-10_pre_swingup_guide/`。

## 4. 构建与运行命令

无构建步骤（纯 Python）。所有命令在项目根目录、使用 `.venv` 运行：

```bash
# 训练（结束时在 runs/ 下写出 <algo>_<env>_seed<n>.json 日志与 .pkl checkpoint）
.venv/bin/python train.py --algo gippo --env cartpole
.venv/bin/python train.py --algo rpo   --env ant
.venv/bin/python train.py --algo ppo   --env hopper

# 评测 + 渲染（--video mp4 需要离屏 GL；失败会自动回退到 HTML，纯 CPU 可跑）
.venv/bin/python evaluate.py --ckpt runs/rpo_ant_seed0.pkl --episodes 128 --video mp4
.venv/bin/python evaluate.py --ckpt "runs/*_ant_*.pkl" --episodes 128 --no-video

# 画学习曲线 / 诊断指标
.venv/bin/python plot.py --runs "runs/*_ant_*.json" --diagnostics
```

常用消融（改超参一律走命令行覆盖，不必改代码）：

```bash
.venv/bin/python train.py --algo rpo --env ant --rpo-epochs 1        # 无样本复用
.venv/bin/python train.py --algo rpo --env ant --lambda-kl 0.0       # 无 KL 正则
.venv/bin/python train.py --algo gippo --env ant --delta-oorr 1.0    # GI-PPO 附录 7.3.4
```

## 5. 代码风格约定

- **注释与文档字符串一律使用中文**（公式与论文引用保留英文术语），新代码必须延续
  这一约定；matplotlib 图内标签用英文（避免缺字体显示成方块）。
- 遵循现有代码风格：4 空格缩进、模块级 docstring 开头说明设计意图与论文出处、
  用 `# ======` 分节注释划分一次迭代的步骤。
- 所有训练计算必须能整段 `jax.jit`：minibatch 循环用 `jax.lax.scan`
  （见 `common.run_epochs`），不要在 jit 内部使用 Python 数据依赖的控制流。
- 涉及概率比 / 指数的数值稳定性套路：`jnp.clip(logp_diff, -20, 20)`、
  `jnp.nan_to_num`、`logaddexp`——新代码遇到类似场景请沿用。
- 论文公式与代码的对应关系写在 README 第 4 节的表格里；修改算法逻辑时请同步
  更新 README 与相关 docstring 中的公式编号。

## 6. 测试与验证

- **项目中没有自动化测试套件，也没有 CI 配置。** 验证改动的方式是：
  1. 导入冒烟测试：`.venv/bin/python -c "from algos import get_config, make_agent"`；
  2. 小规模训练冒烟：用小 `--total-steps`（如 `--total-steps 8192`）在 cartpole 上
     跑通三个算法，确认 `runs/` 下产出 json 与 pkl；
  3. 用 `evaluate.py --no-video` 与 `plot.py` 验证产出物可读。
- 注意 brax 导入时会打印 `Failed to import warp` 警告，这是可选依赖缺失的提示，
  不影响运行。
- DDCG 专用超参：`--ddcg-samples`（N，默认 32）、`--ddcg-envs`（M，默认 8）、
  `--ddcg-sigma`（参数平滑 std，默认 0.02）、`--ddcg-c`（检测松弛，默认 0.3）。

## 7. 安全注意事项

- **Checkpoint 是 pickle 文件**（`common.save_checkpoint/load_checkpoint`）。
  `evaluate.py` 会直接 unpickle `--ckpt` 指定的文件——绝对不要加载来源不明的
  `.pkl`，存在任意代码执行风险。
- `evaluate.py` / `plot.py` 的 `--ckpt`、`--runs` 参数支持 glob，批量操作前注意
  shell 引号，避免误匹配。
- 本仓库没有密钥或网络服务；依赖安装只应发生在项目内的 `.venv`，不要装到系统环境。

## 8. 已知事项与历史变更

- **mujoco_playground 接入（2026-08）**：安装的是 PyPI 包 `playground==0.1.0`
  （0.2.0 要求 Python>=3.11，本机 venv 为 3.10）。兼容性补丁在
  `algos/playground_env.py` 导入期应用：`mjx.make_data` 的 `nconmax` → `naconmax`
  别名（playground 0.1.0 用的是旧 API）。环境名前缀 `pg:`（如
  `pg:CartpoleSwingup`），产出文件名会去掉冒号（`ppo_pgPendulumSwingup_seed0`）。
  改动备份 `backup/2026-08-12_pre_playground/`。注意 locomotion/manipulation 环境
  首次加载会联网下载 menagerie 资产；dm_control 套件无此依赖。**playground 没有
  经典 Ant**（locomotion 套件为 Go1/G1 等四足/人形）；hopper 用 dm_control 版
  `pg:HopperHop` / `pg:HopperStand`（已注册）。

- **MJXCartpole 按需重置优化（2026-08）**：原实现 `step()` 每步无条件
  `vmap(_reset_one)`（内含 `mjx.forward`），前向开销翻倍且撑大 BPTT 图。
  已改为 `lax.cond(any(done), 算新的, 复用缓存)`（缓存放 `info["reset_data"]`）。
  A/B 实测（PPO/cartpole 稳态）：10.6s → 5.8s/迭代（约 1.8×）。备份
  `backup/envs.py.20260812_pre_condreset`。

- **episode_return 日志修复（2026-08）**：原实现中 rollout 的回报累加器每次迭代
  清零，日志值只是 horizon 步窗口内的部分和（上限被钉在 horizon=32），无完成的
  轮次记 0，曲线呈锯齿且严重失真。已改为在环境 info 中维护 `ep_ret` /
  `last_ep_ret`（跨窗口保留），日志报告真实完整回合回报。改动前版本备份于
  `backup/2026-08-07_pre_epret_fix/`。注意：修复前后日志的 episode_return 数值
  **口径不同，不能直接对比**（如旧摆起 run 的 -32 实际是窗口和，真实回合约 -240）。

- **MJXCartpole 滑轨碰撞 bug（2026-08 修复）**：`_CARTPOLE_XML` 中装饰用的
  rail 胶囊体原本参与碰撞，它从小车中心穿过，与小车/杆产生嵌入接触，把整个
  机构“焊死”（50N 推 1 秒小车只挪 4mm），导致摆起任务学到的是“半抡”局部最优
  （return ≈ -32）。已修复：rail 加 `contype="0" conaffinity="0"`（纯装饰）。
  修复前在坏环境上的所有 swing-up 训练结果作废。该版本 envs.py 备份于
  `backup/envs.py.20260807_prerailfix`。

- **PPO tanh squash 改造（2026-08）**：摆起 cartpole 上 PPO 陷入局部最优
  （return 卡在 -32），定位为无界高斯 + env 内硬 clip 的信用分配问题。已改为
  SB3 风格常规实现：`Config.squash_actions=True`（仅 PPO；RPO/GI-PPO 数学上
  要求无界高斯重参数化，必须保持 False），rollout 中动作用 tanh 压缩、logp 带
  `tanh_squash_logp_correction` 修正，PPO 超参对齐 SB3（恒定 lr、10 epochs ×
  minibatch 64、critic 10 epochs）。改动前代码备份在
  `backup/2026-08-07_pre_ppo_conventional/`。

- **共享 GPU 的 MPS 问题（2026-08）**：机器上 MPS 控制守护进程属于其他用户时，
  JAX 初始化会报 `CUDA_ERROR_MPS_CONNECTION_FAILED` 并静默回退 CPU（训练极慢）。
  解法：以当前用户启动独立 MPS 守护进程后，训练命令前加
  `CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh`（目录名可自定）：
  `mkdir -p /tmp/mps-wyh /tmp/mps-wyh-log && CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh CUDA_MPS_LOG_DIRECTORY=/tmp/mps-wyh-log nvidia-cuda-mps-control -d`。
  无显示的机器上渲染 mp4 需加 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。

- **MJX 求解器反向传播补丁（2026-08）**：mujoco 3.11 的 `mjx._src.solver.solve`
  主循环用 `jax.lax.while_loop`，反向模式微分直接报错，导致 GI-PPO / RPO 无法训练。
  `algos/envs.py` 导入期通过 `_patch_mjx_solver()` 将其替换为库自带、语义等价且可微的
  `_while_loop_scan`（lax.scan 实现，前向结果不变，代价是始终跑满 `opt.iterations` 次）。
  该补丁依赖 mujoco==3.11.0 的私有符号；升级 mujoco 后需重新核对该函数。

- 根目录的 `__init__.py` 原本被误放在项目根（其内容是 `from .config import ...`
  的相对导入，只有作为 `gippo` 包的 `__init__.py` 才能工作，导致
  `from algos import get_config` 直接 ImportError）。已于 2026-08 移动到
  `algos/__init__.py` 并验证导入通过。若发现该文件又出现在根目录，即是回归。
- Ant / Hopper 的 MJX 接触求解分段光滑，解析梯度带偏差与高方差：若日志中
  `grad_a_norm` 频繁爆炸，可开 `--clip-grad-a 1.0` 或缩短 `--horizon`。
- 与论文的已知实现差异（状态无关 std、并行环境数、模拟器不同导致绝对分数不可比等）
  见 README 第 6 节。
