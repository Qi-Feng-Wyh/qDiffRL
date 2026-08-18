# BUG 排查与修复经过汇报

> 时间范围：2026-08-04 ~ 2026-08-07
> 涉及任务：cartpole（静态平衡 → swing-up 摆起）、hopper、ant
> 涉及算法：GI-PPO / RPO / PPO
> 本文档面向后续维护者，完整记录本轮排查出的所有问题、定位过程、修复方案与验证结果。
> 仓库结构与算法说明见 `AGENTS.md` / `README.md`；另一份独立诊断记录见 `CLAUDE.md`。

---

## 0. 总结

本轮共修复 **2 个代码 bug**、**1 个指标口径 bug**，完成 **1 项实现对齐改造**，
并澄清 **3 个环境/运维层面的非代码问题**。核心剧情线：

> PPO 在 swing-up cartpole 上"卡在 -32"这一悬案，实际是**三个独立问题叠加**的产物：
> 装饰滑轨未关碰撞把物理机构焊死（真因）+ 日志指标把窗口部分和当回合回报（掩盖真相）
> + PPO 动作表示与主流实现不一致（次要因素）。
> 逐一修复后，PPO 在同一任务上达到 **确定性评测 ≈ +190**（任务上界参考 +173~+200），
> 摆起 + 顶端镇定全部学会。

---

## 1. 问题清单总览

| # | 类型 | 问题 | 根因 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | 代码 bug | GI-PPO/RPO 训练直接崩溃：`Reverse-mode differentiation does not work for lax.while_loop` | mujoco 3.11 MJX 约束求解器主循环用 `jax.lax.while_loop`，不支持反向模式微分 | ✅ 已修复（猴子补丁） |
| 2 | 运维 | JAX 报 `CUDA_ERROR_MPS_CONNECTION_FAILED` 并静默回退 CPU | 共享 GPU 的 MPS 控制守护进程属于其他用户 | ✅ 已规避（独立 MPS 通道） |
| 3 | 配置 | 训练一启动就占 18G 显存 | JAX 默认预分配 75% 显存 | ✅ 已约定关闭预分配 |
| 4 | **代码 bug（核心）** | swing-up 任务完全学不动，PPO 卡在 "-32" | **XML 中装饰滑轨 rail 未关碰撞**，从小车中心穿过，接触约束把整个机构焊死 | ✅ 已修复（一行属性） |
| 5 | 指标口径 bug | `episode_return` 日志长期显示 0 或 -32 锯齿 | rollout 的回报累加器每次迭代清零，日志值只是 32 步窗口内的部分和 | ✅ 已修复（累加器移入 env info） |
| 6 | 实现对齐 | PPO 与主流实现（SB3）不一致 | 无界高斯 + env 内硬 clip，无 logp 修正；lr 线性衰减；全批量 5 epochs | ✅ 已改造（tanh squash，仅 PPO） |
| 7 | 遗留疑点 | XML 写 `integrator="RK4"` 但实际生效 Euler | 未解析生效，原因未查（不影响正确性） | ⬜ 未处理 |
| 8 | 遗留优化 | `MJXCartpole.step` 每步无条件 vmap 一次 reset（含 `mjx.forward`） | 实现浪费，开销翻倍、撑大 BPTT 图 | ⬜ 未处理（来自 CLAUDE.md §4） |

---

## 2. 各问题详细经过

### 2.1 MJX 求解器不可反传（问题 1）

**现象**（2026-08-04）：`train.py --algo gippo --env cartpole` 首次迭代即报错
`ValueError: Reverse-mode differentiation does not work for lax.while_loop...`。

**定位**：堆栈指向 `mujoco/mjx/_src/solver.py` 的 `solve()`——mujoco 3.11 的约束求解
主循环使用 `jax.lax.while_loop`（动态终止条件），JAX 反向模式不支持。GI-PPO/RPO 需要
`jax.grad` 穿过 `mjx.step`，必然触发。

**修复**：`gippo/envs.py` 导入期执行 `_patch_mjx_solver()`，逐行复制原版 `solve`，
仅将主循环替换为库自带的 `_while_loop_scan`（`lax.scan` 实现，收敛后 no-op，
官方注明 reverse-mode autodiff ok）。前向数值结果不变，代价是始终跑满
`opt.iterations` 次。该补丁依赖 mujoco==3.11.0 的私有符号，升级依赖需重新核对。

**验证**：GI-PPO / RPO cartpole 冒烟训练均跑通，`grad_a_norm` 正常非零。

---

### 2.2 共享 GPU 的 MPS 问题（问题 2）

**现象**（2026-08-05）：训练突然极慢，日志显示 JAX 回退 CPU：
`cuInit(0) failed: CUDA_ERROR_MPS_CONNECTION_FAILED`。

**定位**：本机 MPS 控制守护进程（`/tmp/nvidia-mps`）属于其他用户，且当前用户有一个
残留的僵尸 MPS server。GPU 本身空闲。

**修复**：以当前用户启动独立 MPS 守护进程，之后所有命令带环境变量：

```bash
mkdir -p /tmp/mps-wyh /tmp/mps-wyh-log
CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh CUDA_MPS_LOG_DIRECTORY=/tmp/mps-wyh-log nvidia-cuda-mps-control -d
# 之后运行训练/评测时加：CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-wyh
```

已写入 `AGENTS.md` 第 8 节与 `run_cartpole_benchmark.sh`。

---

### 2.3 JAX 显存预分配（问题 3）

**现象**：训练进程一启动 `nvidia-smi` 即显示占用 ~18G。

**定位**：JAX 默认预分配 75% 显存（24G × 75% ≈ 18G），是预留池而非实际用量。
实测对照：默认 18750 MiB vs 关闭后 472 MiB（仅初始化）。

**约定**：共享 GPU 上所有训练/评测命令必须带
`XLA_PYTHON_CLIENT_PREALLOCATE=false`（已写入基准脚本与 AGENTS.md）。
注意：JAX 分配器仍会保留高水位显存，但峰值远低于 18G（cartpole 约 1~3G）。

---

### 2.4 滑轨碰撞 bug——本轮核心问题（问题 4）

**现象**：PPO 在 swing-up cartpole 上 `episode_return` 长期卡在 -32；
换种子、加熵正则、换实现（squash 前后）均收敛到**完全相同的 -32**。

**排查过程**（关键：全部靠实际运行物理，而非读代码）：

1. 零动作 rollout：return = **-240**（杆垂到底，cos θ ≈ -1 全程）——奖励定义与观测
   方向正确，物理在"无输入"下正常；
2. 手写能量泵 + PD 镇定控制器：**两种符号、满幅 ±50N 打 6 秒，杆连 8° 都抬不起来**
   ——开始可疑；
3. 恒力测试（决定性证据）：恒定 ctrl=+1（=50N）推 1 秒，小车只移动 **4~11 cm**，
   且速度方向与力不符。50N 作用于 1kg 小车应产生 ~45 m/s² 加速度；
4. CPU 原版 MuJoCo 对照：**表现完全一致** → 排除 MJX、排除求解器补丁，问题在
   XML 模型本身；
5. 根因确认：`<geom name="rail" ...>`（装饰滑轨，胶囊体）沿 x 轴从小车中心穿过，
   未设 `contype/conaffinity`，与小车（及下垂时的杆）产生深度嵌入接触，
   接触约束力把整个机构"焊死"。

**修复**（`gippo/envs.py`，一行属性）：

```xml
<geom name="rail" ... contype="0" conaffinity="0"/>
```

**验证**：修复后恒力测试小车 0.4s 冲到轨道尽头（x=2，v=9.3 m/s），杆被正常甩起；
MJX 与 CPU 数值一致。修复前 envs.py 备份于 `backup/envs.py.20260807_prerailfix`。

**教训**：在坏环境里，-32 是"机构被焊死"下的理论最优（小幅晃动杆）；
两个不同实现收敛到同一数值正是因为它们面对的是同一个物理上限。

---

### 2.5 episode_return 指标口径 bug（问题 5）

**现象**：训练日志中 `episode_return` 长期为 0，非零时恒为 -32 的锯齿。

**根因**（由 CLAUDE.md 首次正确指出，纠正了排查过程中"−32 是完整回合回报"的误判）：
`rollout.py` 的回报累加器每次 rollout（horizon=32 步）清零，而 swing-up 环境
无提前终止、回合固定 240 步。日志值只是窗口内的部分和（上限被钉在 32），
且 64 个环境相位同步 → 仅每 7~8 次迭代有一个窗口含 done，其余记 0。
旧 run 的 "-32" 实际是 "32 步窗口 × -1.0/步"（杆全程垂着），
**真实回合回报约 -240**。

**修复**（采纳 CLAUDE.md §1 方案）：`ep_ret` / `last_ep_ret` 移入环境 `info`
（`MJXCartpole` 与 brax `DiffAutoResetWrapper` 同步修改），随 `env_state` 跨窗口
保留；`rollout.py` 的 `episode_return` 改为报告 `last_ep_ret` 均值。
备份：`backup/2026-08-07_pre_epret_fix/`。

**验证**：cartpole 冒烟 iter 8 = -207.6（首批完整回合）→ iter 15 = -178.8（上升），
窗口间保持前值不再归零；hopper（brax 路径）冒烟正常。

**注意**：修复前后日志口径不同，不能直接对比数值。

---

### 2.6 PPO 对齐 SB3 常规实现（问题 6）

**背景**：排查初期怀疑 PPO 实现有误（用户参照成品库经验）。逐行审计
（`ppo.py` / `rollout.py` / `common.py` / `networks.py`，另有 CLAUDE.md §4 独立复核）
**未发现数学错误**，但与 SB3 存在设计差异。用户批准后按 SB3 风格改造：

| 改动 | 内容 |
| --- | --- |
| tanh squash | 策略动作 `tanh(a)`，logp 带 Jacobian 修正 `Σ log(1−tanh²u)`（数值稳定形式），采样侧（rollout）与损失侧（ppo.py）一致；`Config.squash_actions` 开关，**仅 PPO 开启**（RPO/GI-PPO 的数学依赖无界高斯重参数化，不可 squash） |
| 超参对齐 | 恒定 lr=3e-4（原线性衰减）、10 epochs × minibatch 64（原全批量 5 epochs）、critic 10 epochs |
| 评测一致 | `evaluate.py` 的动作压缩与训练侧同步 |

备份：`backup/2026-08-07_pre_ppo_conventional/`。
备注：squash 改造的动机（信用分配污染）后来被证明是次要因素——真因是 rail bug；
但该改造本身无害且更标准，予以保留。

---

## 3. 修复后的最终验证（PPO, swing-up cartpole, seed 114514）

训练：1M 步（squash 版 PPO，`--ent-coef 0.01`，修复后环境 + 修复后指标）。

**训练曲线**（新口径 episode_return）：

| 阶段 | 表现 |
| --- | --- |
| iter 8 | -203（起步接近悬挂） |
| iter 16~112 | -108 → +122 快速上升 |
| iter 200~232 | **峰值 +184**（达到/超过 +173 参考上界） |
| iter 240 附近 | 一度塌陷至 -45（entropy 降至负值，std≈0.13，探索塌缩的残余影响） |
| iter 360~488 | 回升，最终批次 +98 |

**最终 checkpoint 评测（128 episodes）**：

| 协议 | 回报 |
| --- | --- |
| 随机评测 | ≈ +170（均值），分布主体 150~190，少量失败拖尾 |
| **确定性评测** | **≈ +190，128 个 episode 高度集中** |

逐帧奖励曲线：-1.0 起步 → 约 50 步（0.5s）内甩到竖直 → 之后稳定保持 +0.9/步
直到回合结束。轨迹视频确认摆起 + 镇定均成功。**任务已解决。**

---

## 4. 经验与遗留事项

### 方法论教训

1. **"跑物理"和"读代码"必须互补**：rail bug 只能通过实际仿真实验发现（恒力测试），
   纯代码审查（包括外部 AI 的复核）无法暴露；反之指标口径 bug 读代码更快。
2. **先修指标，再调算法**：指标失真会让所有后续判断建立在错误事实上
   （一度把"杆全程垂着"误读为"半抡局部最优"）。
3. **环境改动必须做物理验证**：零动作基准 + 恒力基准是成本最低的体检手段。
4. 两个实现收敛到同一数值 ≠ 实现正确，可能是共享环境的物理上限。

### 遗留事项（按建议优先级）

1. `horizon` 32 → 64：让单个窗口装得下完整起摆（~60 步），GAE 与 BPTT 同受益；
2. reset 时随机化 `steps` 计数器，错开 64 环境的回合相位（提升批次多样性，
   顺带消除日志的同步完成现象）；
3. 训练中期塌陷问题（iter 240 附近 entropy 掉负、回报 -45）：可加大 `--ent-coef`
   或限制 log_std 下界；
4. 可选：`action_repeat=4~8`（CLAUDE.md §2：100Hz 白噪声探索效率低）；
5. `MJXCartpole.step` 每步无条件 vmap reset 的性能瑕疵（CLAUDE.md §4）；
6. XML 中 `integrator="RK4"` 未生效的疑点（目前实际 Euler，训练正常）。

### 备份清单

| 路径 | 内容 |
| --- | --- |
| `backup/2026-08-07_pre_ppo_conventional/` | squash 改造前的 gippo/ + 三个入口脚本 |
| `backup/envs.py.20260807_prerailfix` | rail 修复前的 envs.py |
| `backup/2026-08-07_pre_epret_fix/` | 指标修复前的 envs.py + rollout.py |
| `archive/2026-08-06_pre_cartpole_benchmark/` | 早期全部 runs/eval/figures 实验产物 |
