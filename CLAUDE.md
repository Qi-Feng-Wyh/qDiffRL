# CLAUDE.md — PPO × Cartpole(swing-up) 卡在 -32 的诊断记录

> 本文件面向后续接手该仓库的 AI 编码代理与开发者，记录 2026-08 一次问题定位的完整结论。
> 仓库结构、算法与论文对应关系见 `AGENTS.md` / `README.md`。

## 0. TL;DR

现象：`python train.py --algo ppo --env cartpole` 的 `episode_return` 长期停在 -32。

结论：**PPO 的算法实现本身没有数学错误**。-32 是两件事叠加的产物：

1. `rollout.py` 里回合回报累加器每次 rollout 都被清零，`episode_return` 记的是
   32 步窗口内的部分和，**上限被硬钉在 `horizon = 32`**（日志 bug）。
2. 策略确实完全没学会摆起，杆全程垂在底部，每步奖励恰好 \(-1.000\)，
   真实回合回报是 \(240 \times (-1) = -240\)（任务可达约 **+170**）。

\[
\text{episode\_return}_{\text{logged}} = \text{horizon} \times (-1.000) = -32
\]

学不动的**首要原因是控制频率过高导致白噪声探索完全失效**（100 Hz + \(\sigma=0.368\)，
随机策略连一次接近顶端的样本都采不到）。修复优先级见第 5 节。

---

## 1. 日志 bug：`episode_return` 统计的不是回合回报

### 位置

`gippo/rollout.py`，`make_rollout_fn._surrogate`：

```python
cnt0 = jnp.zeros((n_env,))
ep0  = jnp.zeros((n_env,))          # <-- 每次调用 rollout 都归零
(final_state, _, _), outs = jax.lax.scan(body, (state0, cnt0, ep0), (eps_seq, u_seq))
```

一次 rollout 只跑 `horizon = 32` 步，而 `episode_length = 240`。`ep_ret` 因此只累加
**当前 32 步窗口内**的奖励，得到的是"窗口起点到 done 为止"的部分和。

### 为什么恰好是 -32（而不是别的数）

* 奖励 \(r = \cos\theta - 0.05x^2 - 0.001\dot\theta^2 - 0.01a^2\)；杆垂着不动时
  \(\cos\pi = -1\)、其余三项 \(\approx 0\)，每步奖励恰好 \(-1.000\)。
* 64 个环境同相位（同时 reset、同时 truncate），回合在全局第 \(240k\) 步结束，
  落在第 \(7.5k\) 次迭代。\(k = 4, 8, 12, \dots\) 对应迭代 30 / 60 / 90 …
  （正是 `log_interval = 10` 会打印的那些），且 done 恰好落在窗口的**最后一步**，
  累加满 32 步 → \(-32\)。
* 不含 done 的迭代因为 `jnp.sum(ep_ret) / jnp.maximum(n_done, 1.0)` 被记成 0。
  所以真实曲线是"0 …… 0 …… -32"的锯齿，非零时永远是 -32。

### 临时验证手段

json 日志里已经记录了 `mean_reward`。若它 \(\approx -1.0\)，即实锤"杆全程垂着"。
在修好之前，用 `mean_reward * episode_length` 作为代理指标。

### 修法

把累加器放进环境 state，使其跨迭代保留（`MJXCartpole` 的 `info` 已随 `env_state`
在迭代间传递）：

```python
# envs.py: MJXCartpole.reset —— info 中增加两个字段
info = {..., "ep_ret": zeros, "last_ep_ret": zeros}

# envs.py: MJXCartpole.step
ep_ret = state.info["ep_ret"] + reward
info.update(ep_ret      = jnp.where(done > 0, 0.0, ep_ret),
            last_ep_ret = jnp.where(done > 0, ep_ret, state.info["last_ep_ret"]))

# rollout.py: metrics 改用最后一帧的 last_ep_ret 均值，并去掉 ep_ret / ep0 那套逻辑
episode_return = jnp.mean(outs["last_ep_ret"][-1])
```

`BraxMJXEnv` 分支需同样处理（`DiffAutoResetWrapper` 里加同名字段）。

---

## 2. 真正学不动的原因：100 Hz 下白噪声探索等于没有探索

按 XML 参数（\(M=1.0,\ m=0.1,\ L=0.6,\ \text{gear}=50,\ b_x=0.1,\ b_\theta=0.02,\ \Delta t=0.01\)）
用等价解析动力学复现该环境后的对照实验（240 步一回合，16 个随机种子取均值）：

| 策略 | 回合回报 | 每步奖励 | 全程最大 \(\cos\theta\) |
| --- | --- | --- | --- |
| 什么都不做 \(u=0\) | -240.1 | -1.000 | -1.00 |
| 100 Hz 白噪声 \(\sigma=0.37\)（**当前初始策略**） | -239.6 | -0.998 | -0.77 |
| 100 Hz 白噪声 \(\sigma=1.0\) | -219.9 | -0.916 | -0.35 |
| 同噪声、每 8 步换一次动作（12.5 Hz） | -128.6 | -0.536 | **+0.86** |
| 同噪声、每 16 步换一次动作（6.25 Hz） | -93.6 | -0.390 | **+0.95** |
| 手写能量整形控制器（任务上界参考） | **+173.1** | +0.72 | +1.00 |

要点：

* `init_log_std = -1.0` → \(\sigma = 0.368\)。逐步独立采样的动作噪声，在摆的固有周期
  \(T = 2\pi\sqrt{2L/3g} \approx 1.27\ \mathrm{s}\)（127 步）尺度上被完全平均掉，
  杆最多晃到 ±40°。**随机策略采不到任何接近顶端的样本**，PPO 没有可利用的正优势信号，
  只能收敛到"不动"这个局部最优。
* 只要把动作保持几步（等价 `action_repeat`），同样强度的噪声就能把杆甩到顶端。
  这是最省事也最有效的一处改动。
* 任务本身完全可解，可达回报约 **+170**，摆起只需约 **60 步（0.6 s）**。
  不是执行器不够、也不是回合时长不够。
* 杆的阻尼比 \(\zeta = b_\theta / \big(2\sqrt{m g l \cdot J}\big) \approx 0.17\)，
  每周期振幅衰减到 35%，"温和共振式起摆"这条路被阻尼堵死，进一步放大了上述问题。

---

## 3. 叠加的次要因素

1. **`horizon = 32` 短于摆起所需的 60 步。** 没有任何一个 rollout 窗口装得下一次完整
   起摆，信用分配全靠 critic 外推，而 critic 的训练数据里根本没有顶端状态。
   \(\gamma = 0.99\) 在 100 Hz 下有效视野仅 100 步 = 1 s，起摆收益要打
   \(0.99^{60} \approx 0.55\) 折扣，而 \(-0.05x^2\)、\(-0.001\dot\theta^2\) 是立即支付的。

2. **奖励在 \(\theta = \pi\) 处是驻点。** \(\frac{d}{d\theta}\cos\theta\big|_{\pi} = -\sin\pi = 0\)，
   一阶梯度为零、只有二阶信号；速度项与位移项恰好惩罚起摆所需的能量注入。
   对 GI-PPO / RPO 更致命——解析 action-gradient 在底部近似为 0，
   预期 `grad_a_norm` 会一直很小。

3. **批次多样性极低。** 64 个环境同时 reset、初值均为 \(\theta = \pi \pm 0.1\)、相位完全同步，
   一次迭代的 2048 个样本实际只是 32 个时刻 × 64 条几乎重合的轨迹。
   建议 reset 时随机化 `steps`（`jax.random.randint(k, (), 0, episode_length)`）错开相位。

4. **熵塌缩风险。** `ent_coef = 1e-3` 偏小；`ppo_epochs = 10 × 32` 个 minibatch =
   每迭代 320 次梯度步；`normalize_adv` 会把纯噪声优势放大到单位尺度。
   若日志中 `entropy` 单调下降，说明 \(\sigma\) 在快速塌缩、更快锁死局部最优。
   另外 `target_kl = 0.008` 在 PPO 分支中定义了但 `adaptive_lr = False`，
   既没做自适应学习率也没做 SB3 式 KL early-stop；至少应监控 `kl` 与 `clip_frac`。

---

## 4. PPO 实现本身的复核结果（逐行核对，未发现数学错误）

* `logprob_from_eps(eps, log_std)` 与 `gaussian_logprob(mu + σε, mu, log_std)` 数值一致，
  首个 epoch 的 ratio 严格等于 1。✔
* tanh squash 的 Jacobian 修正 \(\sum \log(1 - \tanh^2 u)\) 在采样侧与 loss 侧都对
  `batch["act"]`（pre-tanh）计算，与参数无关、在 ratio 中相消，做法正确。✔
* clip 目标 \(\min\big(\rho A,\ \mathrm{clip}(\rho, 1 \pm \epsilon) A\big)\)。✔
* GAE 递推与 truncation 自举（`terminated = done * (1 - trunc)`；本环境只有超时，
  故恒为 0、始终自举）。✔
* `run_epochs` 的 permutation / minibatch / `lax.scan`。✔

已知的实现瑕疵（非本次问题主因）：

* `MJXCartpole.step` 每一步都无条件调用 `jax.vmap(self._reset_one)`（内含 `mjx.forward`），
  把每步开销翻倍；对 GI-PPO / RPO 还会额外撑大 BPTT 计算图。
  可改为复用 `reset` 时缓存的 `first_data`，或仅在需要时计算。

---

## 5. 修改优先级（建议按序执行）

1. **修 `episode_return`**（第 1 节）。否则无法判断任何后续改动是否有效。
   过渡期先看 `mean_reward`。
2. **加 `action_repeat`**：`MJXCartpole.step` 中对 `mjx.step` 循环 4~8 次、奖励求和
   （或取末帧），控制频率降到 12~25 Hz；同时把 `episode_length` 改为 `240 / repeat`
   以保持 2.4 s 的物理时长。按第 2 节的表，这一条足以让随机策略摸到顶端。
   注意：`obs_before_reset` / `truncation` 的语义需在循环外维持不变。
3. **放大探索**：`init_log_std = 0.0`（\(\sigma = 1\)）、`ent_coef = 0.01`。
4. **拉长视野**：`horizon` 保持 32（配合 repeat 后已覆盖 1.3~2.6 s）；
   若暂不加 action_repeat，则把 `gamma` 提到 0.997。
5. **错开各环境的回合相位**（reset 时随机化 `steps`）。
6. **仍不动再做奖励整形**：先把 \(-0.001\dot\theta^2\) 置 0，
   或加一项能量误差项 \(-k\left|\tfrac{1}{2}J\dot\theta^2 + mgl(\cos\theta - 1)\right|\)，
   给底部提供一阶梯度。

**验收标准**：先做 1 + 2，回报应从 -240 明显上行；达到 **+150 左右**即说明
摆起 + 顶端镇定都已学会（参考上界 +173）。

---

## 6. 对 GI-PPO / RPO 的连带影响

* 第 1 节的日志 bug 是三个算法共用的（`rollout.py`），同样影响 gippo / rpo 的曲线。
* 第 3 节第 2 点（底部是奖励驻点）对两个解析梯度算法影响更大：
  \(\nabla_a A\) / \(\nabla_a R\) 在底部趋近于 0，GI-PPO 的 `alpha` 会被压小、
  RPO 的 `grad_a_norm` 会长期偏低。若在 cartpole 上观察到这一现象，
  应先排除本文件所列的环境侧问题，再去怀疑算法实现。
* `horizon = 32`（0.32 s）远短于 60 步的摆起动作，两个算法的 BPTT 窗口内
  根本不包含"起摆 → 到顶"的完整因果链；加 `action_repeat` 后窗口的物理时长
  会随之放大，对它们同样是必要修复。
