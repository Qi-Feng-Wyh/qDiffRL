# Cartpole Swing-up：MuJoCo 建模与奖励函数设计

本文给出一份可直接运行的 MuJoCo XML 模型，并讨论 swing-up 任务的观测设计、奖励函数的几种思路以及常见的踩坑点。

---

## 一、MuJoCo XML

```xml
<mujoco model="cartpole_swingup">
  <compiler inertiafromgeom="true"/>
  <option timestep="0.005" integrator="RK4"/>

  <default>
    <geom contype="0" conaffinity="0" friction="0 0 0" rgba="0.7 0.7 0 1"/>
  </default>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="rail" type="capsule" fromto="-2 0 0 2 0 0" size="0.02"
          rgba="0.3 0.3 0.3 1"/>

    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"
             range="-1.8 1.8" damping="0.1" limited="true"/>
      <geom name="cart" type="box" size="0.1 0.05 0.05" mass="1.0"/>

      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.005"/>
        <geom name="pole" type="capsule" fromto="0 0 0 0 0 0.6"
              size="0.025" mass="0.1" rgba="0 0.7 0.7 1"/>
        <site name="tip" pos="0 0 0.6" size="0.02" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="slide" joint="slider" gear="10"
           ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>

  <keyframe>
    <!-- 摆自然下垂：hinge = pi -->
    <key name="hanging" qpos="0 3.14159265"/>
  </keyframe>
</mujoco>
```

### 关键设计点

| 项目 | 说明 |
| --- | --- |
| 关节顺序 | `qpos[0] = x`（小车位置），`qpos[1] = θ`（摆角）；`qvel` 同理 |
| 角度约定 | 摆杆沿自身 \(+z\)，绕 \(y\) 轴转 \(\theta\) 后世界方向为 \((\sin\theta,\,0,\,\cos\theta)\)。故 \(\theta=0\) 为**竖直向上**，\(\theta=\pi\) 为下垂；杆尖高度即 \(l\cos\theta\)，可直接用作奖励 |
| hinge 不加 `range` | swing-up 必须允许摆自由旋转多圈，加限位会直接破坏任务 |
| `contype/conaffinity=0` | cartpole 无需碰撞检测，关掉可加速并避免杆撞导轨的数值抖动 |
| `gear=10` + `ctrlrange=-1 1` | 策略输出归一化到 \([-1,1]\)，实际推力 \(\pm 10\,\mathrm{N}\) |
| 阻尼 | 摆的 `damping` 要小（0.001～0.01），否则能量泵不起来 |

关于 `gear` 的取值：这个值必须卡准——太大则一步就能甩上去（任务退化为倒立摆平衡），太小则永远甩不上去。对上表的质量参数，5～15 N 是合理区间，**必须小于**能一次性拉起摆杆的力，才能逼出「来回摆荡蓄能」的行为。

控制频率建议 `frame_skip=4`（即 50 Hz 决策、200 Hz 物理仿真），episode 长度取 500～1000 步。

---

## 二、观测设计

**不要直接把 \(\theta\) 喂给网络**，因为 \(0\) 与 \(2\pi\) 在数值上不连续。推荐使用：

\[
s = [\,x,\ \sin\theta,\ \cos\theta,\ \dot{x},\ \dot{\theta}\,]
\]

\(\dot\theta\) 建议 clip 到 \([-10,\ 10]\) 左右，防止早期随机策略产生极端值把网络打炸。

---

## 三、奖励函数

### 方案 A：加性密集奖励

最直观，但容易调坏。

\[
r_t = \cos\theta_t \;-\; 0.1\,x_t^2 \;-\; 0.001\,u_t^2
\]

取值范围大致为 \([-1,\ 1]\)。

> **注意**：不要无条件惩罚 \(\dot\theta^2\)。这是最常见的错误，会直接抑制蓄能摆荡，导致策略卡在底部不动。

如果需要抑制顶部抖动，改成门控形式：

\[
r_t = \cos\theta_t - 0.1\,x_t^2 - 0.001\,u_t^2 - 0.05\,\mathbb{1}[\cos\theta_t > 0.9]\cdot\dot\theta_t^2
\]

### 方案 B：乘性奖励（推荐）

dm_control 的做法。各项都归一化到 \([0,1]\) 后相乘，任何一项不满足都会拉低总分，无需手调各项权重的相对大小：

\[
r_t = r_{\text{up}}\cdot r_{\text{cen}}\cdot r_{\text{vel}}\cdot r_{\text{ctrl}}
\]

\[
r_{\text{up}} = \frac{1+\cos\theta}{2},\qquad
r_{\text{cen}} = \frac{1 + \sigma(x;\,2.0)}{2}
\]

\[
r_{\text{vel}} = \frac{1 + \sigma(\dot\theta;\,5.0)}{2},\qquad
r_{\text{ctrl}} = \frac{4 + \sigma(u;\,1.0)}{5}
\]

其中 \(\sigma\) 为高斯容差函数：

\[
\sigma(z; m) = \exp\!\left(-\tfrac{1}{2}\big(z\,\lambda/m\big)^2\right),\quad
\lambda = \sqrt{-2\ln(0.1)}
\]

即在 \(|z|=m\) 处衰减到 0.1。

注意 \(r_{\text{vel}}\)、\(r_{\text{ctrl}}\) 的基底偏移（\(+1\) 除以 2、\(+4\) 除以 5）：它们只做温和调制，最低也有 0.5 和 0.8，不会把主奖励压死——这是乘性设计能 work 的关键。

### 方案 C：能量整形

收敛快，但引入了较强的先验。摆的机械能为

\[
E = \tfrac{1}{2}ml^2\dot\theta^2 + mgl\cos\theta
\]

目标能量 \(E^\ast = mgl\)，则

\[
r_t = -\,|E_t - E^\ast| \;-\; 0.1\,x_t^2
\]

配合顶部切换到 LQR，或再叠加一项 \(\cos\theta\)。适合做 baseline 对比；纯 RL 场景一般用方案 A 或 B。

### 参考实现

```python
import numpy as np

LAMBDA = np.sqrt(-2.0 * np.log(0.1))

def tol(z, margin):
    return np.exp(-0.5 * (z * LAMBDA / margin) ** 2)

def reward(data, action):
    x, theta = data.qpos[0], data.qpos[1]
    x_dot, theta_dot = data.qvel[0], data.qvel[1]

    upright  = (1.0 + np.cos(theta)) / 2.0
    centered = (1.0 + tol(x, 2.0)) / 2.0
    slow     = (1.0 + tol(theta_dot, 5.0)) / 2.0
    quiet    = (4.0 + tol(action[0], 1.0)) / 5.0
    return upright * centered * slow * quiet
```

---

## 四、常见踩坑点

### 1. 初始状态随机化

这是能否学出来的分水岭。固定从 \(\theta=\pi\) 出发时探索会非常慢，建议：

\[
\theta_0 \sim \mathcal{U}(\pi - 0.3,\ \pi + 0.3),\quad
x_0 \sim \mathcal{U}(-0.1,\ 0.1),\quad
\dot{q}_0 \sim \mathcal{N}(0,\ 0.05^2)
\]

更激进的做法是让 \(\theta_0 \sim \mathcal{U}(-\pi,\ \pi)\) 覆盖全状态空间，相当于自带课程学习，SAC / PPO 都会明显更好学。

### 2. 不要设失败终止

swing-up 是纯 shaping 任务，提前终止（例如杆倒了就 `done`）会让智能体学会「快速结束以避免负奖励」。应让 episode 跑满固定步数，只在小车撞限位时给一次性惩罚，或者干脆依靠 XML 中 `range` 的物理约束、不给额外惩罚。

### 3. 算法选择

- **SAC**：在该任务上基本 100k～200k step 内稳定收敛，首选。
- **PPO**：需要更多样本（约 1M step），且对奖励尺度更敏感；使用时记得开启 observation normalization 与 reward scaling。
