"""
AoBG (alpha-order batched gradient) —— Suh et al., ICML 2022 的实现。

论文：Suh, Simchowitz, Zhang, Tedrake,
"Do Differentiable Simulators Give Better Policy Gradients?"（arXiv:2202.00817）

与 DDCG 同属参数空间复合梯度（随机平滑 + 0/1 阶混合），共享 ddcg.py 的
扰动 rollout 骨架，仅权重选择规则不同。AoBG 的规则是一个带精度约束的
方差最小化（论文 Eq. 4）：

    min_alpha  alpha^2 v1 + (1-alpha)^2 v0
    s.t.       eps + alpha * B <= gamma

其中
  * B = ||g1 - g0||           1 阶与 0 阶均值梯度之差（偏差的替代度量）
  * eps                       ZoBG 的置信半径，经验 Bernstein 向量集中界
                              （附录 C.4，置信度 delta=0.05）：
        eps = sqrt(2 * sigma_bar0^2 * log((d+1)/delta) / N)
              + (2R / 3N) * log((d+1)/delta)
      sigma_bar0^2 = sum_i ||g0_i||^2（经验方差上界），
      R = max_i ||g0_i - g0||（样本最大偏离的"教育猜测"）
  * gamma                     精度阈值，**逐任务调参**（这是 AoBG 被诟病之处，
                              DDCG 论文附录 K：cartpole=1, hopper=1e5, ant=1e6）

闭式解（论文 Lemma 4.4 / Eq. 5）：
    alpha = alpha_inf                若 alpha_inf * B <= gamma - eps
          = (gamma - eps) / B        否则（压缩 1 阶权重以满足精度）
    若 eps > gamma（样本不足，不可行）→ alpha = 0，完全退回 0 阶。

与原文的实现差异（均已文档化）：
  * 原文 FoBG / ZoBG 用独立轨迹采样（附录 C.1），这里与 DDCG 实现一致，
    共用同一批扰动样本（更省模拟量，对决策影响很小）；
  * R 的"先验上界"按原文建议取批内样本最大偏离。
"""

import jax.numpy as jnp

from .ddcg import DDCG


class AoBG(DDCG):

    log_keys = ("episode_return", "alpha", "feasible", "B", "eps_conf",
                "ret_mean", "actor_lr")

    # ------------------------------------------------------------------
    def _select_gradient(self, st):
        """AoBG 的权重选择：Eq. 4 的约束优化 + Lemma 4.4 闭式解。"""
        cfg = self.cfg
        g0, g1, v0, v1 = st["g0"], st["g1"], st["v0"], st["v1"]
        gamma, delta = cfg.aobg_gamma, cfg.aobg_delta
        N, d = st["N"], st["d"]

        # ---- 偏差替代度量 B = ||g1 - g0||（Eq. 4 约束项）----
        B = jnp.linalg.norm(g1 - g0)

        # ---- 置信半径 eps：经验 Bernstein 向量集中界（附录 C.4）----
        G0f = st["G0f"]                                        # (N, D)
        sigma_bar0_sq = jnp.sum(jnp.sum(G0f ** 2, axis=1))     # sum_i ||g0_i||^2
        R_bound = jnp.max(jnp.linalg.norm(G0f - g0[None], axis=1))
        log_term = jnp.log((d + 1.0) / delta)
        eps = (jnp.sqrt(2.0 * sigma_bar0_sq * log_term / N)
               + (2.0 * R_bound) / (3.0 * N) * log_term)

        # ---- 闭式解（Lemma 4.4 / Eq. 5）----
        alpha_inf = v0 / (v0 + v1 + 1e-12)                     # IVW 权重
        feasible = eps <= gamma                                # 样本量可行性
        within = alpha_inf * B <= gamma - eps                  # 约束内直接用 IVW
        alpha_constrained = (gamma - eps) / jnp.maximum(B, 1e-12)
        alpha = jnp.where(feasible,
                          jnp.where(within, alpha_inf, alpha_constrained),
                          0.0)
        alpha = jnp.clip(alpha, 0.0, 1.0)

        g = alpha * g1 + (1.0 - alpha) * g0
        return alpha, g, dict(
            alpha_opt=alpha_inf, B=B, eps_conf=eps,
            feasible=feasible.astype(jnp.float32),
            within=within.astype(jnp.float32),
        )
