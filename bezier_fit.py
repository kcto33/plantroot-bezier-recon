# -*- coding: utf-8 -*-
"""
bezier_fit.py
真正的最小二乘三次贝塞尔拟合。

给定一组三维点(曲线上的采样点云), 拟合一条三次贝塞尔曲线:
    P(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

做法(标准最小二乘):
  - 端点固定: P0 = 首点, P3 = 末点。
  - 用累计弦长参数化每个点得 t_i ∈ [0,1]。
  - 对每个坐标, 解线性最小二乘求两个控制点 P1, P2。
"""
import numpy as np


def chord_length_param(pts):
    """累计弦长参数化, 返回 t ∈ [0,1]。pts: (N,3)。"""
    diff = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diff)])
    total = cum[-1]
    if total < 1e-9:
        return np.linspace(0, 1, len(pts))
    return cum / total


def basis(t):
    """三次贝塞尔基函数。t: (N,) -> (N,4) [b0,b1,b2,b3]。"""
    t = np.asarray(t, dtype=float)
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3
    return np.stack([b0, b1, b2, b3], axis=1)


def fit_cubic_bezier(pts):
    """对 pts: (N,3) 拟合三次贝塞尔。返回 (P0,P1,P2,P3) 及残差。"""
    pts = np.asarray(pts, dtype=float)
    N = pts.shape[0]
    if N < 4:
        raise ValueError("至少需要 4 个点来拟合三次贝塞尔")
    P0 = pts[0]
    P3 = pts[-1]
    t = chord_length_param(pts)
    B = basis(t)
    b1 = B[:, 1]
    b2 = B[:, 2]
    b0 = B[:, 0]
    b3 = B[:, 3]

    # 残差: R_i = Q_i - b0_i*P0 - b3_i*P3 ;  A_i = [b1_i, b2_i]
    A = np.stack([b1, b2], axis=1)          # (N,2)
    P = np.zeros((3, 2))
    residual = np.zeros((N, 3))
    for d in range(3):
        R = pts[:, d] - b0 * P0[d] - b3 * P3[d]
        sol, res, *_ = np.linalg.lstsq(A, R, rcond=None)
        P[d] = sol
        residual[:, d] = R - A @ sol
    P1 = P[:, 0]
    P2 = P[:, 1]
    # 拟合残差
    fitted = b0[:, None] * P0[None, :] + b1[:, None] * P1[None, :] + \
             b2[:, None] * P2[None, :] + b3[:, None] * P3[None, :]
    err = np.linalg.norm(pts - fitted, axis=1)
    return P0, P1, P2, P3, err


def bernstein_basis(degree, t):
    """n 次贝塞尔基函数。t: (N,) -> (N, degree+1)。"""
    t = np.asarray(t, dtype=float)
    from scipy.special import comb
    n = degree
    B = []
    for i in range(n + 1):
        B.append(comb(n, i) * (1 - t) ** (n - i) * t ** i)
    return np.stack(B, axis=1)


def fit_bezier_any(pts, degree=3):
    """对 pts: (N,3) 拟合 n 次贝塞尔(端点固定, 最小二乘反解中间 n-1 个控制点)。
    返回 (controls, err)。controls: (degree+1, 3)。"""
    pts = np.asarray(pts, dtype=float)
    N = pts.shape[0]
    n = degree
    if N < n + 1:
        raise ValueError(f"需要至少 {n+1} 个点拟合 {n} 次贝塞尔")
    t = chord_length_param(pts)
    B = bernstein_basis(n, t)            # (N, n+1)
    P0 = pts[0]
    Pn = pts[-1]
    controls = np.zeros((n + 1, 3))
    controls[0] = P0
    controls[-1] = Pn
    # 未知的是中间 n-1 个控制点
    mid_idx = np.arange(1, n)            # 控制点 index 1..n-1
    A = B[:, mid_idx]                    # (N, n-1)
    b0 = B[:, 0]
    bn = B[:, n]
    for d in range(3):
        R = pts[:, d] - b0 * P0[d] - bn * Pn[d]
        sol, *_ = np.linalg.lstsq(A, R, rcond=None)
        controls[mid_idx, d] = sol
    # 残差
    fitted = B @ controls
    err = np.linalg.norm(pts - fitted, axis=1)
    return controls, err


def sample_bezier_any(controls, n=80):
    """由 (degree+1,3) 控制点采样 n 个曲线点。"""
    controls = np.asarray(controls, dtype=float)
    degree = controls.shape[0] - 1
    t = np.linspace(0, 1, n)[:, None]
    B = bernstein_basis(degree, np.linspace(0, 1, n))
    return B @ controls


def sample_cubic(ctrl, n=50):
    """由 (P0,P1,P2,P3) 采样 n 个点。"""
    P0, P1, P2, P3 = ctrl
    t = np.linspace(0, 1, n)[:, None]
    B = np.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                  3 * (1 - t) * t ** 2, t ** 3], axis=2)
    B = B.reshape(n, 4)
    return (B[:, 0, None] * P0[None, :] + B[:, 1, None] * P1[None, :] +
            B[:, 2, None] * P2[None, :] + B[:, 3, None] * P3[None, :])


def fit_model_curves(model, n_out=50):
    """对 model.curves 里每条曲线的采样点做最小二乘贝塞尔拟合。
    返回: fitted 列表, 每项 {curve_id, depth, ctrl, err_max, err_mean, points, radii}。
    粗细由原始曲线各点半径沿 t 平滑得到(端点半径用附近点平均)。
    """
    curves = model.curves
    out = []
    for ci, c in enumerate(curves):
        pts = c['points']
        radii = c['radii']
        P0, P1, P2, P3, err = fit_cubic_bezier(pts)
        t = chord_length_param(pts)

        # 沿拟合曲线采样点
        spts = sample_cubic((P0, P1, P2, P3), n=n_out)
        # 采样点的半径: 用拟合后的参数 t 在 0..1 均匀采样, 插值原始半径
        t_out = np.linspace(0, 1, n_out)
        srad = np.interp(t_out, t, radii)

        out.append({
            'curve_id': ci,
            'depth': c['depth'],
            'ctrl': (P0, P1, P2, P3),
            'err_max': float(err.max()),
            'err_mean': float(err.mean()),
            'points': spts,
            'radii': srad,
        })
    return out
