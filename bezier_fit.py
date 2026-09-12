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


def fit_bezier_any(pts, degree=3, lam=0.0, t=None):
    """对 pts: (N,3) 拟合 n 次贝塞尔(端点固定, 最小二乘反解中间 n-1 个控制点)。
    t: 各点的曲线参数(N,)。缺省用累计弦长参数化; 若测点来自"沿参数均匀采样"
    (本项目标记约定), 传入 t=k/(n-1) 可避免弦长参数化与生成参数失配引起的
    控制点震荡与形状偏差。lam>0 时对控制点二阶差分加正则(lambda*||D2 C||^2)。
    返回 (controls, err)。controls: (degree+1, 3)。"""
    pts = np.asarray(pts, dtype=float)
    N = pts.shape[0]
    n = degree
    if N < n + 1:
        raise ValueError(f"需要至少 {n+1} 个点拟合 {n} 次贝塞尔")
    tt = chord_length_param(pts) if t is None else np.asarray(t, dtype=float)
    B = bernstein_basis(n, tt)           # (N, n+1)
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
    m = len(mid_idx)
    if lam > 0 and m >= 2:
        # 二阶差分矩阵(只含未知控制点部分)与端点常数项
        D = np.zeros((m, m))
        for i in range(m):
            D[i, i] = -2.0
            if i > 0:
                D[i, i - 1] += 1.0
            if i < m - 1:
                D[i, i + 1] += 1.0
        # s_i = C_{i-1} - 2C_i + C_{i+1}; 首末行含固定端点 P0/Pn
        Dd = D.copy()
        const = np.zeros((m, 3))
        Dd[0, 0] = -2.0; const[0] = P0
        Dd[-1, -1] = -2.0; const[-1] = Pn
        AtA = A.T @ A + lam * (Dd.T @ Dd)
        for d in range(3):
            R = pts[:, d] - b0 * P0[d] - bn * Pn[d]
            rhs = A.T @ R - lam * (Dd.T @ const[:, d])
            controls[mid_idx, d] = np.linalg.solve(AtA, rhs)
    else:
        for d in range(3):
            R = pts[:, d] - b0 * P0[d] - bn * Pn[d]
            sol, *_ = np.linalg.lstsq(A, R, rcond=None)
            controls[mid_idx, d] = sol
    # 残差
    fitted = B @ controls
    err = np.linalg.norm(pts - fitted, axis=1)
    return controls, err


def fit_bezier_full(pts, t, degree=5):
    """自由端点的最小二乘贝塞尔拟合(端点不固定, 全部 degree+1 个控制点由 LS 解出)。
    用于"端点测量不可靠/缺失时, 由可靠测点预测端点坐标"。
    返回 (controls (degree+1,3), err)。"""
    pts = np.asarray(pts, dtype=float)
    tt = np.asarray(t, dtype=float)
    n = degree
    if len(pts) < n + 1:
        raise ValueError(f"需要至少 {n+1} 个点拟合 {n} 次贝塞尔")
    B = bernstein_basis(n, tt)
    controls = np.zeros((n + 1, 3))
    for d in range(3):
        controls[:, d], *_ = np.linalg.lstsq(B, pts[:, d], rcond=None)
    err = np.linalg.norm(pts - B @ controls, axis=1)
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


def remove_outliers(P, sigma=3.0):
    """用低阶贝塞尔为基准, 迭代剔除距基准过远的离群测点。返回 (P_clean, 掩码)。"""
    P = np.asarray(P, dtype=float)
    if len(P) <= 5:
        return P, np.ones(len(P), dtype=bool)
    keep = np.ones(len(P), dtype=bool)
    for _ in range(3):
        Pk = P[keep]
        if len(Pk) < 5:
            break
        # 低阶(3次)基准, 稳健不摆动
        try:
            P0, p1, p2, p3, err = fit_cubic_bezier(Pk)
        except Exception:
            break
        t = np.linspace(0, 1, 60)[:, None]
        base = ((1-t)**3*P0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3)
        d = np.linalg.norm(Pk[:, None, :] - base[None, :, :], axis=2).min(axis=1)
        med = np.median(d)
        if med < 1e-4:
            break
        thresh = max(sigma * med, 0.3)     # 阈值: 中位残差*sigma, 至少0.3
        good = d <= thresh
        # 更新原始掩码
        idxs = np.where(keep)[0]
        new_keep = keep.copy()
        new_keep[idxs[~good]] = False
        if new_keep.sum() < 5:
            break
        keep = new_keep
    return P[keep], keep


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
