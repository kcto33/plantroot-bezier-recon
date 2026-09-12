# -*- coding: utf-8 -*-
"""
triangulate.py
多视角三角化(DLT), 用已知相机投影矩阵 P_i 与 2D 同名点 x_i 反求 3D 点 X。
"""
import numpy as np


def triangulate_dlt(projections, points2d):
    """DLT 多视角三角化。
    projections: list[np (3,4)] 或 (M,3,4) 投影矩阵
    points2d:    (M,2) 各视角同名像素坐标
    返回齐次 4 向量归一化后的世界坐标 (3,)。
    """
    projections = np.atleast_3d(np.asarray(projections, dtype=float))
    points2d = np.atleast_2d(np.asarray(points2d, dtype=float))
    M = projections.shape[0]
    A = []
    for i in range(M):
        P = projections[i]
        u, v = points2d[i]
        x = np.array([u, v, 1.0])
        # x × (P X) = 0, 取两行为独立方程
        A.append(x[1] * P[2, :] - x[2] * P[1, :])
        A.append(x[2] * P[0, :] - x[0] * P[2, :])
        # 第三行 (x[0]*P[1]-x[1]*P[0]) 与前两行线性相关, 不重复使用
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]                      # 最小奇异值对应的右奇异向量
    X = Xh[:3] / Xh[3]               # 齐次归一化
    return X


def reproject_error(projection, point3d, point2d):
    """单个视角的重投影误差(像素)。"""
    Xh = np.append(np.asarray(point3d, dtype=float), 1.0)
    proj = projection @ Xh
    proj = proj[:2] / proj[2]
    return float(np.linalg.norm(proj - np.asarray(point2d, dtype=float)))


def triangulate_robust(projections, points2d, tol_px=2.0, min_views=3,
                       max_rounds=None):
    """迭代剔除坏视角的多视角三角化(P1-5)。
    每轮: DLT -> 各视角重投影误差 + cheirality(正深度)检查 -> 剔除误差最大
    (或深度为负)的视角, 直到最大残差 <= tol_px 或只剩 min_views 个视角。
    projections: (M,3,4); points2d: (M,2)。
    返回 (X, keep_idx, resids); X 为 None 表示无法三角化。"""
    P = np.atleast_3d(np.asarray(projections, dtype=float))
    uv = np.atleast_2d(np.asarray(points2d, dtype=float))
    M = P.shape[0]
    keep = list(range(M))
    if max_rounds is None:
        max_rounds = max(0, M - min_views)
    X = None
    for _ in range(max_rounds + 1):
        if len(keep) < 2:
            return None, [], []
        X = triangulate_dlt(P[keep], uv[keep])
        Xh = np.append(X, 1.0)
        depths = np.array([(P[i] @ Xh)[2] for i in keep])   # 相机系深度
        errs = np.array([np.linalg.norm((P[i] @ Xh)[:2] / (P[i] @ Xh)[2] - uv[i])
                         for i in keep])
        bad = (errs > tol_px) | (depths <= 0)
        if not bad.any() or len(keep) <= min_views:
            break
        score = errs.astype(float).copy()
        score[depths <= 0] = np.inf      # 负深度(点在相机后方)优先剔除
        del keep[int(np.argmax(score))]
    if X is None:
        return None, [], []
    resids = []
    for i in keep:
        pr = P[i] @ np.append(X, 1.0)
        pr = pr[:2] / pr[2]
        resids.append(float(np.linalg.norm(pr - uv[i])))
    return X, keep, resids
