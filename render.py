# -*- coding: utf-8 -*-
"""
render.py
用自定义针孔相机 + z-buffer 在 numpy 中把三维根系渲染成 2D 图像。
点状渲染: 每个采样点投影为像素圆盘, 半径由该点三维半径/深度决定, 带 z-buffer 遮挡。
目标点用醒目颜色单独渲染, 便于后续颜色定位。
"""
import numpy as np
from PIL import Image


def render_points(pts, radii, cam, width=640, height=640,
                  bg=(255, 255, 255), color=(90, 55, 25),
                  k_scale=0.5, zmin=None, zmax=None):
    """把三维点渲染为图像。
    pts: (N,3) 世界坐标; radii: (N,) 各点半径; cam: PinholeCamera。
    返回 RGB 图像 + z-buffer (像素处的世界 z 深度, 用于遮挡/重投影比较)。
    """
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:] = bg

    # 投影到像素
    uv = cam.project(pts)
    Xc = (cam.R @ pts.T + cam.t).T
    Z = Xc[:, 2]

    zbuf = np.full((height, width), np.inf)

    # 圆盘半径(像素) = f * 三维半径 / 深度, 乘一个比例增强可见度
    r_pix = cam.fx * radii / np.maximum(Z, 1e-6) * k_scale

    for i in range(len(pts)):
        u, v = uv[i]
        if u < 0 or u >= width or v < 0 or v >= height:
            continue
        z = Z[i]
        r = max(1.0, r_pix[i])
        u0, v0 = int(u), int(v)
        y0, y1 = max(0, v0 - int(r)), min(height, v0 + int(r) + 1)
        x0, x1 = max(0, u0 - int(r)), min(width, u0 + int(r) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        mask = (xs - u) ** 2 + (ys - v) ** 2 <= r ** 2
        if not mask.any():
            continue
        sub_z = np.where(mask, z, np.inf)
        upd = sub_z < zbuf[y0:y1, x0:x1]
        zbuf[y0:y1, x0:x1] = np.where(upd, sub_z, zbuf[y0:y1, x0:x1])
        for c in range(3):
            sub = image[y0:y1, x0:x1, c]
            sub[mask] = np.where(upd[mask], color[c], sub[mask])

    return image.astype(np.uint8), zbuf


def render_curves(curves, cam, width=640, height=640,
                  bg=(255, 255, 255), color=(90, 55, 25),
                  k_scale=0.6, subdiv=8, flip_z=False):
    """把多条曲线渲染成连续带粗细的管状线条。
    curves: list of dict, 每条含 'points'(N,3) 和 'radii'(N,)。
    做法: 对每条曲线的相邻采样点, 用弧长线性插值加密出 subdiv 个中间点,
          再把每个(加密)点画成深度感知的圆盘, 圆盘间彼此重叠 -> 连续曲线。
    flip_z: True 时把世界坐标 z 取反后再投影(仅显示翻转, 不改几何数据)。
    返回 (image, zbuf)。
    """
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:] = bg
    zbuf = np.full((height, width), np.inf)

    for c in curves:
        pts = np.asarray(c['points'], dtype=float)
        radii = np.asarray(c['radii'], dtype=float)
        n = pts.shape[0]
        if n < 2:
            continue
        # 弧长参数化
        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = seglen.sum()
        if total < 1e-9:
            continue

        # 逐段加密(线性插值点 + 线性插值半径)
        dense_pts = []
        dense_r = []
        for i in range(n - 1):
            p0, p1 = pts[i], pts[i + 1]
            r0, r1 = radii[i], radii[i + 1]
            for s in range(subdiv):
                t = s / subdiv
                dense_pts.append(p0 + (p1 - p0) * t)
                dense_r.append(r0 + (r1 - r0) * t)
        dense_pts.append(pts[-1])
        dense_r.append(radii[-1])
        dense_pts = np.array(dense_pts)
        dense_r = np.array(dense_r)

        # 仅显示翻转: 投影前把世界坐标 z 取反(不改几何数据)
        if flip_z:
            dense_pts = dense_pts.copy()
            dense_pts[:, 2] = -dense_pts[:, 2]

        # 投影
        uv = cam.project(dense_pts)
        Xc = (cam.R @ dense_pts.T + cam.t).T
        Z = Xc[:, 2]
        r_pix = cam.fx * dense_r / np.maximum(Z, 1e-6) * k_scale

        for i in range(len(dense_pts)):
            u, v = uv[i]
            if u < 0 or u >= width or v < 0 or v >= height:
                continue
            z = Z[i]
            r = max(1.0, r_pix[i])
            u0, v0 = int(u), int(v)
            y0, y1 = max(0, v0 - int(r)), min(height, v0 + int(r) + 1)
            x0, x1 = max(0, u0 - int(r)), min(width, u0 + int(r) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            mask = (xs - u) ** 2 + (ys - v) ** 2 <= r ** 2
            if not mask.any():
                continue
            sub_z = np.where(mask, z, np.inf)
            upd = sub_z < zbuf[y0:y1, x0:x1]
            zbuf[y0:y1, x0:x1] = np.where(upd, sub_z, zbuf[y0:y1, x0:x1])
            for cc in range(3):
                sub = image[y0:y1, x0:x1, cc]
                sub[mask] = np.where(upd[mask], color[cc], sub[mask])

    return image.astype(np.uint8), zbuf


def render_marker(point3d, cam, image, zbuf, color=(0, 200, 0), marker_r=12):
    """在已渲染图像上以醒目颜色画一个目标点标记(投影该三维点)。"""
    u, v = cam.project(point3d)
    Xc = (cam.R @ np.asarray(point3d).reshape(3, 1) + cam.t).ravel()
    z = Xc[2]
    width, height = image.shape[1], image.shape[0]
    if 0 <= u < width and 0 <= v < height:
        u0, v0 = int(u), int(v)
        y0, y1 = max(0, v0 - marker_r), min(height, v0 + marker_r + 1)
        x0, x1 = max(0, u0 - marker_r), min(width, u0 + marker_r + 1)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        mask = (xs - u) ** 2 + (ys - v) ** 2 <= marker_r ** 2
        # 只在与目标点深度接近处着色(受 z-buffer 约束, 避免穿透写)
        if mask.any():
            for c in range(3):
                sub = image[y0:y1, x0:x1, c]
                sub[mask] = color[c]
    return u, v, z


def save_image(image, path):
    Image.fromarray(image.astype(np.uint8)).save(path)


def detect_colored_blob(image, color, tol=40):
    """按颜色阈值定位彩色标记的质心像素。返回 2D 质心或 None。"""
    rgb = image.astype(np.int64)
    mask = np.all(np.abs(rgb - np.asarray(color)) <= tol, axis=-1)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])
