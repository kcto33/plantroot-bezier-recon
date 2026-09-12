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


def _densify(pts, radii, subdiv):
    """弧长线性插值加密采样点与半径。"""
    dense_pts = []
    dense_r = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        r0, r1 = radii[i], radii[i + 1]
        for s in range(subdiv):
            t = s / subdiv
            dense_pts.append(p0 + (p1 - p0) * t)
            dense_r.append(r0 + (r1 - r0) * t)
    dense_pts.append(pts[-1])
    dense_r.append(radii[-1])
    return np.array(dense_pts), np.array(dense_r)


def render_curves(curves, cam, width=640, height=640,
                  bg=(255, 255, 255), color=(90, 55, 25),
                  k_scale=0.6, subdiv=8, flip_z=False,
                  auto_fit=False, margin=24):
    """把多条曲线渲染成连续带粗细的管状线条。
    curves: list of dict, 每条含 'points'(N,3) 和 'radii'(N,)。
    做法: 对每条曲线的相邻采样点, 用弧长线性插值加密出 subdiv 个中间点,
          再把每个(加密)点画成深度感知的圆盘, 圆盘间彼此重叠 -> 连续曲线。
    flip_z: True 时把世界坐标 z 取反后再投影(仅显示翻转, 不改几何数据)。
    auto_fit: True 时对全部投影点做包围盒, 缩放平移使整根完整入画
              (仅用于展示用途; 需保持原始投影对齐标记的场景必须用 False)。
    返回 (image, zbuf)。
    """
    image = np.zeros((height, width, 3), dtype=np.float64)
    image[:] = bg
    zbuf = np.full((height, width), np.inf)

    # ---- 自动取景: 预投影全部加密点, 求包围盒 -> 等比缩放 + 平移 ----
    fit_scale, fit_dx, fit_dy = 1.0, 0.0, 0.0
    if auto_fit:
        all_uv = []
        for c in curves:
            pts = np.asarray(c['points'], dtype=float)
            if pts.shape[0] < 2:
                continue
            dp = _densify(pts, np.asarray(c['radii'], dtype=float), subdiv)[0]
            if flip_z:
                dp = dp.copy(); dp[:, 2] = -dp[:, 2]
            all_uv.append(cam.project(dp))
        if all_uv:
            uv_cat = np.vstack(all_uv)
            uv_cat = uv_cat[np.isfinite(uv_cat).all(axis=1)]
        if all_uv and len(uv_cat):
            x0, y0 = uv_cat.min(axis=0)
            x1, y1 = uv_cat.max(axis=0)
            bw, bh = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            s = min((width - 2 * margin) / bw, (height - 2 * margin) / bh)
            fit_scale = min(s, 8.0)   # 包围盒极小时限制过度放大
            fit_dx = width / 2 - (x0 + x1) / 2 * fit_scale
            fit_dy = height / 2 - (y0 + y1) / 2 * fit_scale

    def _fit_uv(uv):
        if fit_scale == 1.0 and fit_dx == 0.0 and fit_dy == 0.0:
            return uv
        return uv * fit_scale + np.array([fit_dx, fit_dy])

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

        dense_pts, dense_r = _densify(pts, radii, subdiv)

        # 仅显示翻转: 投影前把世界坐标 z 取反(不改几何数据)
        if flip_z:
            dense_pts = dense_pts.copy()
            dense_pts[:, 2] = -dense_pts[:, 2]

        # 投影
        uv = _fit_uv(cam.project(dense_pts))
        Xc = (cam.R @ dense_pts.T + cam.t).T
        Z = Xc[:, 2]
        r_pix = cam.fx * dense_r / np.maximum(Z, 1e-6) * k_scale * fit_scale

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


def render_marker(point3d, cam, image, zbuf, color=(0, 200, 0), marker_r=12,
                  occupied=None):
    """在已渲染图像上以醒目颜色画一个目标点标记(投影该三维点), 写完整圆盘并写 zbuf。
    occupied: 已画标记的占用掩码(可选)。若本标记圆盘与已画标记重叠则整个跳过
    (互斥): 被部分覆盖的标记会剩新月形残片, 检测质心会被拉偏; 互斥保证
    每个画出的标记都是完整圆盘, 检测质心无偏(P1-4)。"""
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
        if mask.any():
            if occupied is not None:
                if (occupied[y0:y1, x0:x1] & mask).any():
                    return u, v, z      # 与其他标记重叠: 整个不画(不留残片)
                occupied[y0:y1, x0:x1] |= mask
            zbuf[y0:y1, x0:x1] = np.where(
                mask, np.minimum(z, zbuf[y0:y1, x0:x1]), zbuf[y0:y1, x0:x1])
            for c in range(3):
                sub = image[y0:y1, x0:x1, c]
                sub[mask] = color[c]
    return u, v, z


def save_image(image, path):
    Image.fromarray(image.astype(np.uint8)).save(path)


def occlusion_visible(point3d, cam, zbuf, tol=0.15):
    """判断 point3d 在该相机下是否未被遮挡(深度不大于该处 zbuf)。"""
    u, v = cam.project(point3d)
    Xc = (cam.R @ np.asarray(point3d).reshape(3, 1) + cam.t).ravel()
    z = Xc[2]
    if z <= 0:
        return False
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or ui >= zbuf.shape[1] or vi < 0 or vi >= zbuf.shape[0]:
        return False
    return z <= zbuf[vi, ui] + tol


def draw_markers(targets, cam, image, zbuf, k_scale=0.6,
                 min_marker_r=5, max_marker_r=22.0, order_shift=0):
    """绘制带互斥的标记列表(P1-4)。
    targets: 每项含 'coord'(3,), 'color'((r,g,b)), 'radius'(三维半径)。

    绘制顺序 = 基础随机排列按视角滚动(order_shift 为视角索引):
    分叉节点处常有多个标记互相重叠(同一节点既是父曲线测点又是子曲线起点,
    且各点深度次序在环视相机下基本固定), 若按固定顺序绘制, 其中某个标记
    会在所有视角都被覆盖成新月残片甚至完全消失。互斥(与已画标记重叠者
    本视角整体跳过、不留残片)保证每个画出的标记都是完整圆盘、检测质心无偏;
    而按视角滚动绘制顺序使重叠簇内各成员确定性地轮流做"第一画者",
    各获得约 1/簇大小 的视角, 不再有永久输家。
    每个标记仍先过中心可见性测试(被表面遮挡的点本视角不画)。
    """
    zs = []
    for tp in targets:
        Xc = (cam.R @ np.asarray(tp['coord']).reshape(3, 1) + cam.t).ravel()
        zs.append(Xc[2])
    zs = np.asarray(zs)
    # 多套基础排列轮换: 若两个簇成员在某套排列中恰好相邻(其"优先窗口"过小),
    # 在其他排列中大概率不相邻, 从而避免饿死。
    perms = [np.random.default_rng(s).permutation(len(zs))
             for s in (12345, 24680, 13579)]
    occupied = np.zeros(image.shape[:2], dtype=bool)
    # 滚动步长取与 n 互素的数(37 与常见 n=120/8 互素), 使各视角的起点
    # 均匀铺满整个循环 => 重叠簇内各成员确定性地轮流做"第一画者"。
    perm = perms[order_shift % len(perms)]
    shift = (order_shift * 37) % max(len(zs), 1)
    for ti in np.roll(perm, shift):
        tp = targets[ti]
        if zs[ti] <= 0 or not occlusion_visible(tp['coord'], cam, zbuf):
            continue
        r_pix = min(max_marker_r, cam.fx * tp['radius'] / max(zs[ti], 1e-6) * k_scale)
        render_marker(tp['coord'], cam, image, zbuf, color=tp['color'],
                      marker_r=int(round(max(min_marker_r, r_pix))),
                      occupied=occupied)


def detect_colored_blob(image, color, tol=40):
    """按颜色阈值定位彩色标记的质心像素。返回 2D 质心或 None。"""
    rgb = image.astype(np.int64)
    mask = np.all(np.abs(rgb - np.asarray(color)) <= tol, axis=-1)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def export_model_json(curves, path, max_pts=140):
    """把重建曲线导出为前端 3D 查看器用的 JSON(每条曲线降采样)。"""
    import json
    out = []
    for c in curves:
        pts = np.asarray(c['points'], dtype=float)
        radii = np.asarray(c['radii'], dtype=float)
        n = pts.shape[0]
        if n < 2:
            continue
        idx = np.unique(np.linspace(0, n - 1, min(max_pts, n)).astype(int))
        out.append({'pts': [[round(float(v), 4) for v in p] for p in pts[idx]],
                    'r': [round(float(r), 4) for r in radii[idx]]})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'curves': out}, f, ensure_ascii=False)
    return path
