# -*- coding: utf-8 -*-
"""
detect_dense_and_rebuild.py   (加密重建点: 检测 -> 三角化 -> 按曲线分组 -> 贝塞尔拟合)

仅凭 views_dense/cam0..7.png 的像素 + 已知相机:
  - 对每个调色板颜色(最近邻切分)定位 2D 质心。
  - 用 dense_meta.json 里的"颜色->所属曲线/次序"把检测结果对应到各根。
  - 多视角 DLT 三角化 -> 3D 点。
  - 按曲线分组, 对每条曲线的重建点做最小二乘三次贝塞尔拟合, 重建整棵树并绘图。
"""
import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for _f in ('Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC'):
    try:
        matplotlib.rcParams['font.sans-serif'] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

from camera import ring_of_cameras
from triangulate import triangulate_robust, reproject_error
from web_core_images import detect_palette_blobs

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 8            # 与 views_dense/ 实际视角数一致(此前误设 72)


def build_cams():
    return ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))


def nearest_palette_centroid(image, palette, tol=70):
    """按最近调色板颜色定位各颜色的 2D 质心(含连通域兜底)。
    与主链路共用实现(web_core_images.detect_palette_blobs)。"""
    return detect_palette_blobs(image, palette, tol=tol)


def load_views(dirname="views_dense"):
    return [np.asarray(Image.open(os.path.join(dirname, f"cam{i}.png")).convert('RGB'))
            for i in range(N_VIEWS)]


def main():
    cams = build_cams()
    imgs = load_views()

    with open("dense_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    points_meta = meta['points']
    palette = [tuple(pm['color']) for pm in points_meta]
    n_targets = len(points_meta)

    print("=" * 62)
    print("  加密重建点: 颜色定位 -> 三角化 -> 按根分组 -> 贝塞尔拟合重建")
    print("=" * 62)

    print(f"\n[1/4] 逐视角按最近调色板色定位 {n_targets} 个重建点 ...")
    # dets[ti][vi] = (uv, r_pix) 或 None
    dets = [[None] * N_VIEWS for _ in range(n_targets)]
    for vi, img in enumerate(imgs):
        cc = nearest_palette_centroid(img, palette)
        found = sum(1 for k in range(n_targets) if cc[k] is not None)
        for k in range(n_targets):
            dets[k][vi] = cc[k]
        print(f"  视角 cam{vi}: 定位到 {found}/{n_targets}")

    print("\n[2/4] 多视角 DLT 三角化 (每条重建点) ...")
    rec_points = [None] * n_targets
    rec_err = [None] * n_targets
    for ti in range(n_targets):
        seen = [(vi, dets[ti][vi]) for vi in range(N_VIEWS) if dets[ti][vi] is not None]
        if len(seen) < 2:
            continue
        Pset = np.array([cams[vi].P for vi, _ in seen])
        uvset = np.array([d[0] for _, d in seen])
        rec, keep, resids = triangulate_robust(Pset, uvset)
        if rec is None:
            continue
        rec_points[ti] = rec
        rec_err[ti] = float(np.mean(resids))

    n_ok = sum(1 for p in rec_points if p is not None)
    print(f"  成功三角化: {n_ok}/{n_targets}")

    # 统计: 重建点 vs 真值 3D 误差
    errs3d = []
    for ti, pm in enumerate(points_meta):
        if rec_points[ti] is not None:
            errs3d.append(np.linalg.norm(rec_points[ti] - np.array(pm['coord_true'])))
    if errs3d:
        print(f"  重建点 3D 误差: 均值={np.mean(errs3d):.4f} 最大={np.max(errs3d):.4f} "
              f"(模型尺寸 ~18)")

    print("\n[3/4] 按根分组 + 分段三次贝塞尔(Catmull-Rom 过所有点)重建 ...")
    # 按 curve_id 分组, 组内按 t_index 排序
    curves = {}
    for ti, pm in enumerate(points_meta):
        if rec_points[ti] is None:
            continue
        curves.setdefault(pm['curve_id'], []).append((pm['t_index'], rec_points[ti],
                                                      np.array(pm['coord_true'])))
    for cid in curves:
        curves[cid].sort(key=lambda x: x[0])

    def catmull_rom_segments(P):
        """把点列转成分段三次贝塞尔控制点(端点自然边界)。返回每个分段的 (P0,P1,P2,P3)。"""
        P = np.atleast_2d(P)
        n = len(P)
        if n < 2:
            return []
        # 扩展端点
        Pext = np.vstack([P[0], P, P[-1]])
        segs = []
        for i in range(n - 1):
            p0, p1, p2, p3 = Pext[i], Pext[i + 1], Pext[i + 2], Pext[i + 3]
            c1 = p1 + (p2 - p0) / 6.0
            c2 = p2 - (p3 - p1) / 6.0
            segs.append((p1, c1, c2, p2))
        return segs

    def sample_segments(segs, per_seg=25):
        pts = []
        for (P0, P1, P2, P3) in segs:
            t = np.linspace(0, 1, per_seg)[:, None]
            pts.append(((1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3))
        return np.vstack(pts)

    def arclength_resample(ctrl_pts_fn, m):
        """在由 ctrl_pts_fn() 生成的分段贝塞尔上按弧长参数取 m 个点。"""
        dense = sample_segments(ctrl_pts_fn(), per_seg=50)
        diff = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        c = np.concatenate([[0.0], np.cumsum(diff)])
        t = c / max(c[-1], 1e-9)
        idx = np.clip((t * (len(dense) - 1)).round().astype(int), 0, len(dense) - 1)
        # 按目标点数 m 重新取弧长等分点
        tt = np.linspace(0, 1, m)
        j = np.interp(tt, t, np.arange(len(dense)))
        return dense[np.clip(j.round().astype(int), 0, len(dense) - 1)]

    print(f"  重建 {len(curves)} 条根 (分段贝塞尔)")
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    fit_errs = []
    palette = ['#e0433a', '#1f8fe0', '#2ca02c', '#ff7f0e', '#9467bd',
               '#d62728', '#17becf', '#bcbd22', '#7f7f7f', '#e377c2']
    for ci, (cid, items) in enumerate(curves.items()):
        P = np.array([it[1] for it in items])     # 重建点
        Pt = np.array([it[2] for it in items])    # 真值
        segs = catmull_rom_segments(P)
        s = sample_segments(segs)
        col = palette[ci % len(palette)]
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=col, lw=1.8,
                label=f"根{cid}" if ci < 6 else None)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='black', s=20)
        # 拟合曲线 vs 真值采样点误差(按弧长对齐到真值点数)
        sr = arclength_resample(lambda: catmull_rom_segments(P), len(Pt))
        e = np.linalg.norm(sr - Pt, axis=1)
        fit_errs.append(float(e.max()))
    ax.scatter(0, 0, 0, color='green', s=70, marker='*', label='地表原点')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("加密重建点 -> 三角化 -> 按根分组分段贝塞尔重建")
    ax.view_init(elev=22, azim=-58)
    ax.legend()
    fig.tight_layout()
    fig.savefig("dense_reconstructed_model.png", dpi=115)
    plt.close(fig)
    print("  已输出 dense_reconstructed_model.png")

    print("\n[4/4] 精度汇总 ...")
    if fit_errs:
        print(f"  各根拟合曲线 vs 真值采样点: 最大={max(fit_errs):.4f} "
              f"平均均值={np.mean(fit_errs):.4f}")
    print("-" * 62)


if __name__ == "__main__":
    main()
