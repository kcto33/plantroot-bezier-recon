# -*- coding: utf-8 -*-
"""
generate_dense_marked_views.py   (加密重建点: 生成带大量彩色标记的 8 视角图)

沿每条根取若干重建点, 每点一个唯一颜色。渲染成 8 视角带标记图, 并记录元数据:
  - 每个点的 (曲线号, 沿曲线t次序, 真值3D坐标, 颜色)
重建阶段用这个元数据把三角化出的 3D 点按所属曲线分组, 再做贝塞尔拟合。

输出:
  views_dense/cam0..7.png
  dense_meta.json           (每点: 颜色/曲线id/次序/真值坐标)
"""
import os
import json
import numpy as np
from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import render_points, draw_markers, save_image
from web_core_images import make_palette

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 8            # 与实际渲染的视角数一致(此前误设 72, 与 views_dense/ 不符)
K_SCALE = 0.6

POINTS_PER_CURVE = 12  # 每条曲线取 12 个曲线上点(与 dense_meta.json/主链路一致)
MIN_MARKER_R = 3       # 标记最小半径(像素), 保证小点可见


def hsv_palette(n):
    """调色板: 与主链路一致, 使用贪心最远点选色(相邻色距离大, 检测不易误配)。
    此前的 HSV 色相均分法相邻色仅差 3° 色相, 检测时极易误配到邻近色。"""
    return make_palette(n)


def collect_points(model, per_curve=POINTS_PER_CURVE):
    """从每条曲线取重建点。返回列表:
       each: {curve_id, t_index, coord, radius}  (t_index 为沿曲线次序 0..per_curve-1)"""
    pts_all = []
    for ci, c in enumerate(model.curves):
        pts = c['points']
        radii = c['radii']
        n = pts.shape[0]
        # 取均匀的 per_curve 个索引 (含首末)
        idxs = np.linspace(0, n - 1, per_curve).round().astype(int)
        for k, ii in enumerate(idxs):
            # 起止点不重复取第一个采样点的t_center, 用累积法确定顺序即可, 简化用linspace顺序
            pts_all.append({
                'curve_id': ci,
                't_index': k,
                'coord': pts[ii].copy(),
                'radius': float(radii[ii]),
            })
    return pts_all


def occlusion_visible(point3d, cam, zbuf, tol=0.15):
    u, v = cam.project(point3d)
    Xc = (cam.R @ np.asarray(point3d).reshape(3, 1) + cam.t).ravel()
    z = Xc[2]
    if z <= 0:
        return False
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or ui >= zbuf.shape[1] or vi < 0 or vi >= zbuf.shape[0]:
        return False
    return z <= zbuf[vi, ui] + tol


def main():
    model = RootModel(RootParams(), seed=0)
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))
    pts, radii = model.all_points()

    targets = collect_points(model)
    palette = hsv_palette(len(targets))
    for ti, tp in enumerate(targets):
        tp['color'] = palette[ti]

    os.makedirs("views_dense", exist_ok=True)
    print(f"重建点数量: {len(targets)} (每条曲线 {POINTS_PER_CURVE} 点)")

    # 每点在各视角可见性(同时用于渲染标注)
    vis = np.zeros((len(targets), len(cams)), dtype=int)
    for ci, c in enumerate(cams):
        img, zbuf = render_points(pts, radii, c,
                                  width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=K_SCALE)
        for ti, tp in enumerate(targets):
            if occlusion_visible(tp['coord'], c, zbuf):
                vis[ti, ci] = 1
        # 标记: 深度排序 + 互斥(重叠标记整个跳过), 保证检测质心不被拉偏(P1-4)
        draw_markers(targets, c, img, zbuf, k_scale=K_SCALE,
                     min_marker_r=MIN_MARKER_R, max_marker_r=24.0, order_shift=ci)
        save_image(img, os.path.join("views_dense", f"{c.name}.png"))
        print(f"  {c.name} 已生成")

    # 可见性统计: 以"该视角实际画出了标记"为准(互斥跳过/遮挡均视为不可见)
    # 简化处理: 检测端自会丢弃缺失视角, 这里按遮挡测试统计供参考
    n_ok = sum(1 for ti in range(len(targets)) if vis[ti].sum() >= 2)
    print(f"\n能在>=2个视角看到的重建点: {n_ok}/{len(targets)}")

    meta = []
    for ti, tp in enumerate(targets):
        meta.append({
            'index': ti,
            'curve_id': tp['curve_id'],
            't_index': tp['t_index'],
            'color': tp['color'],
            'coord_true': [float(x) for x in tp['coord']],
            'visible_views': [int(ci) for ci in range(len(cams)) if vis[ti, ci]],
        })
    with open("dense_meta.json", "w", encoding="utf-8") as f:
        json.dump({'points': meta, 'n_views': N_VIEWS}, f, ensure_ascii=False, indent=2)
    print("已写出 dense_meta.json")


if __name__ == "__main__":
    main()
