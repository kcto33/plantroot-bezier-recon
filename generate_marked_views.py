# -*- coding: utf-8 -*-
"""
generate_marked_views.py   (阶段1: 生成带彩色分支点标记的视角图)

把三维根系模型的分支点(分叉点)用互不相同的颜色标注, 渲染成 8 个视角的带标记图。
遮挡感知: 某个分支点在该视角被其它根/模型遮挡时不画标记(下阶段就不检测它)。

输出: views_branch/cam0.png ... cam7.png
"""
import os
import numpy as np
from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import render_points, render_marker, save_image

# 与 pipeline.py 一致的相机参数
IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 8
K_SCALE = 0.6   # 与 render.py render_points 的 k_scale 一致(圆盘缩放)

# 8 个互不相同的颜色, 依次给分支点用 (索引0->分支点1 ...)
PALETTE = [
    (255, 0, 0),      # red
    (0, 255, 0),      # lime
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (0, 255, 255),    # cyan
    (255, 0, 255),    # magenta
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
]


def build(model_seed=0):
    model = RootModel(RootParams(), seed=model_seed)
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))
    pts, radii = model.all_points()
    return model, cams, pts, radii


def occlusion_visible(point3d, cam, zbuf, tol=0.15):
    """判断 point3d 在该相机下是否未被遮挡(深度小于等于该处 zbuf)。"""
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
    model, cams, pts, radii = build()
    os.makedirs("views_branch", exist_ok=True)
    bps = model.bifurcations
    print(f"分支点数量: {len(bps)} (颜色数 {len(PALETTE)})")

    # 可见性矩阵: 每个分支点是否在某视角可见(未被遮挡)
    vis_matrix = np.zeros((len(bps), len(cams)), dtype=int)
    for ci, c in enumerate(cams):
        img, zbuf = render_points(pts, radii, c,
                                  width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=K_SCALE)
        for bi, bp in enumerate(bps):
            pt = bp['coord']
            if occlusion_visible(pt, c, zbuf):
                vis_matrix[bi, ci] = 1
                # 标记半径 = 该点真实投影半径(r_pix = f*r3d/Z*k) 并设下限保证可见,
                # 这样阶段2能通过彩色圆盘尺寸反推粗细。
                r3d = bp['radius']
                Xc = (c.R @ pt.reshape(3, 1) + c.t).ravel()
                Z = Xc[2]
                r_pix = min(40.0, c.fx * r3d / max(Z, 1e-6) * K_SCALE)
                marker_r = int(max(3.0, r_pix))
                render_marker(pt, c, img, zbuf,
                              color=PALETTE[bi % len(PALETTE)], marker_r=marker_r)
        save_image(img, os.path.join("views_branch", f"{c.name}.png"))
        print(f"  {c.name} 已生成")

    print("\n各分支点在不同视角的可见性:")
    for bi, bp in enumerate(bps):
        print(f"  分支点{bi+1} (色{PALETTE[bi]}): "
              f"可见视角={np.where(vis_matrix[bi]==1)[0].tolist()}")


if __name__ == "__main__":
    main()
