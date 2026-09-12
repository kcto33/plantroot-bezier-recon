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
from render import render_points, draw_markers, save_image, occlusion_visible

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


def branch_palette(n):
    """分支点颜色表(P2-11): 分支点数 > 8 时用贪心最远点选色扩展,
    不再 PALETTE[i % 8] 取模复用 -- 取模会让不同分支点共享同一颜色,
    跨视角按色配对时会把不同 3D 点合并。"""
    if n <= len(PALETTE):
        return list(PALETTE)
    from web_core_images import make_palette
    return make_palette(n)


def build(model_seed=0):
    model = RootModel(RootParams(), seed=model_seed)
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))
    pts, radii = model.all_points()
    return model, cams, pts, radii


def main():
    model, cams, pts, radii = build()
    os.makedirs("views_branch", exist_ok=True)
    bps = model.bifurcations
    palette = branch_palette(len(bps))
    print(f"分支点数量: {len(bps)} (颜色数 {len(palette)})")

    # 可见性矩阵: 每个分支点是否在某视角可见(未被遮挡)
    vis_matrix = np.zeros((len(bps), len(cams)), dtype=int)
    for ci, c in enumerate(cams):
        img, zbuf = render_points(pts, radii, c,
                                  width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=K_SCALE)
        marks = [{'coord': bp['coord'], 'radius': bp['radius'],
                  'color': palette[bi]}
                 for bi, bp in enumerate(bps)]
        for bi, bp in enumerate(bps):
            if occlusion_visible(bp['coord'], c, zbuf):
                vis_matrix[bi, ci] = 1
        # 标记: 深度排序 + 互斥(重叠标记整个跳过), 保证检测质心不被拉偏(P1-4)
        draw_markers(marks, c, img, zbuf, k_scale=K_SCALE,
                     min_marker_r=3.0, max_marker_r=40.0, order_shift=ci)
        save_image(img, os.path.join("views_branch", f"{c.name}.png"))
        print(f"  {c.name} 已生成")

    # 元数据: 检测端(detect_and_triangulate)据此对齐颜色表与层级
    import json
    with open(os.path.join("views_branch", "meta.json"), "w", encoding="utf-8") as f:
        json.dump({'n_branch': len(bps), 'palette': [list(c) for c in palette],
                   'depths': [int(bp['depth']) + 1 for bp in bps],  # 分支点层级 = 子曲线层级
                   'n_views': N_VIEWS}, f, ensure_ascii=False, indent=2)

    print("\n各分支点在不同视角的可见性:")
    for bi, bp in enumerate(bps):
        print(f"  分支点{bi+1} (色{palette[bi]}): "
              f"可见视角={np.where(vis_matrix[bi]==1)[0].tolist()}")


if __name__ == "__main__":
    main()
