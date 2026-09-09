# -*- coding: utf-8 -*-
"""
render_rebuilt_compare.py
用与连续曲线渲染相同的相机(up=[0,0,-1], 根朝下), 把"原模型"和"从表重建的分段贝塞尔"
渲染成相同视角的对比图(左=原, 右=重建; 另存一张叠加图), 直观判断还原是否一致。
"""
import os
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

from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import render_curves

IMG_W, IMG_H = 640, 640
# 与 views_cont/ 主查看器、重建完全同一套相机, 保证取景一致不裁切
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24   # 15°一张, 与重建/主查看器一致


def main():
    model = RootModel(RootParams(), seed=0)
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))

    # 原模型曲线
    orig_curves = [{'points': c['points'], 'radii': c['radii']} for c in model.curves]

    # 从表重建的分段贝塞尔
    rebuilt = list(np.load("rebuilt_curves.npy", allow_pickle=True))

    os.makedirs("compare", exist_ok=True)
    for vi, c in enumerate(cams):
        imgL, _ = render_curves(orig_curves, c, width=IMG_W, height=IMG_H,
                                bg=(255, 255, 255), color=(90, 55, 25),
                                k_scale=0.6, subdiv=8)
        imgR, _ = render_curves(rebuilt, c, width=IMG_W, height=IMG_H,
                                bg=(255, 255, 255), color=(224, 67, 58),
                                k_scale=0.6, subdiv=8)
        # 拼: 左原 右重建
        canvas = Image.new("RGB", (IMG_W*2 + 8, IMG_H), (245, 246, 248))
        canvas.paste(Image.fromarray(imgL), (0, 0))
        canvas.paste(Image.fromarray(imgR), (IMG_W + 8, 0))
        canvas.save(os.path.join("compare", f"rebuilt_vs_orig_cam{vi}.png"))

    print(f"已生成 compare/rebuilt_vs_orig_cam0..{N_VIEWS-1}.png (左=原模型, 右=从表重建)")

    # 叠加图(0°视角): 原模型灰 + 重建红
    c = cams[0]
    img, _ = render_curves(orig_curves, c, width=IMG_W*2, height=IMG_H*2,
                           bg=(255, 255, 255), color=(150, 150, 150),
                           k_scale=0.6, subdiv=8)
    imgR, zbuf = render_curves(rebuilt, c, width=IMG_W*2, height=IMG_H*2,
                               bg=(255, 255, 255), color=(224, 67, 58),
                               k_scale=0.6, subdiv=8)
    # 红色叠加到灰色上
    rgb = img.astype(int)
    mask = (imgR[:, :, 0] > 200) & (imgR[:, :, 1] < 120) & (imgR[:, :, 2] < 120)
    rgb[mask] = (224, 67, 58)
    Image.fromarray(rgb.astype(np.uint8)).save("compare/rebuilt_overlay.png")
    print("已生成 compare/rebuilt_overlay.png (灰=原模型, 红=从表重建, 0°视角)")


if __name__ == "__main__":
    main()
