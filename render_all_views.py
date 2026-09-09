# -*- coding: utf-8 -*-
"""
render_all_views.py
生成 15°一张(24 张)的连续曲线渲染图 —— 与"图→表→重建"共用同一套 24 台相机。

用 render_curves 把每条根渲染成"连续带粗细的曲线"(而非离散点)。
输出: views_cont/cam0.png ... cam23.png   (连续曲线, 15°步长)
"""
import os
import numpy as np
from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import render_curves, save_image

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24
STEP = 360.0 / N_VIEWS


def main():
    model = RootModel(RootParams(), seed=0)
    # up=-z: 让竖直根显示为 地表(z=0)在上、深处(z=-14)在下 —— 正确的"根朝下"
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))

    os.makedirs("views_cont", exist_ok=True)
    print(f"渲染 {N_VIEWS} 个视角(每 {STEP:.1f}°一张), 连续曲线渲染(根朝下) ...")
    for ci, c in enumerate(cams):
        img, zbuf = render_curves(model.curves, c,
                                  width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=0.6, subdiv=8)
        save_image(img, os.path.join("views_cont", f"{c.name}.png"))
        if (ci + 1) % 8 == 0 or ci == 0:
            print(f"  {c.name} ({ci+1}/{N_VIEWS})")
    print("完成: 详见 views_cont/")


if __name__ == "__main__":
    main()
