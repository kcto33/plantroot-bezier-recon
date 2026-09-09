# -*- coding: utf-8 -*-
"""
render_side_by_side.py
把"原模型"和"拟合贝塞尔重建模型"都用连续曲线渲染器在相同视角渲染,
生成并排对比图(左: 原模型, 右: 拟合重建)。

用途: 直观对比"有模型几何时, 从表/拟合重建出来的模型"在视觉上与原模型的相似度。
"""
import os
import numpy as np
from PIL import Image
from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import render_curves
from bezier_fit import fit_model_curves

IMG_W, IMG_H = 420, 420
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 12   # 有代表性的几个视角做对比(0,30,60,...330)


def curves_to_render(model):
    """模型原曲线 -> render_curves 需要的 list of {'points','radii'}。"""
    return [{'points': c['points'], 'radii': c['radii']} for c in model.curves]


def fitted_to_render(fitted):
    """拟合结果 -> render_curves 需要的 list of {'points','radii'}。"""
    return [{'points': f['points'], 'radii': f['radii']} for f in fitted]


def main():
    model = RootModel(RootParams(), seed=0)
    fitted = fit_model_curves(model, n_out=60)

    orig_curves = curves_to_render(model)
    fit_curves = fitted_to_render(fitted)

    cam_target = [0.0, 0.0, -7.0]
    cam_height = -2.0
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, cam_height, cam_target,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))

    os.makedirs("compare", exist_ok=True)
    out = "compare/original_vs_fitted.png"

    # 每行: 左原模型, 右拟合重建
    rows = []
    for ci, c in enumerate(cams):
        imgL, _ = render_curves(orig_curves, c, width=IMG_W, height=IMG_H,
                                bg=(255, 255, 255), color=(90, 55, 25),
                                k_scale=0.6, subdiv=8)
        imgR, _ = render_curves(fit_curves, c, width=IMG_W, height=IMG_H,
                                bg=(255, 255, 255), color=(90, 55, 25),
                                k_scale=0.6, subdiv=8)
        rows.append((c.name, imgL, imgR, ci * (360 // N_VIEWS)))

    # 拼接: 4 列 x 3 行, 每格左右并排
    cols, rows_n = 4, 3
    cell_w = IMG_W * 2 + 8
    canvas_w = cols * cell_w + (cols - 1) * 14
    canvas_h = rows_n * (IMG_H + 40) + 30
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 246, 248))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("msyh.ttc", 18)
    except Exception:
        font = ImageFont.load_default()

    seen = 0
    for ci, (name, imgL, imgR, ang) in enumerate(rows):
        if seen >= cols * rows_n:
            break
        col = seen % cols
        row = seen // cols
        x0 = col * (cell_w + 14)
        y0 = row * (IMG_H + 40) + 28
        # 左图
        canvas.paste(Image.fromarray(imgL), (x0, y0))
        # 右图(带框)
        canvas.paste(Image.fromarray(imgR), (x0 + IMG_W + 8, y0))
        # 标注: 原模型(左)/拟合重建(右) + 角度
        draw.text((x0, y0 - 24), "原模型(连续渲染)", fill=(90, 55, 25), font=font)
        draw.text((x0 + IMG_W + 8, y0 - 24), "拟合贝塞尔重建(连续渲染)",
                  fill=(224, 67, 58), font=font)
        draw.text((x0 + cell_w // 2, y0 + IMG_H + 4),
                  f"视角 {ang}°", fill=(90, 90, 90), font=font)
        seen += 1

    canvas.save(out)
    print(f"已输出 {out}  (尺寸 {canvas.size})")
    print("每格: 左=原模型, 右=拟合贝塞尔重建(同视角)")


if __name__ == "__main__":
    main()
