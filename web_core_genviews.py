# -*- coding: utf-8 -*-
"""
web_core_genviews.py
输入 matlab .m -> 模型 -> 生成 24 张连续曲线图(纯曲线, 15°/张) + 相机位姿文件(JSON)。
生成的 24 张图 + 位姿文件 可作为"入口②"的输入。

位姿文件格式(json):
{
  "n_views": 24, "step_deg": 15,
  "cameras": [ {"index":0, "name":"cam0", "K":[[..]], "R":[[..]], "t":[...], "P":[[3x4]]}, ... ]
}
"""
import os
import json
import numpy as np
from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import render_curves, save_image
from web_core_matlab import parse_matlab_params

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24
STEP = 360.0 / N_VIEWS
K_SCALE = 0.6


def build_cameras():
    """生成 24 台相机(默认位姿)。"""
    return ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))


def export_pose(cams):
    """导出 24 台相机的位姿文件(JSON dict), 含 K/R/t/P。"""
    cameras = []
    for i, c in enumerate(cams):
        cameras.append({
            'index': i, 'name': c.name,
            'K': [[float(x) for x in row] for row in c.K],
            'R': [[float(x) for x in row] for row in c.R],
            't': [float(x) for x in c.t.ravel()],
            'P': [[float(x) for x in row] for row in c.P],
        })
    return {'n_views': N_VIEWS, 'step_deg': STEP,
            'radius': CAM_RADIUS, 'height': CAM_HEIGHT,
            'target': CAM_TARGET, 'up': [0, 0, -1],
            'cameras': cameras}


def generate_views_from_matlab(m_file_path, outdir):
    """matlab -> 模型 -> 24 张连续曲线图 + 位姿文件。返回结果路径。"""
    os.makedirs(outdir, exist_ok=True)
    with open(m_file_path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    params = parse_matlab_params(text)
    model = RootModel(params, seed=0)

    cams = build_cameras()
    imgdir = os.path.join(outdir, "views")
    os.makedirs(imgdir, exist_ok=True)
    for ci, c in enumerate(cams):
        img, zbuf = render_curves(model.curves, c, width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=K_SCALE, subdiv=8)
        save_image(img, os.path.join(imgdir, f"{c.name}.png"))

    pose = export_pose(cams)
    pose_path = os.path.join(outdir, "pose.json")
    with open(pose_path, 'w', encoding='utf-8') as f:
        json.dump(pose, f, ensure_ascii=False, indent=2)

    return {
        'n_views': N_VIEWS, 'step_deg': STEP,
        'view_dir': "/gen/" + os.path.basename(outdir) + "/views",
        'pose_file': "/gen/" + os.path.basename(outdir) + "/pose.json",
        'curves': len(model.curves), 'nodes': len(model.nodes),
        'params': {'mainLength': params.mainLength, 'maxDepth': int(params.maxDepth),
                   'branchNum': params.branchNum},
    }


if __name__ == "__main__":
    r = generate_views_from_matlab("plantroot06.m", "gen_views_demo")
    print("完成:", r)
