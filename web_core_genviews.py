# -*- coding: utf-8 -*-
"""
web_core_genviews.py
输入 matlab .m -> 模型 -> 生成 24 张带颜色标记的曲线图(15°/张) + 相机位姿文件(JSON)。
生成的 24 张图 + 位姿文件 可作为"入口③"的输入:
  每条曲线取 12 个曲线上点, 每点一个唯一调色板色(与入口③的检测约定一致),
  这样入口③仅凭上传图像素即可做颜色检测与三角化, 无需模型几何。

位姿文件格式(json):
{
  "n_views": 24, "step_deg": 15,
  "mark_meta": {"n_curves":10, "points_per_curve":12, "n_points":120},
  "cameras": [ {"index":0, "name":"cam0", "K":[[..]], "R":[[..]], "t":[...], "P":[[3x4]]}, ... ]
}
"""
import os
import json
import numpy as np
from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import render_curves, draw_markers, save_image
from web_core_matlab import parse_matlab_params
from web_core_images import make_palette

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24
STEP = 360.0 / N_VIEWS
K_SCALE = 0.6
MARKS_PER_CURVE = 12   # 每条曲线取 12 个曲线上点做颜色标记(与入口③约定一致)
MIN_MARKER_R = 3


def build_cameras():
    """生成 24 台相机(默认位姿)。"""
    return ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))


def collect_mark_targets(model, per_curve=MARKS_PER_CURVE):
    """每条曲线取 起点/终点 + (per_curve-2) 个中间采样点, 顺序 = ti = 曲线id*per_curve + k。"""
    targets = []
    for ci, c in enumerate(model.curves):
        pts = c['points']
        radii = c['radii']
        n = pts.shape[0]
        idxs = np.linspace(0, n - 1, per_curve).round().astype(int)
        for k, ii in enumerate(idxs):
            targets.append({'curve_id': ci, 't_index': k,
                            'coord': pts[ii].copy(), 'radius': float(radii[ii])})
    return targets


def export_pose(cams, mark_meta=None):
    """导出 24 台相机的位姿文件(JSON dict), 含 K/R/t/P 与标记约定元数据。"""
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
            'mark_meta': mark_meta or {},
            'cameras': cameras}


def generate_views_from_matlab(m_file_path, outdir, seed=0, progress_cb=None):
    """matlab -> 模型 -> 24 张带标记曲线图 + 位姿文件。返回结果路径。
    progress_cb(stage, done, total): 每生成一张视图回调一次, 供流式进度展示。"""
    def _cb(stage, done, total):
        if progress_cb:
            try:
                progress_cb(stage, done, total)
            except Exception:
                pass

    os.makedirs(outdir, exist_ok=True)
    with open(m_file_path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    params = parse_matlab_params(text)
    model = RootModel(params, seed=seed)

    cams = build_cameras()
    imgdir = os.path.join(outdir, "views")
    os.makedirs(imgdir, exist_ok=True)

    # 颜色标记: 每条曲线 12 点, 每点唯一调色板色(与入口③检测约定一致)
    targets = collect_mark_targets(model)
    palette = make_palette(len(targets))
    for ti, tp in enumerate(targets):
        tp['color'] = palette[ti]

    for ci, c in enumerate(cams):
        img, zbuf = render_curves(model.curves, c, width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=K_SCALE, subdiv=8)
        # 标记: 深度排序 + 互斥(重叠标记整个跳过), 保证检测质心不被拉偏(P1-4)
        draw_markers(targets, c, img, zbuf, k_scale=K_SCALE,
                     min_marker_r=MIN_MARKER_R, order_shift=ci)
        save_image(img, os.path.join(imgdir, f"{c.name}.png"))
        _cb("生成视图", ci + 1, len(cams))

    mark_meta = {'n_curves': len(model.curves),
                 'points_per_curve': MARKS_PER_CURVE,
                 'n_points': len(targets),
                 'curve_depths': [int(c['depth']) for c in model.curves],
                 'max_depth': int(params.maxDepth)}
    pose = export_pose(cams, mark_meta)
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
