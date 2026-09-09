# -*- coding: utf-8 -*-
"""
重建管线(三维重建特征点匹配 -> 三角还原)。

流程:
  1. 生成三维根系几何模型。
  2. 放置绕 Z 轴一圈的 8 个针孔相机, 渲染各视角 2D 图像。
  3. 指定目标点(给出 3D 坐标或从分叉点清单选序号), 在 8 个视角中用醒目颜色标注。
  4. 颜色定位每个视角中标记的 2D 像素质心。
  5. 用已知投影矩阵做多视角三角化(DLT), 反求 3D 坐标, 并与模型真值比较。
"""
import os
import numpy as np

from root_model import RootModel, RootParams
from camera import ring_of_cameras
from render import (render_points, render_curves, render_marker, save_image,
                    detect_colored_blob)
from triangulate import triangulate_dlt, reproject_error


# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
MARKER_COLOR = (0, 200, 0)     # 目标点醒目标记色
IMG_W, IMG_H = 640, 640
CAM_RADIUS = 15.0              # 相机绕 Z 轴的半径
CAM_HEIGHT = -2.0              # 相机高度
CAM_TARGET = [0.0, 0.0, -7.0]  # 相机看向的点
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 72                   # 5°一张: 360/5 = 72
STEP_DEG = 360.0 / N_VIEWS


def build_geometry(seed=0):
    m = RootModel(RootParams(), seed=seed)
    return m


def build_cameras():
    return ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))


def render_views(model, cams, outdir="views", marker_pt=None):
    """渲染多个视角(5°一张)。marker_pt 为 None 时渲染裸模型, 否则叠加彩色标记。
    返回 (images, zbufs)。"""
    images, zbufs = [], []
    os.makedirs(outdir, exist_ok=True)
    for c in cams:
        img, zbuf = render_curves(model.curves, c,
                                  width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=0.6, subdiv=8)
        if marker_pt is not None:
            render_marker(marker_pt, c, img, zbuf,
                          color=MARKER_COLOR, marker_r=13)
        save_image(img, os.path.join(outdir, f"{c.name}.png"))
        images.append(img)
        zbufs.append(zbuf)
    return images, zbufs


def project_marker_in_views(marker_pt, cams):
    """把目标点投影到每个相机, 得到它"应出现在"的像素坐标与可见性。"""
    projs = []
    for c in cams:
        uv = c.project(marker_pt)
        Xc = (c.R @ np.asarray(marker_pt).reshape(3, 1) + c.t).ravel()
        projs.append({'uv': uv, 'z': Xc[2], 'visible': Xc[2] > 0})
    return projs


def detect_marker_in_views(images):
    """在已渲染图像上按颜色定位标记质心。"""
    dets = []
    for img in images:
        blob = detect_colored_blob(img, MARKER_COLOR, tol=45)
        dets.append(blob)
    return dets


def select_bifurcation(model, index):
    """从分叉点清单里按序号取一个 3D 坐标 (1-based)。"""
    if index < 1 or index > len(model.bifurcations):
        raise IndexError(f"分叉点序号超出范围 [1, {len(model.bifurcations)}]")
    return model.bifurcations[index - 1]['coord'].copy()


def pick_visible_bifurcation(model, cams):
    """自动挑一个分叉点, 要求它在所有 8 个相机中都可见(深度>0)。"""
    for i, bp in enumerate(model.bifurcations, start=1):
        pt = bp['coord']
        all_visible = all(c.project(pt) is not None and
                          (c.R @ pt.reshape(3, 1) + c.t).ravel()[2] > 0
                          for c in cams)
        if all_visible:
            return i, pt
    return 1, model.bifurcations[0]['coord'].copy()


def select_by_pixel(model, cams, u, v, view_idx):
    """在一个视角给像素坐标 -> 反向投影射线, 找与该射线最近的模型采样点(最近点)，
    把它作为目标点(近似交互: 用户点图上的某处)。"""
    cam = cams[view_idx]
    ray = cam.project_ray(np.array([[u, v]]))[0]   # 相机系方向
    # 转换为世界系方向
    R = cam.R
    d_world = R.T @ ray                            # 世界系射线方向
    origin = -R.T @ cam.t.ravel()                  # 相机光心在世界系

    pts, radii = model.all_points()
    # 点到射线的距离: 在射线上投影 t, 再算垂距
    vec = pts - origin
    t = vec @ d_world / (d_world @ d_world)
    t = np.maximum(t, 0.0)
    closest = origin[None, :] + t[:, None] * d_world[None, :]
    dist = np.linalg.norm(pts - closest, axis=1)
    idx = int(np.argmin(dist))
    print(f"  像素({u},{v})在视角{cam.name}最近端采样点 idx={idx}, "
          f"垂距={dist[idx]:.2f}")
    return pts[idx]


def main(target_index=None, pixel=None, view_idx=0):
    print("=" * 62)
    print("  三维重建(特征点颜色匹配) + 多视角三角还原")
    print("=" * 62)

    print("\n[1/5] 生成三维根系几何 ...")
    model = build_geometry(seed=0)
    pts, radii = model.all_points()
    print(f"  曲线数={len(model.curves)} 采样点={len(pts)} 分叉点={len(model.bifurcations)}")

    print("\n[2/5] 放置 8 个相机并渲染各视角 ...")
    cams = build_cameras()
    images, zbufs = render_views(model, cams, outdir="views", marker_pt=None)
    print(f"  已生成 {len(images)} 张视角图 -> views/")

    # ---- 指定目标点 ----
    if pixel is not None:
        target_pt = select_by_pixel(model, cams, pixel[0], pixel[1], view_idx)
        print(f"\n[3/5] 通过像素坐标指定 (视角{view_idx}): 得到目标点 {np.round(target_pt,4)}")
    elif target_index is not None:
        target_pt = select_bifurcation(model, target_index)
        print(f"\n[3/5] 指定分叉点序号 {target_index}")
    else:
        target_index, target_pt = pick_visible_bifurcation(model, cams)
        print(f"\n[3/5] 自动选择可见分叉点: 序号 {target_index}")
    print(f"  目标点 3D 真值: {np.round(target_pt, 4)}")

    # ---- 带标记渲染 ----
    print("\n[4/5] 在 8 个视角中标注目标点(颜色标记) ...")
    images_marked, _ = render_views(model, cams, outdir="views_marked",
                                    marker_pt=target_pt)

    # ---- 颜色定位 + 三角化 ----
    print("\n[5/5] 颜色定位 + DLT 三角化 ...")
    dets = detect_marker_in_views(images_marked)

    valid_cam = []
    valid_uv = []
    for i, c in enumerate(cams):
        if dets[i] is None:
            print(f"  视角 {c.name}: 未检测到标记(被遮挡/超出视界)")
            continue
        # 用真实投影矩阵把 3D 真值投到该视角得到"理论"像素, 与颜色质心对比
        gt_uv = c.project(target_pt)
        print(f"  视角 {c.name}: 检测质心={np.round(dets[i],2)} "
              f"真值投影={np.round(gt_uv,2)} "
              f"偏差={np.linalg.norm(dets[i]-gt_uv):.2f}px")
        valid_cam.append(c)
        valid_uv.append(dets[i])

    if len(valid_cam) < 2:
        print("可三角化的视角不足 2 个, 无法还原。")
        return

    Ps = np.array([c.P for c in valid_cam])
    uv = np.array(valid_uv)
    rec = triangulate_dlt(Ps, uv)

    print("\n" + "-" * 62)
    print("三角化结果")
    print("-" * 62)
    print(f"  重建 3D 坐标 X = {np.round(rec, 4)}")
    print(f"  模型真值 3D 坐标 X_t = {np.round(target_pt, 4)}")
    err = np.linalg.norm(rec - target_pt)
    print(f"  三维误差 ||X - X_t|| = {err:.4f} (占模型尺寸比例 "
          f"{err/np.linalg.norm(model.bbox[1]-model.bbox[0]):.4f})")

    # 逐视角重投影误差
    print("\n  各视角重投影误差(像素):")
    for i, c in enumerate(valid_cam):
        rerr = reproject_error(c.P, rec, valid_uv[i])
        print(f"    {c.name}: {rerr:.3f} px")
    print("-" * 62)

    return rec, target_pt, model, cams


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="三维重建 + 多视角三角还原")
    ap.add_argument("--from-list", type=int, default=None,
                    help="从分叉点清单里按序号(1-based)指定目标点")
    ap.add_argument("--pixel", type=float, nargs=2, metavar=("U", "V"),
                    default=None, help="在一个视角给出的像素坐标, 如 --pixel 320 340")
    ap.add_argument("--view", type=int, default=0,
                    help="像素坐标属于哪个视角(0-7), 默认 0")
    args = ap.parse_args()
    main(target_index=args.from_list, pixel=args.pixel, view_idx=args.view)
