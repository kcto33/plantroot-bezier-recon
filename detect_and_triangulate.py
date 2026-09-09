# -*- coding: utf-8 -*-
"""
detect_and_triangulate.py   (阶段2: 仅凭图片像素 + 已知相机, 自动重建分支点并填表)

只用:
  - views_branch/cam0..7.png 的像素(自动检测彩色分支点标记的 2D 位置)
  - 已知的 8 个相机投影矩阵(与 generate_marked_views 相同)

流程:
  1) 对每个视角, 按 8 个颜色(阈值)检测每个分支点标记的 2D 质心。
  2) 按颜色 id 跨视角配对 -> 每个分支点得到一组 (视角, 像素) 对应。
  3) 用已知相机矩阵做多视角 DLT 三角化 -> 反求 3D 坐标。
  4) 把 3D 分支点写入之前的根表结构 (root_branch_table.csv / .sql)。
  5) 把重建出的 3D 点重投影回各视角, 与检测到的像素对比, 给出重投影误差。
"""
import os
import numpy as np
from PIL import Image
import csv

from camera import ring_of_cameras
from triangulate import triangulate_dlt, reproject_error
from table_build import HEADERS, maturity_of

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 8

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255),
]


def detect_blob(image, color, tol=40):
    """在一张图里找指定颜色的连通区。返回 (u,v,半径像素) 或 None。
    半径通过等效面积估算: r = sqrt(area/pi)。"""
    rgb = image.astype(np.int64)
    mask = np.all(np.abs(rgb - np.asarray(color)) <= tol, axis=-1)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    area = mask.sum()
    cx, cy = xs.mean(), ys.mean()
    r_pix = float(np.sqrt(area / np.pi))
    return (np.array([cx, cy]), r_pix)


def build_cams():
    return ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))


def load_views(dirname="views_branch"):
    imgs = []
    for i in range(N_VIEWS):
        p = os.path.join(dirname, f"cam{i}.png")
        imgs.append(np.asarray(Image.open(p).convert('RGB')))
    return imgs


def main():
    cams = build_cams()
    imgs = load_views()

    print("=" * 62)
    print("  仅凭图片像素 + 已知相机 -> 自动检测分支点并三角化")
    print("=" * 62)

    print("\n[1/4] 逐视角检测彩色分支点标记的 2D 位置 ...")
    # detections[bi][vi] = (u,v) 或 None
    num_colors = len(PALETTE)
    detections = [[None] * N_VIEWS for _ in range(num_colors)]
    for vi, img in enumerate(imgs):
        for bi in range(num_colors):
            detections[bi][vi] = detect_blob(img, PALETTE[bi])
        found = sum(1 for bi in range(num_colors) if detections[bi][vi] is not None)
        print(f"  视角 cam{vi}: 检测到 {found}/{num_colors} 个彩色分支点")

    def mean_2d(seen):
        pts = np.array([p for _, p in seen])
        return pts.mean(axis=0)

    print("\n[2/4] 按颜色跨视角配对 + 多视角 DLT 三角化 ...")
    rows = []
    rec_points = []
    K_SCALE = 0.6   # 与生成阶段一致, 用于反推粗细
    max_depth_assumed = 3
    for bi in range(num_colors):
        # detections[bi][vi] = (uv, r_pix) 或 None
        seen = [(vi, detections[bi][vi]) for vi in range(N_VIEWS)
                if detections[bi][vi] is not None]
        if len(seen) < 2:
            print(f"  分支点{bi+1} (色{PALETTE[bi]}): 可见视角 {len(seen)} < 2, 跳过后无法三角化。")
            continue
        Pset = np.array([cams[vi].P for vi, _ in seen])
        uvset = np.array([det[0] for _, det in seen])
        rec = triangulate_dlt(Pset, uvset)
        rec_points.append(rec)

        # 重投影误差
        rerr = [reproject_error(cams[vi].P, rec, det[0]) for vi, det in seen]

        # 粗细估计: 用每个可见视角反推 r3d = r_pix * Z / (f * k_scale),
        # Z 为该点在该相机下的深度。取各视角的中位数。
        r3d_est = []
        for vi, det in seen:
            uv, r_pix = det
            Xc = (cams[vi].R @ rec.reshape(3, 1) + cams[vi].t).ravel()
            Z = Xc[2]
            if Z > 0 and r_pix > 0:
                r3d_est.append(r_pix * Z / (cams[vi].fx * K_SCALE))
        thick = float(np.median(r3d_est)) if r3d_est else -1.0

        print(f"  分支点{bi+1} (色{PALETTE[bi]}): {len(seen)}视角 -> "
              f"X={np.round(rec,3)}  重投影误差均值={np.mean(rerr):.2f}px  "
              f"粗细={thick:.4f}")

        # 填根表: 该分支点作为"首节点"行, 记录其3D坐标、粗细、成熟度
        rows.append({
            '序号': len(rows),
            '级别': 1,               # 分支点层级(简化, 可后续按需定)
            '编号1': bi,             # 分支点 id
            '编号2': 0,
            '编号3': None,
            '编号4': None,
            '类别': '首节点',
            'x': round(float(rec[0]), 4),
            'y': round(float(rec[1]), 4),
            'z': round(float(rec[2]), 4),
            '粗细': round(thick, 4),  # 由圆盘尺寸+深度反推
            '成熟度': maturity_of(1, max_depth_assumed),
            '上节点': None,
            '下节点': None,
        })

    print("\n[3/4] 写入根表结构 ...")
    from table_build import rows_to_csv, rows_to_sql
    rows_to_csv(rows, "root_branch_table.csv")
    rows_to_sql(rows, table_name="root_branch_table", path="root_branch_table.sql")
    print(f"  已写出 root_branch_table.csv ({len(rows)} 行) / root_branch_table.sql")

    print("\n[4/4] 验证: 重建3D点重投影回各视角 vs 检测像素 ...")
    if rec_points:
        pts3d = np.array(rec_points)
        # rec_points按bi顺序(跳过不可三角化的), 建立 rec_index 到 bi 映射
        rec_bi = []
        for bi in range(num_colors):
            if any(detections[bi][vi] is not None for vi in range(N_VIEWS)) and \
               sum(1 for vi in range(N_VIEWS) if detections[bi][vi] is not None) >= 2:
                rec_bi.append(bi)
        for vi, c in enumerate(cams):
            uv = c.project(pts3d)
            errs = []
            for k, bi in enumerate(rec_bi):
                det = detections[bi][vi]
                if det is not None:
                    errs.append(np.linalg.norm(uv[k] - det[0]))
            if errs:
                print(f"  视角 cam{vi}: 重投影偏差均值={np.mean(errs):.2f}px "
                      f"(n={len(errs)})")
    print("-" * 62)


if __name__ == "__main__":
    main()
