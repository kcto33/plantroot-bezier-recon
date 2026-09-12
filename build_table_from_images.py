# -*- coding: utf-8 -*-
"""
build_table_from_images.py
完整闭环: 72 张连续曲线图 -> 匹配每段曲线的 起点/终点/中间点 -> 颜色特征匹配
        -> 多视角三角化 -> 空间坐标表 -> 从表取点做最小二乘贝塞尔拟合重建。

流程:
  [1] 生成 72 个相机, 每条曲线取 4 个曲线上点(起点/2中间/终点), 每点唯一颜色,
      渲染成 72 张带色标图 (views_pts4/cam0..71.png)。
  [2] 每个视角按"最近调色板色"定位各点 2D 质心。
  [3] 跨视角(按颜色id配对) 多视角 DLT 三角化 -> 每点 3D 空间坐标。
  [4] 记录空间坐标表 reconstructed_points.csv。
  [5] 从表按曲线分组取出 起点/中间/终点, 用最小二乘三次贝塞尔拟合重建并绘图。
"""
import os
import json
import csv
import colorsys
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
from render import render_curves, draw_markers, save_image
from triangulate import triangulate_robust
from bezier_fit import fit_cubic_bezier
from web_core_images import (detect_palette_blobs as web_detect_palette_blobs,
                             make_palette, estimate_radius_3d,
                             ENDPOINT_MERGE_TOL, TRIANGULATION_TOL_PX)

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24
K_SCALE = 0.6
POINTS_PER_CURVE = 12      # 每段测 起点+中间+终点 共12个曲线上点, 稳住拟合
MIN_MARKER_R = 3
OUTDIR_VIEWS = "views_pts4"
POINTS_CSV = "reconstructed_points.csv"
FIT_IMG = "rebuilt_from_table.png"


def hsv_palette(n):
    """生成 n 个尽量区分的颜色。
    用"贪心最远点"法从 RGB 立方体候选集中挑彼此距离最大的颜色,
    避免 HSV 色相均分导致相近色混淆。返回 [(r,g,b),...]。"""
    import colorsys
    rng = np.random.default_rng(0)
    # 生成大量候选(高饱和/中亮度范围, 便于在图上区分), 用随机+网格
    cand = []
    # HSV 色相环 + 亮度变化 生成候选
    for i in range(2000):
        h = rng.random()
        s = 0.85 + 0.15 * rng.random()      # 高饱和, 颜色鲜艳易区分
        v = 0.75 + 0.25 * rng.random()      # 中高亮度, 与白底区分
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cand.append((r * 255, g * 255, b * 255))
    cand = np.array(cand)
    # 贪心: 依次选离已选集合最远的点
    chosen = []
    first = int(rng.integers(len(cand)))
    chosen_idx = [first]
    chosen = [cand[first]]
    while len(chosen) < n:
        arr = np.array(chosen)
        # 候选到已选的最小距离
        d = np.linalg.norm(cand[:, None, :] - arr[None, :, :], axis=2).min(axis=1)
        idx = int(np.argmax(d))
        chosen.append(cand[idx])
        # 移除该候选, 避免重复
        cand = np.delete(cand, idx, axis=0)
        if len(cand) == 0:
            break
    return [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in chosen]


def collect_points(model, per_curve=POINTS_PER_CURVE):
    """每条曲线取 起点/终点 + (per_curve-2) 个中间采样点。返回列表。"""
    targets = []
    for ci, c in enumerate(model.curves):
        pts = c['points']
        radii = c['radii']
        n = pts.shape[0]
        idxs = np.linspace(0, n - 1, per_curve).round().astype(int)
        for k, ii in enumerate(idxs):
            # 标记类别: 首=起点, 末=终点, 中间=中间点
            cat = '起点' if k == 0 else ('终点' if k == per_curve - 1 else '中间点')
            targets.append({
                'curve_id': ci, 't_index': k, 'category': cat,
                'coord': pts[ii].copy(), 'radius': float(radii[ii]),
            })
    return targets


def occlusion_visible(p3, cam, zbuf, tol=0.15):
    u, v = cam.project(p3)
    Xc = (cam.R @ np.asarray(p3).reshape(3, 1) + cam.t).ravel()
    z = Xc[2]
    if z <= 0:
        return False
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or ui >= zbuf.shape[1] or vi < 0 or vi >= zbuf.shape[0]:
        return False
    return z <= zbuf[vi, ui] + tol


def nearest_palette_centroid(image, palette, tol=70, subsample=1):
    """按最近调色板色定位各颜色质心(带连通域兜底与像素半径)。
    实现与 web 入口③共用(web_core_images.detect_palette_blobs), 避免两份逻辑漂移。
    subsample 参数仅为兼容旧签名, 现统一不降采样(P0-2: 下采样会稀释小标记)。"""
    return web_detect_palette_blobs(image, palette, tol=tol)


def main():
    model = RootModel(RootParams(), seed=0)
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))
    pts_full, radii_full = model.all_points()

    # [1] 生成带色标图
    targets = collect_points(model)
    palette = hsv_palette(len(targets))
    for ti, tp in enumerate(targets):
        tp['color'] = palette[ti]
    os.makedirs(OUTDIR_VIEWS, exist_ok=True)
    print(f"[1/5] 渲染 {N_VIEWS} 视角, 每条曲线 {POINTS_PER_CURVE} 点(起/中/终), "
          f"共 {len(targets)} 点 ...")
    for ci, c in enumerate(cams):
        img, zbuf = render_curves(model.curves, c, width=IMG_W, height=IMG_H,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=K_SCALE, subdiv=8)
        # 标记: 深度排序 + 互斥(重叠标记整个跳过), 保证检测质心不被拉偏(P1-4)
        draw_markers(targets, c, img, zbuf, k_scale=K_SCALE,
                     min_marker_r=MIN_MARKER_R, order_shift=ci)
        save_image(img, os.path.join(OUTDIR_VIEWS, f"{c.name}.png"))

    # [2] 每个视角定位各点
    print("[2/5] 逐视角按最近颜色定位 2D 点 ...")
    dets = [[None] * N_VIEWS for _ in range(len(targets))]   # [ti][vi] = uv
    rpix = [[None] * N_VIEWS for _ in range(len(targets))]   # [ti][vi] = 像素半径
    for vi in range(N_VIEWS):
        img = np.asarray(Image.open(os.path.join(OUTDIR_VIEWS, f"cam{vi}.png")).convert('RGB'))
        cc = nearest_palette_centroid(img, palette)
        for k in range(len(targets)):
            if cc[k] is not None:
                dets[k][vi], rpix[k][vi] = cc[k]

    # [3] 三角化 (迭代剔除坏视角 + cheirality 正深度校验, P1-5) + 半径反演(P2-8)
    print("[3/5] 多视角 DLT 三角化(迭代剔除配错视角) ...")
    rec = [None] * len(targets)
    rad3d = [None] * len(targets)
    kept_views = [set() for _ in range(len(targets))]
    kept_resids = [[] for _ in range(len(targets))]
    for ti in range(len(targets)):
        seen = [(vi, dets[ti][vi]) for vi in range(N_VIEWS) if dets[ti][vi] is not None]
        if len(seen) < 2:
            continue
        Pset = np.array([cams[vi].P for vi, _ in seen])
        uvset = np.array([p for _, p in seen])
        # 三角化: 迭代剔除重投影残差大的视角(P1-5), 阈值 1.2px 与入口③一致
        X, keep, _res = triangulate_robust(Pset, uvset, tol_px=TRIANGULATION_TOL_PX)
        if X is None:
            continue
        rec[ti] = X
        kept_views[ti] = set(seen[i][0] for i in keep)
        kept_resids[ti] = list(_res)
        rad3d[ti] = estimate_radius_3d(
            [rpix[ti][seen[i][0]] for i in keep],
            [cams[seen[i][0]] for i in keep], X)
    n_ok = sum(1 for p in rec if p is not None)
    print(f"  成功三角化 {n_ok}/{len(targets)}")

    # 保存逐点检测/残差(供 evaluate_reconstruction.py 校验 C4/B2)
    detections = {'points': {}}
    for ti in range(len(targets)):
        if rec[ti] is None:
            continue
        Xh = np.append(rec[ti], 1.0)
        det_entry = {'dets': {}, 'reproj_errs': []}
        for vi in range(N_VIEWS):
            if dets[ti][vi] is not None:
                pr = cams[vi].P @ Xh; pr = pr[:2] / pr[2]
                det_entry['dets'][str(vi)] = [float(x) for x in dets[ti][vi]]
                if vi in kept_views[ti]:
                    det_entry['reproj_errs'].append(
                        float(np.linalg.norm(pr - dets[ti][vi])))
        detections['points'][str(ti)] = det_entry
    with open("detections.json", "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False)

    # [4] 写空间坐标表
    print("[4/5] 写空间坐标表 ->", POINTS_CSV)
    with open(POINTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["点序号", "曲线id", "段内序号", "类别", "x", "y", "z"])
        for ti, tp in enumerate(targets):
            if rec[ti] is None:
                continue
            w.writerow([ti, tp['curve_id'], tp['t_index'], tp['category'],
                        round(float(rec[ti][0]), 4), round(float(rec[ti][1]), 4),
                        round(float(rec[ti][2]), 4)])
    print(f"  已写入 {POINTS_CSV} (有效点 {n_ok})")

    # [5] 从表分组 + 反解控制点 -> 一条三次贝塞尔
    print("[5/5] 每条曲线的曲线上点(起/中/终) -> 反解控制点1/2 -> 一条三次贝塞尔 ...")

    curves = {}
    for ti, tp in enumerate(targets):
        if rec[ti] is None or rad3d[ti] is None:
            continue
        curves.setdefault(tp['curve_id'], []).append(
            (tp['t_index'], rec[ti], rad3d[ti]))
    for cid in curves:
        curves[cid].sort(key=lambda x: x[0])

    from bezier_fit import fit_bezier_any, sample_bezier_any, remove_outliers
    DEGREE = 5   # 5次贝塞尔(6控制点): 比三次更贴复杂弯曲, 又不至于过拟合摆动

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    rebuilt_curves = []
    ctrl_rows = []       # 存 每段反解出的 6 控制点
    n_ctrl_ok = 0

    def smooth_points(P, win=3):
        """曲线测点平滑(保留端点), 用滑动平均去个别误差点毛刺。"""
        P = np.asarray(P, dtype=float)
        if len(P) < win:
            return P
        out = P.copy()
        for i in range(1, len(P) - 1):
            lo, hi = max(0, i - win // 2), min(len(P), i + win // 2 + 1)
            out[i] = P[lo:hi].mean(axis=0)
        return out

    # ---- 节点连续性: 端点对齐 ----
    # 收集所有段的 首点(起点)/末点(终点), 仅合并"同一节点的重复测量"
    # (同一次分叉被两条曲线共享的端点)。阈值必须远小于相邻分支节点的间距
    # (一个采样步长, 细根约 0.06), 否则会把不同的分支节点错误合并、拉偏端点。
    endpoint_groups = []   # 每个聚类: 共享坐标
    e_assign = {}          # (cid, 'start'/'end') -> 聚类索引
    for cid, items in curves.items():
        P = np.array([it[1] for it in items])
        if len(P) < 2:
            continue
        for k, tag in ((0, 'start'), (len(P) - 1, 'end')):
            pt = P[k]
            placed = False
            for gi, g in enumerate(endpoint_groups):
                if np.linalg.norm(g['coord'] - pt) < ENDPOINT_MERGE_TOL:
                    g['pts'].append(pt)
                    g['coord'] = np.mean(g['pts'], axis=0)
                    e_assign[(cid, tag)] = gi
                    placed = True
                    break
            if not placed:
                endpoint_groups.append({'coord': pt, 'pts': [pt]})
                e_assign[(cid, tag)] = len(endpoint_groups) - 1

    n_out = 0
    controls_by_cid = {}       # cid -> 拟合控制点(供根表)
    curve_meta = {}            # cid -> 12 槽测点/半径(供根表)
    for cid, items in curves.items():
        P = np.array([it[1] for it in items])          # 坐标
        R = np.array([it[2] for it in items])          # 半径
        # 各测点的生成参数 t = t_index/(每曲线点数-1): 与标记采样约定一致,
        # 用它拟合可避免弦长参数化失配导致的控制点震荡(P2-10)。
        t_orig = np.array([it[0] for it in items], dtype=float) / max(POINTS_PER_CURVE - 1, 1)
        # 端点用对齐后坐标
        if len(P) >= 2:
            P[0] = endpoint_groups[e_assign[(cid, 'start')]]['coord']
            P[-1] = endpoint_groups[e_assign[(cid, 'end')]]['coord']
        if len(P) >= DEGREE + 1:
            P, keep = remove_outliers(P)               # 剔除离群测点
            n_out += int((~keep).sum())
            t_k = t_orig[keep]
            P = smooth_points(P, win=3)                # 再轻度平滑
            # 同步半径掩码: 剔除/平滑后的半径
            Rk = R[keep]
            if len(Rk) >= 2:
                # 半径也轻度平滑
                Rs = Rk.copy()
                for i in range(1, len(Rs) - 1):
                    lo, hi = max(0, i - 1), min(len(Rs), i + 2)
                    Rs[i] = Rk[lo:hi].mean()
            else:
                Rs = Rk
            deg = min(DEGREE, len(P) - 1)
            ctrl, err = fit_bezier_any(P, degree=deg, t=t_k)
            s = sample_bezier_any(ctrl, n=100)
            n_ctrl_ok += 1
            controls_by_cid[cid] = ctrl
            # 沿贝塞尔插值半径(用测点的生成参数 t 对应半径)
            from bezier_fit import chord_length_param
            radii_curve = np.interp(np.linspace(0, 1, len(s)), t_k, Rs)
            for i, p in enumerate(ctrl):
                c = 'red' if i in (0, deg) else 'blue'
                ax.scatter(*p, color=c, s=12)
                ctrl_rows.append([cid, i, '端点' if i in (0, deg) else '控制点',
                                  round(float(p[0]), 4), round(float(p[1]), 4),
                                  round(float(p[2]), 4)])
        else:
            s = P
            radii_curve = np.full(len(s), 0.08)
        rebuilt_curves.append({'points': s, 'radii': radii_curve})
        # 根表用 12 槽测点/半径(缺失槽位用拟合曲线补齐), 端点用拟合端点
        pts_full = np.full((POINTS_PER_CURVE, 3), np.nan)
        rad_full = np.full(POINTS_PER_CURVE, np.nan)
        for (k, xyz, r) in items:
            pts_full[k] = xyz
            rad_full[k] = r
        for k in range(POINTS_PER_CURVE):
            if np.isnan(pts_full[k]).any():
                from bezier_fit import bernstein_basis
                if cid in controls_by_cid:
                    tt = k / max(POINTS_PER_CURVE - 1, 1)
                    dg = len(controls_by_cid[cid]) - 1
                    pts_full[k] = bernstein_basis(dg, np.array([tt]))[0] @ controls_by_cid[cid]
                else:
                    pts_full[k] = np.nan_to_num(pts_full[k])
                rad_full[k] = np.nanmax(rad_full) if np.isfinite(rad_full).any() else 0.08
        if cid in controls_by_cid:
            pts_full[0] = controls_by_cid[cid][0]
            pts_full[-1] = controls_by_cid[cid][-1]
        curve_meta[cid] = {'points': pts_full, 'radii': rad_full}
        # 用粗细反映到线宽(模拟渐变)
        for j in range(0, len(s), 1):
            lw = 0.8 + 6.0 * (radii_curve[j] / max(radii_curve.max(), 1e-6))
            ax.plot(s[j:j+2, 0], s[j:j+2, 1], s[j:j+2, 2], color='#e0433a',
                    lw=lw, alpha=0.9)
    ax.scatter(0, 0, 0, color='green', s=70, marker='*')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"高阶贝塞尔/段 (deg={DEGREE}) {n_ctrl_ok}段, 剔除{n_out}离群点")
    # 更俯视视角, 便于看清分叉结构(和连续渲染方向一致: 地表在上、深处在下)
    ax.view_init(elev=38, azim=-78)
    # 使三轴比例一致, 避免主干显得过高
    ax.set_box_aspect((1, 1, 1.1))
    fig.tight_layout()
    fig.savefig(FIT_IMG, dpi=115)
    plt.close(fig)
    print(f"  剔除 {n_out} 个离群测点; 反解出 {n_ctrl_ok} 段高阶贝塞尔; 已输出", FIT_IMG)

    # 写控制点表: 每条曲线的贝塞尔定义点(端点+控制点)
    with open("control_points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["曲线id", "控制点序号", "类别", "x", "y", "z"])
        for r in ctrl_rows:
            w.writerow(r)
    print(f"  已写 control_points.csv ({len(ctrl_rows)} 个定义点)")

    # 保存重建曲线, 供同视角连续渲染对比
    np.save("rebuilt_curves.npy", np.array([
        {'points': c['points'], 'radii': c['radii']} for c in rebuilt_curves
    ], dtype=object), allow_pickle=True)
    print("  已保存 rebuilt_curves.npy 供同视角对比渲染")

    # [6] 根表(P2-9): 由重建结果推导 "主根 -> 分支点 -> 侧根" 树,
    #     输出与 table_build 相同 14 列结构的 root_table.csv/.sql
    from root_topology import write_root_table
    depths = {cid: int(model.curves[cid]['depth']) for cid in curve_meta}
    write_root_table(curve_meta, depths, max_depth=model.params.maxDepth,
                     csv_path="root_table.csv", sql_path="root_table.sql")
    print(f"  已写 root_table.csv / root_table.sql ({len(curve_meta)} 条曲线)")
    print("-" * 60)


if __name__ == "__main__":
    main()
