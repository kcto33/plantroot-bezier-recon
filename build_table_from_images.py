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
from render import render_curves, render_marker, save_image
from triangulate import triangulate_dlt, reproject_error
from bezier_fit import fit_cubic_bezier

IMG_W, IMG_H = 640, 640
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24
K_SCALE = 0.6
POINTS_PER_CURVE = 12      # 每段测 起点+中间+终点 共12个曲线上点, 稳住拟合
MIN_MARKER_R = 5
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
    """按最近调色板色定位各颜色质心。subsample>1 时先块均值下采样提速, 质心乘回原尺度。
    默认为 1(不降采样)以避免质心偏移。"""
    rgb = image.astype(np.int64)
    H, W = rgb.shape[:2]
    # 下采样(块均值), 减少像素数
    if subsample > 1:
        H2, W2 = H // subsample, W // subsample
        rgb2 = rgb[:H2*subsample, :W2*subsample].reshape(
            H2, subsample, W2, subsample, 3).mean(axis=(1, 3)).astype(np.int64)
    else:
        H2, W2, rgb2 = H, W, rgb
    pal = np.asarray(palette, dtype=np.int64)
    flat = rgb2.reshape(-1, 3)
    # 分块计算到调色板的距离, 避免一次性大矩阵
    out = [None] * len(palette)
    d_min = np.full(len(flat), np.inf)
    n_idx = np.full(len(flat), -1, dtype=int)
    CH = 256
    for s in range(0, len(flat), CH):
        blk = flat[s:s+CH]
        dd = np.linalg.norm(blk[:, None, :] - pal[None, :, :], axis=2)  # (B,K)
        mn = dd.argmin(axis=1)
        mnd = dd[np.arange(len(blk)), mn]
        seg = slice(s, s+CH)
        d_min[seg] = mnd
        n_idx[seg] = mn
    for k in range(len(palette)):
        m = (n_idx == k) & (d_min <= tol)
        if not m.any():
            continue
        ys, xs = np.divmod(np.nonzero(m)[0], W2)
        if len(xs) < 3:
            continue
        out[k] = np.array([xs.mean() * subsample, ys.mean() * subsample])
    return out


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
        for ti, tp in enumerate(targets):
            if occlusion_visible(tp['coord'], c, zbuf):
                Xc = (c.R @ tp['coord'].reshape(3, 1) + c.t).ravel()
                Z = Xc[2]
                r_pix = min(22.0, c.fx * tp['radius'] / max(Z, 1e-6) * K_SCALE)
                render_marker(tp['coord'], c, img, zbuf,
                              color=tp['color'], marker_r=int(max(MIN_MARKER_R, r_pix)))
        save_image(img, os.path.join(OUTDIR_VIEWS, f"{c.name}.png"))

    # [2] 每个视角定位各点
    print("[2/5] 逐视角按最近颜色定位 2D 点 ...")
    dets = [[None] * N_VIEWS for _ in range(len(targets))]
    for vi in range(N_VIEWS):
        img = np.asarray(Image.open(os.path.join(OUTDIR_VIEWS, f"cam{vi}.png")).convert('RGB'))
        cc = nearest_palette_centroid(img, palette)
        for k in range(len(targets)):
            dets[k][vi] = cc[k]

    # [3] 三角化 (带视角一致性剔除)
    print("[3/5] 多视角 DLT 三角化(剔除配错视角) ...")
    rec = [None] * len(targets)
    n_view_rej = 0
    for ti in range(len(targets)):
        seen = [(vi, dets[ti][vi]) for vi in range(N_VIEWS) if dets[ti][vi] is not None]
        if len(seen) < 3:
            # 视角太少, 直接用全部
            if len(seen) < 2:
                continue
            Pset = np.array([cams[vi].P for vi, _ in seen])
            uvset = np.array([p for _, p in seen])
            rec[ti] = triangulate_dlt(Pset, uvset)
            continue
        keep = list(range(len(seen)))
        for _ in range(2):   # 迭代2轮剔除坏视角
            Pset = np.array([cams[seen[i][0]].P for i in keep])
            uvset = np.array([seen[i][1] for i in keep])
            X = triangulate_dlt(Pset, uvset)
            # 重投影误差
            errs = []
            for i in keep:
                P = cams[seen[i][0]].P
                Xh = np.append(X, 1.0)
                pr = P @ Xh
                pr = pr[:2] / pr[2]
                errs.append(np.linalg.norm(pr - seen[i][1]))
            errs = np.array(errs)
            if errs.max() <= 2.0 or len(keep) < 3:
                rec[ti] = X
                break
            # 剔除误差最大的视角
            bad = int(np.argmax(errs))
            del keep[bad]
            n_view_rej += 1
        else:
            rec[ti] = X
    n_ok = sum(1 for p in rec if p is not None)
    print(f"  成功三角化 {n_ok}/{len(targets)} (剔除 {n_view_rej} 个配错视角)")

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
        if rec[ti] is None:
            continue
        curves.setdefault(tp['curve_id'], []).append(
            (tp['t_index'], rec[ti], tp['radius'],
             np.array(tp['coord_true']) if 'coord_true' in tp else None))
    for cid in curves:
        curves[cid].sort(key=lambda x: x[0])

    from bezier_fit import fit_bezier_any, sample_bezier_any
    DEGREE = 5   # 5次贝塞尔(6控制点): 比三次更贴复杂弯曲, 又不至于过拟合摆动
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    rebuilt_curves = []
    ctrl_rows = []       # 存 每段反解出的 6 控制点
    n_ctrl_ok = 0

    def remove_outliers(P, sigma=3.0):
        """用低阶贝塞尔为基准, 迭代剔除距基准过远的离群测点。返回 (P_clean, 掩码)。"""
        P = np.asarray(P, dtype=float)
        if len(P) <= 5:
            return P, np.ones(len(P), dtype=bool)
        from bezier_fit import fit_cubic_bezier
        keep = np.ones(len(P), dtype=bool)
        for _ in range(3):
            Pk = P[keep]
            if len(Pk) < 5:
                break
            # 低阶(3次)基准, 稳健不摆动
            try:
                P0, p1, p2, p3, err = fit_cubic_bezier(Pk)
            except Exception:
                break
            t = np.linspace(0, 1, 60)[:, None]
            base = ((1-t)**3*P0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3)
            d = np.linalg.norm(Pk[:, None, :] - base[None, :, :], axis=2).min(axis=1)
            med = np.median(d)
            if med < 1e-4:
                break
            thresh = max(sigma * med, 0.3)     # 阈值: 中位残差*sigma, 至少0.3
            good = d <= thresh
            # 更新原始掩码
            idxs = np.where(keep)[0]
            new_keep = keep.copy()
            new_keep[idxs[~good]] = False
            if new_keep.sum() < 5:
                break
            keep = new_keep
        return P[keep], keep

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
    # 收集所有段的 首点(起点)/末点(终点), 空间上很近的(同一分叉节点的多次三角化)
    # 合并取均值, 使相邻共享节点的各段端点用同一坐标。
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
                if np.linalg.norm(g['coord'] - pt) < 0.35:
                    g['pts'].append(pt)
                    g['coord'] = np.mean(g['pts'], axis=0)
                    e_assign[(cid, tag)] = gi
                    placed = True
                    break
            if not placed:
                endpoint_groups.append({'coord': pt, 'pts': [pt]})
                e_assign[(cid, tag)] = len(endpoint_groups) - 1

    n_out = 0
    for cid, items in curves.items():
        P = np.array([it[1] for it in items])          # 坐标
        R = np.array([it[2] for it in items])          # 半径
        # 端点用对齐后坐标
        if len(P) >= 2:
            P[0] = endpoint_groups[e_assign[(cid, 'start')]]['coord']
            P[-1] = endpoint_groups[e_assign[(cid, 'end')]]['coord']
        if len(P) >= DEGREE + 1:
            P, keep = remove_outliers(P)               # 剔除离群测点
            n_out += int((~keep).sum())
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
            ctrl, err = fit_bezier_any(P, degree=DEGREE)
            s = sample_bezier_any(ctrl, n=100)
            n_ctrl_ok += 1
            # 沿贝塞尔插值半径(用测点的弧长参数化 -> 贝塞尔参数 t 对应半径)
            from bezier_fit import chord_length_param
            t_meas = chord_length_param(P)
            radii_curve = np.interp(np.linspace(0, 1, len(s)), t_meas, Rs)
            # 粗细向末端渐细(原模型 taper), 并按测点半径整体缩放
            radii_curve = radii_curve * (1.0 - 0.6 * np.linspace(0, 1, len(s)))
            for i, p in enumerate(ctrl):
                c = 'red' if i in (0, DEGREE) else 'blue'
                ax.scatter(*p, color=c, s=12)
                ctrl_rows.append([cid, i, '端点' if i in (0, DEGREE) else '控制点',
                                  round(float(p[0]), 4), round(float(p[1]), 4),
                                  round(float(p[2]), 4)])
        else:
            s = P
            radii_curve = np.full(len(s), 0.08)
        rebuilt_curves.append({'points': s, 'radii': radii_curve})
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
    print("-" * 60)


if __name__ == "__main__":
    main()
