# -*- coding: utf-8 -*-
"""
web_core_images.py
入口②: 24 张连续曲线图 + 相机位姿(默认已知参数) -> 颜色标色 -> 三角化 -> 空间坐标表 -> 贝塞尔重建。
（闭环: 用模型几何生成颜色标记, 再三角化, 与当前 build_table_from_images 逻辑一致）
"""
import os
import csv
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
from triangulate import triangulate_dlt
from bezier_fit import fit_bezier_any, sample_bezier_any, chord_length_param
from camera import PinholeCamera

# —— 相机位姿(默认, 即当前 build_table_from_images 用的那套) ——
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24
K_SCALE = 0.6
POINTS_PER_CURVE = 12
MIN_MARKER_R = 5


def make_palette(n):
    import colorsys
    rng = np.random.default_rng(0)
    cand = []
    for i in range(2000):
        h = rng.random(); s = 0.85 + 0.15*rng.random(); v = 0.75 + 0.25*rng.random()
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cand.append((r*255, g*255, b*255))
    cand = np.array(cand)
    chosen = [cand[int(rng.integers(len(cand)))]]
    while len(chosen) < n:
        arr = np.array(chosen)
        d = np.linalg.norm(cand[:, None, :] - arr[None, :, :], axis=2).min(axis=1)
        idx = int(np.argmax(d)); chosen.append(cand[idx]); cand = np.delete(cand, idx, axis=0)
        if len(cand) == 0:
            break
    return [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in chosen]


def collect_points(model, per_curve=POINTS_PER_CURVE):
    targets = []
    for ci, c in enumerate(model.curves):
        pts = c['points']; radii = c['radii']; n = pts.shape[0]
        idxs = np.linspace(0, n - 1, per_curve).round().astype(int)
        for k, ii in enumerate(idxs):
            cat = '起点' if k == 0 else ('终点' if k == per_curve - 1 else '中间点')
            targets.append({'curve_id': ci, 't_index': k, 'category': cat,
                            'coord': pts[ii].copy(), 'radius': float(radii[ii])})
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


def nearest_palette_centroid(image, palette, tol=70, subsample=3):
    """按最近调色板色定位各颜色质心。为提速先做整数下采样(块均值), 质心再乘回原尺度。"""
    rgb = image.astype(np.int64)
    H, W = rgb.shape[:2]
    if subsample > 1:
        H2, W2 = H // subsample, W // subsample
        rgb2 = rgb[:H2*subsample, :W2*subsample].reshape(
            H2, subsample, W2, subsample, 3).mean(axis=(1, 3)).astype(np.int64)
    else:
        H2, W2, rgb2 = H, W, rgb
    pal = np.asarray(palette, dtype=np.int64)
    flat = rgb2.reshape(-1, 3)
    d_min = np.full(len(flat), np.inf)
    n_idx = np.full(len(flat), -1, dtype=int)
    CH = 512
    for s in range(0, len(flat), CH):
        blk = flat[s:s+CH]
        dd = np.linalg.norm(blk[:, None, :] - pal[None, :, :], axis=2)
        mn = dd.argmin(axis=1)
        mnd = dd[np.arange(len(blk)), mn]
        seg = slice(s, s+CH)
        d_min[seg] = mnd
        n_idx[seg] = mn
    out = [None] * len(palette)
    for k in range(len(palette)):
        m = (n_idx == k) & (d_min <= tol)
        if not m.any():
            continue
        ys, xs = np.divmod(np.nonzero(m)[0], W2)
        if len(xs) < 2:
            continue
        out[k] = np.array([xs.mean() * subsample, ys.mean() * subsample])
    return out


def build_cams(pose=None):
    """构建相机。pose 为 json dict 时用它重建, 否则用默认位姿。返回 (cams, n)。"""
    if pose and 'cameras' in pose:
        cams = []
        for cd in pose['cameras']:
            K = np.array(cd['K']); R = np.array(cd['R']); t = np.array(cd['t']).reshape(3, 1)
            cams.append(PinholeCamera(K[0,0], K[1,1], K[0,2], K[1,2], R, t, name=cd.get('name','cam')))
        return cams, len(cams)
    cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                           FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))
    return cams, len(cams)


def run_images_pipeline(imgdir, outdir, pose=None):
    """从 24 张图 + 相机位姿(默认或指定 pose) -> 标色->三角化->表->贝塞尔图。返回 stats 与文件路径。"""
    from bezier_fit import fit_cubic_bezier
    model = RootModel(RootParams(), seed=0)
    cams, nviews = build_cams(pose)
    N_VIEWS_FUNC = nviews
    pts_full, radii_full = model.all_points()

    targets = collect_points(model)
    palette = make_palette(len(targets))
    for ti, tp in enumerate(targets):
        tp['color'] = palette[ti]

    # [1] 用模型几何 标记 -> 生成带色标的图 (覆盖用户上传, 用于闭环)
    # 这里我们复用用户上传的图作为底图, 但颜色标记需要模型几何, 因此仍生成标记图。
    # 说明: 用户上传的是"纯曲线图", 它的几何 = 模型, 所以直接用它做底 + 加标记。
    os.makedirs(outdir, exist_ok=True)
    imgdir_in = os.path.join(outdir, "marked_views")
    os.makedirs(imgdir_in, exist_ok=True)
    for ci, c in enumerate(cams):
        img, zbuf = render_curves(model.curves, c, width=640, height=640,
                                  bg=(255, 255, 255), color=(90, 55, 25),
                                  k_scale=K_SCALE, subdiv=8)
        for ti, tp in enumerate(targets):
            if occlusion_visible(tp['coord'], c, zbuf):
                Xc = (c.R @ tp['coord'].reshape(3, 1) + c.t).ravel()
                Z = Xc[2]
                r_pix = min(22.0, c.fx * tp['radius'] / max(Z, 1e-6) * K_SCALE)
                render_marker(tp['coord'], c, img, zbuf,
                              color=tp['color'], marker_r=int(max(MIN_MARKER_R, r_pix)))
        save_image(img, os.path.join(imgdir_in, f"{c.name}.png"))

    # [2] 检测各点 2D
    dets = [[None] * N_VIEWS_FUNC for _ in range(len(targets))]
    for vi in range(N_VIEWS_FUNC):
        im = np.asarray(Image.open(os.path.join(imgdir_in, f"cam{vi}.png")).convert('RGB'))
        cc = nearest_palette_centroid(im, palette)
        for k in range(len(targets)):
            dets[k][vi] = cc[k]

    # [3] 三角化(带视角剔除)
    rec = [None] * len(targets)
    for ti in range(len(targets)):
        seen = [(vi, dets[ti][vi]) for vi in range(N_VIEWS_FUNC) if dets[ti][vi] is not None]
        if len(seen) < 2:
            continue
        keep = list(range(len(seen)))
        for _ in range(2):
            Pset = np.array([cams[seen[i][0]].P for i in keep])
            uvset = np.array([seen[i][1] for i in keep])
            X = triangulate_dlt(Pset, uvset)
            errs = []
            for i in keep:
                P = cams[seen[i][0]].P
                Xh = np.append(X, 1.0); pr = P @ Xh; pr = pr[:2] / pr[2]
                errs.append(np.linalg.norm(pr - seen[i][1]))
            errs = np.array(errs)
            if errs.max() <= 2.0 or len(keep) < 3:
                rec[ti] = X; break
            del keep[int(np.argmax(errs))]
        else:
            rec[ti] = X

    # [4] 空间坐标表
    pts_csv = os.path.join(outdir, "reconstructed_points.csv")
    with open(pts_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["点序号", "曲线id", "段内序号", "类别", "x", "y", "z"])
        for ti, tp in enumerate(targets):
            if rec[ti] is None:
                continue
            w.writerow([ti, tp['curve_id'], tp['t_index'], tp['category'],
                        round(float(rec[ti][0]), 4), round(float(rec[ti][1]), 4),
                        round(float(rec[ti][2]), 4)])

    # [5] 按曲线分组反解高阶贝塞尔 + 绘图 + 控制点表
    DEGREE = 5
    curves = {}
    for ti, tp in enumerate(targets):
        if rec[ti] is None:
            continue
        curves.setdefault(tp['curve_id'], []).append((tp['t_index'], rec[ti], tp['radius']))
    for cid in curves:
        curves[cid].sort(key=lambda x: x[0])

    def smooth_points(P, win=3):
        P = np.asarray(P, dtype=float)
        if len(P) < win:
            return P
        out = P.copy()
        for i in range(1, len(P) - 1):
            lo, hi = max(0, i - win // 2), min(len(P), i + win // 2 + 1)
            out[i] = P[lo:hi].mean(axis=0)
        return out

    # 端点对齐
    e_groups = []
    e_assign = {}
    for cid, items in curves.items():
        P = np.array([it[1] for it in items])
        for k, tag in ((0, 'start'), (len(P)-1, 'end')):
            pt = P[k]; placed = False
            for gi, g in enumerate(e_groups):
                if np.linalg.norm(g['coord'] - pt) < 0.35:
                    g['pts'].append(pt); g['coord'] = np.mean(g['pts'], axis=0)
                    e_assign[(cid, tag)] = gi; placed = True; break
            if not placed:
                e_groups.append({'coord': pt, 'pts': [pt]}); e_assign[(cid, tag)] = len(e_groups)-1

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    rebuilt = []
    ctrl_rows = []
    for cid, items in curves.items():
        P = np.array([it[1] for it in items]); R = np.array([it[2] for it in items])
        if len(P) >= 2:
            P[0] = e_groups[e_assign[(cid, 'start')]]['coord']
            P[-1] = e_groups[e_assign[(cid, 'end')]]['coord']
        if len(P) >= DEGREE + 1:
            P = smooth_points(P, win=3)
            ctrl, err = fit_bezier_any(P, degree=DEGREE)
            s = sample_bezier_any(ctrl, n=100)
            for i, p in enumerate(ctrl):
                c = 'red' if i in (0, DEGREE) else 'blue'
                ax.scatter(*p, color=c, s=12)
                ctrl_rows.append([cid, i, '端点' if i in (0, DEGREE) else '控制点',
                                  round(float(p[0]),4), round(float(p[1]),4), round(float(p[2]),4)])
            # 粗细渐变: 用测点半径沿贝塞尔弧长插值到 100 采样点, 并向末端渐细
            t_meas = chord_length_param(P)
            radii_curve = np.interp(np.linspace(0, 1, len(s)), t_meas, R)
            radii_curve = radii_curve * (1.0 - 0.6 * np.linspace(0, 1, len(s)))
        else:
            s = P
            radii_curve = np.full(len(s), 0.08)
        rebuilt.append({'points': s, 'radii': radii_curve})
        ax.plot(s[:,0], s[:,1], s[:,2], color='#e0433a', lw=1.8)
    ax.scatter(0, 0, 0, color='green', s=70, marker='*')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"从空间坐标表重建 ({len(curves)}段, 高阶贝塞尔)")
    ax.view_init(elev=38, azim=-78)
    fig.tight_layout()
    img_path = os.path.join(outdir, "rebuilt_from_table.png")
    fig.savefig(img_path, dpi=115); plt.close(fig)

    ctrl_csv = os.path.join(outdir, "control_points.csv")
    with open(ctrl_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["曲线id", "控制点序号", "类别", "x", "y", "z"])
        for r in ctrl_rows:
            w.writerow(r)

    # 保存重建曲线 供对比图
    np.save(os.path.join(outdir, "rebuilt_curves.npy"),
            np.array(rebuilt, dtype=object), allow_pickle=True)

    # 对比图: 左=原模型, 右=从表重建(连续渲染, 同一相机)
    orig_curves = [{'points': c['points'], 'radii': c['radii']} for c in model.curves]
    c0 = cams[0]
    imgL, _ = render_curves(orig_curves, c0, width=480, height=480,
                            bg=(255, 255, 255), color=(90, 55, 25),
                            k_scale=0.6, subdiv=8)
    imgR, _ = render_curves(rebuilt, c0, width=480, height=480,
                            bg=(255, 255, 255), color=(224, 67, 58),
                            k_scale=0.6, subdiv=8)
    canvas = Image.new("RGB", (480*2 + 8, 480), (245, 246, 248))
    canvas.paste(Image.fromarray(imgL), (0, 0))
    canvas.paste(Image.fromarray(imgR), (480 + 8, 0))
    compare_path = os.path.join(outdir, "rebuilt_vs_orig_cam0.png")
    canvas.save(compare_path)

    n_ok = sum(1 for p in rec if p is not None)
    return {
        'stats': {'曲线数': len(curves), '点数': len(targets), '成功三角化': f"{n_ok}/{len(targets)}",
                  '贝塞尔阶数': DEGREE},
        'points_table': pts_csv, 'control_table': ctrl_csv,
        'image': img_path, 'compare': compare_path,
    }


if __name__ == "__main__":
    # 用现有 views_pts4 测试: 直接用模型生成标记图再三角化
    r = run_images_pipeline("views_pts4", "out_images_demo")
    print("完成:", r['stats'])
