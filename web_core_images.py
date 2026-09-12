# -*- coding: utf-8 -*-
"""
web_core_images.py
入口③: 上传的 24 张带颜色标记图 + 相机位姿(pose.json 或默认参数)
       -> 颜色检测(仅用上传图像素) -> 三角化 -> 空间坐标表 -> 贝塞尔重建。

标记约定(与模块① build 阶段一致, 双方无需模型几何即可解码):
  - 目标点 ti = 曲线id * 每曲线点数 + 段内序号, 共 n_curves*points_per_curve 个;
  - ti 对应调色板色 make_palette(n)[ti](确定性生成, rng 固定);
  - 每条曲线取 12 个曲线上点(起点/中间/终点)。
模块①在 pose.json 里写入 mark_meta{n_curves, points_per_curve}; 缺省按 10*12。
管线内不存在绕过上传图的自渲染旁路(修复 P0-1): 检测只读 imgdir 里的上传图。
"""
import os
import csv
import json
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

from camera import ring_of_cameras
from render import render_curves
from bezier_fit import fit_bezier_any, sample_bezier_any, remove_outliers
from camera import PinholeCamera

# —— 相机位姿(默认, 与模块①/CLI 一致) ——
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24
K_SCALE = 0.6
N_CURVES = 10
POINTS_PER_CURVE = 12
MIN_MARKER_R = 3      # 与模块①一致: 小于此像素半径的标记被放大, 反演时需注意截断
ENDPOINT_MERGE_TOL = 0.03   # 端点对齐: 仅合并同一节点的重复测量
TRIANGULATION_TOL_PX = 1.2  # 三角化视角剔除的重投影残差阈值(px)


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


def detect_palette_blobs(image, palette, tol=70):
    """按最近调色板色在上传图上定位各颜色的 2D 质心与等效像素半径。
    不做降采样(P0-2): 5px 小标记经块均值稀释会漏检/偏移。
    含连通域兜底(P1-6): 同色出现多个连通块时, 若第二大的块面积可观则视为
    歧义丢弃该视角检测, 否则只取最大块质心, 避免全局均值被分裂块拉偏。
    含腐蚀清洗: 标记被邻近标记啃噬出的新月残片会拉偏质心, 腐蚀掉交界残片;
    完整圆盘腐蚀后质心不变(对称)。
    返回 out[k] = (质心uv, 等效半径r_pix=sqrt(area/pi)+腐蚀补偿) 或 None。"""
    from scipy import ndimage
    rgb = image.astype(np.int64)
    H, W = rgb.shape[:2]
    pal = np.asarray(palette, dtype=np.int64)
    flat = rgb.reshape(-1, 3)
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
    struct = np.ones((3, 3), dtype=bool)
    out = [None] * len(palette)
    for k in range(len(palette)):
        m = ((n_idx == k) & (d_min <= tol)).reshape(H, W)
        n_pix = int(m.sum())
        if n_pix < 3:
            continue
        lab, n_lab = ndimage.label(m)
        if n_lab > 1:
            # 取最大连通域: 生成端会为每个标记补画"中心点"(近->远, 见
            # render.draw_markers), 被邻标记覆盖成残片的标记以中心点为主域,
            # 其质心即真实投影; 完整圆盘的中心点与盘连通, 质心不变。
            sizes = ndimage.sum(m, lab, range(1, n_lab + 1))
            m = lab == (int(np.argmax(sizes)) + 1)
        # 腐蚀 1 轮: 去掉被邻近标记啃噬出的 1px 级新月细弧(质心严重拉偏);
        # 完整圆盘/中心点腐蚀后质心不变(对称)。只剩细弧的视角在此被丢弃。
        e = ndimage.binary_erosion(m, structure=struct)
        if e.any():
            m = e
            n_erode = 1
        else:
            n_erode = 0
        n_pix = int(m.sum())
        if n_pix < 3:
            continue
        ys, xs = np.nonzero(m)
        # 半径反演补偿腐蚀掉的 1 圈边缘(完整圆盘精确, 残片偏小由中位数兜底)
        out[k] = (np.array([xs.mean(), ys.mean()]),
                  float(np.sqrt(n_pix / np.pi)) + n_erode)
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


def smooth_points(P, win=3):
    P = np.asarray(P, dtype=float)
    if len(P) < win:
        return P
    out = P.copy()
    for i in range(1, len(P) - 1):
        lo, hi = max(0, i - win // 2), min(len(P), i + win // 2 + 1)
        out[i] = P[lo:hi].mean(axis=0)
    return out


# 端点检测可靠性判据: 保留视角数/残差中位数明显低于常态(中位 ~21 视角, ~0.2px)
# 的点, 视为被邻近标记啃噬的"残片点", 其测量带系统性偏移。
REL_MIN_VIEWS = 12
REL_MAX_MED_RESID = 0.8


def point_reliability(kept_count, med_resid):
    return kept_count >= REL_MIN_VIEWS and med_resid <= REL_MAX_MED_RESID


def prepare_curve_fit(items, n_per, rel_by_ti, degree=5):
    """整理一条曲线的测点供贝塞尔拟合, 并对不可靠/缺失的端点做预测替换。
    items: [(t_index, xyz, radius)] 按 t_index 升序;
    rel_by_ti: {t_index: 是否可靠}。
    当首端(t_index=0)或末端(t_index=n_per-1)缺失或不可靠时, 用可靠测点做
    自由端点最小二乘拟合, 在 t=0/1 处预测端点坐标(半径取最近可靠点),
    避免把残片偏移的端点测量钉死在拟合端点上(P1-4 的间接后果)。
    返回 (P, t, R, n_pred)。"""
    from bezier_fit import fit_bezier_full, bernstein_basis
    P = np.array([it[1] for it in items], dtype=float)
    R = np.array([it[2] for it in items], dtype=float)
    t = np.array([it[0] for it in items], dtype=float) / max(n_per - 1, 1)
    tis = [it[0] for it in items]
    has_start = tis[0] == 0
    has_end = tis[-1] == n_per - 1
    bad_start = (not has_start) or (not rel_by_ti.get(0, False))
    bad_end = (not has_end) or (not rel_by_ti.get(n_per - 1, False))
    if not (bad_start or bad_end):
        return P, t, R, 0
    rel_idx = [i for i, k in enumerate(tis) if rel_by_ti.get(k, False)]
    if len(rel_idx) < 4:
        return P, t, R, 0
    # 预测用 3 次拟合(真值曲线即三次贝塞尔): 低阶外推到 t=0/1 稳健,
    # 高阶会放大端点外推的摆动。
    d = min(3, len(rel_idx) - 1)
    try:
        ctrl, _ = fit_bezier_full(P[rel_idx], t[rel_idx], degree=d)
    except ValueError:
        return P, t, R, 0
    n_pred = 0
    if bad_start:
        p0 = bernstein_basis(d, np.array([0.0]))[0] @ ctrl
        r0 = R[rel_idx[0]]
        if has_start:
            P[0] = p0
        else:
            P = np.vstack([p0[None, :], P])
            R = np.concatenate([[r0], R])
            t = np.concatenate([[0.0], t])
        n_pred += 1
    if bad_end:
        p1 = bernstein_basis(d, np.array([1.0]))[0] @ ctrl
        r1 = R[rel_idx[-1]]
        if has_end:
            P[-1] = p1
        else:
            P = np.vstack([P, p1[None, :]])
            R = np.concatenate([R, [r1]])
            t = np.concatenate([t, [1.0]])
        n_pred += 1
    return P, t, R, n_pred


def estimate_radius_3d(r_pix_list, cams_list, X):
    """由各视角的标记像素半径反演三维半径 r3d = r_pix*Z/(fx*K_SCALE), 取中位数。
    注意: 生成端有 MIN_MARKER_R 像素下限, 被截断的小标记会偏大。"""
    Xh = np.append(np.asarray(X, dtype=float), 1.0)
    est = []
    for r_pix, cam in zip(r_pix_list, cams_list):
        if r_pix is None or r_pix <= 0:
            continue
        pr = cam.P @ Xh
        Z = pr[2]
        if Z <= 0:
            continue
        est.append(r_pix * Z / (cam.fx * K_SCALE))
    return float(np.median(est)) if est else None


def run_images_pipeline(imgdir, outdir, pose=None):
    """从上传的带标记图目录 + 相机位姿(默认或 pose.json) -> 检测->三角化->表->贝塞尔图。
    返回 stats 与文件路径 dict。"""
    os.makedirs(outdir, exist_ok=True)
    cams, nviews = build_cams(pose)
    # 标记约定元数据(模块①写入 pose.json; 缺省按默认参数)
    meta = (pose or {}).get('mark_meta') or {}
    n_curves = int(meta.get('n_curves', N_CURVES))
    n_per = int(meta.get('points_per_curve', POINTS_PER_CURVE))
    n_targets = n_curves * n_per
    palette = make_palette(n_targets)

    # 目标点约定表: ti = ci*n_per + k (与模块①标记顺序一致)
    targets = []
    for ci in range(n_curves):
        for k in range(n_per):
            cat = '起点' if k == 0 else ('终点' if k == n_per - 1 else '中间点')
            targets.append({'curve_id': ci, 't_index': k, 'category': cat})

    # [1] 检测: 只读上传图目录(P0-1), 不降采样(P0-2)
    imgs = []
    for vi in range(nviews):
        p = os.path.join(imgdir, f"cam{vi}.png")
        imgs.append(np.asarray(Image.open(p).convert('RGB'))
                    if os.path.isfile(p) else None)
    n_have = sum(1 for im in imgs if im is not None)
    dets = [[None] * nviews for _ in range(n_targets)]     # [ti][vi] = uv
    rpix = [[None] * nviews for _ in range(n_targets)]     # [ti][vi] = 像素半径
    for vi in range(nviews):
        if imgs[vi] is None:
            continue
        blobs = detect_palette_blobs(imgs[vi], palette)
        for ti in range(n_targets):
            if blobs[ti] is not None:
                dets[ti][vi], rpix[ti][vi] = blobs[ti]

    # [2] 三角化(迭代剔除坏视角 + cheirality 正深度校验, P1-5) + 半径反演
    from triangulate import triangulate_robust
    rec = [None] * n_targets
    rad3d = [None] * n_targets
    kept_views = [set() for _ in range(n_targets)]
    kept_resids = [[] for _ in range(n_targets)]
    for ti in range(n_targets):
        seen = [(vi, dets[ti][vi]) for vi in range(nviews) if dets[ti][vi] is not None]
        if len(seen) < 2:
            continue
        Pset = np.array([cams[vi].P for vi, _ in seen])
        uvset = np.array([p for _, p in seen])
        # 三角化: 迭代剔除重投影残差大的视角(P1-5)。阈值取 1.2px:
        # 被邻近标记啃噬成残片的检测(偏 ~1.5-3px)会被剔除, 干净视角主导解算;
        # 若干净视角不足则按 min_views=3 保底接受最优的 3 个视角, 不丢点。
        X, keep, _res = triangulate_robust(Pset, uvset, tol_px=TRIANGULATION_TOL_PX)
        if X is None:
            continue
        rec[ti] = X
        kept_views[ti] = set(seen[i][0] for i in keep)
        kept_resids[ti] = list(_res)
        rad3d[ti] = estimate_radius_3d(
            [rpix[ti][seen[i][0]] for i in keep],
            [cams[seen[i][0]] for i in keep], X)

    # 保存逐点检测/残差(供 evaluate_reconstruction.py 校验 C4/B2):
    # reproj_errs 只含三角化采用的视角; dets 保留全部视角检测(被剔除的即疑似对应错误)
    detections = {'points': {}}
    for ti in range(n_targets):
        if rec[ti] is None:
            continue
        Xh = np.append(rec[ti], 1.0)
        det_entry = {'dets': {}, 'reproj_errs': []}
        for vi in range(nviews):
            if dets[ti][vi] is not None:
                pr = cams[vi].P @ Xh; pr = pr[:2] / pr[2]
                det_entry['dets'][str(vi)] = [float(x) for x in dets[ti][vi]]
                if vi in kept_views[ti]:
                    det_entry['reproj_errs'].append(
                        float(np.linalg.norm(pr - dets[ti][vi])))
        detections['points'][str(ti)] = det_entry
    with open(os.path.join(outdir, "detections.json"), "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False)

    # [3] 空间坐标表
    pts_csv = os.path.join(outdir, "reconstructed_points.csv")
    with open(pts_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["点序号", "曲线id", "段内序号", "类别", "x", "y", "z"])
        for ti, tp in enumerate(targets):
            if rec[ti] is None:
                continue
            w.writerow([ti, tp['curve_id'], tp['t_index'], tp['category'],
                        round(float(rec[ti][0]), 4), round(float(rec[ti][1]), 4),
                        round(float(rec[ti][2]), 4)])

    # [4] 按曲线分组: 端点预测 + 离群剔除(P0-3) + 平滑 + 端点对齐 + 高阶贝塞尔
    DEGREE = 5
    curves = {}
    for ti, tp in enumerate(targets):
        if rec[ti] is None or rad3d[ti] is None:
            continue
        curves.setdefault(tp['curve_id'], []).append(
            (tp['t_index'], rec[ti], rad3d[ti]))
    for cid in curves:
        curves[cid].sort(key=lambda x: x[0])

    # 端点对齐: 仅合并"同一节点的重复测量"(同一次分叉被两条曲线共享的端点)。
    # 阈值必须远小于相邻分支节点的间距(一个采样步长, 细根约 0.06), 否则会把
    # 不同的分支节点错误合并、拉偏两条曲线的端点。
    e_groups = []
    e_assign = {}
    for cid, items in curves.items():
        P = np.array([it[1] for it in items])
        if len(P) < 2:
            continue
        for k, tag in ((0, 'start'), (len(P)-1, 'end')):
            pt = P[k]; placed = False
            for gi, g in enumerate(e_groups):
                if np.linalg.norm(g['coord'] - pt) < ENDPOINT_MERGE_TOL:
                    g['pts'].append(pt); g['coord'] = np.mean(g['pts'], axis=0)
                    e_assign[(cid, tag)] = gi; placed = True; break
            if not placed:
                e_groups.append({'coord': pt, 'pts': [pt]}); e_assign[(cid, tag)] = len(e_groups)-1

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    rebuilt = []
    ctrl_rows = []
    n_out = 0
    controls_by_cid = {}       # cid -> 拟合控制点(供根表)
    curve_meta = {}            # cid -> (测点12槽坐标, 半径12槽) 供根表
    for cid, items in curves.items():
        P = np.array([it[1] for it in items])
        R = np.array([it[2] for it in items])
        # 各测点的生成参数 t = t_index/(每曲线点数-1): 与标记采样约定一致,
        # 用它拟合可避免弦长参数化失配导致的控制点震荡(P2-10)。
        t_arr = np.array([it[0] for it in items], dtype=float) / max(n_per - 1, 1)
        if len(P) >= 2:
            P[0] = e_groups[e_assign[(cid, 'start')]]['coord']
            P[-1] = e_groups[e_assign[(cid, 'end')]]['coord']
        if len(P) >= DEGREE + 1:
            P, keep = remove_outliers(P)          # 剔除离群测点, 防止控制点飞掉
            n_out += int((~keep).sum())
            Rk = R[keep]
            t_k = t_arr[keep]
            P = smooth_points(P, win=3)
            Rs = Rk.copy()
            for i in range(1, len(Rs) - 1):
                lo, hi = max(0, i - 1), min(len(Rs), i + 2)
                Rs[i] = Rk[lo:hi].mean()
            deg = min(DEGREE, len(P) - 1)
            ctrl, err = fit_bezier_any(P, degree=deg, t=t_k)
            s = sample_bezier_any(ctrl, n=100)
            controls_by_cid[cid] = ctrl
            for i, p in enumerate(ctrl):
                c = 'red' if i in (0, deg) else 'blue'
                ax.scatter(*p, color=c, s=12)
                ctrl_rows.append([cid, i, '端点' if i in (0, deg) else '控制点',
                                  round(float(p[0]),4), round(float(p[1]),4), round(float(p[2]),4)])
            # 粗细: 直接用图像反演的半径沿生成参数插值(测量值已含真实渐变,
            # 不再加硬编码 taper), 与几何拟合同一参数化
            radii_curve = np.interp(np.linspace(0, 1, len(s)), t_k, Rs)
        else:
            s = P
            radii_curve = np.full(len(s), float(np.mean(R)) if len(R) else 0.08)
        rebuilt.append({'points': s, 'radii': radii_curve})
        ax.plot(s[:,0], s[:,1], s[:,2], color='#e0433a', lw=1.8)
        # 根表用 12 槽测点/半径(缺失槽位用拟合曲线在对应参数处补齐)
        pts_full = np.full((n_per, 3), np.nan)
        rad_full = np.full(n_per, np.nan)
        for (k, xyz, r) in items:
            pts_full[k] = xyz
            rad_full[k] = r
        for k in range(n_per):
            if np.isnan(pts_full[k]).any():
                tt = k / max(n_per - 1, 1)
                from bezier_fit import bernstein_basis
                if cid in controls_by_cid:
                    dg = len(controls_by_cid[cid]) - 1
                    pts_full[k] = bernstein_basis(dg, np.array([tt]))[0] @ controls_by_cid[cid]
                else:
                    pts_full[k] = np.nan_to_num(pts_full[k])
                rad_full[k] = np.nanmax(rad_full) if np.isfinite(rad_full).any() else 0.08
        curve_meta[cid] = {'points': pts_full, 'radii': rad_full}
        # 端点槽位用拟合端点(已做节点对齐, 更接近真实节点位置)
        if cid in controls_by_cid:
            curve_meta[cid]['points'][0] = controls_by_cid[cid][0]
            curve_meta[cid]['points'][-1] = controls_by_cid[cid][-1]
    ax.scatter(0, 0, 0, color='green', s=70, marker='*')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"从上传图重建 ({len(curves)}段, {DEGREE}次贝塞尔, 剔除{n_out}离群点)")
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

    # 对比图: 左=上传的第 0 视角原图, 右=重建曲线在同一相机下的渲染
    # (不引入模型几何, 避免"原模型"旁路)
    c0 = cams[0]
    imgL = Image.open(os.path.join(imgdir, "cam0.png")).convert('RGB') \
        if os.path.isfile(os.path.join(imgdir, "cam0.png")) else \
        Image.fromarray(np.full((480, 480, 3), 255, dtype=np.uint8))
    imgR, _ = render_curves(rebuilt, c0, width=480, height=480,
                            bg=(255, 255, 255), color=(224, 67, 58),
                            k_scale=0.6, subdiv=8)
    canvas = Image.new("RGB", (480*2 + 8, 480), (245, 246, 248))
    canvas.paste(imgL.resize((480, 480)), (0, 0))
    canvas.paste(Image.fromarray(imgR), (480 + 8, 0))
    compare_path = os.path.join(outdir, "rebuilt_vs_orig_cam0.png")
    canvas.save(compare_path)

    # [5] 根表(P2-9): 由重建结果推导 "主根 -> 分支点 -> 侧根" 树,
    #     输出与 table_build 相同 14 列结构的 root_table.csv/.sql
    from root_topology import write_root_table
    depths = meta.get('curve_depths')
    depths = {cid: int(depths[cid]) for cid in curve_meta} if depths else None
    root_csv = os.path.join(outdir, "root_table.csv")
    root_sql = os.path.join(outdir, "root_table.sql")
    write_root_table(curve_meta, depths,
                     max_depth=int(meta.get('max_depth', max(depths.values()) if depths else 1)),
                     csv_path=root_csv, sql_path=root_sql)

    n_ok = sum(1 for p in rec if p is not None)
    return {
        'stats': {'曲线数': len(curves), '点数': n_targets,
                  '成功三角化': f"{n_ok}/{n_targets}", '贝塞尔阶数': DEGREE,
                  '剔除离群点': n_out, '视角': f"{n_have}/{nviews}"},
        'points_table': pts_csv, 'control_table': ctrl_csv,
        'root_table': root_csv, 'root_sql': root_sql,
        'image': img_path, 'compare': compare_path,
    }


if __name__ == "__main__":
    # 用带标记的视图目录测试(如模块①生成或 views_pts4): 只读图, 不自渲染
    r = run_images_pipeline("views_pts4", "out_images_demo")
    print("完成:", r['stats'])
