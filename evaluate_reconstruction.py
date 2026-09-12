# -*- coding: utf-8 -*-
"""
evaluate_reconstruction.py
一键自动评估: 把重建产物(reconstructed_points.csv / control_points.csv / 根表)
与真值(dense_meta.json 的 coord_true, 及由 RootModel 重建的曲线/分支点真值)对比,
输出 JSON + markdown 表 + 逐条 pass/fail(对应 docs/重建优化-prompt与完成标准.md 第二部分)。

用法:
  python evaluate_reconstruction.py                    # 评估根目录 CLI 产物(seed=0)
  python evaluate_reconstruction.py --points results/job_images/reconstructed_points.csv \
      --controls results/job_images/control_points.csv --label web
  python evaluate_reconstruction.py --seed 1 --views-dir <seed=1 视图集>   # seed 回归

指标来源:
  - B/C 组: reconstructed_points.csv 的 (x,y,z) vs 真值 coord_true(按 点序号 对齐)。
  - D 组: control_points.csv 的贝塞尔控制点/端点 vs 真值曲线。
  - C4/B2: 需管线写出的 detections.json(逐点逐视角检测+残差); 旧产物没有则记 n/a。
  - E/F 组: 重建产出的根表(CSV+SQL); 未生成则记 fail(待阶段3)。
"""
import os
import json
import csv
import sqlite3
import argparse
import numpy as np

from root_model import RootModel, RootParams
from camera import ring_of_cameras

# 与 build_table_from_images / web_core_images 一致的相机参数(默认位姿)
CAM_RADIUS, CAM_HEIGHT = 15.0, -2.0
CAM_TARGET = [0.0, 0.0, -7.0]
FX, FY, CX, CY = 600.0, 600.0, 320.0, 340.0
N_VIEWS = 24

N_CURVES = 10
POINTS_PER_CURVE = 12
N_POINTS = N_CURVES * POINTS_PER_CURVE

# ---- 完成标准阈值(docs 第二部分) ----
TH = {
    'C1_median': 0.02,
    'C2_mean': 0.05,
    'C3_max': 0.30,
    'C4_reproj_median_px': 1.0,
    'C4_reproj_max_px': 2.0,
    'D1_curve_max_dev': 0.40,
    'D2_ctrl_bbox_dist': 2.0,
    'D3_endpoint_err': 0.05,
    'E2_branch_err': 0.05,
    'F1_radius_rel_err': 0.30,
}


# ----------------------------------------------------------------------
def read_points_csv(path):
    """读 reconstructed_points.csv -> {ti: (curve_id, t_index, xyz)}。"""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            ti = int(d['点序号'])
            out[ti] = (int(d['曲线id']), int(d['段内序号']),
                       np.array([float(d['x']), float(d['y']), float(d['z'])]))
    return out


def read_control_csv(path):
    """读 control_points.csv -> {cid: [(序号, 类别, xyz), ...]}(按控制点序号排序)。"""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            cid = int(d['曲线id'])
            out.setdefault(cid, []).append(
                (int(d['控制点序号']), d['类别'],
                 np.array([float(d['x']), float(d['y']), float(d['z'])])))
    for cid in out:
        out[cid].sort(key=lambda x: x[0])
    return out


def load_meta_truth(seed=0):
    """真值: seed=0 优先用 dense_meta.json 的 coord_true; 其余 seed 由模型生成。
    返回 (truth_pts: {ti: xyz}, model)。"""
    model = RootModel(RootParams(), seed=seed)
    truth_pts = {}
    if seed == 0 and os.path.isfile('dense_meta.json'):
        with open('dense_meta.json', encoding='utf-8') as f:
            meta = json.load(f)['points']
        for m in meta:
            truth_pts[m['index']] = np.array(m['coord_true'], dtype=float)
        # 与模型交叉校验(真值来源一致性)
        for ci, c in enumerate(model.curves):
            idxs = np.linspace(0, c['points'].shape[0] - 1,
                               POINTS_PER_CURVE).round().astype(int)
            for k, ii in enumerate(idxs):
                ti = ci * POINTS_PER_CURVE + k
                if ti in truth_pts:
                    assert np.linalg.norm(truth_pts[ti] - c['points'][ii]) < 1e-6, \
                        f"dense_meta.json 真值与 seed={seed} 模型不一致(ti={ti})"
    else:
        for ci, c in enumerate(model.curves):
            idxs = np.linspace(0, c['points'].shape[0] - 1,
                               POINTS_PER_CURVE).round().astype(int)
            for k, ii in enumerate(idxs):
                truth_pts[ci * POINTS_PER_CURVE + k] = c['points'][ii].copy()
    return truth_pts, model


def point_to_polyline_dist(p, poly):
    """点 p 到折线 poly (N,3) 的最小距离。"""
    seg_a = poly[:-1]
    seg_b = poly[1:]
    ab = seg_b - seg_a                               # (N-1,3)
    ap = p[None, :] - seg_a
    t = np.sum(ap * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = seg_a + t[:, None] * ab
    return float(np.linalg.norm(proj - p[None, :], axis=1).min())


def dist_to_aabb(p, lo, hi):
    """点到轴对齐包围盒的距离(内部为 0)。"""
    d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return float(np.linalg.norm(d))


# ----------------------------------------------------------------------
def eval_points(points, truth_pts, model, detections=None):
    """B1/B2/C1-C4。返回 (指标 dict, 检查项 list)。"""
    res = {}
    checks = []
    n_total = N_POINTS
    n_ok = len(points)
    res['B1_成功点数'] = f"{n_ok}/{n_total}"
    checks.append({'组': 'B1', '项': '标记点重建成功率',
                   '值': f"{n_ok}/{n_total}", '阈值': f"{n_total}/{n_total}",
                   'pass': n_ok == n_total})

    errs, per_curve_max = [], {}
    for ti, (cid, k, xyz) in sorted(points.items()):
        if ti not in truth_pts:
            continue
        e = float(np.linalg.norm(xyz - truth_pts[ti]))
        errs.append((ti, cid, e))
        per_curve_max[cid] = max(per_curve_max.get(cid, 0.0), e)
    if errs:
        arr = np.array([e for _, _, e in errs])
        res['C1_中位误差'] = float(np.median(arr))
        res['C2_平均误差'] = float(arr.mean())
        res['C3_最大误差'] = float(arr.max())
        res['C3_最差点'] = int(max(errs, key=lambda x: x[2])[0])
    else:
        res['C1_中位误差'] = res['C2_平均误差'] = res['C3_最大误差'] = None
    for key, th, name in (('C1_中位误差', TH['C1_median'], 'C1 中位误差'),
                          ('C2_平均误差', TH['C2_mean'], 'C2 平均误差'),
                          ('C3_最大误差', TH['C3_max'], 'C3 最大误差')):
        v = res[key]
        ok = (v is not None) and (v <= th)
        checks.append({'组': name.split()[0], '项': name, '值':
                       ('n/a' if v is None else f"{v:.4f}"),
                       '阈值': f"≤{th}", 'pass': ok})

    # C4 重投影残差(需管线保存的逐点残差)
    reproj = None
    if detections:
        rs = []
        for ti, det in detections.items():
            for r in det.get('reproj_errs', []):
                rs.append(r)
        if rs:
            rs = np.array(rs)
            reproj = (float(np.median(rs)), float(rs.max()))
    if reproj:
        res['C4_重投影残差px'] = {'中位': reproj[0], '最大': reproj[1]}
        checks.append({'组': 'C4', '项': '重投影残差(中位/最大)',
                       '值': f"{reproj[0]:.2f}/{reproj[1]:.2f}px",
                       '阈值': f"≤{TH['C4_reproj_median_px']}/{TH['C4_reproj_max_px']}px",
                       'pass': reproj[0] <= TH['C4_reproj_median_px'] and
                               reproj[1] <= TH['C4_reproj_max_px']})
    else:
        res['C4_重投影残差px'] = None
        checks.append({'组': 'C4', '项': '重投影残差(中位/最大)', '值': 'n/a(产物无残差数据)',
                       '阈值': f"≤{TH['C4_reproj_median_px']}/{TH['C4_reproj_max_px']}px",
                       'pass': False})

    # B2 跨视角对应错误率(用颜色 id 真值: 检测像素应落在真值点投影附近)
    if detections:
        cams = ring_of_cameras(N_VIEWS, CAM_RADIUS, CAM_HEIGHT, CAM_TARGET,
                               FX, FY, CX, CY, name_prefix="cam", up=(0, 0, -1))
        n_det, n_wrong = 0, 0
        for ti_str, det in detections.items():
            ti = int(ti_str)
            if ti not in truth_pts:
                continue
            tp = truth_pts[ti]
            for vi_str, uv in det.get('dets', {}).items():
                vi = int(vi_str)
                if vi < 0 or vi >= len(cams):
                    continue
                pr = cams[vi].project(tp)
                n_det += 1
                if np.linalg.norm(pr - np.asarray(uv, dtype=float)) > 8.0:
                    n_wrong += 1      # 偏离真值投影超过标记盘尺度 => 对应错误
        rate = (n_wrong / n_det) if n_det else None
        res['B2_对应错误率'] = rate
        checks.append({'组': 'B2', '项': '跨视角对应错误率',
                       '值': ('n/a' if rate is None else f"{rate:.1%} ({n_wrong}/{n_det})"),
                       '阈值': '0', 'pass': rate == 0.0})
    else:
        res['B2_对应错误率'] = None
        checks.append({'组': 'B2', '项': '跨视角对应错误率', '值': 'n/a(产物无检测数据)',
                       '阈值': '0', 'pass': False})
    return res, checks


def eval_curves(controls, model):
    """D1/D2/D3: 控制点/端点 vs 真值曲线。"""
    res, checks = {}, []
    devs, bbox_dists, ep_errs = [], [], []
    main_start_err = None
    for cid, rows in sorted(controls.items()):
        if cid >= len(model.curves):
            continue
        tc = model.curves[cid]
        tpoly = tc['points']                     # 真值曲线采样折线
        lo, hi = tpoly.min(axis=0), tpoly.max(axis=0)
        # D1: 拟合曲线(由控制点采样) vs 真值曲线 —— 双向最大偏差
        from bezier_fit import bernstein_basis
        ctrl = np.array([r[2] for r in rows])
        deg = len(ctrl) - 1
        t = np.linspace(0, 1, 120)
        samp = bernstein_basis(deg, t) @ ctrl
        d1 = max(point_to_polyline_dist(p, tpoly) for p in samp)
        # 真值折线每个顶点到拟合曲线的距离(用 120 采样近似)
        d2 = max(point_to_polyline_dist(p, samp) for p in tpoly)
        devs.append(max(d1, d2))
        # D2: 控制点到真值曲线包围盒距离
        bbox_dists.append(max(dist_to_aabb(p, lo, hi) for p in ctrl))
        # D3: 端点误差
        e0 = float(np.linalg.norm(ctrl[0] - tc['start']))
        e1 = float(np.linalg.norm(ctrl[-1] - tc['end']))
        ep_errs.append(max(e0, e1))
        if cid == 0:
            main_start_err = e0

    res['D1_曲线最大偏差'] = max(devs) if devs else None
    res['D2_控制点离包围盒最远'] = max(bbox_dists) if bbox_dists else None
    res['D3_端点最大误差'] = max(ep_errs) if ep_errs else None
    res['D3_主根起点误差'] = main_start_err

    checks.append({'组': 'D1', '项': '曲线最大偏差',
                   '值': ('n/a' if res['D1_曲线最大偏差'] is None
                          else f"{res['D1_曲线最大偏差']:.4f}"),
                   '阈值': f"≤{TH['D1_curve_max_dev']}",
                   'pass': res['D1_曲线最大偏差'] is not None and
                           res['D1_曲线最大偏差'] <= TH['D1_curve_max_dev']})
    checks.append({'组': 'D2', '项': '控制点离曲线包围盒最大距离',
                   '值': ('n/a' if res['D2_控制点离包围盒最远'] is None
                          else f"{res['D2_控制点离包围盒最远']:.4f}"),
                   '阈值': f"≤{TH['D2_ctrl_bbox_dist']}",
                   'pass': res['D2_控制点离包围盒最远'] is not None and
                           res['D2_控制点离包围盒最远'] <= TH['D2_ctrl_bbox_dist']})
    checks.append({'组': 'D3', '项': '端点最大误差(主根起点≈原点)',
                   '值': ('n/a' if res['D3_端点最大误差'] is None else
                          f"{res['D3_端点最大误差']:.4f} (主根起点 {main_start_err:.4f})"),
                   '阈值': f"≤{TH['D3_endpoint_err']}",
                   'pass': res['D3_端点最大误差'] is not None and
                           res['D3_端点最大误差'] <= TH['D3_endpoint_err'] and
                           main_start_err is not None and
                           main_start_err <= TH['D3_endpoint_err']})
    return res, checks


def read_root_table(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            rows.append(d)
    return rows


def eval_root_table(root_csv, root_sql, model, seed):
    """E1-E4 / F1: 重建产出的根表(14 列结构) 与 .sql。"""
    res, checks = {}, []
    rows = read_root_table(root_csv)
    res['根表行数'] = len(rows)
    HEADERS = ["序号", "级别", "编号1", "编号2", "编号3", "编号4", "类别",
               "x", "y", "z", "粗细", "成熟度", "上节点", "下节点"]

    if not rows:
        for grp, item in (('E1', '曲线条数与层级'), ('E2', '分支点误差'),
                          ('E3', '上/下节点构成有效树'), ('E4', 'CSV+SQL 生成且可导入'),
                          ('F1', '粗细由图像估计')):
            checks.append({'组': grp, '项': item, '值': 'n/a(根表未生成)',
                           '阈值': '-', 'pass': False})
        return res, checks

    # E1: 曲线条数与层级
    curve_levels = {}
    for r in rows:
        if r['类别'] in ('首节点', '采样点', '末节点') and r['编号1'] != '':
            curve_levels.setdefault(int(r['编号1']), set()).add(int(r['级别']))
    n_curves = len(curve_levels)
    lv_ok = all(len(s) == 1 for s in curve_levels.values())
    expect_depths = sorted(c['depth'] for c in model.curves)
    got_depths = sorted(next(iter(s)) for s in curve_levels.values())
    e1_ok = n_curves == len(model.curves) and lv_ok and got_depths == expect_depths
    checks.append({'组': 'E1', '项': f'曲线条数与层级(期望 {len(model.curves)} 条: '
                                    '主根1+一级3+二级6)',
                   '值': f"{n_curves} 条, 层级{got_depths}", '阈值': str(expect_depths),
                   'pass': e1_ok})
    res['E1_曲线数'] = n_curves
    res['E1_层级'] = got_depths

    # E2: 分支点位置误差(重建表中的 分支点 行 vs 模型 bifurcations)
    bp_rows = [r for r in rows if r['类别'] == '分支点']
    bps = model.bifurcations
    if bp_rows and bps:
        rec = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in bp_rows])
        tru = np.array([b['coord'] for b in bps])
        d = np.linalg.norm(rec[:, None, :] - tru[None, :, :], axis=2).min(axis=1)
        # 每个真值分支点都应被覆盖(用双向最近距离)
        d_t2r = np.linalg.norm(tru[:, None, :] - rec[None, :, :], axis=2).min(axis=1)
        e2 = max(d.max(), d_t2r.max())
    else:
        e2 = None
    res['E2_分支点最大误差'] = e2
    checks.append({'组': 'E2', '项': f'全部分支点({len(bps)}个)位置误差',
                   '值': ('n/a' if e2 is None else f"{e2:.4f}"),
                   '阈值': f"≤{TH['E2_branch_err']}",
                   'pass': e2 is not None and e2 <= TH['E2_branch_err']})

    # E3: 从"原点"行出发, 沿 上节点/下节点 构成的边(无向) 可遍历到全部曲线行。
    # 每行的 上节点/下节点 各指向一个关联行, 两者互为父子边(单列下节点
    # 无法列出全部子行, 故按无向树边遍历)。
    row_by_id = {int(r['序号']): r for r in rows if r['序号'] != ''}
    origin_ids = [int(r['序号']) for r in rows if r['类别'] == '原点']
    adj = {}
    for rid, r in row_by_id.items():
        for col in ('上节点', '下节点'):
            v = r[col]
            if v not in ('', None, 'None'):
                try:
                    vid = int(float(v))
                except ValueError:
                    continue
                if vid in row_by_id:
                    adj.setdefault(rid, set()).add(vid)
                    adj.setdefault(vid, set()).add(rid)
    visited = set()
    if origin_ids:
        stack = list(origin_ids)
        while stack:
            rid = stack.pop()
            if rid in visited:
                continue
            visited.add(rid)
            stack.extend(adj.get(rid, ()))
    # "到达全部曲线行"判定: 每条曲线的首节点行可从遍历集合经 上节点 链接到
    reach_ok = bool(origin_ids) and n_curves > 0
    missing_curves = []
    for r in rows:
        if r['类别'] == '首节点' and r['编号1'] != '':
            up = r['上节点']
            try:
                up_id = int(float(up)) if up not in ('', None) else None
            except ValueError:
                up_id = None
            if up_id is None or up_id not in visited:
                reach_ok = False
                missing_curves.append(r['编号1'])
    res['E3_可遍历'] = reach_ok
    checks.append({'组': 'E3', '项': "从'原点'出发可遍历到全部曲线行(无孤行/断链)",
                   '值': ('通过' if reach_ok else f"失败(未达曲线 {missing_curves[:5]})"),
                   '阈值': '通过', 'pass': reach_ok})

    # E4: SQL 存在且 sqlite3 实际导入验证
    sql_ok = False
    sql_msg = ''
    if os.path.isfile(root_sql):
        try:
            with open(root_sql, encoding='utf-8') as f:
                sql_text = f.read()
            con = sqlite3.connect(':memory:')
            con.executescript(sql_text)
            cur = con.execute(f"SELECT COUNT(*) FROM {os.path.splitext(os.path.basename(root_csv))[0]}")
            cnt = cur.fetchone()[0]
            con.close()
            sql_ok = (cnt == len(rows))
            sql_msg = f"导入 {cnt} 行"
        except Exception as e:
            sql_msg = f"导入失败: {e}"
    else:
        sql_msg = '.sql 不存在'
    checks.append({'组': 'E4', '项': 'CSV+SQL 同时生成, SQL 可实际导入',
                   '值': sql_msg, '阈值': f"与 CSV 行数一致({len(rows)})",
                   'pass': sql_ok})
    res['E4_sql'] = sql_msg

    # F1: 粗细由图像估计 -> 主根起点半径相对误差 + 单调渐变
    # 主根起点: 主根曲线(级别=1)的首节点行
    main_first = [r for r in rows if r['类别'] == '首节点' and r['级别'] == '1']
    if main_first:
        r0 = float(main_first[0]['粗细'])
        true_r0 = model.curves[0]['radii'][0]
        rel = abs(r0 - true_r0) / true_r0
        # 沿曲线无跳变: 同曲线相邻采样行粗细比有界
        jump = 0.0
        by_curve = {}
        for r in rows:
            if r['编号1'] != '' and r['类别'] in ('首节点', '采样点', '末节点'):
                by_curve.setdefault(int(r['编号1']), []).append(float(r['粗细']))
        for cid, rs in by_curve.items():
            rs_sorted = rs   # 行序即沿曲线序
            for a, b in zip(rs_sorted, rs_sorted[1:]):
                if a > 1e-6:
                    jump = max(jump, abs(b - a) / a)
        res['F1_主根起点粗细'] = r0
        res['F1_相对误差'] = rel
        res['F1_最大跳变率'] = jump
        f1_ok = rel <= TH['F1_radius_rel_err'] and jump <= 0.6
        checks.append({'组': 'F1', '项': '主根起点半径相对误差/沿曲线无跳变',
                       '值': f"相对误差 {rel:.1%}, 最大跳变 {jump:.1%}",
                       '阈值': f"≤{TH['F1_radius_rel_err']:.0%} / ≤60%",
                       'pass': f1_ok})
    else:
        res['F1_主根起点粗细'] = None
        checks.append({'组': 'F1', '项': '主根起点半径', '值': 'n/a(未找到主根首节点行)',
                       '阈值': '-', 'pass': False})
    return res, checks


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='重建产物一键评估(完成标准 B/C/D/E/F)')
    ap.add_argument('--points', default='reconstructed_points.csv')
    ap.add_argument('--controls', default='control_points.csv')
    ap.add_argument('--root-csv', default=None, help='重建产出的根表 CSV(缺省不评 E/F)')
    ap.add_argument('--root-sql', default=None)
    ap.add_argument('--detections', default=None,
                    help='管线保存的逐点检测/残差 JSON(缺省 C4/B2 记 n/a)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label', default='CLI')
    ap.add_argument('--outdir', default='evaluation')
    args = ap.parse_args()

    if args.root_csv is None:
        # 根表仅在显式提供时评估(项目根目录的 root_table.csv 是"模型→表"产物, 不是重建产物)
        args.root_csv = None
    if args.detections is None:
        cand = os.path.join(os.path.dirname(args.points) or '.', 'detections.json')
        if os.path.isfile(cand):
            args.detections = cand

    truth_pts, model = load_meta_truth(args.seed)
    points = read_points_csv(args.points)
    controls = read_control_csv(args.controls)
    detections = None
    if args.detections and os.path.isfile(args.detections):
        with open(args.detections, encoding='utf-8') as f:
            detections = json.load(f).get('points')

    os.makedirs(args.outdir, exist_ok=True)

    all_checks = []
    res_pts, ck = eval_points(points, truth_pts, model, detections)
    all_checks += ck
    res_ctrl, ck = eval_curves(controls, model)
    all_checks += ck
    res_root = {}
    if args.root_csv:
        res_root, ck = eval_root_table(args.root_csv, args.root_sql, model, args.seed)
        all_checks += ck
    else:
        for grp, item in (('E1', '曲线条数与层级'), ('E2', '分支点误差'),
                          ('E3', "上/下节点构成有效树"), ('E4', 'CSV+SQL 生成且可导入'),
                          ('F1', '粗细由图像估计')):
            all_checks.append({'组': grp, '项': item, '值': 'n/a(重建根表未提供/未生成)',
                               '阈值': '-', 'pass': False})

    n_pass = sum(1 for c in all_checks if c['pass'])
    summary = {
        'label': args.label, 'seed': args.seed,
        'points_csv': args.points, 'controls_csv': args.controls,
        'root_csv': args.root_csv,
        'metrics': {**res_pts, **res_ctrl, **res_root},
        'checks': all_checks, 'pass_count': n_pass, 'total': len(all_checks),
    }
    json_path = os.path.join(args.outdir, f"eval_{args.label}_seed{args.seed}.json")
    md_path = os.path.join(args.outdir, f"eval_{args.label}_seed{args.seed}.md")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    lines = [f"# 重建评估报告 — {args.label} (seed={args.seed})",
             "",
             f"- 点表: `{args.points}`",
             f"- 控制点表: `{args.controls}`",
             f"- 根表: `{args.root_csv}`",
             f"- 真值: dense_meta.json(seed=0) / RootModel(seed={args.seed})",
             "",
             "| 组 | 检查项 | 实测 | 阈值 | 结果 |",
             "|---|---|---|---|---|"]
    for c in all_checks:
        lines.append(f"| {c['组']} | {c['项']} | {c['值']} | {c['阈值']} | "
                     f"{'✅ pass' if c['pass'] else '❌ fail'} |")
    lines += ["", f"**通过 {n_pass}/{len(all_checks)}**"]
    md = "\n".join(lines)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(md)
    print(f"\n已写出: {json_path}\n已写出: {md_path}")
    return 0 if n_pass == len(all_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
