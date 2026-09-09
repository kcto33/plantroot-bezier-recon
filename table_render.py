# -*- coding: utf-8 -*-
"""
table_render.py
从"每行=一个采样点"的表反推并绘制原根系模型。

两种绘制方式:
  1. 按链表连接采样点 (用 下节点/编号4 把点串成折线) —— 忠实还原, 无需贝塞尔。
  2. 拆出每条曲线的 首节点/控制点1/控制点2/末节点, 用三次贝塞尔重建, 再采样绘制。
     同时验证: 用表里的控制点重建的曲线, 是否与原模型采样点重合。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 使用系统中文字体, 避免中文标题/标签乱码
for _f in ('Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC'):
    try:
        matplotlib.rcParams['font.sans-serif'] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

from table_build import HEADERS


def read_rows_csv(path):
    import csv
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            for k in d:
                if k not in ('类别',):
                    d[k] = None if d[k] in ('', 'NULL', 'None') else float(d[k])
            d['类别'] = d['类别']
            rows.append(d)
    return rows


def extract_curves(rows):
    """按编号1(曲线id)分组采样点行, 并按编号2排序。返回 {curve_id: 有序行列表}。
    只包含沿曲线的采样点(首节点/末节点/采样点), 跳过 控制点1/控制点2 行,
    以及节点行(编号1 为 None 的 分支点/末端点/原点)。"""
    groups = {}
    for r in rows:
        if r['编号1'] is None:
            continue
        if r['类别'] in ('控制点1', '控制点2'):
            continue
        cid = int(r['编号1'])
        groups.setdefault(cid, []).append(r)
    for cid in groups:
        groups[cid].sort(key=lambda r: r['编号2'])
    return groups


def rebuild_beziers(rows):
    """从表里拆出每条曲线的 4 个控制点(首节点/控制点1/控制点2/末节点)。
    返回 {curve_id: (P0,P1,P2,P3)}。跳过节点行(编号1 为 None)。"""
    curves = {}
    for r in rows:
        if r['编号1'] is None:
            continue
        cid = int(r['编号1'])
        cat = r['类别']
        p = np.array([r['x'], r['y'], r['z']])
        if cat in ('首节点', '末节点'):
            if cat == '首节点':
                curves.setdefault(cid, [None, None, None, None])[0] = p
            else:
                curves.setdefault(cid, [None, None, None, None])[3] = p
        elif cat == '控制点1':
            curves.setdefault(cid, [None, None, None, None])[1] = p
        elif cat == '控制点2':
            curves.setdefault(cid, [None, None, None, None])[2] = p
    # 过滤缺失
    return {k: v for k, v in curves.items() if all(x is not None for x in v)}


def sample_cubic(P0, P1, P2, P3, n=50):
    t = np.linspace(0, 1, n)[:, None]
    c0 = (1 - t) ** 3
    c1 = 3 * (1 - t) ** 2 * t
    c2 = 3 * (1 - t) * t ** 2
    c3 = t ** 3
    return c0 * P0 + c1 * P1 + c2 * P2 + c3 * P3


def plot_model(rows, out_path, title="由数据表重建的根系模型", projection='3d',
               show_controls=False):
    groups = extract_curves(rows)
    fig = plt.figure(figsize=(9, 8))
    if projection == '3d':
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    # 方式1: 按链表(下节点)连接采样点
    for cid in groups:
        pts = np.array([[r['x'], r['y'], r['z']] for r in groups[cid]])
        if projection == '3d':
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color='#5a3719', lw=1.2, alpha=0.9)
        else:
            ax.plot(pts[:, 0], pts[:, 1],
                    color='#5a3719', lw=1.2, alpha=0.9)

    # 叠加方式2: 贝塞尔控制点重建, 用不同颜色验证是否重合
    if show_controls:
        bz = rebuild_beziers(rows)
        for cid, (P0, P1, P2, P3) in bz.items():
            s = sample_cubic(P0, P1, P2, P3, 60)
            if projection == '3d':
                ax.plot(s[:, 0], s[:, 1], s[:, 2], '--', color='#e04a4a',
                        lw=1.0, alpha=0.7)
            else:
                ax.plot(s[:, 0], s[:, 1], '--', color='#e04a4a',
                        lw=1.0, alpha=0.7)

    # 目标点标记(表内 上下节点 使用的行/首节点)
    # 画各曲线控制点
    if show_controls:
        bz = rebuild_beziers(rows)
        for cid, (P0, P1, P2, P3) in bz.items():
            for p, c in [(P0, 'red'), (P1, 'blue'), (P2, 'blue'), (P3, 'red')]:
                if projection == '3d':
                    ax.scatter(*p, color=c, s=8, alpha=0.6)
                else:
                    ax.scatter(p[0], p[1], color=c, s=8, alpha=0.6)

    if projection == '3d':
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=-60)
    else:
        ax.set_aspect('equal')
        ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return groups


def verify_against_original(rows, model):
    """对比: 由表重建的曲线与原始模型采样点是否一致性(控制点重建 vs 原模型点)。
    返回每条曲线最大/平均偏差。"""
    bz = rebuild_beziers(rows)
    curves = model.curves
    results = []
    for cid, (P0, P1, P2, P3) in bz.items():
        if cid >= len(curves):
            continue
        orig = curves[cid]['points']
        rebuilt = sample_cubic(P0, P1, P2, P3, n=len(orig))
        # 对每个重建点找最近原模型点距离
        dist = np.linalg.norm(rebuilt[:, None, :] - orig[None, :, :], axis=2).min(axis=1)
        results.append({
            'curve': cid, 'max_err': float(dist.max()),
            'mean_err': float(dist.mean()),
            'n': len(orig),
        })
    return results, bz
