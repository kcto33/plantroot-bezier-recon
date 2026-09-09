# -*- coding: utf-8 -*-
"""
reconstruct_from_table.py
独立脚本: 只凭数据表(如 root_table.csv)还原并绘制原贝塞尔根系模型。
不依赖 root_model / camera / render / make_table 等模型生成代码。

裸依赖: numpy, matplotlib, csv。

用法:
    python reconstruct_from_table.py [表CSV路径] [--outdir 输出目录] [--3d] [--xy] [--title 标题]

绘制方式:
  1) 按链表: 用 编号4(下节点) 把同一 编号1(曲线) 的采样点串成折线。
  2) 按贝塞尔: 从 类别=首节点/控制点1/控制点2/末节点 拆出该曲线 4 控制点,
     用三次贝塞尔公式重新采样(红色虚线)覆盖, 验证是否重合。
"""
import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 系统中文字体, 避免中文乱码
for _f in ('Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC'):
    try:
        matplotlib.rcParams['font.sans-serif'] = [_f]
        break
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False


# ----------------------------------------------------------------------
# 读表
# ----------------------------------------------------------------------
def read_rows(path):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            for k in d:
                if k != '类别':
                    d[k] = None if d[k] in ('', 'NULL', 'None') else float(d[k])
            rows.append(d)
    return rows


# ----------------------------------------------------------------------
# 由表还原
# ----------------------------------------------------------------------
def extract_curves(rows):
    """按编号1分组采样点, 按编号2排序。跳过 控制点1/控制点2 行。返回 {cid: [行,...]}。"""
    groups = {}
    for r in rows:
        if r['类别'] in ('控制点1', '控制点2'):
            continue
        if r['编号1'] is None:
            continue   # 节点行(分支点/末端点/原点)
        groups.setdefault(int(r['编号1']), []).append(r)
    for cid in groups:
        groups[cid].sort(key=lambda r: r['编号2'])
    return groups


def rebuild_control_points(rows):
    """从表拆每条曲线 4 控制点。返回 {cid: (P0,P1,P2,P3)}。跳过节点行。"""
    curves = {}
    for r in rows:
        if r['编号1'] is None:
            continue
        cid = int(r['编号1'])
        cat = r['类别']
        p = np.array([r['x'], r['y'], r['z']])
        slot = {'首节点': 0, '控制点1': 1, '控制点2': 2, '末节点': 3}.get(cat)
        if slot is not None:
            curves.setdefault(cid, [None, None, None, None])[slot] = p
    return {k: v for k, v in curves.items() if all(x is not None for x in v)}


def sample_cubic(P0, P1, P2, P3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1
            + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3)


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def plot(rows, out_path, projection='3d', title="由数据表重建", show_controls=True):
    groups = extract_curves(rows)
    bz = rebuild_control_points(rows) if show_controls else {}

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d') if projection == '3d' else fig.add_subplot(111)

    pts_all = np.array([[r['x'], r['y'], r['z']] for r in rows
                        if r['类别'] not in ('控制点1', '控制点2')])
    mins, maxs = pts_all.min(0), pts_all.max(0)
    span = maxs - mins

    for cid, grp in groups.items():
        pts = np.array([[r['x'], r['y'], r['z']] for r in grp])
        r_g = np.array([r['粗细'] for r in grp])
        # 用粗细作为线宽(按模型尺寸归一化)
        lw = max(0.8, 8.0 * r_g.mean() / max(span.min(), 1e-6))
        if projection == '3d':
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='#5a3719',
                    lw=lw, alpha=0.9)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color='#5a3719', lw=lw, alpha=0.9)

    for cid, (P0, P1, P2, P3) in bz.items():
        s = sample_cubic(P0, P1, P2, P3)
        if projection == '3d':
            ax.plot(s[:, 0], s[:, 1], s[:, 2], '--', color='#e04a4a',
                    lw=1.0, alpha=0.7)
        else:
            ax.plot(s[:, 0], s[:, 1], '--', color='#e04a4a', lw=1.0, alpha=0.7)
        for p, c in [(P0, 'red'), (P1, 'blue'), (P2, 'blue'), (P3, 'red')]:
            if projection == '3d':
                ax.scatter(*p, color=c, s=8, alpha=0.6)
            else:
                ax.scatter(p[0], p[1], color=c, s=8, alpha=0.6)

    if projection == '3d':
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=-60)
    else:
        ax.set_aspect('equal'); ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def summarize(rows):
    groups = extract_curves(rows)
    bz = rebuild_control_points(rows)
    pts_all = np.array([[r['x'], r['y'], r['z']] for r in rows
                        if r['类别'] not in ('控制点1', '控制点2')])
    print("-" * 58)
    print("  由数据表还原统计")
    print("-" * 58)
    print(f"  表总行数        : {len(rows)}")
    print(f"  曲线(分支段)数  : {len(groups)}")
    n_sample = sum(len(g) for g in groups.values())
    print(f"  采样点数        : {n_sample}")
    print(f"  控制点行数      : {len(rows) - n_sample}")
    print(f"  完整4控制点的曲线: {len(bz)}")
    if len(bz) and len(pts_all):
        print(f"  模型XYZ范围 x:[{pts_all[:,0].min():.2f},{pts_all[:,0].max():.2f}] "
              f"y:[{pts_all[:,1].min():.2f},{pts_all[:,1].max():.2f}] "
              f"z:[{pts_all[:,2].min():.2f},{pts_all[:,2].max():.2f}]")
        # 贝塞尔重建 vs 采样点链偏差(min distance)
        errs = []
        for cid, (P0, P1, P2, P3) in bz.items():
            grp = groups.get(cid)
            if not grp:
                continue
            orig = np.array([[r['x'], r['y'], r['z']] for r in grp])
            rebuilt = sample_cubic(P0, P1, P2, P3, n=len(orig))
            d = np.linalg.norm(rebuilt[:, None, :] - orig[None, :, :], axis=2).min(1)
            errs.append(float(d.max()))
        if errs:
            span = np.linalg.norm(pts_all.max(0) - pts_all.min(0))
            print(f"  贝塞尔重建最大偏差: {max(errs):.4f} "
                  f"(相对模型尺寸 {max(errs)/span:.4%})")
    print("-" * 58)


def main():
    ap = argparse.ArgumentParser(description="只凭数据表还原并绘制贝塞尔根系模型")
    ap.add_argument("csv", nargs="?", default="root_table.csv",
                    help="表CSV路径 (默认 root_table.csv)")
    ap.add_argument("--outdir", default=".", help="输出目录")
    ap.add_argument("--title", default="由数据表重建的根系模型")
    ap.add_argument("--no-3d", action="store_true", help="不画3D图")
    ap.add_argument("--no-xy", action="store_true", help="不画俯视XY图")
    ap.add_argument("--no-controls", action="store_true", help="不只画贝塞尔控制点重建")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"找不到表文件: {args.csv}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    rows = read_rows(args.csv)
    print(f"已读取表 {args.csv}: {len(rows)} 行")

    if not args.no_3d:
        out = os.path.join(args.outdir, "rebuilt_3d.png")
        plot(rows, out, '3d', args.title, show_controls=not args.no_controls)
        print(f"  已输出 {out}")
    if not args.no_xy:
        out = os.path.join(args.outdir, "rebuilt_xy.png")
        plot(rows, out, '2d', args.title + " -> 俯视XY", show_controls=not args.no_controls)
        print(f"  已输出 {out}")

    summarize(rows)


if __name__ == "__main__":
    main()
