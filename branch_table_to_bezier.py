# -*- coding: utf-8 -*-
"""
branch_table_to_bezier.py
从图片重建出的分支点表(root_branch_table.csv)做贝塞尔拟合重建模型。

说明:
  - 图片三角化只能重建出离散的分支点(分叉点), 表里没有曲线上的连续采样点。
  - 因此这里把这些重建点作为"骨架关键点", 按空间拓扑(左右侧 + 深度)组织,
    对每一簇做最小二乘三次贝塞尔拟合, 绘制重建的根系贝塞尔曲线。
"""
import csv
import numpy as np
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


def read_points(csv_path):
    pts = []
    with open(csv_path, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            pts.append(np.array([float(d['x']), float(d['y']), float(d['z'])]))
    return np.array(pts)


def chord_param(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    c = np.concatenate([[0.0], np.cumsum(d)])
    return c / max(c[-1], 1e-9)


def fit_cubic(P):
    """对一簇点做最小二乘三次贝塞尔拟合(端点固定)。返回 (P0,P1,P2,P3)。"""
    P = np.atleast_2d(P)
    if len(P) < 3:
        # 点太少: 直接返回直线两端点 + 中点
        if len(P) == 1:
            return P[0], P[0], P[0], P[0]
        return P[0], P[0], P[-1], P[-1]
    t = chord_param(P)
    b0, b1, b2, b3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
    P0, P3 = P[0], P[-1]
    A = np.stack([b1, b2], axis=1)
    sol = np.empty((3, 2))
    for d in range(3):
        R = P[:, d] - b0*P0[d] - b3*P3[d]
        sol[d], *_ = np.linalg.lstsq(A, R, rcond=None)
    return P0, sol[:, 0], sol[:, 1], P3


def sample(P0, P1, P2, P3, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return ((1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3)


def cluster_skeleton(pts):
    """按 x 正负分左右, 每组内按 z(深度, 越深越靠后) 排序, 得到骨架簇。
    返回 聚类后的点列表。"""
    left = pts[pts[:, 0] < 0]
    right = pts[pts[:, 0] >= 0]
    left = left[np.argsort(left[:, 2])]   # z 从小到大(浅->深)
    right = right[np.argsort(right[:, 2])]
    out = []
    if len(left):
        out.append(left)
    if len(right):
        out.append(right)
    return out


def main():
    pts = read_points("root_branch_table.csv")
    print(f"读入 {len(pts)} 个重建分支点:")
    for i, p in enumerate(pts):
        print(f"  {i+1}: {np.round(p,3)}")
    print("引用说明: 图片三角化仅得离散分支点, 此处按空间骨架做贝塞尔拟合。")

    clusters = cluster_skeleton(pts)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#e0433a', '#1f8fe0']
    for ci, cl in enumerate(clusters):
        ctrl = fit_cubic(cl)
        s = sample(*ctrl)
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=colors[ci % 2], lw=2.2,
                label=f"重建贝塞尔骨架{ci+1} ({len(cl)}点)")
        ax.scatter(cl[:, 0], cl[:, 1], cl[:, 2], color='black', s=40, zorder=5)

    # 地表原点参考
    ax.scatter(0, 0, 0, color='green', s=70, marker='*', label='地表原点')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("由图片重建分支点 -> 贝塞尔拟合重建模型")
    ax.view_init(elev=22, azim=-58)
    ax.legend()
    fig.tight_layout()
    fig.savefig("branch_reconstructed_model.png", dpi=115)
    plt.close(fig)
    print("已输出 branch_reconstructed_model.png")


if __name__ == "__main__":
    main()
