# -*- coding: utf-8 -*-
"""
branch_table_to_bezier.py
从图片重建出的分支点表(root_branch_table.csv)做贝塞尔拟合重建模型。

说明:
  - 图片三角化只能重建出离散的分支点(分叉点), 表里没有曲线上的连续采样点。
  - 表中已含"原点/分支点"节点行与 上节点 链接(见 detect_and_triangulate),
    这里按树结构(而非早期"x 正负分组 + z 排序"启发式)把分支点串成
    骨架链, 对每一簇做最小二乘三次贝塞尔拟合, 绘制重建的根系骨架。
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


def read_table(csv_path):
    """读分支点表 -> (rows, row_by_id)。"""
    rows = []
    with open(csv_path, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            rows.append(d)
    return rows, {int(r['序号']): r for r in rows if r['序号'] != ''}


def xyz(r):
    return np.array([float(r['x']), float(r['y']), float(r['z'])])


def tree_chains(rows, row_by_id):
    """按 上节点 链接把分支点组织成骨架链: 从 原点 的每个一级分支起,
    沿"以本行为父"的子分支递归。返回若干条点链(每条 >=2 点)。"""
    kids = {}
    for r in rows:
        if r['类别'] == '分支点' and r['上节点'] not in ('', None, 'None'):
            pid = int(float(r['上节点']))
            kids.setdefault(pid, []).append(int(r['序号']))
    chains = []
    def walk(rid, chain):
        kids_here = sorted(kids.get(rid, []))
        if not kids_here:
            chains.append(chain)
            return
        for k, cid in enumerate(kids_here):
            ext = chain + [cid] if k == 0 else [row_by_id[chain[-1]] and cid]
            # 分叉处: 主链延续, 其余子链以"父链 + 该子分支"成新链
            if k == 0:
                walk(cid, chain + [cid])
            else:
                walk(cid, [cid])
    origin = [int(r['序号']) for r in rows if r['类别'] == '原点']
    for o in origin:
        for cid in sorted(kids.get(o, [])):
            walk(cid, [cid])
    if not chains:   # 兜底: 无链接信息时退化为按序单链
        chains = [[int(r['序号']) for r in rows if r['类别'] == '分支点']]
    return [[xyz(row_by_id[rid]) for rid in ch] for ch in chains
            if len(ch) >= 1]


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


def main():
    rows, row_by_id = read_table("root_branch_table.csv")
    bps = [r for r in rows if r['类别'] == '分支点']
    print(f"读入 {len(bps)} 个重建分支点:")
    for i, r in enumerate(bps):
        print(f"  {i+1}: {np.round(xyz(r),3)} (级别 {r['级别']})")
    print("引用说明: 图片三角化仅得离散分支点, 此处按树结构骨架做贝塞尔拟合。")

    chains = tree_chains(rows, row_by_id)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#e0433a', '#1f8fe0', '#2ca02c', '#ff7f0e', '#9467bd',
              '#d62728', '#17becf', '#bcbd22', '#7f7f7f', '#e377c2']
    for ci, cl in enumerate(chains):
        cl = np.array(cl)
        ctrl = fit_cubic(cl)
        s = sample(*ctrl)
        ax.plot(s[:, 0], s[:, 1], s[:, 2], color=colors[ci % len(colors)], lw=2.2,
                label=f"重建贝塞尔骨架{ci+1} ({len(cl)}点)")
        ax.scatter(cl[:, 0], cl[:, 1], cl[:, 2], color='black', s=40, zorder=5)

    # 地表原点参考
    ax.scatter(0, 0, 0, color='green', s=70, marker='*', label='地表原点')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("由图片重建分支点 -> 按树结构骨架贝塞尔拟合")
    ax.view_init(elev=22, azim=-58)
    ax.legend()
    fig.tight_layout()
    fig.savefig("branch_reconstructed_model.png", dpi=115)
    plt.close(fig)
    print("已输出 branch_reconstructed_model.png")


if __name__ == "__main__":
    main()
