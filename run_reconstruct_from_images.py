# -*- coding: utf-8 -*-
"""
run_reconstruct_from_images.py
一键复现"仅凭图片 + 已知相机 -> 自动检测分支点 -> 三角化 -> 填表"。

流程:
  阶段1 generate_marked_views.py : 生成带彩色分支点标记的 8 视角图 (views_branch/)
  阶段2 detect_and_triangulate.py: 仅凭图片像素 + 已知相机, 自动检测/配对/三角化/填表
  (可选) 阶段3: 用重建出的分支点 + 贝塞尔拟合, 由点反推曲线并绘图

用法:
  python run_reconstruct_from_images.py
"""
import subprocess
import sys
import os
import numpy as np

from table_build import rows_to_csv
from table_build import HEADERS


def run_stage(script):
    print(f"\n{'#'*62}\n# 阶段: {script}\n{'#'*62}")
    r = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__))
    if r.returncode != 0:
        print(f"阶段 {script} 失败, 退出。")
        sys.exit(r.returncode)


def fit_bezier_with_points(branch_csv="root_branch_table.csv"):
    """用一个简单的贝塞尔拟合示例: 把重建出的分支点串成贝塞尔控制点, 绘制重建分支。
    这里只演示"从点反推曲线"的思路, 不替代真实拟合。"""
    import csv
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

    pts = []
    with open(branch_csv, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            pts.append([float(d['x']), float(d['y']), float(d['z'])])
    if len(pts) < 2:
        print("分支点不足, 跳过拟合绘图。")
        return
    pts = np.array(pts)
    print(f"\n[阶段3] 用 {len(pts)} 个重建分支点绘图 (示意: 按顺序连接)")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o-', color='#5a3719', lw=1.5)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='red', s=25)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("由图片重建的分支点 (示意连接)")
    fig.savefig("rebuilt_branchpoints_3d.png", dpi=110)
    plt.close(fig)
    print("  已输出 rebuilt_branchpoints_3d.png")


def main():
    print("=" * 62)
    print("  仅凭图片 + 已知相机 -> 自动检测分支点 -> 三角还原填表")
    print("=" * 62)
    print("说明: 阶段1需模型几何生成带色标记图; 阶段2只用图片像素+相机三角化。")
    run_stage("generate_marked_views.py")
    run_stage("detect_and_triangulate.py")
    fit_bezier_with_points()
    print("\n全部完成。")


if __name__ == "__main__":
    main()
