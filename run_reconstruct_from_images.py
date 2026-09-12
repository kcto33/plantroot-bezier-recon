# -*- coding: utf-8 -*-
"""
run_reconstruct_from_images.py
一键复现"仅凭图片 + 已知相机 -> 自动检测分支点 -> 三角化 -> 填表"。

流程:
  阶段1 generate_marked_views.py : 生成带彩色分支点标记的 8 视角图 (views_branch/)
  阶段2 detect_and_triangulate.py: 仅凭图片像素 + 已知相机, 自动检测/配对/三角化/填表
  阶段3 branch_table_to_bezier   : 按树结构把分支点串成骨架链,
                                   每条链做最小二乘三次贝塞尔拟合并绘图

另: 完整的 120 点全链路重建(含 14 列根表)见 build_table_from_images.py。

用法:
  python run_reconstruct_from_images.py
"""
import subprocess
import sys
import os


def run_stage(script):
    print(f"\n{'#'*62}\n# 阶段: {script}\n{'#'*62}")
    r = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        print(f"阶段 {script} 失败, 退出。")
        sys.exit(r.returncode)


def main():
    print("=" * 62)
    print("  仅凭图片 + 已知相机 -> 自动检测分支点 -> 三角还原填表")
    print("=" * 62)
    print("说明: 阶段1需模型几何生成带色标记图; 阶段2只用图片像素+相机三角化。")
    run_stage("generate_marked_views.py")
    run_stage("detect_and_triangulate.py")
    run_stage("branch_table_to_bezier.py")   # 阶段3: 树链骨架 + 最小二乘贝塞尔拟合
    print("\n全部完成。")


if __name__ == "__main__":
    main()
