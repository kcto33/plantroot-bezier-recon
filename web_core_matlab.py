# -*- coding: utf-8 -*-
"""
web_core_matlab.py
入口①: 从 matlab 文件(plantroot06.m 结构) 提取参数 -> 生成模型 -> 根表 + 重建图。
"""
import os
import re
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

from root_model import RootModel, RootParams
from table_build import build_table, rows_to_csv, rows_to_sql
from bezier_fit import fit_model_curves, sample_cubic


def parse_matlab_params(text):
    """从 plantroot06.m 之类脚本的文本里提取 params 数值。
    支持: params.mainLength = 12; params.branchNum = [3,2,8]; 等。
    返回 RootParams 实例(填入能解析到的字段)。
    """
    p = RootParams()
    mapping = {
        'maxDepth': 'maxDepth', 'mainLength': 'mainLength', 'mainRadius': 'mainRadius',
        'branchScale': 'branchScale', 'radiusScale': 'radiusScale',
        'curveStrength': 'curveStrength', 'pointsPerCurve': 'pointsPerCurve',
        'gravityBend': 'gravityBend', 'taperRatio': 'taperRatio',
    }
    # 处理 branchNum 数组 [a,b,c] 与单个数值
    m = re.search(r'params\.branchNum\s*=\s*\[([^\]]+)\]', text)
    if m:
        arr = [int(x) for x in re.findall(r'\d+', m.group(1))]
        if arr:
            p.branchNum = arr
    # angleRange [a,b]
    m = re.search(r'params\.angleRange\s*=\s*\[([^\]]+)\]', text)
    if m:
        arr = [float(x) for x in re.findall(r'\d+\.?\d*', m.group(1))]
        if len(arr) >= 2:
            p.angleRange = arr[:2]
    for key, attr in mapping.items():
        m = re.search(r'params\.' + key + r'\s*=\s*([-+0-9.]+)', text)
        if m:
            val = float(m.group(1))
            # 整数字段转 int(linspace 等需要 int)
            if key in ('pointsPerCurve', 'maxDepth'):
                val = int(val)
            setattr(p, attr, val)
    return p


def run_matlab_pipeline(m_file_path, outdir, progress_cb=None):
    """matlab -> 模型 -> 根表 + 重建图。返回结果文件路径 dict。
    progress_cb(stage, done, total): 粗粒度阶段进度。"""
    def _cb(stage, done, total):
        if progress_cb:
            try:
                progress_cb(stage, done, total)
            except Exception:
                pass

    os.makedirs(outdir, exist_ok=True)
    _cb("解析参数与生成模型", 1, 4)
    with open(m_file_path, encoding='utf-8', errors='ignore') as f:
        text = f.read()
    params = parse_matlab_params(text)
    model = RootModel(params, seed=0)

    _cb("生成根表", 2, 4)
    # 根表
    rows, curve_rows = build_table(model)
    csv_path = os.path.join(outdir, "root_table.csv")
    sql_path = os.path.join(outdir, "root_table.sql")
    rows_to_csv(rows, csv_path)
    rows_to_sql(rows, table_name="root_table", path=sql_path)

    _cb("渲染重建图", 3, 4)
    # 重建图: 拟合贝塞尔 vs 原始
    fitted = fit_model_curves(model, n_out=60)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    for c in model.curves:
        pts = c['points']
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='#bbbbbb', lw=1.0, alpha=0.5)
    for fc in fitted:
        spts = fc['points']
        ax.plot(spts[:, 0], spts[:, 1], spts[:, 2], color='#e0433a', lw=1.6)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"matlab→模型→根表→贝塞尔重建 ({len(model.curves)}根)")
    ax.view_init(elev=38, azim=-78)
    fig.tight_layout()
    img_path = os.path.join(outdir, "model_rebuilt_3d.png")
    fig.savefig(img_path, dpi=115)
    plt.close(fig)

    # 交互查看器数据
    from render import export_model_json
    export_model_json(fitted, os.path.join(outdir, "model3d.json"))
    _cb("完成", 4, 4)

    return {
        'curves': len(model.curves), 'nodes': len(model.nodes),
        'table_csv': csv_path, 'table_sql': sql_path, 'image': img_path,
        'model3d_json': os.path.join(outdir, "model3d.json"),
        'params': {'mainLength': params.mainLength, 'maxDepth': int(params.maxDepth),
                   'branchNum': params.branchNum},
    }


if __name__ == "__main__":
    r = run_matlab_pipeline("plantroot06.m", "out_matlab_demo")
    print("完成:", r)
