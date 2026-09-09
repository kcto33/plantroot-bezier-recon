# -*- coding: utf-8 -*-
"""
fit_and_rebuild.py
用"真正的最小二乘三次贝塞尔拟合"重建根系模型并重新填根表。

用途:
  - 给模型的每条曲线做最小二乘贝塞尔拟合, 得到精确的 4 控制点。
  - 用拟合出的控制点重建曲线, 与原始采样点对比精度。
  - 由拟合曲线沿程得到更合理的粗细(半径渐变)。
  - 把 拟合控制点 + 采样点 写入根表结构。
  - 绘制"由拟合贝塞尔重建"的模型图, 与原模型叠加对比。

注: 这条链是针对"有模型几何(=有采样点云)"的场景做最小二乘拟合。
    (当没有模型时, 用上面 detect_and_triangulate 得到的重建分支点也能拟合,
     只是点更稀疏、精度更低。)
"""
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
from bezier_fit import fit_model_curves, sample_cubic
from table_build import rows_to_csv, rows_to_sql, HEADERS, maturity_of


def build_fitted_table(fitted, params):
    """由拟合结果生成根表: 每条曲线 -> 首节点/控制点1/控制点2/末节点 + 采样点行。"""
    max_depth = params.maxDepth
    rows = []
    for fc in fitted:
        cid = fc['curve_id']
        depth = fc['depth']
        P0, P1, P2, P3 = fc['ctrl']
        spts = fc['points']
        srad = fc['radii']
        n = len(spts)

        # 采样点(含首末节点)
        sample_rows = []
        for i in range(n):
            cat = '首节点' if i == 0 else ('末节点' if i == n - 1 else '采样点')
            rows.append({
                '序号': len(rows), '级别': depth, '编号1': cid, '编号2': i,
                '编号3': None, '编号4': None, '类别': cat,
                'x': round(float(spts[i, 0]), 4),
                'y': round(float(spts[i, 1]), 4),
                'z': round(float(spts[i, 2]), 4),
                '粗细': round(float(srad[i]), 4),
                '成熟度': maturity_of(depth, max_depth),
                '上节点': None, '下节点': None,
            })
        # 控制点行(编号2 用 n/n+1 区分)
        for cap, tag, idx2 in ((P1, '控制点1', n), (P2, '控制点2', n + 1)):
            rows.append({
                '序号': len(rows), '级别': depth, '编号1': cid, '编号2': idx2,
                '编号3': None, '编号4': None, '类别': tag,
                'x': round(float(cap[0]), 4), 'y': round(float(cap[1]), 4),
                'z': round(float(cap[2]), 4),
                '粗细': round(float(srad[0]), 4),
                '成熟度': maturity_of(depth, max_depth),
                '上节点': None, '下节点': None,
            })
    return rows


def plot_fitted(fitted, model, out_path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 原始模型(灰)
    for c in model.curves:
        pts = c['points']
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='#bbbbbb', lw=1.0, alpha=0.5,
                label='原始采样点' if c is model.curves[0] else None)
    # 拟合贝塞尔(红)
    for fc in fitted:
        spts = fc['points']
        ax.plot(spts[:, 0], spts[:, 1], spts[:, 2], color='#e0433a', lw=1.6,
                label='拟合贝塞尔' if fc is fitted[0] else None)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("最小二乘贝塞尔拟合重建(红) vs 原始采样点(灰)")
    ax.view_init(elev=25, azim=-60)
    if len(fitted):
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    plt.close(fig)


def main():
    model = RootModel(RootParams(), seed=0)
    print("=" * 62)
    print("  最小二乘三次贝塞尔拟合 -> 重建模型 + 填根表")
    print("=" * 62)

    print("\n[1/4] 对每条曲线做最小二乘贝塞尔拟合 ...")
    fitted = fit_model_curves(model, n_out=50)
    errs_max = [f['err_max'] for f in fitted]
    errs_mean = [f['err_mean'] for f in fitted]
    print(f"  拟合 {len(fitted)} 条曲线: 最大误差最大={max(errs_max):.4f} "
          f"平均均值={np.mean(errs_mean):.4f}")

    print("\n[2/4] 用拟合控制点重建并填根表 ...")
    rows = build_fitted_table(fitted, model.params)
    rows_to_csv(rows, "root_fitted_table.csv")
    rows_to_sql(rows, table_name="root_fitted_table", path="root_fitted_table.sql")
    print(f"  已写出 root_fitted_table.csv ({len(rows)} 行) / root_fitted_table.sql")

    print("\n[3/4] 绘制拟合重建 vs 原始 ...")
    plot_fitted(fitted, model, "fitted_rebuilt_3d.png")
    print("  已输出 fitted_rebuilt_3d.png")

    print("\n[4/4] 精度汇总 ...")
    span = np.linalg.norm(model.bbox[1] - model.bbox[0])
    print(f"  拟合曲线最大几何误差: {max(errs_max):.4f} (相对模型尺寸 "
          f"{max(errs_max)/span:.4%})")
    print(f"  每条曲线拟合误差:")
    for f in fitted:
        print(f"    曲线{f['curve_id']} (depth={f['depth']}): "
              f"max={f['err_max']:.4f} mean={f['err_mean']:.4f}")
    print("-" * 62)


if __name__ == "__main__":
    main()
