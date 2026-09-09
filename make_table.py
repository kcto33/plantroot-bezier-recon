# -*- coding: utf-8 -*-
"""
make_table.py
入口: 生成数据表(CSV + SQL) -> 从表反推绘图 -> 验证贝塞尔重建精度。
"""
import numpy as np
from root_model import RootModel, RootParams
from table_build import build_table, rows_to_csv, rows_to_sql, print_table
from table_render import (read_rows_csv, plot_model, verify_against_original,
                          rebuild_beziers)


def main():
    model = RootModel(RootParams(), seed=0)
    print("=" * 62)
    print("  从贝塞尔根系模型生成数据表 -> 由表反推绘制原模型")
    print("=" * 62)

    print("\n[1/4] 生成三维模型 ...")
    print(f"  曲线数={len(model.curves)} 采样点总数={sum(c['points'].shape[0] for c in model.curves)}")

    print("\n[2/4] 生成数据表(每行=一个采样点) ...")
    rows, curve_rows = build_table(model)
    rows_to_csv(rows, "root_table.csv")
    rows_to_sql(rows, table_name="root_table", path="root_table.sql")
    print(f"  已写出 root_table.csv ({len(rows)} 行) 和 root_table.sql")
    print("\n  前 12 行:")
    print_table(rows, max_rows=12)

    print("\n[3/4] 由表同时用 采样点链表 和 贝塞尔控制点 重建并绘图 ...")
    plot_model(rows, "model_rebuilt_3d.png",
               title="由数据表重建(采样点链) + 贝塞尔控制点重建(红色虚线/点)",
               projection='3d', show_controls=True)
    plot_model(rows, "model_rebuilt_xy.png",
               title="由数据表重建 -> 俯视 XY 投影",
               projection='2d', show_controls=False)
    print("  已写出 model_rebuilt_3d.png / model_rebuilt_xy.png")

    print("\n[4/4] 验证: 用表里控制点重建的曲线 vs 原模型采样点 ...")
    bz = rebuild_beziers(rows)
    print(f"  成功拆出 {len(bz)} 条曲线的完整4控制点 (共{len(model.curves)}条)")
    results, _ = verify_against_original(rows, model)
    if results:
        errs = np.array([r['max_err'] for r in results])
        means = np.array([r['mean_err'] for r in results])
        print(f"  每条曲线最大偏差(模型单位): 最大={errs.max():.4f} 平均均值={means.mean():.4f}")
        print(f"  模型整体尺寸 = {np.linalg.norm(model.bbox[1]-model.bbox[0]):.2f}, "
              f"相对误差 = {errs.max()/np.linalg.norm(model.bbox[1]-model.bbox[0]):.4%}")
    print("-" * 62)


if __name__ == "__main__":
    main()
