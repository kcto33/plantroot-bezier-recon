# -*- coding: utf-8 -*-
"""
root_topology.py
从重建结果(每条曲线的测点/拟合控制点/图像反演半径)重建
"主根 -> 分支点 -> 侧根" 树(P2-9), 并输出与 table_build.py 相同
14 列结构的完整根表(CSV + SQL)。

树结构推导(不依赖模型几何, 仅用重建结果 + 曲线层级):
  - 主根 = 层级 1 的曲线(唯一);
  - 每条层级 d>1 的曲线(侧根)的起点即一个分支点, 其父曲线 = 层级 d-1
    的曲线中采样折线离该起点最近者;
  - 多条子曲线共用同一分支节点(生成端允许在同一处分支)时, 按
    BRANCH_MERGE_TOL 贪心聚类为一个"分支点"行(与 root_model 的
    分叉点去重阈值一致)。

表结构(14 列, 语义同 table_build.HEADERS):
  节点行: 原点 / 分支点 / 末端点
  曲线行: 首节点 / 采样点... / 末节点 / 控制点1..4
  链接:   原点.下节点 -> 主根首节点;  分支点.下节点 -> 子曲线首节点;
          曲线首节点.上节点 -> 起点节点行;  分支点.上节点 -> 父曲线首节点行;
          曲线末节点.下节点 -> 末端点行 —— 从"原点"出发可遍历全部曲线行。
"""
import numpy as np

from table_build import HEADERS, maturity_of, rows_to_csv, rows_to_sql

BRANCH_MERGE_TOL = 0.15   # 与生成端 root_model.bifurcations 的去重阈值一致


def curve_depths_geometric(curves, depths_hint=None):
    """由几何推导各曲线层级: 主根 = 起点离原点最近的曲线;
    其余曲线的层级 = 其父曲线(采样折线离它起点最近的已定级曲线)层级 + 1。
    depths_hint 提供时直接校验/使用(模块①经 pose.json 的 mark_meta 传入)。"""
    if depths_hint is not None:
        return dict(depths_hint)
    info = []
    for cid, cd in curves.items():
        pts = cd['points']
        info.append((cid, pts[0], pts))
    main = min(info, key=lambda x: float(np.linalg.norm(x[1]))) [0]
    depths = {main: 1}
    remaining = [x[0] for x in info if x[0] != main]
    while remaining:
        progressed = False
        for cid in list(remaining):
            start = curves[cid]['points'][0]
            best, best_d = None, np.inf
            for pid, dd in depths.items():
                poly = curves[pid]['points']
                d = np.linalg.norm(poly - start[None, :], axis=1).min()
                if d < best_d:
                    best, best_d = pid, d
            if best is not None:
                depths[cid] = depths[best] + 1
                remaining.remove(cid)
                progressed = True
        if not progressed:
            for cid in remaining:
                depths[cid] = 2
            break
    return depths


def assign_parents(curves, depths):
    """每条侧根的父曲线 = 层级-1 的曲线中采样折线离其起点最近者。
    返回 {cid: parent_cid}。"""
    parents = {}
    for cid, cd in curves.items():
        d = depths[cid]
        if d <= 1:
            continue
        start = cd['points'][0]
        best, best_d = None, np.inf
        for pid, pd in depths.items():
            if pd != d - 1:
                continue
            poly = curves[pid]['points']
            dist = float(np.linalg.norm(poly - start[None, :], axis=1).min())
            if dist < best_d:
                best, best_d = pid, dist
        parents[cid] = best
    return parents


def cluster_branch_nodes(curves, parents, depths):
    """把各子曲线的起点聚成分支节点(贪心, 按 curve_id 升序 = 生成顺序)。
    返回 nodes: [{coord, child_cids:[...], depth}]。"""
    nodes = []
    for cid in sorted(curves.keys()):
        if depths[cid] <= 1:
            continue
        start = curves[cid]['points'][0]
        placed = False
        for nd in nodes:
            if np.linalg.norm(nd['coord'] - start) < BRANCH_MERGE_TOL:
                nd['child_cids'].append(cid)
                placed = True
                break
        if not placed:
            nodes.append({'coord': np.asarray(start, dtype=float).copy(),
                          'child_cids': [cid],
                          'depth': depths[cid]})
    return nodes


def build_root_table(curves, depths, max_depth):
    """构建 14 列根表行。curves: {cid: {'points': (n,3) 按 t 序测点,
    'radii': (n,), 'controls': (d+1,3) 拟合控制点}}; depths: {cid: 层级}。
    返回 rows(dict 列表, 键为 HEADERS)。"""
    depths = curve_depths_geometric(curves, depths)
    parents = assign_parents(curves, depths)
    branch_nodes = cluster_branch_nodes(curves, parents, depths)
    main_cid = next(cid for cid, d in depths.items() if d == 1)

    rows = []
    node_row = {}          # (kind, key) -> 行序号
    branch_of_child = {}   # cid -> 分支点节点序号(节点 id, 非行号)

    # ---- 节点行: 原点 ----
    origin_pt = curves[main_cid]['points'][0]
    rid = len(rows)
    node_row[('origin', 0)] = rid
    rows.append({
        '序号': rid, '级别': 1, '编号1': None, '编号2': 0,
        '编号3': None, '编号4': None, '类别': '原点',
        'x': round(float(origin_pt[0]), 4), 'y': round(float(origin_pt[1]), 4),
        'z': round(float(origin_pt[2]), 4), '粗细': round(float(curves[main_cid]['radii'][0]), 4),
        '成熟度': maturity_of(1, max_depth),
        '上节点': None, '下节点': None,
    })

    # ---- 节点行: 分支点(一条子曲线的起点, 多条子曲线共享时聚为一个) ----
    for bi, nd in enumerate(branch_nodes):
        rid = len(rows)
        node_row[('branch', bi)] = rid
        for cid in nd['child_cids']:
            branch_of_child[cid] = bi
        rows.append({
            '序号': rid, '级别': nd['depth'], '编号1': None, '编号2': bi,
            '编号3': None, '编号4': None, '类别': '分支点',
            'x': round(float(nd['coord'][0]), 4), 'y': round(float(nd['coord'][1]), 4),
            'z': round(float(nd['coord'][2]), 4),
            '粗细': round(float(curves[nd['child_cids'][0]]['radii'][0]), 4),
            '成熟度': maturity_of(nd['depth'], max_depth),
            '上节点': None, '下节点': None,
        })

    # ---- 节点行: 末端点(每条曲线的终点) ----
    for cid in sorted(curves.keys()):
        cd = curves[cid]
        rid = len(rows)
        node_row[('end', cid)] = rid
        rows.append({
            '序号': rid, '级别': depths[cid], '编号1': None, '编号2': cid,
            '编号3': None, '编号4': None, '类别': '末端点',
            'x': round(float(cd['points'][-1][0]), 4), 'y': round(float(cd['points'][-1][1]), 4),
            'z': round(float(cd['points'][-1][2]), 4), '粗细': round(float(cd['radii'][-1]), 4),
            '成熟度': maturity_of(depths[cid], max_depth),
            '上节点': None, '下节点': None,
        })

    # ---- 曲线行: 首节点/采样点/末节点 + 控制点 ----
    first_row_of = {}      # cid -> 首节点行序号
    last_row_of = {}       # cid -> 末节点行序号
    ctrl_rows_of = {}      # cid -> [(行序号, 分支点行序号或None)...] 记录曲线上的分支点
    branch_rows_on = {}    # cid -> [分支点行序号...] (该曲线上的分叉, 按序)
    for cid in sorted(curves.keys()):
        cd = curves[cid]
        pts, radii = cd['points'], cd['radii']
        n = len(pts)
        depth = depths[cid]
        sample_ids = []
        for i in range(n):
            rid = len(rows)
            cat = '首节点' if i == 0 else ('末节点' if i == n - 1 else '采样点')
            rows.append({
                '序号': rid, '级别': depth, '编号1': cid, '编号2': i,
                '编号3': None, '编号4': None, '类别': cat,
                'x': round(float(pts[i, 0]), 4), 'y': round(float(pts[i, 1]), 4),
                'z': round(float(pts[i, 2]), 4), '粗细': round(float(radii[i]), 4),
                '成熟度': maturity_of(depth, max_depth),
                '上节点': None, '下节点': None,
            })
            sample_ids.append(rid)
        first_row_of[cid] = sample_ids[0]
        last_row_of[cid] = sample_ids[-1]
        ctrl_rows_of[cid] = sample_ids

        # 控制点行(5 次贝塞尔的 4 个中间控制点; 端点即首/末节点行)
        controls = cd.get('controls')
        if controls is not None:
            for ci_, p in enumerate(controls[1:-1], start=1):
                rid = len(rows)
                rows.append({
                    '序号': rid, '级别': depth, '编号1': cid, '编号2': n + ci_ - 1,
                    '编号3': None, '编号4': None, '类别': f'控制点{ci_}',
                    'x': round(float(p[0]), 4), 'y': round(float(p[1]), 4),
                    'z': round(float(p[2]), 4), '粗细': round(float(radii[0]), 4),
                    '成熟度': maturity_of(depth, max_depth),
                    '上节点': None, '下节点': None,
                })

    # ---- 填写链接(上节点/下节点) ----
    # 原点 -> 主根首节点
    rows[node_row[('origin', 0)]]['下节点'] = first_row_of[main_cid]
    rows[first_row_of[main_cid]]['上节点'] = node_row[('origin', 0)]
    # 分支点行: 上节点 = 父曲线首节点行; 下节点 = 第一个子曲线首节点行
    for bi, nd in enumerate(branch_nodes):
        brow = node_row[('branch', bi)]
        parent_cid = parents[nd['child_cids'][0]]
        rows[brow]['上节点'] = first_row_of[parent_cid]
        rows[brow]['下节点'] = first_row_of[nd['child_cids'][0]]
    # 每条子曲线首节点行: 上节点 = 其分支点行
    for cid, bi in branch_of_child.items():
        rows[first_row_of[cid]]['上节点'] = node_row[('branch', bi)]
    # 曲线末节点行 -> 末端点行
    for cid in sorted(curves.keys()):
        rows[last_row_of[cid]]['下节点'] = node_row[('end', cid)]
        rows[node_row[('end', cid)]]['上节点'] = last_row_of[cid]
    # 曲线首节点行 -> 该曲线上的分支点行(下节点; 供从主根向下遍历)
    for bi, nd in enumerate(branch_nodes):
        parent_cid = parents[nd['child_cids'][0]]
        branch_rows_on.setdefault(parent_cid, []).append(node_row[('branch', bi)])
    for cid, brows in branch_rows_on.items():
        rows[first_row_of[cid]]['下节点'] = brows[0]

    # ---- 编号3/编号4: 曲线内采样链 ----
    for cid, ids in ctrl_rows_of.items():
        for k, rid in enumerate(ids):
            rows[rid]['编号3'] = ids[k - 1] if k > 0 else None
            rows[rid]['编号4'] = ids[k + 1] if k < len(ids) - 1 else None

    return rows


def write_root_table(curves, depths, max_depth, csv_path, sql_path,
                     table_name='root_table'):
    """构建根表并写出 CSV + SQL。返回 rows。"""
    rows = build_root_table(curves, depths, max_depth)
    rows_to_csv(rows, csv_path)
    rows_to_sql(rows, table_name=table_name, path=sql_path)
    return rows
