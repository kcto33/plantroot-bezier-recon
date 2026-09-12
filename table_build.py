# -*- coding: utf-8 -*-
"""
table_build.py
把三维根系模型转换成"每行=一个曲线采样点"的数据表。

表结构(与目标 MySQL 表一致):
  序号 | 级别 | 编号1 | 编号2 | 编号3 | 编号4 | 类别 | x | y | z | 粗细 | 成熟度 | 上节点 | 下节点

语义:
  - 每行是沿某条贝塞尔曲线采样的一个点(Node 类)。
  - 编号1: 所属曲线/分支段的全局 ID (branch 编号)。
  - 编号2: 该点在曲线内的序号(0-based)。
  - 编号3: 上一节点序号(Node.last), 编号4: 下一节点序号(Node.next) —— 用于把采样点串成链。
  - 类别: 区分 首节点/控制点1/控制点2/末节点/采样点 (对应 beiseer 类)。
  - 上节点: 本段起点相连的分叉父节点序号; 下节点: 本段末点相连的子分支首点序号。
  - 粗细=半径, 成熟度=归一化成熟度。
"""
import numpy as np


HEADERS = ["序号", "级别", "编号1", "编号2", "编号3", "编号4", "类别",
           "x", "y", "z", "粗细", "成熟度", "上节点", "下节点"]


def maturity_of(depth, max_depth):
    """主根(深度1)成熟度高, 细根成熟度低。返回 [0,1]。
    depth=1 -> 1.0, depth=max_depth -> 0.3 之类, 再线性减淡。"""
    if max_depth <= 1:
        return 1.0
    return round(1.0 - 0.35 * (depth - 1) / (max_depth - 1), 3)


def build_table(model, random_seed=0):
    """生成表行列表(每个元素为 dict, 键为 HEADERS)。
    节点图结构:
      - 每个 分支点/末端点 一行(类别=节点/分支点/末端点), 序号全局唯一。
      - 每条曲线一行组: 首节点/控制点1/控制点2/末节点/采样点。
      - 曲线的 首节点行的上节点 = 起点节点行序号, 末节点行的下节点 = 终点节点行序号,
        这样曲线空间位置被两个节点行固定。
    返回 (rows, curve_rows)。
    """
    curves = model.curves
    max_depth = model.params.maxDepth
    rng = np.random.default_rng(random_seed)

    rows = []

    # ---- 先写出所有节点行(用 node id -> 表中序号 映射) ----
    node_row = {}          # node id -> 行序号
    for nd in model.nodes:
        nid = nd['id']
        rid = len(rows)
        node_row[nid] = rid
        cat = '分支点' if nd['is_branch'] else '末端点'
        if nd['parent_id'] is None:
            cat = '原点'
        rows.append({
            '序号': rid,
            '级别': nd['depth'],
            '编号1': None,  # 节点行不属于具体曲线, 节点 id 记录在编号2
            '编号2': nid,
            '编号3': None,
            '编号4': None,
            '类别': cat,
            'x': round(float(nd['coord'][0]), 4),
            'y': round(float(nd['coord'][1]), 4),
            'z': round(float(nd['coord'][2]), 4),
            '粗细': round(float(nd['radius']), 4),
            '成熟度': maturity_of(nd['depth'], max_depth),
            '上节点': node_row[nd['parent_id']] if nd['parent_id'] is not None else None,
            '下节点': None,
        })

    # ---- 每条曲线: 采样点行 + 控制点行 ----
    curve_rows = []
    for ci, c in enumerate(curves):
        pts = c['points']
        radii = c['radii']
        npts = pts.shape[0]
        depth = c['depth']
        sn_row = node_row[c['start_node']]
        en_row = node_row[c['end_node']]

        # 分类采样点
        categories = ['采样点'] * npts
        categories[0] = '首节点'
        categories[-1] = '末节点'

        sample_ids = []
        for i in range(npts):
            rid = len(rows)
            rows.append({
                '序号': rid,
                '级别': depth,
                '编号1': ci,
                '编号2': i,
                '编号3': None,   # last
                '编号4': None,   # next
                '类别': categories[i],
                'x': round(float(pts[i, 0]), 4),
                'y': round(float(pts[i, 1]), 4),
                'z': round(float(pts[i, 2]), 4),
                '粗细': round(float(radii[i]), 4),
                '成熟度': maturity_of(depth, max_depth),
                # 首节点行的上节点=起点节点行, 末节点行的下节点=终点节点行
                '上节点': sn_row if i == 0 else None,
                '下节点': en_row if i == npts - 1 else None,
            })
            sample_ids.append(rid)

        # 控制点1/控制点2 两行
        for cap, cp_tag in ((c['ctrl1'], '控制点1'), (c['ctrl2'], '控制点2')):
            rid = len(rows)
            rows.append({
                '序号': rid,
                '级别': depth,
                '编号1': ci,
                '编号2': npts if cp_tag == '控制点1' else npts + 1,
                '编号3': None,
                '编号4': None,
                '类别': cp_tag,
                'x': round(float(cap[0]), 4),
                'y': round(float(cap[1]), 4),
                'z': round(float(cap[2]), 4),
                '粗细': round(float(radii[0]), 4),
                '成熟度': maturity_of(depth, max_depth),
                '上节点': None,
                '下节点': None,
            })

        curve_rows.append({
            'id': ci, 'sample_row_ids': sample_ids,
            'first_row': sample_ids[0], 'last_row': sample_ids[-1],
            'depth': depth,
            'start_node_row': sn_row, 'end_node_row': en_row,
            'parent_curve_id': None, 'child_curve_ids': [],
        })

    # ---- 填写 编号3(last)/编号4(next) 采样点链表 ----
    for cr in curve_rows:
        ids = cr['sample_row_ids']
        for k, rid in enumerate(ids):
            rows[rid]['编号3'] = ids[k - 1] if k > 0 else None
            rows[rid]['编号4'] = ids[k + 1] if k < len(ids) - 1 else None

    return rows, curve_rows


def rows_to_csv(rows, path, headers=None):
    headers = headers or HEADERS
    import csv
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in headers})


def rows_to_sql(rows, table_name='root_table', path=None):
    """生成 MySQL 建表 + 插入语句(字符串), 可选写文件。"""
    col_defs = [
        "序号 INT", "级别 INT", "编号1 INT", "编号2 INT",
        "编号3 INT", "编号4 INT", "类别 VARCHAR(20)",
        "x DOUBLE", "y DOUBLE", "z DOUBLE",
        "粗细 DOUBLE", "成熟度 DOUBLE", "上节点 INT", "下节点 INT",
    ]
    create = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  " + \
             ",\n  ".join(col_defs) + ",\n  PRIMARY KEY (序号)\n);"

    def sqlval(v):
        if v is None:
            return 'NULL'
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)

    inserts = []
    for r in rows:
        vals = [sqlval(r[k]) for k in
                ('序号', '级别', '编号1', '编号2', '编号3', '编号4', '类别',
                 'x', 'y', 'z', '粗细', '成熟度', '上节点', '下节点')]
        inserts.append(f"INSERT INTO {table_name} VALUES ({', '.join(vals)});")
    text = create + "\n\n" + "\n".join(inserts)
    if path:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
    return text


def print_table(rows, max_rows=25, headers=None):
    headers = headers or HEADERS
    col_w = {k: max(len(k), 8) for k in headers}
    for r in rows[:max_rows]:
        for k in headers:
            v = r[k]
            col_w[k] = max(col_w[k], len(str(v)) if v is not None else 4)
    def fmt(v):
        return '  ' if v is None else str(v)
    print("  |  " + "  |  ".join(k.ljust(col_w[k]) for k in headers) + "  |")
    print("  |" + "-----|" * len(headers))
    for r in rows[:max_rows]:
        print("  |  " + "  |  ".join(fmt(r[k]).ljust(col_w[k]) for k in headers) + "  |")
    if len(rows) > max_rows:
        print(f"  ... (共 {len(rows)} 行)")
