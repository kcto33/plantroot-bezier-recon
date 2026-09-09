# -*- coding: utf-8 -*-
"""
root_model.py
把 plantroot06.m 的三维贝塞尔树状根系算法移植为 Python，
并组织成"节点图(node-edge graph)"结构：

  - 每个 分支点/末端点 都是一个 Node (有 id, 坐标, 层级, 半径, 父节点, 子节点)。
  - 每条曲线是一条 Edge，严格从起点节点连到终点节点:
        curve.start == start_node.coord
        curve.end   == end_node.coord
    即曲线的空间位置由两个节点坐标唯一确定, 不会浮动。
  - 末端叶子端点也登记为节点(Node)，保证所有曲线首尾都锚定在节点上。

坐标系约定与 MATLAB 一致：地表 z=0，根系沿 z 负方向向下生长。
"""
import numpy as np


class RootParams:
    """与 plantroot06.m 中 params 对应的可调参数。"""

    def __init__(self):
        self.maxDepth = 3          # 根系分支层级
        self.mainLength = 12       # 主根长度
        self.mainRadius = 0.2      # 主根初始半径(顶部粗)
        self.branchNum = [3, 2, 8]  # 每层分支数量
        self.branchScale = 0.5     # 子根长度缩放
        self.radiusScale = 0.4     # 子根半径缩放
        self.angleRange = [20, 70]  # 分支角度范围(度)
        self.curveStrength = 0.4   # 贝塞尔弯曲强度
        self.pointsPerCurve = 50   # 每条曲线采样点数
        self.gravityBend = 0.15    # 重力弯曲系数(正值向下)
        self.taperRatio = 0.3      # 根系变细系数


def random3DPoint(p1, p2, maxDist, rng):
    """生成三维随机偏移点(贝塞尔控制点)。"""
    mid = (p1 + p2) / 2.0
    offset = (rng.random(3) - 0.5) * 2.0 * maxDist
    return mid + offset


def randomBranchDir(parentDir, angleRange, rng):
    """生成随机分支方向(与 MATLAB 的 randomBranchDir 对应)。"""
    parentDir = parentDir / np.linalg.norm(parentDir)
    if abs(parentDir[2]) < 0.9:
        temp = np.array([0.0, 0.0, 1.0])
    else:
        temp = np.array([1.0, 0.0, 0.0])
    ortho1 = np.cross(parentDir, temp)
    ortho1 = ortho1 / np.linalg.norm(ortho1)
    ortho2 = np.cross(parentDir, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)

    angle = np.deg2rad(angleRange[0] + rng.random() * (angleRange[1] - angleRange[0]))
    theta = rng.random() * 2.0 * np.pi
    offset = np.cos(theta) * ortho1 + np.sin(theta) * ortho2
    newDir = parentDir * np.cos(angle) + offset * np.sin(angle)

    # 确保分支也向下生长(负 z 方向分量)
    if newDir[2] > 0:
        newDir[2] = -newDir[2]
    return newDir / np.linalg.norm(newDir)


class RootModel:
    """生成并保存根系的节点图几何。"""

    def __init__(self, params=None, seed=0):
        self.params = params if params is not None else RootParams()
        self.rng = np.random.default_rng(seed)
        self.curves = []       # 每条曲线(edge)，含 start_node/end_node 索引
        self.nodes = []        # 所有节点
        self.bifurcations = [] # 兼容旧接口：分叉点(分支节点)清单
        self._node_id = 0
        self._generate()

    # ------------------------------------------------------------------
    def _new_node(self, coord, depth, radius, parent_id, is_branch):
        nid = self._node_id
        self._node_id += 1
        self.nodes.append({
            'id': nid, 'coord': np.array(coord, dtype=float).copy(),
            'depth': depth, 'radius': float(radius),
            'parent_id': parent_id, 'child_ids': [],
            'is_branch': bool(is_branch),
        })
        if parent_id is not None:
            self.nodes[parent_id]['child_ids'].append(nid)
        return nid

    # ------------------------------------------------------------------
    def _generate(self):
        p = self.params
        startCoord = np.array([0.0, 0.0, 0.0])
        initDir = np.array([0.0, 0.0, -1.0])
        startR = p.mainRadius
        # 地表原点节点 (深度1)
        root_node = self._new_node(startCoord, 1, startR, None, is_branch=True)
        self._growEdge(root_node, initDir, 1, p)

    # ------------------------------------------------------------------
    def _growEdge(self, start_node_id, dir, depth, p):
        """从 start_node_id 长出一条曲线，终点登记为新节点，并递归分支。"""
        if depth > p.maxDepth:
            return

        start_node = self.nodes[start_node_id]
        startP = start_node['coord']

        # 当前边的长度、起止半径
        dir = dir / np.linalg.norm(dir)
        rootLen = p.mainLength * (p.branchScale ** (depth - 1))
        startR = start_node['radius'] if start_node['radius'] > 0 else \
            p.mainRadius * (p.radiusScale ** (depth - 1))
        endR = startR * p.taperRatio
        endP = startP + dir * rootLen

        # 贝塞尔随机控制点 + 重力下偏
        ctrlP1 = random3DPoint(startP, endP, p.curveStrength * rootLen, self.rng)
        ctrlP2 = random3DPoint(startP, endP, p.curveStrength * rootLen, self.rng)
        ctrlP1[2] -= p.gravityBend * rootLen
        ctrlP2[2] -= p.gravityBend * rootLen

        # 三次贝塞尔采样
        t = np.linspace(0.0, 1.0, p.pointsPerCurve)
        c0 = (1 - t) ** 3
        c1 = 3 * (1 - t) ** 2 * t
        c2 = 3 * (1 - t) * t ** 2
        c3 = t ** 3
        pts = (
            c0[:, None] * startP[None, :]
            + c1[:, None] * ctrlP1[None, :]
            + c2[:, None] * ctrlP2[None, :]
            + c3[:, None] * endP[None, :]
        )
        radii = startR * (1 - t) + endR * t

        # —— 终点登记为新节点（末端也是节点，保证曲线终点锚定）——
        end_node_id = self._new_node(endP, depth, endR, start_node_id, is_branch=False)

        self.curves.append({
            'points': pts, 'radii': radii, 'depth': depth,
            'start': startP.copy(), 'ctrl1': ctrlP1.copy(),
            'ctrl2': ctrlP2.copy(), 'end': endP.copy(),
            'start_node': start_node_id,      # 起点节点 id
            'end_node': end_node_id,          # 终点节点 id
            'start_coord': startP.copy(),     # 起点坐标(=起点节点)
            'end_coord': endP.copy(),         # 终点坐标(=终点节点)
        })

        # 分叉节点清单(兼容)
        # 子分支
        if depth < p.maxDepth:
            branchStartIdx = np.arange(
                round(p.pointsPerCurve * 0.1), round(p.pointsPerCurve * 0.8) + 1
            )
            numBranches = p.branchNum[depth - 1]
            for _ in range(numBranches):
                idx = branchStartIdx[self.rng.integers(len(branchStartIdx))]
                bStart = pts[idx, :]
                pointRadius = radii[idx]

                # 在父边登记一个 分支节点(split node)，子曲线起点 = 该节点坐标
                split_node_id = self._new_node(bStart, depth + 1, pointRadius,
                                               start_node_id, is_branch=True)

                # 分叉点清单(去重)
                isDuplicate = False
                for bp in self.bifurcations:
                    if np.linalg.norm(bp['coord'] - bStart) < 0.15:
                        isDuplicate = True
                        break
                if not isDuplicate:
                    self.bifurcations.append({
                        'coord': bStart.copy(),
                        'depth': depth,
                        'radius': pointRadius,
                        'node_id': split_node_id,
                    })

                bDir = randomBranchDir(dir, p.angleRange, self.rng)
                self._growEdge(split_node_id, bDir, depth + 1, p)

    # ------------------------------------------------------------------
    def all_points(self):
        """返回所有曲线采样点(用于渲染)。"""
        all_pts = []
        all_r = []
        for c in self.curves:
            all_pts.append(c['points'])
            all_r.append(c['radii'])
        return np.vstack(all_pts), np.concatenate(all_r)

    def all_nodes(self):
        """返回所有节点坐标 (M,3)。"""
        return np.array([n['coord'] for n in self.nodes])

    @property
    def bbox(self):
        pts, _ = self.all_points()
        return pts.min(axis=0), pts.max(axis=0)
