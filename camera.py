# -*- coding: utf-8 -*-
"""
camera.py
定义针孔相机(内参 K + 外参 [R|t])与投影函数。

约定与 OpenCV 一致:
  - 相机坐标系: X 向右, Y 向下, Z 朝前(沿光轴)。
  - 世界->相机: Xc = R @ Xw + t
  - 投影: u = fx * X/Z + cx ; v = fy * Y/Z + cy
"""
import numpy as np


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """构造从 eye 看向 target 的相机外参 (R, t)。
    返回 R, t 满足 Xc = R @ Xw + t。
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    z_axis = target - eye          # 光轴方向 (相机朝前)
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(up, z_axis)  # 相机 X (向右)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)  # 相机 Y (向下)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # 世界->相机的旋转
    t = -R @ eye
    return R, t


class PinholeCamera:
    """针孔相机, 封装 K, R, t 与投影矩阵 P = K[R|t]。"""

    def __init__(self, fx, fy, cx, cy, R, t, name=""):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.t = np.asarray(t, dtype=float).reshape(3, 1)
        self.P = self.K @ np.hstack([self.R, self.t])
        self.name = name

    def project(self, Xw):
        """把世界坐标点投影到图像像素。Xw: (N,3) 或 (3,)。返回 (N,2)。"""
        single = Xw.ndim == 1
        Xw = np.atleast_2d(Xw)
        Xc = (self.R @ Xw.T + self.t).T      # N x 3
        Z = Xc[:, 2]
        u = self.fx * Xc[:, 0] / Z + self.cx
        v = self.fy * Xc[:, 1] / Z + self.cy
        out = np.stack([u, v], axis=1)
        return out[0] if single else out

    def project_ray(self, uv):
        """由像素 (u,v) 出发得到相机系下的归一化方向向量 (反向投影射线)。"""
        uv = np.atleast_2d(uv).astype(float)
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        return np.stack([x, y, np.ones(uv.shape[0])], axis=1)   # N x 3，相机系方向


def ring_of_cameras(n, radius, height, target, fx, fy, cx, cy,
                    start_deg=0.0, name_prefix="cam", up=(0, 0, 1)):
    """生成绕 Z 轴一圈的 n 个相机。
    radius: 相机到 Z 轴的距离; height: 相机 Z 高度; target: 看向的点;
    up: 相机 up 向量(默认 +z; 传 (0,0,-1) 可让竖直根显示为地表在上、深处在下)。
    返回 PinholeCamera 列表。
    """
    cams = []
    for i in range(n):
        ang = np.deg2rad(start_deg + 360.0 * i / n)
        eye = np.array([radius * np.cos(ang), radius * np.sin(ang), height])
        R, t = look_at(eye, np.asarray(target, dtype=float), np.asarray(up, dtype=float))
        cams.append(PinholeCamera(fx, fy, cx, cy, R, t, name=f"{name_prefix}{i}"))
    return cams
