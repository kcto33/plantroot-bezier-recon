# 贝塞尔根系模型：多视角重建 + 数据表还原

本仓库围绕 `plantroot06.m`（三维贝塞尔树状根系，向下生长）实现两条完整链路：

1. **模型 → 数据表 → 由表还原模型**（有几何算法时）
2. **图片 + 已知相机 → 自动检测分支点 → 三角还原 → 填表**（只有渲染图时）

两种方式最终都产出符合下图表格结构的数据表，且**用表里的点能控制贝塞尔曲线绘出原模型**。

```
序号 | 级别 | 编号1 | 编号2 | 编号3 | 编号4 | 类别 | x | y | z | 粗细 | 成熟度 | 上节点 | 下节点
```

---

## 一、项目结构

| 文件 | 作用 |
|---|---|
| `plantroot06.m` | 原始 MATLAB 根系生成脚本（参考） |
| `root_model.py` | 把 MATLAB 算法移植为 Python 三维几何模型 |
| `camera.py` | 针孔相机（内参 K + 外参 [R,t] + 投影矩阵 P）、绕 Z 轴一圈排布 |
| `render.py` | 用自定义相机 + z-buffer 程序化渲染 2D 图、给目标点加颜色标记、颜色定位 |
| `triangulate.py` | 多视角 DLT 三角化 + 重投影误差 |
| `table_build.py` | 模型 → 数据表（CSV + MySQL SQL），并打印表 |
| `table_render.py` | 从表还原模型：按链表串联采样点 + 拆四控制点重建贝塞尔 |
| `bezier_fit.py` | **真正的最小二乘三次贝塞尔拟合** |
| `reconstruct_from_table.py` | **独立脚本**：只凭数据表还原并绘图（不依赖模型代码） |

### 链路一（模型 → 表 → 还原）
| 文件 | 作用 |
|---|---|
| `make_table.py` | 生成表（`root_table.csv/.sql`）+ 由表反推绘图 + 验证贝塞尔重建精度 |
| `fit_and_rebuild.py` | 对每条曲线做最小二乘贝塞尔拟合，重建模型并回填根表 |

### 链路二（图片 → 表）
| 文件 | 作用 |
|---|---|
| `generate_marked_views.py` | 生成带彩色分支点标记的 8 视角图（遮挡感知） |
| `detect_and_triangulate.py` | 仅凭图片像素 + 已知相机：自动检测 → 跨视角配对 → DLT 三角化 → 填表 |
| `run_reconstruct_from_images.py` | 一键跑链路二（阶段1+2+画图） |

> 另：`pipeline.py` 是最早的多视角三角化演示（指定点 . 无需迁移）。

---

## 二、环境

- Python 3.10+
- numpy、Pillow（PIL）、matplotlib
- 不需要 OpenCV（颜色标记代替特征匹配）

```bash
pip install numpy pillow matplotlib
```

Windows 下若中文乱码，先执行：

```powershell
chcp 65001
$env:PYTHONIOENCODING='utf-8'
```

---

## 三、用法

### 链路一：模型 → 表 → 由表还原（有模型几何）

```bash
python make_table.py
```
产物：
- `root_table.csv` / `root_table.sql`：模型转成"每行一个采样点"的表（含各曲线 4 控制点）
- `model_rebuilt_3d.png` / `model_rebuilt_xy.png`：由表重建的 3D 图 / 俯视 XY 图

**单段最小二乘贝塞尔拟合重建：**

```bash
python fit_and_rebuild.py
```
产物：
- `root_fitted_table.csv` / `root_fitted_table.sql`：由拟合贝塞尔重建的表（采样点 + 控制点）
- `fitted_rebuilt_3d.png`：拟合重建（红）vs 原始采样点（灰）

**只凭表还原（不依赖模型代码）：**

```bash
python reconstruct_from_table.py root_table.csv --outdir rebuilt_out
```

### 链路二：图片 + 已知相机 → 自动检测分支点 → 三角化填表

```bash
python run_reconstruct_from_images.py
```
或分步：
```bash
python generate_marked_views.py      # 生成带彩色标记的 8 视角图 -> views_branch/
python detect_and_triangulate.py     # 检测/配对/三角化/填表
```
产物：
- `views_branch/cam0~7.png`：带彩色分支点标记的视角图
- `root_branch_table.csv` / `root_branch_table.sql`：重建出的分支点 3D 坐标（含粗细估计）
- `rebuilt_branchpoints_3d.png`

**多视角三角化演示（指定点）：**
```bash
python pipeline.py                                  # 自动选一个可见分支点
python pipeline.py --from-list 4                    # 从分叉点清单按序号指定
python pipeline.py --pixel 320 340 --view 0         # 在某视角用像素坐标指定
```

---

## 四、表结构语义

| 列 | 含义 |
|---|---|
| 序号 | 全局行号 |
| 级别 | 根的分支层级（1=主根 …） |
| 编号1 | 所属曲线/分支段 ID |
| 编号2 | 点在曲线内的序号 |
| 编号3 | 上一节点（Node 类 `last`） |
| 编号4 | 下一节点（Node 类 `next`） |
| 类别 | `首节点`/`控制点1`/`控制点2`/`末节点`/`采样点` |
| x,y,z | 坐标 |
| 粗细 | 半径（由圆盘尺寸+深度反推，或拟合曲线沿程插值） |
| 成熟度 | 归一化成熟度（主根 1.0 → 细根 0.3） |
| 上节点 | 分段起点所连接的分叉父节点 |
| 下节点 | 分段末点所连接的子分支首点 |

**用表还原模型有两种方式（供绘图/拟合）**：
1. **按链表**：用 `编号4`(下一节点) 把同一 `编号1` 的采样点串成折线。
2. **拆贝塞尔控制点**：从每条曲线的 `类别=首节点/控制点1/控制点2/末节点` 四行取 4 控制点，用三次贝塞尔公式重建。

---

## 五、精度一览

| 场景 | 结果 |
|---|---|
| 模型 → 表 → 拆4控制点还原 | 最大偏差 0.0001（相对 0.0004%） |
| 最小二乘贝塞尔拟合（单段） | 整体形态还原良好；主根中段几何误差约 3%（单段三次贝塞尔欠拟合复杂弯曲） |
| 图片自动检测 → 三角化 | 分支点坐标误差 < 0.05，重投影误差约 0.1px |
| 粗细估计（圆盘+深度反推） | 粗分支点较准；细小分支点因 3px 可见性下限有上偏 |

---

## 六、限制与取舍

- **链路二依赖"给图加彩色标记"**。若坚持纯黑白渲染图不加标记，跨视角自动匹配在低纹理下不可靠（需引入 OpenCV 特征匹配或手工给坐标）。
- **`粗细`**：链路二用彩色圆盘尺寸 + 深度反推（小分支受 3px 下限影响）；链路一（有拟合曲线）用拟合曲线沿程半径渐变，更合理。
- **`成熟度`**：目前按"层级越深越不成熟"线性公式填充，可替换为真实数据。
- **单段三次贝塞尔**对复杂主根欠拟合（约 3% 几何误差），如需更高精度可改分段拟合。
- `上节点/下节点`、`编号3/编号4` 父子/链表关系当前用几何最近邻自动推断，可按实际规则替换。
