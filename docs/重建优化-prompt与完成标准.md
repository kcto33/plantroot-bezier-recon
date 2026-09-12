# 仿真根系多视角三维重建 —— 优化任务 Prompt 与完成标准

> 本文档由项目现状诊断生成（2026-09-12）。
> 【使用方法】第一部分是给 AI 编码助手的完整任务 prompt，开新会话直接投喂（建议连同本文件一起提供，助手可直接引用其中的行号与真值说明）；第二部分是配套的完成标准，已写入 prompt 中，供验收时逐条核对。

---

# 第一部分：优化后的任务 Prompt

## 【目标】
修复并优化本项目"仿真根系多视角截图 → 三维重建 → 主根/一级侧根提取 → 根数据库"的全链路重建质量。当前 web 模块③（上传 24 图 + pose.json → 重建）输出严重失真。最终必须通过第二部分的全部量化完成标准，并交付一键自动评估脚本。

## 【背景与实测基线 —— 已完成诊断，可直接采信】
- 项目闭环链路：`plantroot06.m`（参数）→ `root_model.py` 节点-边图（seed=0 时共 10 条曲线：主根 1 条 + 一级侧根 3 条 + 二级侧根 6 条，实测 8 个分支点）→ `camera.py` 24 台环形相机 + `render.py` 自绘渲染 → 24 张图 + `pose.json` → 颜色标记检测 → DLT 三角化 → 5 次贝塞尔 → 根表（CSV + SQL，14 列，结构见 `table_build.py:3-22`）。
- 真值来源：`dense_meta.json` 存有每个标记点的 `coord_true`（真值坐标），可直接用于自动评估。
- 实测基线（按 `dense_meta.json` 真值逐点计算 3D 误差）：

  | 链路 | 结果 | 误差（中位 / 最大） | 结论 |
  |---|---|---|---|
  | CLI `build_table_from_images.py`（根目录 `reconstructed_points.csv`） | 120/120 点 | 0.014 / 0.122 | 基本可用，仍有提升空间 |
  | Web 模块③ `web_core_images.py`（`results/job_images/`） | 119/120 点（曲线 7 一点三角化失败；最差点误差 5.91） | 0.43 / 5.91 | 崩坏，本次主要修复对象 |
  | 分支点 8 视角链（`root_branch_table.csv`） | 8/8 分支点 | — / 0.003 | 精度高，但无拓扑 |

- 相机位姿（`pose.json` 与渲染同源）已验证一致，不是问题来源。

## 【已定位薄弱点 —— 按优先级修复】
### P0（直接导致崩坏，先修）
1. **上传图被完全忽略（最严重）**：`web_core_images.py:130` 的 `run_images_pipeline(imgdir, ...)` 中 `imgdir` 参数从未被使用；`:143-160` 用 `RootModel(RootParams(), seed=0)` 重新自渲染标记图再检测——重建输出与用户上传的 24 张图完全无关。必须改为真正读取上传图目录 + `pose.json`。
2. **检测下采样破坏颜色**：`web_core_images.py:82` 检测用 `subsample=3` 块均值（CLI 版 `build_table_from_images.py:115` 为 1），5px 小标记被稀释 → 漏检 / 误配到邻近调色板色 → 跨视角对应错乱。
3. **web 版缺离群剔除**：`web_core_images.py:247-249` 无 `remove_outliers`（CLI 版 `build_table_from_images.py:263-294` 有：3 次拟合基准 + 3σ 剔除），错点直通 5 次贝塞尔拟合 → 控制点飞掉。

### P1（精度与鲁棒性）
4. **标记渲染无深度测试**：`render.py:134-151` 的 `render_marker` 注释称受 zbuf 约束，实际 mask 内无条件覆盖颜色、不写 zbuf（`:146-150`）→ 标记重叠互相覆盖拉偏质心；`occlusion_visible`（`build_table_from_images.py:103-112`）只测中心像素。
5. **三角化鲁棒性弱**：`build_table_from_images.py:190-225` / `web_core_images.py:170-191` 剔除仅 2 轮、2px 阈值硬编码、剩余视角 <3 时接受未验证解（`:217` / web 版 `:187`）；`triangulate.py:9` 的 DLT 无 cheirality（正深度）检查。
6. **颜色分割无兜底**：`nearest_palette_centroid`（`build_table_from_images.py:115-150`）为全局最近邻，无连通域分析——同色多 blob 或邻近色（120 色、tol=70）误配时无任何校验。
7. **主管线无真值评估**：`build_table_from_images.py:251` 的 `tp['coord_true']` 是死代码；dense 评估链已损坏（`generate_dense_marked_views.py:24` N_VIEWS=72 而 `views_dense/` 只有 8 张图；`detect_dense_and_rebuild.py:33` 同样 72；`dense_meta.json` 每曲线 12 点与现码 `POINTS_PER_CURVE=4`（`:27`）不符）。

### P2（全链路完善）
8. **半径未从图像估计**：直接取模型真值半径（`build_table_from_images.py:98` / `web_core_images.py:66`）再乘硬编码 taper `(1-0.6t)`（`:360` / web 版 `:259`）。应改为从图像可观测信息估计（如标记盘像素半径反演 `r = px·Z/fx`，注意 `MIN_MARKER_R=5` 截断的影响），taper 参数化；若论证不可行需给出分析并经确认。
9. **分支拓扑缺失**：`detect_and_triangulate.py:122-137` 分支表只写"首节点"行、级别硬编码 1、上下节点留空；`branch_table_to_bezier.py:63-75` 靠"x 正负分组 + z 排序"纯启发式。需重建"主根 → 分支点 → 侧根"树，并输出与 `table_build.py` 相同 14 列结构的完整根表。
10. **贝塞尔阶数固定致震荡**：`bezier_fit.py:81-108` 固定 5 次最小二乘（端点固定 + 弦长参数化，无正则），近直主根控制点摆幅达 ±7.5（根目录 `control_points.csv`）。改为阶数自适应或加正则/限幅。
11. **潜在 bug**：`generate_marked_views.py:83` 的 `PALETTE[bi % len(PALETTE)]` 颜色取模复用——分支点 >8 时不同 3D 点共享颜色 → 跨视角按色配对会合并不同点；当前参数恰好未触发。
12. **未完成代码与魔法数**：`run_reconstruct_from_images.py:31-64` 阶段 3 仅按 CSV 顺序连线画示意、未真正拟合；端点对齐阈值 0.35、2px、0.6 等魔法数应集中为常量/配置。

## 【范围 —— 只允许以下操作】
- 项目根目录：`D:\2StudyRelate\05Implement\plantroot-bezier-recon-main`
- 可读：仅项目目录内的文件
- 可写/新建/删除：仅项目目录内的文件
- 可运行：项目内脚本（环境已有 numpy / Pillow / matplotlib / scipy）

## 【边界 —— 禁止】
- 不访问项目路径之外的任何位置（包括其他盘符、C:\Users、系统目录、注册表、环境变量）
- 尽量不安装新依赖（numpy/Pillow/matplotlib/scipy 之外如确需引入，可以允许，但是不能破坏其他环境，比如可以创建一个新的venv）；不改系统/编辑器配置
- 不执行 git commit / push（除非明确要求）
- 不改 `plantroot06.m` 的根系生成语义；不改根表 14 列字段语义（`table_build.py:3-22`）；不把渲染器重写为 OpenGL/外部库；不更换 web 框架（保持 http.server 纯标准库）

## 【完成标准】
见本文档第二部分，由 `evaluate_reconstruction.py` 逐条自动校验。

## 【约束】
- 保持现有代码风格；最小改动原则，只改必须改的
- 每阶段结束运行评估并报告指标变化，未通过当前阶段验收不得进入下一阶段
- 回复使用中文

## 【执行方式 —— 分阶段推进】
- **阶段 0（评估先行）**：新建 `evaluate_reconstruction.py` 一键评估（读 `reconstructed_points.csv` / `control_points.csv` / 根表，与 `dense_meta.json` 真值对比），输出 JSON + markdown 表 + 逐条 pass/fail；先跑出修复前基线报告。
- **阶段 1**：修 P0 三项 → web 模块③ 指标必须达到完成标准 A/B/C/D 组。
- **阶段 2**：修 P1（渲染遮挡、三角化鲁棒性、颜色连通域兜底、评估链修复）→ 全指标回归。
- **阶段 3**：修 P2（半径估计、拓扑 + 根表、贝塞尔阶数、latent bug、常量集中）→ 全指标回归 + 换 seed=1/2 回归。
- **阶段 4**：端到端验收：web 上传 `data/plantroot_views (1)/` 的 24 图 + `pose.json` 走通全流程并达标；CLI 同样达标；更新 README 中过时的精度声明（`README.md:135-141` 与 web 实测不符），如实注明评估方法与真值来源。
- 有歧义或需要越出边界时：停下来问，不要自行发挥；与目标无关的"顺便改进"一律不做。

---

# 第二部分：完成标准（量化自动指标）

> 世界尺度：主根长度 mainLength=12。全部指标由 `evaluate_reconstruction.py` 自动计算并输出 pass/fail。
> 数值为本次诊断给出的建议默认值，阶段 0 出基线后如有不合理可提出并经确认调整。

## A. 输入真实性
- A1 web 模块③必须消费上传图：上传不同参数/seed 生成的视图集时，重建输出必须随之变化（管线内不得存在绕过上传图的自渲染旁路）。

## B. 检测与对应
- B1 标记点重建成功率 120/120（web 基线 119/120）。
- B2 跨视角对应错误率 0（用颜色 id 真值校验）。

## C. 三维点精度（`reconstructed_points.csv` vs 真值 `coord_true`）
- C1 中位误差 ≤ 0.02（≈主根长度的 0.17%；CLI 基线 0.014 可达）。
- C2 平均误差 ≤ 0.05。
- C3 最大误差 ≤ 0.30（CLI 基线 0.122；web 基线 5.91）。
- C4 每点重投影残差：中位 ≤ 1.0px，最大 ≤ 2.0px。

## D. 曲线级精度（每条拟合贝塞尔采样点 vs 真值曲线）
- D1 曲线最大偏差 ≤ 0.40。
- D2 控制点不发散：任一控制点到所属曲线弦包围盒的距离 ≤ 2.0（消除主根 ±7.5 震荡）。
- D3 端点误差 ≤ 0.05，主根起点须 ≈ (0,0,0)。

## E. 拓扑与根表（最低要求：主根 + 一级侧根入库；目标：10 条曲线全入库）
- E1 曲线条数与层级正确（seed=0 时：主根 1 + 一级侧根 3 + 二级侧根 6），每条侧根可追溯到所属父分支点。
- E2 全部分支点（seed=0 实测 8 个）位置误差 ≤ 0.05。
- E3 根表 14 列完整；"上节点/下节点"构成有效树：脚本从"原点"节点行出发可遍历到达全部曲线行，无孤行、无断链。
- E4 `root_table.csv` 与 `.sql` 同时生成，SQL 通过 sqlite3 实际导入验证（保持 `table_build.py:168` 现有建表风格）。

## F. 半径
- F1 根表"粗细"列不再直接读取模型真值字段：由图像可观测信息估计，主根起点半径相对误差 ≤ 30%，且沿曲线单调渐变无跳变；若论证不可行，给出分析并经用户确认后降级。

## G. 鲁棒性（防对 seed=0 过拟合）
- G1 用 seed=1、seed=2 生成新视图集重跑全链路，C/D/E 组指标同样达标。
- G2 该回归可一条命令重复执行（生成视图 → 重建 → 评估）。

## H. 交付物与验收
- H1 `evaluate_reconstruction.py`：一键输出全部指标（JSON + markdown 表 + 逐条 pass/fail），修复前后均可运行。
- H2 web 与 CLI 两条入口均达标；`results/job_images/` 重新生成且失真消除。
- H3 README 更新：以实测数字替换过时精度声明（`README.md:135-141`），注明评估方法与真值来源。
- H4 修复前后 `rebuilt_vs_orig` 对比图各一组，视觉可辨改善。
