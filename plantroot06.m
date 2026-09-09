% ==============================================
% 三维贝塞尔曲线 树状根系生成程序（向下生长版本）
% 核心：递归分支 + 3D三次贝塞尔曲线 
% ==============================================
clear; clc; close all;

%% 1. 根系参数（可自由调整）
params.maxDepth = 3;             % 根系分支层级（1=仅主根，2=主根+一级侧根，3=多级细根）
params.mainLength = 12;          % 主根长度
params.mainRadius = 0.2;         % 主根初始半径（顶部粗）
params.branchNum = [3,2,8];      % 每一层级的分支数量[3,5,8]; 
params.branchScale = 0.5;        % 子根长度缩放比例
params.radiusScale = 0.4;        % 子根半径缩放比例
params.angleRange = [20,70];     % 分支角度范围(度)
params.curveStrength = 0.4;      % 贝塞尔弯曲强度(0-1，越大越弯)
params.pointsPerCurve = 50;      % 每条贝塞尔曲线的采样点数
params.gravityBend = 0.15;       % 重力弯曲系数（正值向下弯曲）
params.taperRatio = 0.3;         % 根系变细系数（0-1，越小末端越细）

% 存储所有分叉点信息
global bifurcationPoints
bifurcationPoints = {};  % {坐标, 层级, 半径}

%% 2. 初始化画面
fig = figure('Color','w','Position',[100,100,900,800]);
ax = gca;  % 获取坐标区句柄
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
title(ax, '3D 贝塞尔曲线树状根系（向下生长）','FontSize',14);
view(ax, 3);  % 设置3D视角
%%%%%%%%%%%%%%%%%%%camlight('headlight'); lighting gouraud;  % 光照效果

%% 3. 递归生成根系
% 初始点：原点(0,0,0) 表示地表
% 初始方向：沿Z轴向下 [0,0,-1]（主根向下生长）
startPoint = [0,0,0];
initDir = [0,0,-1];  % 修改为向下方向
generateRoot(startPoint, initDir, 1, params);

%% 4. 在命令行窗口输出所有分叉点坐标和粗细
fprintf('\n');
fprintf('================================================================\n');
fprintf('                    分叉点坐标及粗细统计\n');
fprintf('================================================================\n');
fprintf('共找到 %d 个分叉点\n\n', length(bifurcationPoints));

if length(bifurcationPoints) > 0
    fprintf('序号   层级      半径         坐标 (x, y, z)\n');
    fprintf('----------------------------------------------------------------\n');
    
    % 提取半径数组用于统计
    radiusArray = zeros(1, length(bifurcationPoints));
    
    for i = 1:length(bifurcationPoints)
        point = bifurcationPoints{i};
        radiusArray(i) = point.radius;
        fprintf('%2d     %2d       %.4f     (%.4f, %.4f, %.4f)\n', ...
                i, point.depth, point.radius, ...
                point.coord(1), point.coord(2), point.coord(3));
    end
    fprintf('================================================================\n');
    fprintf('注：1. 分叉点为主根/侧根上长出子根的位置\n');
    fprintf('     2. 根系从地表 (z=0) 向下生长\n');
    fprintf('     3. 半径单位为模型单位，从顶部向末端逐渐变细\n');
    fprintf('================================================================\n\n');
    
    % 附加统计信息
    fprintf('【统计摘要】\n');
    fprintf('  平均半径: %.4f\n', mean(radiusArray));
    fprintf('  最大半径: %.4f (序号 %d)\n', max(radiusArray), find(radiusArray == max(radiusArray), 1));
    fprintf('  最小半径: %.4f (序号 %d)\n', min(radiusArray), find(radiusArray == min(radiusArray), 1));
    fprintf('  半径标准差: %.4f\n', std(radiusArray));
    fprintf('================================================================\n\n');
else
    fprintf('没有找到分叉点\n');
end

%% 5. 图像优化
axis(ax, 'tight');
rotate3d(ax, 'on');  % 允许鼠标旋转视角

%% 6. 创建视角控制对话框
createViewControlDialog(ax);

% ==============================================
% 函数：创建视角控制对话框
% ==============================================
function createViewControlDialog(ax)
    % 获取当前视角
    [az, el] = view(ax);
    
    % 创建对话框
    dlg = dialog('Name', '视角控制', 'Position', [300, 300, 350, 250], 'Resize', 'off');
    
    % 方位角控制
    uicontrol('Parent', dlg, 'Style', 'text', 'String', '方位角 (Azimuth):', ...
              'Position', [30, 190, 120, 25], 'HorizontalAlignment', 'left');
    azEdit = uicontrol('Parent', dlg, 'Style', 'edit', 'String', num2str(az), ...
                       'Position', [160, 190, 80, 25]);
    azSlider = uicontrol('Parent', dlg, 'Style', 'slider', 'Min', -180, 'Max', 180, ...
                         'Value', az, 'Position', [250, 190, 80, 25]);
    
    % 仰角控制
    uicontrol('Parent', dlg, 'Style', 'text', 'String', '仰角 (Elevation):', ...
              'Position', [30, 140, 120, 25], 'HorizontalAlignment', 'left');
    elEdit = uicontrol('Parent', dlg, 'Style', 'edit', 'String', num2str(el), ...
                       'Position', [160, 140, 80, 25]);
    elSlider = uicontrol('Parent', dlg, 'Style', 'slider', 'Min', -90, 'Max', 90, ...
                         'Value', el, 'Position', [250, 140, 80, 25]);
    
    % 预设视角按钮
    uicontrol('Parent', dlg, 'Style', 'pushbutton', 'String', '默认视角 (3D)', ...
              'Position', [30, 80, 90, 35], 'Callback', @(src,event) setPresetView(ax, dlg, azEdit, elEdit, azSlider, elSlider, 1));
    uicontrol('Parent', dlg, 'Style', 'pushbutton', 'String', '正视图 (XZ)', ...
              'Position', [130, 80, 90, 35], 'Callback', @(src,event) setPresetView(ax, dlg, azEdit, elEdit, azSlider, elSlider, 2));
    uicontrol('Parent', dlg, 'Style', 'pushbutton', 'String', '俯视图 (XY)', ...
              'Position', [230, 80, 90, 35], 'Callback', @(src,event) setPresetView(ax, dlg, azEdit, elEdit, azSlider, elSlider, 3));
    
    % 关闭按钮
    uicontrol('Parent', dlg, 'Style', 'pushbutton', 'String', '关闭', ...
              'Position', [130, 20, 80, 30], 'Callback', @(src,event) delete(dlg));
    
    % 回调函数：编辑框改变时更新滑块和视图
    azEdit.Callback = @(src,event) updateFromEdit(ax, azEdit, azSlider, 'az');
    elEdit.Callback = @(src,event) updateFromEdit(ax, elEdit, elSlider, 'el');
    
    % 回调函数：滑块改变时更新编辑框和视图
    azSlider.Callback = @(src,event) updateFromSlider(ax, azEdit, azSlider, 'az');
    elSlider.Callback = @(src,event) updateFromSlider(ax, elEdit, elSlider, 'el');
end

% ==============================================
% 函数：从编辑框更新视角
% ==============================================
function updateFromEdit(ax, edit, slider, type)
    try
        val = str2double(edit.String);
        if isnan(val)
            return;
        end
        % 限制范围
        if strcmp(type, 'az')
            val = max(-180, min(180, val));
            slider.Value = val;
        else
            val = max(-90, min(90, val));
            slider.Value = val;
        end
        edit.String = num2str(val);
        
        % 更新视图
        [az, el] = view(ax);
        if strcmp(type, 'az')
            view(ax, val, el);
        else
            view(ax, az, val);
        end
    catch ME
        disp(['更新视角出错: ', ME.message]);
    end
end

% ==============================================
% 函数：从滑块更新视角
% ==============================================
function updateFromSlider(ax, edit, slider, type)
    val = slider.Value;
    edit.String = num2str(round(val));
    
    % 更新视图
    [az, el] = view(ax);
    if strcmp(type, 'az')
        view(ax, val, el);
    else
        view(ax, az, val);
    end
end

% ==============================================
% 函数：设置预设视角
% ==============================================
function setPresetView(ax, dlg, azEdit, elEdit, azSlider, elSlider, preset)
    switch preset
        case 1  % 默认3D视角
            az = -37.5;
            el = 30;
        case 2  % 正视图（XZ平面）
            az = 0;
            el = 0;
        case 3  % 俯视图（XY平面）
            az = 0;
            el = 90;
    end
    
    % 更新控件
    azEdit.String = num2str(az);
    elEdit.String = num2str(el);
    azSlider.Value = az;
    elSlider.Value = el;
    
    % 更新视图
    view(ax, az, el);
end

% ==============================================
% 核心函数：递归生成单条根 + 分支
% ==============================================
function generateRoot(startP, dir, depth, p)
    global bifurcationPoints
    
    % 递归终止条件
    if depth > p.maxDepth
        return;
    end

    % ===================== 步骤1：计算当前根的贝塞尔控制点 =====================
    % 归一化方向向量
    dir = dir / norm(dir);
    % 根长度（随层级递减）
    rootLen = p.mainLength * (p.branchScale)^(depth-1);
    % 起点半径（随层级递减，顶部粗底部细）
    startR = p.mainRadius * (p.radiusScale)^(depth-1);
    % 终点半径（渐变变细）
    endR = startR * p.taperRatio;
    % 终点坐标
    endP = startP + dir * rootLen;

    % 生成贝塞尔曲线随机控制点（控制弯曲形态）
    ctrlP1 = random3DPoint(startP, endP, p.curveStrength*rootLen);
    ctrlP2 = random3DPoint(startP, endP, p.curveStrength*rootLen);
    % 重力修正：让控制点继续向下偏移（负z方向）
    ctrlP1(3) = ctrlP1(3) - p.gravityBend * rootLen;
    ctrlP2(3) = ctrlP2(3) - p.gravityBend * rootLen;

    % ===================== 步骤2：计算三维贝塞尔曲线坐标 =====================
    t = linspace(0,1,p.pointsPerCurve);
    curvePoints = zeros(length(t), 3);
    
    % 三次贝塞尔曲线公式
    for i = 1:length(t)
        ti = t(i);
        c0 = (1-ti)^3;
        c1 = 3*(1-ti)^2*ti;
        c2 = 3*(1-ti)*ti^2;
        c3 = ti^3;
        curvePoints(i,1) = c0*startP(1) + c1*ctrlP1(1) + c2*ctrlP2(1) + c3*endP(1);
        curvePoints(i,2) = c0*startP(2) + c1*ctrlP1(2) + c2*ctrlP2(2) + c3*endP(2);
        curvePoints(i,3) = c0*startP(3) + c1*ctrlP1(3) + c2*ctrlP2(3) + c3*endP(3);
    end
    
    % 计算每个点的半径（从起点半径线性过渡到终点半径）
    radii = startR * (1 - t) + endR * t;

    % ===================== 步骤3：绘制3D根系（渐变圆柱） =====================
    drawTaperedCylinder(curvePoints, radii);

    % ===================== 步骤4：递归生成子分支 =====================
    if depth < p.maxDepth
        % 分支起点：在当前根的中段随机选取branchStartIdx = round(p.pointsPerCurve*0.4) : round(p.pointsPerCurve*0.8);
        branchStartIdx = round(p.pointsPerCurve*0.1) : round(p.pointsPerCurve*0.8);
        numBranches = p.branchNum(depth);
        
        for b = 1:numBranches
            % 随机选择分支起点
            idx = branchStartIdx(randi(length(branchStartIdx)));
            bStart = curvePoints(idx, :);
            bStartCoord = [bStart(1), bStart(2), bStart(3)];
            % 获取该点的半径
            pointRadius = radii(idx);
            
            % 记录分叉点（去重：如果坐标太接近就不重复记录）
            isDuplicate = false;
            for k = 1:length(bifurcationPoints)
                if norm(bifurcationPoints{k}.coord - bStartCoord) < 0.15
                    isDuplicate = true;
                    break;
                end
            end
            
            if ~isDuplicate
                bifurcationPoints{end+1} = struct(...
                    'coord', bStartCoord, ...
                    'depth', depth, ...
                    'radius', pointRadius);
            end
            
            % 生成随机分支方向
            bDir = randomBranchDir(dir, p.angleRange);
            % 递归生成子根
            generateRoot(bStart, bDir, depth+1, p);
        end
    end
end

% ==============================================
% 辅助函数1：生成三维随机偏移点
% ==============================================
function pt = random3DPoint(p1, p2, maxDist)
    mid = (p1 + p2) / 2;
    offset = (rand(1,3)-0.5)*2*maxDist;
    pt = mid + offset;
end

% ==============================================
% 辅助函数2：生成随机分支方向
% ==============================================
function newDir = randomBranchDir(parentDir, angleRange)
    % 生成垂直于父方向的正交向量
    if abs(parentDir(3)) < 0.9
        temp = [0,0,1];
    else
        temp = [1,0,0];
    end
    ortho1 = cross(parentDir, temp);
    ortho1 = ortho1 / norm(ortho1);
    ortho2 = cross(parentDir, ortho1);
    ortho2 = ortho2 / norm(ortho2);
    
    % 随机角度偏转
    angle = deg2rad(angleRange(1) + rand*(angleRange(2)-angleRange(1)));
    theta = rand*2*pi;
    offset = cos(theta)*ortho1 + sin(theta)*ortho2;
    newDir = parentDir*cos(angle) + offset*sin(angle);
    
    % 确保分支也向下生长（负z方向分量）
    if newDir(3) > 0
        newDir(3) = -newDir(3);
    end
    newDir = newDir / norm(newDir);
end

% ==============================================
% 辅助函数3：绘制渐变圆柱
% ==============================================
function drawTaperedCylinder(curvePoints, radii)
    % 圆柱圆周采样点
    theta = linspace(0, 2*pi, 16);
    nTheta = length(theta);
    nPoints = size(curvePoints, 1);
    
    % 对曲线上的每一段绘制锥台
    for i = 1:nPoints-1
        p1 = curvePoints(i, :);
        p2 = curvePoints(i+1, :);
        r1 = radii(i);
        r2 = radii(i+1);
        
        % 如果半径太小，跳过
        if r1 < 0.005 && r2 < 0.005
            continue;
        end
        
        % 段方向向量
        segDir = p2 - p1;
        segLen = norm(segDir);
        if segLen < 1e-6
            continue;
        end
        segDir = segDir / segLen;
        
        % 构建局部坐标系的基向量
        if abs(segDir(3)) < 0.9
            temp = [0, 0, 1];
        else
            temp = [1, 0, 0];
        end
        u = cross(segDir, temp);
        u = u / norm(u);
        v = cross(segDir, u);
        v = v / norm(v);
        
        % 计算起点和终点的圆周顶点
        vertsStart = zeros(nTheta, 3);
        vertsEnd = zeros(nTheta, 3);
        
        for j = 1:nTheta
            c = cos(theta(j));
            s = sin(theta(j));
            offsetStart = r1 * (c * u + s * v);
            offsetEnd = r2 * (c * u + s * v);
            vertsStart(j, :) = p1 + offsetStart;
            vertsEnd(j, :) = p2 + offsetEnd;
        end
        
        % 绘制圆柱侧面
        for j = 1:nTheta-1
            vertices = [
                vertsStart(j, :);
                vertsStart(j+1, :);
                vertsEnd(j+1, :);
                vertsEnd(j, :)
            ];
            faces = [1, 2, 3, 4];
            
            % 颜色根据半径大小渐变
            colorValue = 0.2 + 0.3 * (r1 / max(radii));
            patch('Vertices', vertices, 'Faces', faces, ...
                  'FaceColor', [0.3, 0.2+colorValue*0.3, 0.1+colorValue*0.2], ...
                  'EdgeColor', 'none', 'FaceLighting', 'gouraud');
        end
        
        % 封闭首尾端
        if i == 1 && r1 > 0.01
            patch('Vertices', vertsStart, 'Faces', 1:nTheta, ...
                  'FaceColor', [0.3, 0.25, 0.15], 'EdgeColor', 'none');
        end
        if i == nPoints-1 && r2 > 0.005
            patch('Vertices', vertsEnd, 'Faces', 1:nTheta, ...
                  'FaceColor', [0.35, 0.28, 0.18], 'EdgeColor', 'none');
        end
    end
end