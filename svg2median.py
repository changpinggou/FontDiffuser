import os
import json
import cv2
import numpy as np
import cairosvg
import re
from scipy.spatial import KDTree
from scipy import interpolate
import shutil

def smooth_stroke(points, num_samples=30):
    if len(points) < 4: return points
    try:
        pts = np.array(points)
        x, y = pts[:, 0], pts[:, 1]
        tck, u = interpolate.splprep([x, y], s=2)
        u_new = np.linspace(0, 1, num_samples)
        new_points = interpolate.splev(u_new, tck)
        return np.column_stack(new_points).tolist()
    except: return points

def get_svg_path_d(svg_path):
    """从 SVG 文件中提取所有 path 的 d 属性字符串"""
    with open(svg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 使用正则匹配所有 path 标签中的 d 属性内容
    paths = re.findall(r'<path[^>]*d="([^"]+)"', content)
    return paths

# def svg_to_ordered_medians(svg_path, ref_medians, size=1024):
#     # 渲染为位图用于细化
#     png_data = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size)
#     img = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_GRAYSCALE)
#     _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
#     skeleton = cv2.ximgproc.thinning(binary)
    
#     skel_pts = np.column_stack(np.where(skeleton > 0))[:, [1, 0]]
#     if len(skel_pts) == 0: return []
#     tree = KDTree(skel_pts)
    
#     ordered_results = []
#     for stroke in ref_medians:
#         current_stroke = []
#         for ref_pt in stroke:
#             dist, idx = tree.query(ref_pt, distance_upper_bound=50)
#             if dist < float('inf'):
#                 current_stroke.append(skel_pts[idx].tolist())
#         if len(current_stroke) > 1:
#             unique_stroke = []
#             for p in current_stroke:
#                 if not unique_stroke or p != unique_stroke[-1]:
#                     unique_stroke.append(p)
#             ordered_results.append(smooth_stroke(unique_stroke))
#     return ordered_results

def svg_to_ordered_medians_aligned(svg_path, ref_medians, size=1024):
    # 1. 渲染并提取复刻字的骨架点
    png_data = cairosvg.svg2png(url=svg_path, output_width=size, output_height=size)
    img = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = cv2.ximgproc.thinning(binary)
    skel_pts = np.column_stack(np.where(skeleton > 0))[:, [1, 0]]
    
    if len(skel_pts) == 0: 
        print("错误：无法在图片中提取到骨架点，请检查图片是否全黑或全白")
        return []
    
    # --- 关键改进：基于外接矩形的重采样对齐 ---
    # 计算复刻字的边界
    t_min = np.min(skel_pts, axis=0)
    t_max = np.max(skel_pts, axis=0)
    t_w, t_h = t_max - t_min

    # 计算标准参考字的边界
    all_ref_pts = np.concatenate(ref_medians)
    r_min = np.min(all_ref_pts, axis=0)
    r_max = np.max(all_ref_pts, axis=0)
    r_w, r_h = r_max - r_min

    tree = KDTree(skel_pts)
    ordered_results = []

    for stroke in ref_medians:
        current_stroke = []
        for ref_pt in stroke:
            # A. 将标准点映射到 [0, 1] 空间
            norm_x = (ref_pt[0] - r_min[0]) / r_w
            norm_y = (ref_pt[1] - r_min[1]) / r_h
            
            # B. 映射到复刻字的实际像素空间
            ax = t_min[0] + norm_x * t_w
            ay = t_min[1] + norm_y * t_h
            
            # C. 在复刻字骨架上寻找最近点，扩大搜索半径至 100 像素
            dist, idx = tree.query([ax, ay], distance_upper_bound=100)
            if dist < float('inf'):
                current_stroke.append(skel_pts[idx].tolist())
        
        if len(current_stroke) > 1:
            # 增加平滑处理
            ordered_results.append(smooth_stroke(current_stroke, num_samples=100))
            
    return ordered_results

def debug_visualization(img_path, result_medians):
    # 读取图片并转为彩色以便画红线
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1024, 1024))
    
    for stroke in result_medians:
        for i in range(len(stroke) - 1):
            pt1 = tuple(map(int, stroke[i]))
            pt2 = tuple(map(int, stroke[i+1]))
            cv2.line(img, pt1, pt2, (0, 0, 255), 5) # 画红色骨架线
            
    cv2.imwrite('debug_medians.png', img)
    print("调试图片已保存至 debug_medians.png")

# --- 执行配置 ---
target_char = "国" 
svg_file = f'/data/applechang/FontDiffuser/temp_trace/{target_char}.svg'
img_path = f'/data/applechang/FontDiffuser/temp_trace/{target_char}.bmp'
source_file = 'graphicsZhHans.txt'
output_file = 'graphicsZhHanMy.txt'

# 1. 确保目标文件存在
if not os.path.exists(output_file):
    shutil.copy(source_file, output_file)

# 2. 获取参考中线
ref_medians = None
with open(source_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['character'] == target_char:
            ref_medians = data['medians']
            break

if ref_medians is None:
    print(f"错误：未找到字符 '{target_char}'")
    exit()

# 3. 提取新的数据
print(f"正在处理 '{target_char}'...")
new_strokes = get_svg_path_d(svg_file)
new_medians = svg_to_ordered_medians_aligned(svg_file, ref_medians)

# 4. 批量更新并回写
updated_lines = []
with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        char_data = json.loads(line)
        if char_data['character'] == target_char:
            # char_data['strokes'] = new_strokes # 替换轮廓数据
            char_data['medians'] = new_medians # 替换骨架数据
            updated_lines.append(json.dumps(char_data, ensure_ascii=False))
        else:
            updated_lines.append(line.strip())

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(updated_lines) + '\n')
debug_visualization(img_path, new_medians)

print(f"成功更新！轮廓笔画数: {len(new_strokes)}, 骨架笔画数: {len(new_medians)}")