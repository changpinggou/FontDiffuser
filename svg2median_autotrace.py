import os
import json
import subprocess
import re
import numpy as np
import cairosvg
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev

def smooth_path(points, num_samples=50):
    """B样条平滑笔画"""
    if len(points) < 4: return points
    pts = np.array(points)
    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=2)
        u_new = np.linspace(0, 1, num_samples)
        new_pts = splev(u_new, tck)
        return np.column_stack(new_pts).tolist()
    except: return points

def get_autotrace_skeleton(svg_path, size=1024):
    """调用 autotrace 提取矢量骨架"""
    temp_bmp = "temp_input.bmp"
    temp_svg = "temp_out.svg"
    
    # 渲染为 BMP 提高 autotrace 识别率
    cairosvg.svg2png(url=svg_path, write_to=temp_bmp, output_width=size, output_height=size)
    
    # 执行 autotrace: centerline 模式
    cmd = [
        "autotrace", 
        "--centerline",
        "--color-count", "2",
        "--error-threshold", "2.0",  # 降低阈值，捕捉更多细碎线条
        "--despeckle-level", "10",  # 去除噪点
        "--output-format", "svg",
        "--output-file", temp_svg,
        temp_bmp
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # 解析生成的骨架点
    with open(temp_svg, 'r') as f:
        content = f.read()
    
    # 正则提取所有 path 的 d 属性
    paths_d = re.findall(r'd="([^"]+)"', content)
    skeleton_segments = []
    
    for d in paths_d:
        # 提取所有数字坐标
        coords = re.findall(r"(\d+\.\d+|\d+)", d)
        pts = [[float(coords[i]), float(coords[i+1])] for i in range(0, len(coords), 2)]
        if len(pts) > 1:
            skeleton_segments.append(pts)
            
    # 清理临时文件
    for f in [temp_bmp, temp_svg]:
        if os.path.exists(f): os.remove(f)
        
    return skeleton_segments

def reorder_medians(raw_segments, ref_medians):
    """
    改进版：增加多段自动拼接与距离容错
    """
    if not raw_segments: return []
    
    # 1. 整合所有提取出的骨架点
    all_raw_pts = np.concatenate(raw_segments)
    from scipy.spatial import KDTree
    tree = KDTree(all_raw_pts)
    
    final_medians = []
    
    for ref_stroke in ref_medians:
        ref_pts = np.array(ref_stroke)
        matched_stroke = []
        
        # 增加采样点密度，确保能覆盖到每一小段碎片
        for ref_pt in ref_pts:
            # 扩大搜索半径至 150 像素，适应变形较大的书法字
            dist, idx = tree.query(ref_pt, distance_upper_bound=150)
            if dist < float('inf'):
                matched_stroke.append(all_raw_pts[idx].tolist())
        
        # 2. 检查提取质量
        if len(matched_stroke) > 5: # 至少有5个点才认为是一笔
            # 去重并应用更强的平滑
            unique_pts = []
            for p in matched_stroke:
                if not unique_pts or np.linalg.norm(np.array(p) - np.array(unique_pts[-1])) > 2:
                    unique_pts.append(p)
            
            if len(unique_pts) > 2:
                final_medians.append(smooth_path(unique_pts, num_samples=80))
            
    return final_medians

# --- 主执行流程 ---
target_char = "国"
svg_file = f'/data/applechang/FontDiffuser/temp_trace/{target_char}.svg'
source_db = 'graphicsZhHans.txt'
output_db = 'graphicsZhHanMy.txt'

# 1. 获取参考笔顺
ref_medians = None
with open(source_db, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['character'] == target_char:
            ref_medians = data['medians']
            break

if not ref_medians:
    print(f"未找到字符 {target_char}")
    exit()

# 2. 提取并重排骨架
print(f"正在通过 autotrace 提取 '{target_char}' 的骨架...")
raw_segments = get_autotrace_skeleton(svg_file)
new_medians = reorder_medians(raw_segments, ref_medians)

# 3. 更新数据库
updated_lines = []
with open(source_db, 'r', encoding='utf-8') as f:
    for line in f:
        char_data = json.loads(line)
        if char_data['character'] == target_char:
            char_data['medians'] = new_medians
            # 同时更新 strokes 确保渲染底图一致
            # char_data['strokes'] = get_svg_path_d(svg_file) 
            updated_lines.append(json.dumps(char_data, ensure_ascii=False))
        else:
            updated_lines.append(line.strip())

with open(output_db, 'w', encoding='utf-8') as f:
    f.write('\n'.join(updated_lines) + '\n')

print(f"成功！已将精准骨架写入 {output_db}，共提取 {len(new_medians)} 条笔画。")