import os
import cv2
import glob
import subprocess
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from svgpath2mpl import parse_path
import xml.etree.ElementTree as ET

# ================= 配置区域 =================
INPUT_DIR = "outputs"           # FontDiffuser 生成图片的文件夹
OUTPUT_FONT = "MyAiFont.ttf"    # 最终生成的字体文件名
BASE_FONT_PATH = "ttf/KaiXinSongA.ttf"   # 必须提供一个基础字体作为模板 (推荐黑体或宋体)
TEMP_DIR = "temp_trace"         # 临时文件夹
UPM = 1000                      # 字体单位 (Standard for TTF is 1000 or 2048)
# ===========================================

def preprocess_image(img_path):
    """
    读取图片，二值化，并反转颜色（Potrace 需要黑字白底，或者反过来，视配置而定）
    FontDiffuser 生成的通常是黑底白字或白底黑字，这里统一处理成：
    前景(字) = 黑色，背景 = 白色
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. 放大图片 (Super Resolution 的简易替代，为了让边缘更顺滑)
    scale_factor = 4
    img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # 2. 二值化 (Otsu's Thresholding)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 颜色检查：确保字是黑色的 (0)，背景是白色的 (255)
    # 简单判断：如果角落是黑色，说明背景是黑的，需要反转
    if binary[0, 0] == 0:
        binary = cv2.bitwise_not(binary)

    return binary

def image_to_svg_path(img_path, char_str):
    """
    调用 Potrace 转 SVG，并提取所有路径 (XML 解析修复版)
    """
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    # 保存 BMP
    processed_img = preprocess_image(img_path)
    bmp_path = os.path.join(TEMP_DIR, f"{char_str}.bmp")
    svg_path = os.path.join(TEMP_DIR, f"{char_str}.svg")
    cv2.imwrite(bmp_path, processed_img)

    # 调用 Potrace
    # --alphamax 1.1 让线条稍微圆润一点
    cmd = ["potrace", bmp_path, "-s", "-o", svg_path, "--alphamax", "1.1"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # === 核心修复：读取 SVG 文件中所有的 Path ===
    try:
        # 使用 XML 解析器，不再怕多 path 标签
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # SVG 命名空间处理（有时 Potrace 输出带 xmlns，有时不带，iter() 最稳）
        all_paths = []
        for elem in root.iter():
            # 找到所有 tag 结尾是 path 的元素 (忽略命名空间前缀)
            if elem.tag.endswith('path'):
                d = elem.get('d')
                if d:
                    all_paths.append(d)
        
        if not all_paths:
            print(f"[Error] No paths found in {svg_path}")
            return None
        
        # 将所有路径段拼接成一个长字符串 (中间加空格)
        full_path_data = " ".join(all_paths)
        return full_path_data
        
    except Exception as e:
        print(f"[Error] Failed to parse SVG for {char_str}: {e}")
        return None

def update_font_glyph(font, char, svg_path_data):
    """
    将 SVG 路径数据写入到字体的 Glyph 中 (修复基线漂移 + 垂直居中)
    """
    if not svg_path_data:
        return

    path_obj = parse_path(svg_path_data)
    
    # 1. 获取 Bounding Box
    bbox = path_obj.get_extents()
    xmin, ymin, xmax, ymax = bbox.extents
    
    height = ymax - ymin
    width = xmax - xmin
    if height == 0: return

    # 2. 读取字体关键指标 (Ascender/Descender)
    # 这是修复"字飘在上面"的关键！
    upm = font['head'].unitsPerEm
    ascent = font['hhea'].ascent
    descent = font['hhea'].descent
    
    # 计算字体的"视觉中心线"
    # 比如: ascent=859, descent=-141 -> font_height=1000 -> visual_center = (-141 + 859) / 2 = 359
    # 而不是之前的 UPM/2 = 500 (这导致字高了 141 个单位)
    visual_center_y = (ascent + descent) / 2

    # 3. 设定缩放比例 (0.92 比较饱满)
    scale_ratio = 0.92
    target_height = upm * scale_ratio
    scale = target_height / height
    
    # 4. 计算垂直偏移 (基于视觉中心)
    # 目标 Y 范围的中心应该对齐 visual_center_y
    # target_ymin = visual_center_y - (target_height / 2)
    vertical_offset = visual_center_y - (target_height / 2)
    
    # 5. 计算水平偏移 (基于 UPM 居中)
    scaled_width = width * scale
    horizontal_offset = (upm - scaled_width) / 2

    glyph_set = font.getGlyphSet()
    pen = TTGlyphPen(glyph_set)
    
    def transform_point(pt):
        x = float(pt[0])
        y = float(pt[1])
        
        # 坐标映射 (使用新的 vertical_offset)
        new_x = (x - xmin) * scale + horizontal_offset
        new_y = (y - ymin) * scale + vertical_offset
        
        return (int(new_x), int(new_y))

    # 绘制路径 (正向，不反转)
    for polygon in path_obj.to_polygons():
        if len(polygon) < 3: continue
        
        pen.moveTo(transform_point(polygon[0]))
        for pt in polygon[1:]:
            pen.lineTo(transform_point(pt))
        pen.closePath()

    # 写入字体
    uni_code = ord(char)
    cmap = font.getBestCmap()
    
    if uni_code in cmap:
        glyph_name = cmap[uni_code]
        font['glyf'][glyph_name] = pen.glyph()
        
        # 更新字宽 (Advance Width)
        if 'hmtx' in font:
            font['hmtx'][glyph_name] = (upm, int(horizontal_offset))
            
        print(f"[Success] Updated: {char} (Centered at Y={visual_center_y:.1f})")
    else:
        print(f"[Warning] Character {char} not found in base font.")
        
def main():
    if not os.path.exists(BASE_FONT_PATH):
        print(f"Error: Base font {BASE_FONT_PATH} not found!")
        return

    print(f"Loading base font: {BASE_FONT_PATH}...")
    font = TTFont(BASE_FONT_PATH)

    # 遍历 output 文件夹
    # 假设文件名格式为: "anything_字.png" 或 "字.png"
    # 我们取文件名中最后一个汉字
    import re
    
    files = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"Found {len(files)} images to process.")

    for img_file in files:
        filename = os.path.basename(img_file)
        # 提取文件名里的汉字 (假设文件名里包含且仅包含一个目标汉字)
        # 比如 "result_隆.png" -> 提取 "隆"
        chars = re.findall(r'[\u4e00-\u9fa5]', filename)
        if not chars:
            continue
        target_char = chars[-1] # 取最后一个匹配的汉字
        
        print(f"Processing: {target_char} from {filename}")
        
        # 1. 图片转 SVG Path
        svg_d = image_to_svg_path(img_file, target_char)
        
        # 2. 注入字体
        if svg_d:
            update_font_glyph(font, target_char, svg_d)

    # 保存
    print(f"Saving font to {OUTPUT_FONT}...")
    font.save(OUTPUT_FONT)
    print("Done! 🎉")

if __name__ == "__main__":
    main()