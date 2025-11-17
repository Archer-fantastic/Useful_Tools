import os
import json
import random
import numpy as np
from PIL import Image, ImageFilter
import cv2

# 设置随机种子，保证结果可复现
random.seed(42)
np.random.seed(42)

def ensure_dir(path):
    """确保文件夹存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)

def load_annotation(anno_path):
    """加载labelme格式的标注（同时支持多边形和矩形）"""
    if not os.path.exists(anno_path):
        return None, None  # (标注数据, 标注类型)
    
    try:
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"标注文件解析错误 {anno_path}：{e}")
        return None, None
    
    # 优先处理矩形标注（支持2点或4点格式）
    for shape in data.get('shapes', []):
        shape_type = shape.get('shape_type')
        if shape_type == 'rectangle':
            points = shape.get('points', [])
            if len(points) in [2, 4] and all(len(p) == 2 for p in points):
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                return (x1, y1, x2, y2), 'rectangle'
    
    # 处理多边形标注
    for shape in data.get('shapes', []):
        shape_type = shape.get('shape_type')
        if shape_type == 'polygon':
            points = shape.get('points', [])
            if len(points) >= 3 and all(len(p) == 2 for p in points):
                return np.array(points, dtype=np.int32), 'polygon'
    
    return None, None  # 无有效标注

def extract_defect_region(img_path, annotation, shape_type):
    """从缺陷样本中抠取原始区域（带边缘羽化处理）"""
    # 读取图像并转换为RGBA（添加透明通道）
    img = Image.open(img_path).convert('RGBA')
    img_np = np.array(img)
    height, width = img_np.shape[:2]
    
    # 创建对应类型的掩码（标注区域为255，其他为0）
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if shape_type == 'polygon':
        cv2.fillPoly(mask, [annotation], 255)
    elif shape_type == 'rectangle':
        x1, y1, x2, y2 = map(int, annotation)
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # 计算缺陷实际边界框
    coords = np.argwhere(mask == 255)
    if len(coords) == 0:
        return None, (0, 0)  # 空缺陷区域
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 裁剪标注区域和对应的掩码
    defect_region = img_np[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
    
    # 关键1：对掩码边缘进行羽化（模糊），消除硬边界
    # 羽化半径根据缺陷大小动态调整（1-3像素，避免过度模糊）
    defect_size = (defect_region.shape[1], defect_region.shape[0])
    blur_radius = max(1, min(3, int(min(defect_size) * 0.02)))  # 动态半径
    cropped_mask = cv2.GaussianBlur(cropped_mask, (blur_radius*2+1, blur_radius*2+1), 0)
    cropped_mask = np.clip(cropped_mask, 0, 255).astype(np.uint8)  # 确保范围正确
    
    # 将羽化后的掩码应用到alpha通道
    defect_region[:, :, 3] = cropped_mask
    
    return Image.fromarray(defect_region), defect_size

def random_transform(defect_img, defect_size, normal_img_size):
    """对缺陷区域进行随机变换（优化抗锯齿）"""
    normal_w, normal_h = normal_img_size
    defect_w, defect_h = defect_size
    
    # 计算放缩比例
    max_scale_w = normal_w / defect_w
    max_scale_h = normal_h / defect_h
    max_scale = min(max_scale_w, max_scale_h, 2.0)
    min_scale = 0.5
    scale = random.uniform(min_scale, max_scale)
    new_size = (int(defect_w * scale), int(defect_h * scale))
    
    # 关键2：缩放时使用高质量抗锯齿插值
    defect_img = defect_img.resize(new_size, Image.Resampling.LANCZOS)  # 比BILINEAR更平滑
    
    # 随机旋转（-45到45度）
    angle = random.uniform(-45, 45)
    defect_img = defect_img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
    
    # 关键3：对旋转后的边缘二次平滑（轻微模糊）
    defect_img = defect_img.filter(ImageFilter.GaussianBlur(radius=0.5))  # 极轻微模糊
    
    return defect_img

def paste_randomly(normal_img, defect_img):
    """将缺陷粘贴到正常样本上（边缘融合处理）"""
    normal_img = normal_img.convert('RGBA')
    normal_w, normal_h = normal_img.size
    defect_w, defect_h = defect_img.size
    
    # 确保缺陷能被完整粘贴
    if defect_w >= normal_w or defect_h >= normal_h:
        scale = 0.8 * min(normal_w / defect_w, normal_h / defect_h)
        new_size = (int(defect_w * scale), int(defect_h * scale))
        defect_img = defect_img.resize(new_size, Image.Resampling.LANCZOS)
        defect_w, defect_h = new_size
        print(f"缺陷过大，缩小至: {new_size}")
    
    # 计算粘贴位置
    max_x = normal_w - defect_w
    max_y = normal_h - defect_h
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0
    
    # 关键4：提取缺陷的alpha通道，用于边缘融合
    defect_np = np.array(defect_img)
    alpha = defect_np[:, :, 3] / 255.0  # 归一化到0-1
    
    # 提取正常样本中与缺陷重叠的区域
    normal_np = np.array(normal_img)
    overlap = normal_np[y:y+defect_h, x:x+defect_w, :3]  # 正常样本的RGB区域
    defect_rgb = defect_np[:, :, :3]  # 缺陷的RGB区域
    
    # 边缘融合：缺陷边缘（alpha接近0的区域）与正常样本像素混合
    # 混合公式：result = defect_rgb * alpha + overlap * (1 - alpha)
    fused = (defect_rgb.astype(np.float32) * alpha[..., np.newaxis] +
             overlap.astype(np.float32) * (1 - alpha[..., np.newaxis])).astype(np.uint8)
    
    # 将融合结果放回正常样本
    normal_np[y:y+defect_h, x:x+defect_w, :3] = fused
    normal_img = Image.fromarray(normal_np)
    
    return normal_img.convert('RGB')

def get_all_normal_imgs(normal_root):
    """获取所有正常样本的路径列表"""
    normal_imgs = []
    for root, _, files in os.walk(normal_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                normal_imgs.append(os.path.join(root, file))
    if not normal_imgs:
        raise ValueError(f"正常样本目录 {normal_root} 中未找到图片")
    return normal_imgs

def process_defect_samples(defect_root, normal_root, output_root, num_per_defect=3):
    """处理流程：优化边缘平滑和抗锯齿"""
    normal_imgs = get_all_normal_imgs(normal_root)
    print(f"共加载 {len(normal_imgs)} 张正常样本")
    
    for root, dirs, files in os.walk(defect_root):
        relative_path = os.path.relpath(root, defect_root)
        output_dir = os.path.join(output_root, relative_path)
        ensure_dir(output_dir)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                defect_img_path = os.path.join(root, file)
                anno_file = os.path.splitext(file)[0] + '.json'
                anno_path = os.path.join(root, anno_file)
                
                annotation, shape_type = load_annotation(anno_path)
                if annotation is None:
                    print(f"警告：{anno_path} 无有效标注，跳过")
                    continue
                print(f"\n处理 {file}，标注类型：{shape_type}")
                
                try:
                    defect_region, defect_size = extract_defect_region(
                        defect_img_path, annotation, shape_type)
                    if defect_region is None or defect_size[0] < 5 or defect_size[1] < 5:
                        print(f"警告：{file} 缺陷区域无效，跳过")
                        continue
                    print(f"提取原始缺陷尺寸: {defect_size}")
                except Exception as e:
                    print(f"抠取缺陷失败 {defect_img_path}：{e}")
                    continue
                
                base_name = os.path.splitext(file)[0]
                for i in range(num_per_defect):
                    normal_img_path = random.choice(normal_imgs)
                    normal_img = Image.open(normal_img_path)
                    normal_size = normal_img.size
                    
                    transformed_defect = random_transform(
                        defect_region.copy(), defect_size, normal_size)
                    
                    synthetic_img = paste_randomly(normal_img.copy(), transformed_defect)
                    
                    output_file = f"{base_name}_synth_{i+1}{os.path.splitext(file)[1]}"
                    output_path = os.path.join(output_dir, output_file)
                    synthetic_img.save(output_path)
                    print(f"生成合成样本：{output_path}")


if __name__ == "__main__":
    # 配置路径
    DEFECT_ROOT = r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\08.DefectSampleGen\01.DefectCropPaste\缺陷数据\碳渣"
    NORMAL_ROOT = r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\08.DefectSampleGen\01.DefectCropPaste\正常样本"
    OUTPUT_ROOT = r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\08.DefectSampleGen\01.DefectCropPaste\新缺陷样本\碳渣2"
    
    NUM_PER_DEFECT = 20  # 每张缺陷图生成的合成样本数量
    
    # 检查输入路径
    if not os.path.exists(DEFECT_ROOT):
        raise ValueError(f"缺陷样本目录不存在：{DEFECT_ROOT}")
    if not os.path.exists(NORMAL_ROOT):
        raise ValueError(f"正常样本目录不存在：{NORMAL_ROOT}")
    
    # 开始生成合成样本
    print(f"开始生成合成样本，每张缺陷图生成 {NUM_PER_DEFECT} 张...")
    process_defect_samples(DEFECT_ROOT, NORMAL_ROOT, OUTPUT_ROOT, NUM_PER_DEFECT)
    print("所有合成样本生成完成！")