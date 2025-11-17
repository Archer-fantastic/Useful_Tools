import cv2
import numpy as np
import os

def cv2_imread_chinese(path):
    """支持中文路径的图像读取"""
    if not os.path.exists(path):
        print(f"错误：文件路径不存在 - {path}")
        return None
    img_data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return img

def cv2_imwrite_chinese(path, img):
    """支持中文路径的图像保存"""
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.jpg', '.jpeg']:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        result, img_data = cv2.imencode(ext, img, encode_param)
    elif ext == '.png':
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        result, img_data = cv2.imencode(ext, img, encode_param)
    else:
        result, img_data = cv2.imencode(ext, img)
    if result:
        img_data.tofile(path)
        return True
    else:
        print(f"错误：保存图像失败 - {path}")
        return False

def fix_glare_only(image_path, output_path="修复炫光后的图像.jpg"):
    # 1. 读取图像（支持中文路径）
    img = cv2_imread_chinese(image_path)
    if img is None:
        return

    # 2. 纯炫光弱化（仅处理高光区域，无其他额外处理）
    # 2.1 HSV通道分离，只调整亮度通道（避免影响色彩）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 2.2 分强度定位炫光区域（精准识别高光，不影响正常区域）
    _, strong_glare_mask = cv2.threshold(v, 230, 255, cv2.THRESH_BINARY)  # 强炫光
    _, weak_glare_mask = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)   # 弱炫光
    weak_glare_mask = cv2.subtract(weak_glare_mask, strong_glare_mask)    # 排除重复区域
    
    # 2.3 针对性弱化炫光（仅降低高光亮度，保留原图其他细节）
    v = np.where(strong_glare_mask == 255, cv2.add(v, -120), v)  # 强炫光大幅弱化
    v = np.where(weak_glare_mask == 255, cv2.add(v, -60), v)    # 弱炫光适度弱化
    v = np.clip(v, 0, 255)  # 防止亮度溢出导致失真
    
    # 2.4 合并通道，直接输出（无任何额外处理）
    hsv_corrected = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)  # 保持原始彩色，不转灰度

    # 3. 保存结果
    if cv2_imwrite_chinese(output_path, result):
        print(f"炫光弱化完成，结果保存至：{output_path}")
    

# 调用示例（替换为你的中文路径）
fix_glare_only(
    image_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\demo\demo1.bmp",
    output_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\res\demo1_res.bmp"
)
fix_glare_only(
    image_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\demo\demo2.jpg",
    output_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\res\demo2_res.jpg"
)
fix_glare_only(
    image_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\demo\demo3.jpg",
    output_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\res\demo3_res.jpg"
)
fix_glare_only(
    image_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\demo\demo4.jpg",
    output_path=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\09.处理畸变与炫光\res\demo4_res.jpg"
)