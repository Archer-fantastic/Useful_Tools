import cv2
import numpy as np
import os

# 支持中文路径的读取函数
def cv2_imread_chinese(path, flag=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)

# 支持中文路径的保存函数
def cv2_imwrite_chinese(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif ext == '.png':
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    else:
        raise ValueError("仅支持jpg和png格式")
    
    result, buffer = cv2.imencode(ext, img, encode_param)
    if result:
        with open(path, 'wb') as f:
            f.write(buffer)
        return True
    else:
        raise IOError("图像保存失败")

# 读取两幅图像（替换为你的图像路径，支持中文）
image1_path = r'D:\Min\Projects\VSCodeProjects\01.Useful_Tools\10.CV\demo\img (3).png'  # 可替换为含中文的路径，如'图片1.jpg'
image2_path = r'D:\Min\Projects\VSCodeProjects\01.Useful_Tools\10.CV\demo\img (4).png'  # 可替换为含中文的路径，如'图片2.jpg'

# 读取图像
image1 = cv2_imread_chinese(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2_imread_chinese(image2_path, cv2.IMREAD_GRAYSCALE)

# 检查图像是否读取成功
if image1 is None:
    raise FileNotFoundError(f"无法读取图像1：{image1_path}")
if image2 is None:
    raise FileNotFoundError(f"无法读取图像2：{image2_path}")

# 提取特征点
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)

# 特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 按照距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 选择最佳匹配点（取前15%）
good_matches = matches[:int(len(matches) * 0.15)] if matches else []

# 绘制匹配结果
image_matches = cv2.drawMatches(
    image1, kp1, 
    image2, kp2, 
    good_matches, 
    None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 不绘制单个点
)

# 保存结果（支持中文路径和文件名）
save_dir = r'D:\Min\Projects\VSCodeProjects\01.Useful_Tools\10.CV\匹配结果'  # 可改为含中文的目录，如'特征匹配结果'
os.makedirs(save_dir, exist_ok=True)  # 自动创建目录
save_path = os.path.join(save_dir, '特征匹配结果.png')  # 含中文文件名

# 保存匹配图像
cv2_imwrite_chinese(save_path, image_matches)

print(f"特征匹配结果已保存至：{save_path}")