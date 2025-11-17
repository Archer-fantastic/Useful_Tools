from pathlib import Path
import cv2
import numpy as np
import os

# 读取图像（支持中文路径的读取方式）
def cv2_imread_chinese(path, flag=cv2.IMREAD_COLOR):
    """读取含中文路径的图像"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)

# 保存图像（支持中文路径的保存方式）
def cv2_imwrite_chinese(path, img):
    """保存图像到含中文的路径"""
    # 根据图像格式选择编码格式（这里以png为例，也可改为.jpg）
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif ext == '.png':
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    else:
        raise ValueError("仅支持jpg和png格式")
    
    # 编码为二进制数据
    result, buffer = cv2.imencode(ext, img, encode_param)
    if result:
        with open(path, 'wb') as f:
            f.write(buffer)
        return True
    else:
        raise IOError("图像保存失败")

# 主程序
if __name__ == "__main__":
    # 输入图像路径（可含中文）
    input_path = r'D:\Min\Projects\VSCodeProjects\01.Useful_Tools\10.CV\demo\img (1).png'
    # 输出目录（含中文）
    output_dir = r'D:\Min\Projects\VSCodeProjects\01.Useful_Tools\10.CV\demo\处理结果'
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录（若不存在）

    img_name = Path(input_path).name

    # 读取图像（灰度模式）
    image = cv2_imread_chinese(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{input_path}")

    # 1. 灰度变换（线性变换）
    image_linear = cv2.convertScaleAbs(image * 2)
    cv2_imwrite_chinese(f"{output_dir}/{img_name}_线性变换.png", image_linear)

    # 2. 均值滤波
    image_blurred = cv2.blur(image, (5, 5))
    cv2_imwrite_chinese(f"{output_dir}/{img_name}_均值滤波.png", image_blurred)

    # 3. 中值滤波
    image_median = cv2.medianBlur(image, 5)
    cv2_imwrite_chinese(f"{output_dir}/{img_name}_中值滤波.png", image_median)

    # 4. Sobel边缘检测（需转换为8位格式）
    image_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # x方向边缘
    image_sobel = cv2.convertScaleAbs(image_sobel)  # 转换为uint8
    cv2_imwrite_chinese(f"{output_dir}/{img_name}_Sobel边缘.png", image_sobel)

    # 5. Canny边缘检测
    image_canny = cv2.Canny(image, 100, 200)
    cv2_imwrite_chinese(f"{output_dir}/{img_name}_Canny边缘.png", image_canny)

    # 6. 阈值分割（二值化）
    _, image_threshold = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    cv2_imwrite_chinese(f"{output_dir}/{img_name}_阈值分割.png", image_threshold)

    print(f"所有处理结果已保存至：{output_dir}")