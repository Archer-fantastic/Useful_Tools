from PIL import Image
import os

def replace_center_with_top_left(bmp_path, save_path, crop_size=50):
    """
    读取BMP图片，用左上角50×50区域覆盖中心区域并保存
    :param bmp_path: 输入BMP图片路径
    :param save_path: 新图片保存路径
    :param crop_size: 截取的区域尺寸（默认50×50）
    """
    # 1. 读取BMP图片
    if not os.path.exists(bmp_path):
        raise FileNotFoundError(f"输入图片不存在：{bmp_path}")
    
    # 确保读取为RGB模式（避免单色或透明通道问题）
    img = Image.open(bmp_path).convert('RGB')
    img_width, img_height = img.size
    print(f"读取图片成功，尺寸：{img_width}×{img_height}")

    # 2. 截取左上角50×50区域
    # 若图片本身小于50×50，按实际尺寸截取（避免报错）
    actual_crop_w = min(crop_size, img_width)
    actual_crop_h = min(crop_size, img_height)
    top_left_region = img.crop((0, 0, actual_crop_w, actual_crop_h))
    print(f"截取左上角区域尺寸：{actual_crop_w}×{actual_crop_h}")

    # 3. 计算图片中心位置（确保覆盖区域居中）
    # 中心区域的左上角坐标 = (图片宽/2 - 截取宽/2, 图片高/2 - 截取高/2)
    center_x = (img_width - actual_crop_w) // 2
    center_y = (img_height - actual_crop_h) // 2
    print(f"中心覆盖位置：左上角坐标({center_x}, {center_y})")

    # 4. 将左上角区域粘贴到中心位置（覆盖原有像素）
    img.paste(top_left_region, (center_x, center_y))

    # 5. 保存新图片（保持BMP格式）
    # 确保保存路径的文件夹存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img.save(save_path, format='BMP')
    print(f"新图片已保存至：{save_path}")


def main():
    # 配置参数（请根据你的实际路径修改）
    INPUT_BMP_PATH = r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\08.create_samples\sample.bmp"  # 输入BMP路径
    OUTPUT_BMP_PATH = r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\08.create_samples\sample_new.bmp"  # 输出保存路径
    CROP_SIZE = 50  # 固定截取左上角50×50区域

    # 执行替换操作
    try:
        replace_center_with_top_left(INPUT_BMP_PATH, OUTPUT_BMP_PATH, CROP_SIZE)
        print("图片处理完成！")
    except Exception as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    main()