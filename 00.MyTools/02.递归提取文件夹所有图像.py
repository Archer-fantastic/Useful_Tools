import os
import shutil
from argparse import ArgumentParser

def get_image_files(root_dir):
    """递归获取所有图像文件路径"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_files = []
    
    # 递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件扩展名是否为图像格式
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(dirpath, filename)
                image_files.append(image_path)
    
    return image_files

def process_images(source_dir, dest_dir, operation='copy'):
    """处理图像文件（复制或移动）"""
    # 创建目标文件夹（如果不存在）
    os.makedirs(dest_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = get_image_files(source_dir)
    total = len(image_files)
    
    if total == 0:
        print(f"在 {source_dir} 及其子目录中未找到任何图像文件")
        return
    
    print(f"找到 {total} 个图像文件，正在{ '复制' if operation == 'copy' else '移动' }到 {dest_dir}...")
    
    for i, src_path in enumerate(image_files, 1):
        # 获取文件名（处理重名情况）
        filename = os.path.basename(src_path)
        dest_path = os.path.join(dest_dir, filename)
        
        # 处理重名文件
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        # 执行复制或移动操作
        try:
            if operation == 'copy':
                shutil.copy2(src_path, dest_path)  # 保留元数据
            else:
                shutil.move(src_path, dest_path)
            print(f"[{i}/{total}] 已处理: {src_path}")
        except Exception as e:
            print(f"[{i}/{total}] 处理失败 {src_path}: {str(e)}")
    
    print(f"操作完成！共处理 {total} 个文件")

if __name__ == "__main__":
    # 解析命令行参数
    parser = ArgumentParser(description="递归提取文件夹下的所有图像并复制/移动到目标文件夹")
    parser.add_argument("--source_dir", help="源文件夹路径", default=r'D:\Min\Projects\VSCodeProjects\dataset\cls_test_data\data\2025102913')
    parser.add_argument("--dest_dir", help="目标文件夹路径", default=r'D:\Min\Projects\VSCodeProjects\dataset\cls_test_data\data\2025102913')
    parser.add_argument(
        "-o", "--operation", 
        choices=['copy', 'move'], 
        default='move', 
        help="操作类型：copy（复制，默认）或 move（移动）"
    )
    
    args = parser.parse_args()
    
    # 验证源文件夹是否存在
    if not os.path.isdir(args.source_dir):
        print(f"错误：源文件夹 {args.source_dir} 不存在")
        exit(1)
    
    # 执行操作
    process_images(args.source_dir, args.dest_dir, args.operation)