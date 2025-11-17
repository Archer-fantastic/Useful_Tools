import os
import shutil
import random
from pathlib import Path

def copy_small_dataset(src_dir, dst_dir, train_count=50, val_count=50):
    """
    从YOLO格式数据集中抽取小份数据集
    
    参数:
        src_dir: 原始数据集根目录（包含images和labels文件夹）
        dst_dir: 小数据集保存目录
        train_count: 训练集抽取数量
        val_count: 验证集抽取数量
    """
    # 定义原始和目标路径
    splits = ['train', 'val']
    src_images = {split: Path(src_dir) / 'images' / split for split in splits}
    src_labels = {split: Path(src_dir) / 'labels' / split for split in splits}
    
    dst_images = {split: Path(dst_dir) / 'images' / split for split in splits}
    dst_labels = {split: Path(dst_dir) / 'labels' / split for split in splits}
    
    # 创建目标目录
    for split in splits:
        dst_images[split].mkdir(parents=True, exist_ok=True)
        dst_labels[split].mkdir(parents=True, exist_ok=True)
    
    # 抽取并复制文件
    for split in splits:
        # 确定当前 split 要抽取的数量
        count = train_count if split == 'train' else val_count
        
        # 获取原始图像文件列表（过滤非图像文件）
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in src_images[split].iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        # 检查文件数量是否足够
        if len(image_files) < count:
            print(f"警告：{split}集原始图像数量不足{count}，将复制全部{len(image_files)}张")
            selected = image_files
        else:
            # 随机抽取指定数量的图像
            selected = random.sample(image_files, count)
        
        # 复制图像和对应的标注文件
        for img_path in selected:
            # 复制图像
            dst_img = dst_images[split] / img_path.name
            shutil.copy2(img_path, dst_img)  # 保留文件元数据
            
            # 复制标注文件（替换图像后缀为.txt）
            label_name = img_path.stem + '.txt'
            src_label = src_labels[split] / label_name
            if src_label.exists():
                dst_label = dst_labels[split] / label_name
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告：{img_path.name}的标注文件{label_name}不存在，已跳过")
        
        print(f"已完成{split}集复制：{len(selected)}张图像及标注")

if __name__ == '__main__':
    # --------------------------
    # 请修改以下路径和参数
    # --------------------------
    src_dir = r"D:\Min\Projects\VSCodeProjects\dataset\cls_household-trash-recycling-dataset"  # 原始数据集根目录
    dst_dir = r"D:\Min\Projects\VSCodeProjects\dataset\det_household-trash-recycling-mini-dataset"          # 小数据集保存路径
    train_count = 70                    # 训练集抽取数量
    val_count = 30                      # 验证集抽取数量
    # --------------------------
    
    copy_small_dataset(src_dir, dst_dir, train_count, val_count)
    print(f"小数据集已保存至：{dst_dir}")