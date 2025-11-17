import cv2
import albumentations as A
import os
import argparse
from datetime import datetime
import glob
import random
import numpy as np

class BatteryDefectAugmenter:
    def __init__(self):
        self.args = self.parse_arguments()
        # 清理输入路径（去除首尾空格，处理路径分隔符）
        self.args.input_path = self.clean_path(self.args.input_path)
        self.args.output_dir = self.clean_path(self.args.output_dir)
        
        self.aug_samples_config = {
            'rotate': 5,
            'brightness_contrast': 5,
            'vertical_flip': 1,
            'horizontal_flip': 1,
            'gaussian_blur': self.args.samples,
            'random_crop': self.args.samples,
            'gaussian_noise': self.args.samples,
            'salt_pepper_noise': self.args.samples,
            'random_scale': self.args.samples,
            'elastic_transform': self.args.samples
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = os.path.join(self.args.output_dir, f"aug_results_{self.timestamp}")
        self.safe_makedirs(self.output_root)  # 使用安全创建目录方法
        self.log_file = os.path.join(self.output_root, "augmentation_log.txt")
        self.log("Battery defect image augmentation started")
        self.log(f"Timestamp: {self.timestamp}")
        self.log(f"Input path: {self.args.input_path}")
        self.log(f"Output root: {self.output_root}")
        self.log(f"Selected augmentations: {', '.join(self.args.augmentations)}")
        self.log(f"Sample configuration: {self.aug_samples_config}")
        self.log(f"Recursive mode: {'Enabled' if self.args.recursive else 'Disabled'}")
        
        self.augmentations = self.initialize_augmentations()

    def clean_path(self, path):
        """清理路径：去除首尾空格、统一路径分隔符"""
        if not path:
            return path
        # 去除首尾空格（处理复制路径时的意外空格）
        path = path.strip()
        # 统一路径分隔符（Windows用\，Linux用/）
        path = os.path.normpath(path)
        return path

    def safe_makedirs(self, dir_path):
        """安全创建目录（处理中文和特殊字符，避免创建失败）"""
        try:
            # 先转换为绝对路径，避免相对路径问题
            abs_dir = os.path.abspath(dir_path)
            os.makedirs(abs_dir, exist_ok=True)
            self.log(f"Created directory (or exists): {abs_dir}")
        except Exception as e:
            self.log(f"Error creating directory {dir_path}: {str(e)}")
            raise  # 目录创建失败直接终止程序

    def initialize_augmentations(self):
        augs = {}
        
        if 'rotate' in self.args.augmentations:
            augs['rotate'] = A.Rotate(limit=15, p=1.0)
        
        if 'brightness_contrast' in self.args.augmentations:
            augs['brightness_contrast'] = A.RandomBrightnessContrast(
                brightness_limit=(-0.10, 0.15),
                contrast_limit=(-0.10, 0.15),
                p=1.0
            )
        
        if 'horizontal_flip' in self.args.augmentations:
            augs['horizontal_flip'] = A.HorizontalFlip(p=1.0)
        
        if 'vertical_flip' in self.args.augmentations:
            augs['vertical_flip'] = A.VerticalFlip(p=1.0)
        
        if 'gaussian_blur' in self.args.augmentations:
            augs['gaussian_blur'] = A.GaussianBlur(blur_limit=(3, 7), p=1.0)
        
        if 'random_crop' in self.args.augmentations:
            augs['random_crop'] = A.RandomResizedCrop(
                height=224, width=224,
                scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0
            )
        
        if 'gaussian_noise' in self.args.augmentations:
            augs['gaussian_noise'] = A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
        
        if 'salt_pepper_noise' in self.args.augmentations:
            augs['salt_pepper_noise'] = A.ISONoise(intensity=(0.1, 0.5), p=1.0)
        
        if 'random_scale' in self.args.augmentations:
            augs['random_scale'] = A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0)
        
        if 'elastic_transform' in self.args.augmentations:
            augs['elastic_transform'] = A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50, p=1.0
            )
        
        return augs

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except Exception as e:
            print(f"Error writing log: {str(e)}")  # 日志写入失败不影响主程序

    def process_image(self, image_path):
        try:
            # 再次清理图像路径（确保无空格）
            image_path = self.clean_path(image_path)
            
            # 读取图像（增强中文路径兼容性）
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                self.log(f"Failed to read image with cv2: {str(e)}")
                # 尝试用另一种方式读取（处理特殊编码的图像）
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            
            if image is None:
                self.log(f"Error: Could not read image {image_path}")
                return
            
            # 解析路径信息（优化相对路径计算）
            image_basename = os.path.basename(image_path)
            image_name = os.path.splitext(image_basename)[0]
            image_ext = os.path.splitext(image_basename)[1]
            
            # 安全计算相对路径（处理路径不一致问题）
            try:
                relative_dir = os.path.relpath(os.path.dirname(image_path), self.args.input_path)
                # 处理相对路径为"./"的情况（当前目录）
                if relative_dir == '.':
                    relative_dir = ''
            except ValueError:
                # 如果输入路径不在当前图像路径的父目录中，直接使用图像所在目录名
                relative_dir = os.path.basename(os.path.dirname(image_path))
            
            self.log(f"Processing image: {os.path.join(relative_dir, image_basename) if relative_dir else image_basename}")
            
            # 应用增强
            for aug_name, augmentation in self.augmentations.items():
                current_samples = self.aug_samples_config.get(aug_name, self.args.samples)
                if current_samples <= 0:
                    continue
                
                # 构建输出目录（使用安全创建方法）
                aug_output_dir = os.path.join(self.output_root, aug_name, relative_dir)
                self.safe_makedirs(aug_output_dir)  # 安全创建目录
                
                # 生成增强样本
                for sample_idx in range(current_samples):
                    try:
                        augmented = augmentation(image=image)
                        augmented_image = augmented["image"]
                        
                        # 生成输出文件名（避免文件名过长，简化命名）
                        # 简化文件名：原始名称_增强类型_样本序号.扩展名（时间戳已在根目录）
                        output_filename = f"{image_name}_{aug_name}_sample{sample_idx+1}{image_ext}"
                        output_path = os.path.join(aug_output_dir, output_filename)
                        output_path = self.clean_path(output_path)  # 清理输出路径
                        
                        # 保存图像（增强兼容性）
                        try:
                            _, img_encoded = cv2.imencode(image_ext, augmented_image)
                            img_encoded.tofile(output_path)
                        except Exception as e:
                            self.log(f"Failed to save with tofile: {str(e)}")
                            # 尝试用cv2.imwrite（处理特殊格式）
                            cv2.imwrite(output_path, augmented_image)
                        
                        if sample_idx == 0:
                            rel_output_path = os.path.relpath(output_path, self.output_root)
                            self.log(f"Saved {aug_name} sample (total {current_samples}) to {rel_output_path}")
                    except Exception as e:
                        self.log(f"Error generating {aug_name} sample {sample_idx+1}: {str(e)}")
                        continue  # 单个样本失败不影响其他样本
        
        except Exception as e:
            self.log(f"Error processing {image_path}: {str(e)}")

    def get_image_paths(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff',
                           '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.GIF', '*.TIFF']  # 大小写都匹配
        image_paths = []
        
        input_path = self.args.input_path
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            if any(ext == target_ext[1:].lower() for target_ext in image_extensions):
                image_paths.append(input_path)
            else:
                self.log(f"Error: {input_path} is not a supported image file")
        
        elif os.path.isdir(input_path):
            for ext in image_extensions:
                if self.args.recursive:
                    pattern = os.path.join(input_path, '**', ext)
                    found = glob.glob(pattern, recursive=True)
                else:
                    pattern = os.path.join(input_path, ext)
                    found = glob.glob(pattern)
                # 清理找到的路径，去除空路径和无效路径
                found = [self.clean_path(p) for p in found if os.path.exists(p)]
                image_paths.extend(found)
        
        else:
            self.log(f"Error: {input_path} does not exist")
        
        # 去重并排序
        image_paths = list(sorted(set(image_paths)))
        self.log(f"Found {len(image_paths)} valid images to process")
        return image_paths

    def run(self):
        image_paths = self.get_image_paths()
        
        if not image_paths:
            self.log("No images to process. Exiting.")
            return
        
        for img_idx, image_path in enumerate(image_paths, 1):
            self.log(f"\nProcessing image {img_idx}/{len(image_paths)}")
            self.process_image(image_path)
        
        self.log("\nAugmentation process completed (check logs for details)")
        self.log(f"All results saved to: {self.output_root}")
        
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Battery Defect Image Augmentation Tool (Preserve Directory Structure)')
        
        parser.add_argument('--input_path', 
                          default=r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\检测模型\原始数据集\裂纹', 
                          help='Path to input image file or directory (e.g., ./input_data)')
        parser.add_argument('--output_dir', 
                          default=r'Z:\5-标注数据\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\检测模型\原始数据集\裂纹_数据增强', 
                          help='Root directory for output augmented images (e.g., ./aug_output)')
        
        parser.add_argument('--samples', type=int, default=1, 
                          help='Default number of augmented samples to generate per image per augmentation type (default: 1)')
        parser.add_argument('--augmentations', nargs='+', 
                          choices=['rotate', 'brightness_contrast', 'horizontal_flip', 
                                   'vertical_flip', 'gaussian_blur', 'random_crop',
                                   'gaussian_noise', 'salt_pepper_noise', 'random_scale',
                                   'elastic_transform'],
                          default=['rotate', 'brightness_contrast', 'vertical_flip', 'horizontal_flip'],
                          help='List of augmentations to apply (default: rotate brightness_contrast vertical_flip horizontal_flip)')
        
        parser.add_argument('--recursive', action='store_true', default=True,
                          help='Process images in subdirectories recursively (default: Enabled)')
        
        return parser.parse_args()


if __name__ == "__main__":
    try:
        augmenter = BatteryDefectAugmenter()
        augmenter.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)