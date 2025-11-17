import os
import argparse
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from glob import glob
from tqdm import tqdm
from datetime import datetime

def cv2_imread(img_path):
    """读取含中文路径的图像"""
    try:
        with open(img_path, 'rb') as f:
            img_data = f.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取图像失败：{e}")
        return None

def cv2_imwrite(save_path, img):
    """保存图像到含中文的路径"""
    try:
        ext = os.path.splitext(save_path)[1].lower() or '.jpg'
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext in ['.jpg', '.jpeg'] else \
                      [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        
        result, buf = cv2.imencode(ext, img, encode_param)
        if result:
            with open(save_path, 'wb') as f:
                f.write(buf)
            return True
        else:
            print(f"图像编码失败：{save_path}")
            return False
    except Exception as e:
        print(f"保存图像失败 {save_path}：{e}")
        return False

class FeatureExtractor:
    def __init__(self, model_name='resnet18', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化特征提取器，支持多种预训练模型"""
        self.device = device
        self.model_name = model_name.lower()
        
        # 加载对应模型并移除分类层
        if self.model_name == 'mobilenet':
            self.model = torchvision.models.mobilenet_v2(pretrained=True)
            # MobileNetV2的特征提取部分到最后一个卷积块
            self.model = torch.nn.Sequential(*list(self.model.features) + [
                torch.nn.AdaptiveAvgPool2d((1, 1))  # 增加全局池化层
            ])
            self.feature_dim = 1280  # MobileNetV2特征维度
        
        elif self.model_name == 'resnet34':
            self.model = torchvision.models.resnet34(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 512  # ResNet34特征维度
        
        elif self.model_name == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048  # ResNet50特征维度
        
        elif self.model_name == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 512  # ResNet18特征维度
        
        else:
            raise ValueError(f"不支持的模型：{model_name}，可选：mobilenet, resnet18, resnet34, resnet50")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 通用图像预处理（符合所有模型的输入要求）
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"特征提取器初始化完成：{model_name}，使用设备：{self.device}，特征维度：{self.feature_dim}")

    def extract(self, img_path):
        """提取单张图像的特征"""
        try:
            img = cv2_imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像：{img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.model(img_tensor)
            
            return feature.squeeze().cpu().numpy()
        
        except Exception as e:
            print(f"提取特征失败 {img_path}：{str(e)}")
            return None

class ImageCluster:
    def __init__(self, extractor, cluster_method='kmeans', n_clusters=5, eps=0.5, min_samples=5):
        self.extractor = extractor
        self.cluster_method = cluster_method
        self.features = []
        self.image_paths = []
        
        if cluster_method == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
            print(f"使用KMeans聚类，类别数：{n_clusters}")
        elif cluster_method == 'dbscan':
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
            print(f"使用DBSCAN聚类，eps={eps}, min_samples={min_samples}")
        else:
            raise ValueError("聚类方式仅支持 'kmeans' 或 'dbscan'")

    def load_images(self, folder_path, recursive=False):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"文件夹不存在：{folder_path}")
        
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
        img_paths = []
        
        for ext in img_extensions:
            if recursive:
                img_paths.extend(glob(os.path.join(folder_path, '**', ext), recursive=True))
            else:
                img_paths.extend(glob(os.path.join(folder_path, ext)))
        
        if not img_paths:
            print(f"警告：未在 {folder_path} 中找到图像")
            return
        
        print(f"发现 {len(img_paths)} 张图像，开始提取特征...")
        
        for img_path in tqdm(img_paths, desc="特征提取进度"):
            feature = self.extractor.extract(img_path)
            if feature is not None:
                self.features.append(feature)
                self.image_paths.append(img_path)
        
        print(f"成功提取 {len(self.features)} 张图像的特征")

    def cluster(self, save_dir=None):
        if len(self.features) < 2:
            print("特征数量不足，无法进行聚类")
            return
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        print(f"开始{self.cluster_method}聚类...")
        labels = self.model.fit_predict(features_scaled)
        
        unique_labels = np.unique(labels)
        print(f"聚类完成，共得到 {len(unique_labels)} 个类别")
        for label in unique_labels:
            count = np.sum(labels == label)
            print(f"类别 {label}：{count} 张图像")
        
        if save_dir:
            def get_unique_path(base_path):
                if not os.path.exists(base_path):
                    return base_path
                counter = 1
                name, ext = os.path.splitext(base_path)
                while os.path.exists(f"{name}_{counter}{ext}"):
                    counter += 1
                return f"{name}_{counter}{ext}"
            
            for label in unique_labels:
                label_dir = os.path.join(save_dir, f"cluster_{label}")
                os.makedirs(label_dir, exist_ok=True)
                
                for i, img_path in enumerate(self.image_paths):
                    if labels[i] == label:
                        try:
                            img = cv2_imread(img_path)
                            if img is None:
                                continue
                            
                            img_name = os.path.basename(img_path)
                            save_path = os.path.join(label_dir, img_name)
                            save_path = get_unique_path(save_path)
                            
                            if not cv2_imwrite(save_path, img):
                                print(f"警告：无法保存 {img_path} 到 {save_path}")
                        except Exception as e:
                            print(f"处理图像失败 {img_path}：{str(e)}")
            
            print(f"聚类结果已保存至：{save_dir}")
        
        return {
            "labels": labels,
            "image_paths": self.image_paths,
            "unique_labels": unique_labels
        }

def parse_args():
    parser = argparse.ArgumentParser(description='基于多种预训练模型的图像聚类脚本')
    parser.add_argument('--folder', type=str, default=r'D:\Min\Projects\VSCodeProjects\dataset\cls_OK-裂缝-露箔-褶皱\cluster_data',
                        help='图像文件夹路径')
    parser.add_argument('--recursive', action='store_true',
                        help='是否递归读取子文件夹（默认True）')
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['mobilenet', 'resnet18', 'resnet34', 'resnet50'],
                        help='特征提取模型（默认resnet18）')
    parser.add_argument('--method', type=str, default='kmeans', choices=['kmeans', 'dbscan'],
                        help='聚类方法（kmeans或dbscan，默认kmeans）')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='KMeans聚类类别数（默认8）')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='DBSCAN的eps参数（默认0.5）')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='DBSCAN的min_samples参数（默认5）')
    parser.add_argument('--save_dir', type=str, default=r'D:\Min\Projects\VSCodeProjects\dataset\cls_OK-裂缝-露箔-褶皱\cluster_results',
                        help='聚类结果保存根目录（默认cluster_results）')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 生成带时间戳和模型名的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.model}_clusters_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存至：{save_dir}")
    
    # 初始化特征提取器（指定模型）
    extractor = FeatureExtractor(model_name=args.model)
    cluster = ImageCluster(
        extractor,
        cluster_method=args.method,
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    cluster.load_images(args.folder, recursive=args.recursive)
    cluster.cluster(save_dir=save_dir)

if __name__ == '__main__':
    main()