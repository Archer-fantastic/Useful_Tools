import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from pathlib import Path
from datetime import datetime  # 用于生成时间戳

# 确保中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ------------------------------
# 自定义数据集
# ------------------------------
class RecursiveImageDataset(Dataset):
    """用于训练/验证集：按类别文件夹递归读取图像"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []  # (图像路径, 类别索引)
        self.classes = []  # 类别名称
        
        for cls_idx, cls_name in enumerate(sorted(os.listdir(root_dir))):
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_path):
                continue
            self.classes.append(cls_name)
            
            # 递归扫描子文件夹
            for root, _, files in os.walk(cls_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        img_path = os.path.join(root, file)
                        self.images.append((img_path, cls_idx))
        
        print(f"加载数据集：{root_dir}，类别数：{len(self.classes)}，图像总数：{len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, img_path


class ImagePredictDataset(Dataset):
    """用于预测：支持单张图片或文件夹递归读取"""
    def __init__(self, path, transform=None):
        self.transform = transform
        self.images = []  # 存储图像路径
        
        # 判断输入是文件还是文件夹
        if os.path.isfile(path):
            # 单张图片
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                self.images.append(path)
        else:
            # 文件夹递归扫描
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        img_path = os.path.join(root, file)
                        self.images.append(img_path)
        
        print(f"加载待预测图像：{path}，图像总数：{len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path


# ------------------------------
# 数据预处理
# ------------------------------
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

# ------------------------------
# 损失函数定义
# ------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        features = F.normalize(features)
        weight = F.normalize(self.weight)
        cos_theta = F.linear(features, weight).clamp(-1, 1)
        theta = torch.acos(cos_theta)
        
        target_mask = torch.zeros_like(cos_theta)
        target_mask.scatter_(1, labels.view(-1, 1).long(), 1)
        selected = theta[target_mask.bool()]
        theta[target_mask.bool()] = selected + self.m
        
        cos_theta_m = torch.cos(theta) * self.s
        return F.cross_entropy(cos_theta_m, labels)

# ------------------------------
# 模型定义
# ------------------------------
class FeatureResNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=None)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=None)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            self.feature_dim = 2048
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.backbone.fc = nn.Identity()  # 移除原fc层
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # 特征向量（用于比对）
        logits = self.classifier(features)  # 分类头输出
        return logits, features

# ------------------------------
# 训练函数（仅在最后一轮保存一个模型）
# ------------------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载训练/验证集
    train_transform, val_test_transform = get_transforms()
    total_dataset = RecursiveImageDataset(config.train_data_dir, transform=train_transform)
    num_classes = len(total_dataset.classes)
    if num_classes == 0:
        print("错误：训练集未找到任何类别文件夹！")
        return
    
    # 划分训练集和验证集（按配置比例）
    val_size = int(config.val_split_ratio * len(total_dataset))
    train_size = len(total_dataset) - val_size
    train_dataset, val_dataset = random_split(
        total_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_test_transform

    # 数据加载器（使用配置的批处理大小）
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers, pin_memory=True
    )

    # 初始化模型和损失函数
    model = FeatureResNet(config.model, num_classes).to(device)
    
    if config.loss == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    elif config.loss == 'focal':
        criterion = FocalLoss().to(device)
    elif config.loss == 'triplet':
        criterion = TripletLoss().to(device)
    elif config.loss == 'arcface':
        criterion = ArcFaceLoss(model.feature_dim, num_classes).to(device)
    else:
        raise ValueError(f"不支持的损失函数: {config.loss}")

    # 优化器（使用配置的学习率）
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.scheduler_step_size, 
        gamma=config.scheduler_gamma
    )

    # 训练记录
    train_losses, val_accs = [], []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for inputs, labels, _ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if config.loss == 'triplet':
                batch_size = inputs.shape[0]
                if batch_size < 2:
                    continue
                anchor = inputs
                positive_idx = torch.randint(0, batch_size, (batch_size,))
                positive = inputs[positive_idx]
                negative_mask = labels != labels[positive_idx]
                if not negative_mask.any():
                    continue
                negative = inputs[torch.where(negative_mask)[0][0]].unsqueeze(0)
                
                _, anchor_feat = model(anchor)
                _, positive_feat = model(positive)
                _, negative_feat = model(negative)
                loss = criterion(anchor_feat, positive_feat, negative_feat)
            else:
                logits, features = model(inputs)
                loss = criterion(features, labels) if config.loss == 'arcface' else criterion(logits, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _ = model(inputs)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1} - 训练损失: {train_loss:.4f}, 验证准确率: {val_acc:.4f}")

        scheduler.step()

    # 训练结束后只保存一次模型（带时间戳）
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': total_dataset.classes,
        'final_val_acc': val_accs[-1],  # 记录最终验证准确率
        'timestamp': config.timestamp,
        'config': vars(config)  # 保存配置参数，方便回溯
    }, config.model_path)
    print(f"训练完成，模型保存至 {config.model_path}")

    # 绘制训练曲线（保存到模型同目录）
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='验证准确率', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    curve_path = os.path.join(os.path.dirname(config.model_path), f"train_curve_{config.timestamp}.png")
    plt.savefig(curve_path)
    print(f"训练曲线已保存至 {curve_path}")

# ------------------------------
# 构建特征库（自动加载最新模型）
# ------------------------------
def build_feature_database():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 自动获取最新模型路径
    latest_model_path = config.get_latest_model()
    if not latest_model_path:
        print("错误：未找到任何模型文件，请先训练模型！")
        return
    
    checkpoint = torch.load(latest_model_path, map_location=device, weights_only=True)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    model = FeatureResNet(config.model, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"加载最新模型: {latest_model_path}, 类别: {classes}")

    _, val_test_transform = get_transforms()
    dataset = RecursiveImageDataset(config.train_data_dir, transform=val_test_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers
    )

    feature_dict = {cls: [] for cls in classes}
    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader, desc="提取特征"):
            inputs = inputs.to(device)
            _, features = model(inputs)
            features = F.normalize(features).cpu().numpy()
            
            for feat, label_idx in zip(features, labels.numpy()):
                cls_name = classes[label_idx]
                feature_dict[cls_name].append(feat)

    # 过滤空类别
    feature_dict = {k: v for k, v in feature_dict.items() if len(v) > 0}
    if not feature_dict:
        print("错误：特征库为空，未提取到任何有效特征！")
        return

    avg_features = {cls: np.mean(feats, axis=0) for cls, feats in feature_dict.items()}
    os.makedirs(os.path.dirname(config.db_path), exist_ok=True)
    np.save(config.db_path, avg_features)
    print(f"特征库已保存至 {config.db_path}")

# ------------------------------
# 预测函数（自动加载最新模型和特征库）
# ------------------------------
def predict(input_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 自动获取最新模型路径
    latest_model_path = config.get_latest_model()
    if not latest_model_path:
        print("错误：未找到任何模型文件，请先训练模型！")
        return
    
    checkpoint = torch.load(latest_model_path, map_location=device, weights_only=True)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    model = FeatureResNet(config.model, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载与最新模型匹配的特征库
    if not os.path.exists(config.db_path):
        print(f"错误：未找到与最新模型匹配的特征库 {config.db_path}，请先运行build模式！")
        return
    avg_features = np.load(config.db_path, allow_pickle=True).item()
    print(f"加载特征库: {config.db_path}, 类别: {list(avg_features.keys())}")

    # 加载待预测图像（单张或文件夹）
    _, val_test_transform = get_transforms()
    test_dataset = ImagePredictDataset(input_path, transform=val_test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers
    )

    # 创建带时间戳的保存目录（按类别分文件夹）
    save_root = config.predict_save_dir
    for cls in classes:
        os.makedirs(os.path.join(save_root, cls), exist_ok=True)
    print(f"预分类结果将保存至：{save_root}")

    # 批量预测并保存
    results = []
    with torch.no_grad():
        for inputs, img_paths in tqdm(test_loader, desc="预测中"):
            inputs = inputs.to(device)
            _, features = model(inputs)
            features = F.normalize(features).cpu().numpy()  # 归一化特征
            
            for feat, img_path in zip(features, img_paths):
                img_name = os.path.basename(img_path)
                # 计算与每个类别的余弦距离
                distances = {}
                for cls in classes:
                    avg_feat = avg_features[cls]
                    cos_sim = np.dot(feat, avg_feat) / (np.linalg.norm(feat) * np.linalg.norm(avg_feat))
                    distances[cls] = 1 - cos_sim  # 距离越小越相似
                
                # 排序并确定预测类别
                sorted_dist = sorted(distances.items(), key=lambda x: x[1])
                pred_cls = sorted_dist[0][0]
                results.append({
                    'image_path': img_path,
                    'image_name': img_name,
                    'predicted_class': pred_cls,
                    'distances': sorted_dist
                })

                # 终端输出文件名及各类别距离
                print(f"\n文件名: {img_name}")
                print("类别距离（越小越相似）:")
                for cls, dist in sorted_dist:
                    print(f"  {cls}: {dist:.4f}")
                print(f"预测类别: {pred_cls}")

                # 复制图像到对应类别文件夹
                save_path = os.path.join(save_root, pred_cls, img_name)
                # 避免文件名冲突
                if os.path.exists(save_path):
                    name, ext = os.path.splitext(img_name)
                    save_path = os.path.join(save_root, pred_cls, f"{name}_copy{ext}")
                shutil.copy(img_path, save_path)

    # 保存预测结果日志（带时间戳）
    log_file = os.path.join(save_root, f"prediction_log_{config.timestamp}.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("图像路径\t图像名称\t预测类别\t最小距离\n")
        for res in results:
            f.write(f"{res['image_path']}\t{res['image_name']}\t{res['predicted_class']}\t{res['distances'][0][1]:.4f}\n")
    
    print(f"\n预测完成！共处理 {len(results)} 张图像，结果日志：{log_file}")


# ------------------------------
# 全局配置（包含自动查找最新模型功能）
# ------------------------------
class Config:
    # 模型与损失函数配置（三种模式共享）
    model = "resnet18"  # 支持 resnet18/resnet34/resnet50
    loss = "arcface"         # 支持 ce/focal/triplet/arcface
    
    # 训练参数配置
    epochs = 10                # 训练轮次
    batch_size = 32            # 批处理大小
    learning_rate = 1e-4       # 初始学习率
    weight_decay = 1e-5        # 权重衰减（正则化）
    val_split_ratio = 0.2      # 训练集/验证集划分比例（验证集占比）
    num_workers = 4            # 数据加载线程数
    scheduler_step_size = 10   # 学习率调度器步长
    scheduler_gamma = 0.5      # 学习率衰减系数
    
    # 路径配置（根据实际情况修改）
    train_data_dir = r"D:\Min\Projects\VSCodeProjects\dataset\feature_cls_data\train_data"  # 训练数据集目录
    model_save_root = r"D:\Min\Projects\VSCodeProjects\dataset\feature_cls_data\models"    # 模型保存根目录
    db_save_root = r"D:\Min\Projects\VSCodeProjects\dataset\feature_cls_data\feature_database"  # 特征库保存根目录
    predict_save_root = r"D:\Min\Projects\VSCodeProjects\dataset\feature_cls_data\prediction_results"  # 预测结果保存根目录
    test_data_dir = r'D:\Min\Projects\VSCodeProjects\dataset\feature_cls_data\test_data'  # 默认测试数据路径

    # 时间戳（自动生成，用于区分不同训练结果）
    @property
    def timestamp(self):
        """生成格式为YYYYMMDD_HHMMSS的时间戳"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 派生路径（带时间戳，自动生成）
    @property
    def model_path(self):
        """训练好的模型路径（带时间戳）"""
        return os.path.join(self.model_save_root, f"{self.model}_{self.loss}_{self.timestamp}.pth")
    
    @property
    def db_path(self):
        """特征库文件路径（带时间戳，与最新模型匹配）"""
        latest_model = self.get_latest_model()
        if latest_model:
            # 从最新模型文件名提取时间戳
            timestamp = latest_model.split(f"{self.model}_{self.loss}_")[-1].split(".pth")[0]
            return os.path.join(self.db_save_root, f"{self.model}_{self.loss}_{timestamp}_feature_db.npy")
        return os.path.join(self.db_save_root, f"{self.model}_{self.loss}_{self.timestamp}_feature_db.npy")
    
    @property
    def predict_save_dir(self):
        """预测结果保存目录（带时间戳）"""
        return os.path.join(self.predict_save_root, f"prediction_{self.timestamp}")
    
    def get_latest_model(self):
        """自动查找模型保存目录下最新生成的模型文件"""
        if not os.path.exists(self.model_save_root):
            return None
        # 筛选符合命名规则的模型文件（模型名_损失函数_时间戳.pth）
        model_pattern = f"{self.model}_{self.loss}_*.pth"
        model_files = list(Path(self.model_save_root).glob(model_pattern))
        if not model_files:
            return None
        # 按文件创建时间排序，返回最新的模型路径
        latest_model = max(model_files, key=lambda x: x.stat().st_ctime)
        return str(latest_model)

# 实例化配置（全局使用）
config = Config()


# ------------------------------
# 主函数
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="基于ResNet的特征分类器（支持单张/文件夹预测）")
    parser.add_argument('mode', choices=['train', 'build', 'predict'], 
                       help='运行模式：train（训练）/ build（构建特征库）/ predict（预测）')
    parser.add_argument('--input', type=str, 
                       help='预测模式时的输入路径（单张图片或文件夹，仅predict模式需要）')

    args = parser.parse_args()

    # 根据模式调用对应功能
    if args.mode == 'train':
        print("===== 开始训练 =====")
        # 打印当前训练配置
        print("训练配置:")
        print(f"  模型: {config.model}")
        print(f"  损失函数: {config.loss}")
        print(f"  训练轮次: {config.epochs}")
        print(f"  批处理大小: {config.batch_size}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  验证集比例: {config.val_split_ratio}")
        train()
    elif args.mode == 'build':
        print("===== 开始构建特征库 =====")
        build_feature_database()
    elif args.mode == 'predict':
        if not args.input:
            # 未指定输入路径时使用配置中的默认测试路径
            input_path = config.test_data_dir
            print(f"未指定输入路径，使用默认测试路径：{input_path}")
        else:
            input_path = args.input
        if not os.path.exists(input_path):
            print(f"错误：输入路径不存在 - {input_path}")
            return
        print(f"===== 开始预测（输入：{input_path}） =====")
        predict(input_path)


if __name__ == '__main__':
    import sys
    # 如果没有命令行参数，自动使用预设参数运行
    if len(sys.argv) == 1:
        # 预设模式：可根据需要切换为'train'/'build'/'predict'
        preset_mode = 'predict'  # 这里默认是训练模式，可修改
        
        # 根据预设模式设置参数
        if preset_mode == 'train':
            sys.argv.extend(['train'])
        elif preset_mode == 'build':
            sys.argv.extend(['build'])
        elif preset_mode == 'predict':
            # 预测模式使用配置中的默认测试路径
            sys.argv.extend([
                'predict',
                '--input', config.test_data_dir
            ])
    
    main()