import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from collections import defaultdict

# 确保中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


# ------------------------------
# 自定义数据集（保持不变）
# ------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None, dataset_mode=1, cache_dataset=False):
        self.root = root
        self.transform = transform
        self.dataset_mode = dataset_mode
        self.cache_dataset = cache_dataset
        self.images = []  # (图像路径/缓存的图像数据, 类别名)
        self.classes = []  # 类别列表
        self.class_to_idx = {}  # 类别到索引的映射

        self._scan_and_cache_images()
        self.classes = sorted(list(self.class_to_idx.keys()))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = [(p, self.class_to_idx[cls]) for p, cls in self.images]

        print(f"加载数据集完成：共 {len(self.classes)} 个类别，{len(self.images)} 张图像")
        for cls in self.classes:
            count = sum(1 for _, c in self.images if self.classes[c] == cls)
            print(f"  {cls}: {count} 张")

    def _scan_and_cache_images(self):
        for dirpath, _, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    img_path = os.path.join(dirpath, filename)
                    
                    if self.dataset_mode == 1:
                        rel_path = os.path.relpath(dirpath, self.root)
                        cls_name = rel_path.split(os.sep)[0]
                    else:
                        cls_name = os.path.basename(dirpath)
                    
                    if self.cache_dataset:
                        try:
                            img = Image.open(img_path).convert('RGB')
                            self.images.append((img, cls_name))
                        except Exception as e:
                            print(f"警告：无法缓存图像 {img_path}，错误：{str(e)}")
                            self.images.append((img_path, cls_name))
                    else:
                        self.images.append((img_path, cls_name))
                    
                    if cls_name not in self.class_to_idx:
                        self.class_to_idx[cls_name] = len(self.class_to_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        data, label = self.images[idx]
        if self.cache_dataset and isinstance(data, Image.Image):
            img = data.copy()
        else:
            img = Image.open(data).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        return img, label, data if isinstance(data, str) else "cached_image"


# ------------------------------
# 模型构建工具（保持不变）
# ------------------------------
def build_model(model_name, num_classes, pretrained=True):
    model_name = model_name.lower()
    
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    else:
        raise ValueError(f"不支持的模型：{model_name}")
    
    return model


# ------------------------------
# 新增：ONNX模型导出功能
# ------------------------------
def export_onnx(model, save_path, input_shape, device):
    """
    将PyTorch模型导出为ONNX格式
    
    参数:
        model: 训练好的PyTorch模型
        save_path: ONNX模型保存路径
        input_shape: 输入形状 (C, H, W)，如(3, 320, 320)
        device: 设备（cpu或cuda）
    """
    # 切换到评估模式
    model.eval()
    
    # 创建一个随机输入张量（用于追踪模型）
    dummy_input = torch.randn(1, *input_shape).to(device)  # 批量大小为1
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,  # 导出训练好的参数
        opset_version=11,   # ONNX版本（11较为通用）
        do_constant_folding=True,  # 折叠常量以优化
        input_names=['input'],    # 输入节点名称
        output_names=['output'],  # 输出节点名称（分类概率）
        dynamic_axes={            # 支持动态批量大小
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # 验证ONNX模型（可选）
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX模型验证成功：{save_path}")
    except Exception as e:
        print(f"ONNX模型验证警告：{str(e)}")


# ------------------------------
# 训练评估工具（保持不变）
# ------------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device, classes):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    pbar = tqdm(train_loader, desc="训练")
    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            cls_name = classes[label]
            class_total[cls_name] += 1
            if label == pred:
                class_correct[cls_name] += 1
        
        pbar.set_postfix(loss=loss.item())

    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "class_correct": class_correct,
        "class_total": class_total
    }


def evaluate(model, val_loader, criterion, device, classes):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="验证")
        for inputs, labels, _ in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                cls_name = classes[label]
                class_total[cls_name] += 1
                if label == pred:
                    class_correct[cls_name] += 1

    epoch_loss = total_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "class_correct": class_correct,
        "class_total": class_total
    }


# ------------------------------
# 主训练函数（增加ONNX导出）
# ------------------------------
def train(config):
    # 设备配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用CUDA加速：{torch.cuda.get_device_name(0)}")
        print(f"CUDA内存：总容量 {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    else:
        device = torch.device("cpu")
        print("警告：未检测到CUDA设备，将使用CPU训练（速度较慢）")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config.save_root, f"{config.model}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"训练结果将保存至: {save_dir}")

    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(config.rotation_degree),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    full_dataset = CustomImageDataset(
        root=config.data_root,
        transform=train_transform,
        dataset_mode=config.dataset_mode,
        cache_dataset=config.cache_dataset
    )
    classes = full_dataset.classes
    num_classes = len(classes)

    # 划分训练集和验证集
    val_size = int(config.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    val_dataset.dataset.transform = val_transform

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0
    )

    # 构建模型
    model = build_model(
        model_name=config.model,
        num_classes=num_classes,
        pretrained=config.pretrained
    ).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step,
        gamma=config.scheduler_gamma
    )

    # 训练记录
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    best_val_acc = 0.0
    early_stop_counter = 0

    # 保存配置
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    # 开始训练
    for epoch in range(config.epochs):
        print(f"\n===== Epoch {epoch+1}/{config.epochs} =====")
        
        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, classes)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        
        # 打印训练集每类准确率
        print("训练集每类准确率：")
        for cls in classes:
            correct = train_metrics["class_correct"].get(cls, 0)
            total = train_metrics["class_total"].get(cls, 0)
            acc = correct / total if total > 0 else 0.0
            print(f"  {cls}: {correct}/{total} ({acc:.4f})")
        print(f"训练集总准确率：{train_metrics['acc']:.4f}，损失：{train_metrics['loss']:.4f}")

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device, classes)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        
        # 打印验证集每类准确率
        print("验证集每类准确率：")
        for cls in classes:
            correct = val_metrics["class_correct"].get(cls, 0)
            total = val_metrics["class_total"].get(cls, 0)
            acc = correct / total if total > 0 else 0.0
            print(f"  {cls}: {correct}/{total} ({acc:.4f})")
        print(f"验证集总准确率：{val_metrics['acc']:.4f}，损失：{val_metrics['loss']:.4f}")

        # 学习率调度
        scheduler.step()
        print(f"当前学习率：{scheduler.get_last_lr()[0]:.6f}")

        # 早停策略
        is_improved = val_metrics["acc"] > best_val_acc + config.early_stop_min_delta
        if is_improved:
            best_val_acc = val_metrics["acc"]
            early_stop_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"保存最佳模型（验证准确率：{best_val_acc:.4f}）")
        else:
            early_stop_counter += 1
            print(f"早停计数器：{early_stop_counter}/{config.early_stop_patience}")
            if early_stop_counter >= config.early_stop_patience:
                print(f"\n早停触发！验证集连续{config.early_stop_patience}轮未提升")
                print(f"最佳验证准确率：{best_val_acc:.4f}（第{epoch + 1 - config.early_stop_patience}轮）")
                break

    # 保存最后一轮模型
    if early_stop_counter < config.early_stop_patience:
        torch.save({
            "epoch": config.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": history["val_acc"][-1]
        }, os.path.join(save_dir, "last_model.pth"))

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="训练损失")
    plt.plot(history["val_loss"], label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="训练准确率")
    plt.plot(history["val_acc"], label="验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "train_curve.png"))
    # print(f"训练曲线已保存至 {save_dir}/train_curve.png")

    # 保存训练历史
    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # ------------------------------
    # 新增：训练结束后导出ONNX模型
    # ------------------------------
    print("\n开始导出ONNX模型...")
    # 加载最佳模型权重
    best_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        # 重建模型结构
        best_model = build_model(
            model_name=config.model,
            num_classes=num_classes,
            pretrained=False
        ).to(device)
        # 加载权重
        checkpoint = torch.load(best_model_path, map_location=device)
        best_model.load_state_dict(checkpoint["model_state_dict"])
        # 导出ONNX（输入形状为3通道+配置的图像尺寸）
        onnx_save_path = os.path.join(save_dir, "best_model.onnx")
        export_onnx(
            model=best_model,
            save_path=onnx_save_path,
            input_shape=(3, config.img_size, config.img_size),  # 与训练输入一致
            device=device
        )
        print(f"ONNX模型已导出至：{onnx_save_path}")
    else:
        print("警告：未找到最佳模型权重，无法导出ONNX")

    print(f"训练完成！最佳验证准确率：{best_val_acc:.4f}")


# ------------------------------
# 配置类（保持不变）
# ------------------------------
class Config:
    # 数据集配置
    data_root = r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm"
    dataset_mode = 2  # 1或2（两种数据集模式）
    val_split = 0.2   # 验证集比例
    cache_dataset = True  # 是否缓存图像到内存

    # 模型配置
    model = "resnet18"  # 模型名称
    pretrained = True   # 是否使用预训练权重

    # 训练参数
    img_size = 320      # 输入图像尺寸（会影响ONNX导出的输入形状）
    epochs = 30         # 训练轮次
    batch_size = 16     # 批处理大小
    learning_rate = 1e-3  # 初始学习率
    weight_decay = 1e-5  # 权重衰减
    rotation_degree = 15  # 随机旋转角度
    seed = 42           # 随机种子

    # 学习率调度器
    scheduler_step = 10  # 学习率衰减步长
    scheduler_gamma = 0.5  # 学习率衰减系数

    # 早停策略参数
    early_stop_patience = 10  # 连续多少轮无提升则早停
    early_stop_min_delta = 1e-4  # 最小提升幅度

    # 其他配置
    num_workers = 4     # 数据加载线程数
    save_root = r"D:\Min\Projects\VSCodeProjects\dataset\cls_CYS250804阳极涂布机尾外观瑕疵CCD检测_测试_lxm\train_res"  # 训练结果保存根目录


# ------------------------------
# 主程序入口
# ------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("="*50)
        print("警告：未检测到可用的CUDA设备，训练将非常缓慢！")
        print("建议：安装NVIDIA显卡驱动和CUDA工具包以加速训练。")
        print("="*50)
    
    # 安装onnx（如果需要自动验证ONNX模型）
    try:
        import onnx
    except ImportError:
        print("提示：未安装onnx，将跳过ONNX模型验证。可通过 'pip install onnx' 安装。")
    
    config = Config()
    train(config)