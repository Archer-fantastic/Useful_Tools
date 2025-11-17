import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from torch.nn import functional as F

# 设置中文字体，确保可视化时中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_model(model_path=None):
    """加载ResNet18模型，支持加载预训练模型或自定义训练模型"""
    model = resnet18(pretrained=False)  # 先初始化模型结构
    
    if model_path and os.path.exists(model_path):
        # 加载自定义训练的模型权重
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"成功加载自定义模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}，将使用预训练模型")
            model = resnet18(pretrained=True)
    else:
        # 使用预训练模型
        model = resnet18(pretrained=True)
        print("未提供有效模型路径，将使用PyTorch预训练ResNet18模型")
    
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(img_path, size=(224, 224)):
    """预处理图像，使其符合模型输入要求"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像文件不存在: {img_path}")
    
    # 图像预处理管道（需与训练时保持一致）
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet均值
            std=[0.229, 0.224, 0.225]    # ImageNet标准差
        )
    ])
    
    img = Image.open(img_path).convert('RGB')  # 打开图像并转为RGB
    input_tensor = transform(img).unsqueeze(0)  # 增加批量维度
    return img, input_tensor, img.size  # 返回原图尺寸用于后续上采样


def get_stage_features(model, input_tensor):
    """提取模型各stage的特征图"""
    # 用于保存各stage特征图的字典
    stage_features = {
        'stage1': [],  # layer1输出
        'stage2': [],  # layer2输出
        'stage3': [],  # layer3输出
        'stage4': []   # layer4输出
    }
    
    # 定义钩子函数：将特征图保存到字典中
    def hook_fn(name):
        def hook(module, input, output):
            stage_features[name].append(output.detach().cpu())
        return hook
    
    # 为每个stage注册钩子
    handles = [
        model.layer1.register_forward_hook(hook_fn('stage1')),
        model.layer2.register_forward_hook(hook_fn('stage2')),
        model.layer3.register_forward_hook(hook_fn('stage3')),
        model.layer4.register_forward_hook(hook_fn('stage4'))
    ]
    
    # 前向传播，触发钩子获取特征图
    with torch.no_grad():  # 关闭梯度计算，节省内存
        model(input_tensor)
    
    # 移除钩子，避免内存泄漏
    for handle in handles:
        handle.remove()
    
    return stage_features


def overlay_heatmap(original_img, feature_map, top_k=3, stage_name=""):
    """
    将特征图与原图叠加显示热力图效果
    original_img: 原始PIL图像
    feature_map: 特征图张量 [C, H, W]
    top_k: 选择响应最强的top_k个通道
    stage_name: 当前stage名称
    """
    # 将原始图像转为numpy数组
    original_np = np.array(original_img)
    orig_h, orig_w = original_np.shape[:2]
    
    # 计算每个通道的平均响应（用于选择重要通道）
    channel_scores = feature_map.mean(dim=(1, 2))
    top_indices = torch.topk(channel_scores, top_k).indices  # 响应最强的通道索引
    
    # 创建画布
    fig, axes = plt.subplots(1, top_k + 1, figsize=(5 * (top_k + 1), 5))
    fig.suptitle(f"{stage_name} 特征热力图叠加 (原图 + 响应最强的{top_k}个通道)", fontsize=16)
    
    # 显示原图
    axes[0].imshow(original_np)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 逐个处理top通道并叠加
    for i, idx in enumerate(top_indices, 1):
        # 提取单个通道特征图
        channel = feature_map[idx]
        
        # 归一化到[0, 1]
        normalized = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        
        # 上采样到原图尺寸
        upsampled = F.interpolate(
            normalized.unsqueeze(0).unsqueeze(0),  # 增加批次和通道维度
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze()  # 移除多余维度
        
        # 转换为热力图格式
        heatmap = (upsampled.numpy() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        
        # 与原图叠加（权重融合）
        overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
        
        # 显示叠加效果
        axes[i].imshow(overlay)
        axes[i].set_title(f"通道 {idx} (响应值: {channel_scores[idx]:.2f})")
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留标题空间
    plt.show()


def visualize_stage_features(original_img, stage_features, num_channels=16, top_k_overlay=3):
    """可视化所有stage的特征图及热力图叠加效果"""
    # 先显示原始图像
    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    plt.title("原始输入图像")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 逐个可视化每个stage
    for stage_name, features in stage_features.items():
        if not features:
            print(f"未获取到{stage_name}的特征图")
            continue
        
        feat_tensor = features[0][0]  # 取第一个样本的特征图 [C, H, W]
        total_channels = feat_tensor.shape[0]
        display_channels = min(num_channels, total_channels)  # 实际显示的通道数
        
        # 1. 显示单通道特征图网格
        num_cols = 8
        num_rows = (display_channels + num_cols - 1) // num_cols  # 向上取整
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 2.5 * num_rows))
        fig.suptitle(f"{stage_name} 特征图 (共{total_channels}个通道，显示前{display_channels}个)", fontsize=16)
        
        for i in range(display_channels):
            row = i // num_cols
            col = i % num_cols
            
            # 提取单个通道并归一化
            channel_data = feat_tensor[i].numpy()
            channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
            
            # 显示特征图
            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            ax.imshow(channel_data, cmap='jet')
            ax.set_title(f"通道 {i+1}")
            ax.axis('off')
        
        # 隐藏未使用的子图
        for i in range(display_channels, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            if num_rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # 2. 显示热力图叠加效果（每个stage单独展示）
        overlay_heatmap(original_img, feat_tensor, top_k=top_k_overlay, stage_name=stage_name)


def main():
    # 配置参数（请根据你的实际情况修改）
    MODEL_PATH = r"Z:\12-模型备份\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL\分类模型_划痕\20251103_113529\CYS.250804-阳极涂布机尾外观瑕疵CCD检测ATL_划痕分类_20251103_113529_C.pth"  # 你的模型权重路径
    IMG_PATH = r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\07.vis_feature\test_img2.bmp"           # 要可视化的图像路径
    DISPLAY_CHANNELS = 32                 # 每个stage显示的通道数
    IMG_SIZE = (320, 320)                 # 输入图像尺寸
    TOP_K_OVERLAY = 3                     # 每个stage显示的热力图叠加通道数（响应最强的前K个）
    
    # 步骤1：加载模型
    model = load_model(MODEL_PATH)
    
    # 步骤2：预处理图像
    try:
        original_img, input_tensor, _ = preprocess_image(IMG_PATH, size=IMG_SIZE)
    except Exception as e:
        print(f"图像处理失败: {e}")
        return
    
    # 步骤3：提取各stage特征图
    print("正在提取特征图...")
    stage_features = get_stage_features(model, input_tensor)
    
    # 步骤4：可视化特征图及热力图叠加
    print("正在可视化特征图...")
    visualize_stage_features(
        original_img, 
        stage_features, 
        num_channels=DISPLAY_CHANNELS,
        top_k_overlay=TOP_K_OVERLAY
    )
    
    print("特征图可视化完成！")


if __name__ == "__main__":
    main()