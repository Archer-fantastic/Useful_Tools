# 图像分类模型训练框架

## 项目概述
这是一个基于PyTorch的通用图像分类训练框架，支持多种主流预训练模型，提供完整的训练流程管理（数据加载、模型构建、训练评估、结果保存），并包含ONNX模型导出功能，适用于各类图像分类任务。

## 核心功能
1. **灵活的数据集处理**：支持两种目录结构的数据集加载，可选择图像缓存到内存加速训练
2. **多模型支持**：集成ResNet、MobileNet、EfficientNet等系列预训练模型
3. **完整训练流程**：包含数据增强、训练/验证集划分、损失计算、优化器调度
4. **训练监控**：实时输出每类准确率、总准确率和损失值，支持早停策略
5. **结果保存**：自动保存最佳模型、训练日志、配置参数和类别信息
6. **模型导出**：训练完成后自动将最佳模型导出为ONNX格式，便于部署

## 主要组件说明

### 1. 自定义数据集（CustomImageDataset）
- 支持两种数据集目录模式：
  - 模式1：根目录下为类别文件夹，子文件夹包含该类图像
  - 模式2：多级目录结构，以直接包含图像的文件夹作为类别
- 支持图像缓存功能（`cache_dataset=True`），将图像预加载到内存减少IO操作
- 自动统计并显示每个类别的图像数量
- 支持图像变换（transform）应用

### 2. 模型构建工具（build_model）
支持的预训练模型包括：
- ResNet系列：resnet18、resnet34、resnet50
- MobileNet系列：mobilenet_v2、mobilenet_v3_small、mobilenet_v3_large
- EfficientNet系列：efficientnet_b0、efficientnet_b4
- 自动替换最后一层全连接层以适配目标类别数

### 3. 训练与评估工具
- `train_one_epoch`：单轮训练函数，返回训练损失、总准确率及每类准确率
- `evaluate`：验证函数，返回验证损失、总准确率及每类准确率
- 训练过程中使用tqdm显示进度条，实时更新当前批次损失

### 4. ONNX模型导出（export_onnx）
- 将训练好的PyTorch模型导出为ONNX格式
- 支持动态批量大小（batch_size）
- 自动验证导出的ONNX模型有效性（需安装onnx库）
- 导出的模型输入形状与训练配置保持一致

## 使用方法

### 1. 环境准备
安装依赖库：
```bash
pip install torch torchvision sklearn matplotlib pillow tqdm onnx  # onnx为可选
```

### 2. 配置参数
修改`Config`类中的参数以适应你的任务：
```python
class Config:
    # 数据集配置
    data_root = "你的数据集路径"       # 数据集根目录
    dataset_mode = 2                   # 选择数据集目录模式（1或2）
    val_split = 0.2                    # 验证集占比（如0.2表示20%数据作为验证集）
    cache_dataset = True               # 是否缓存图像到内存（大数据集建议设为False）

    # 模型配置
    model = "resnet18"                 # 选择模型名称
    pretrained = True                  # 是否使用预训练权重（建议设为True加速收敛）

    # 训练参数
    img_size = 320                     # 输入图像尺寸（宽高一致）
    epochs = 30                        # 最大训练轮次
    batch_size = 16                    # 批处理大小（根据GPU内存调整）
    learning_rate = 1e-3               # 初始学习率
    weight_decay = 1e-5                # 权重衰减（防止过拟合）
    rotation_degree = 15               # 数据增强中的随机旋转角度
    seed = 42                          # 随机种子（保证结果可复现）

    # 学习率调度器
    scheduler_step = 10                # 学习率衰减间隔（每10轮衰减一次）
    scheduler_gamma = 0.5              # 学习率衰减系数（每次衰减为当前的50%）

    # 早停策略
    early_stop_patience = 10           # 连续多少轮无提升则停止训练
    early_stop_min_delta = 1e-4        # 认为有提升的最小准确率变化值

    # 其他配置
    num_workers = 4                    # 数据加载线程数（建议设为CPU核心数）
    save_root = "训练结果保存路径"      # 模型和日志的保存根目录
```

### 3. 启动训练
直接运行脚本即可开始训练：
```bash
python 脚本文件名.py
```

## 输出内容说明
训练过程中会在`save_root`下创建以`模型名_时间戳`命名的文件夹，包含以下内容：
1. `best_model.pth`：验证集准确率最高的模型权重
2. `last_model.pth`：最后一轮训练的模型权重（若未触发早停）
3. `best_model.onnx`：导出的ONNX格式模型
4. `config.json`：训练时使用的完整配置参数
5. `classes.txt`：数据集中的类别列表
6. `history.json`：训练过程中的损失和准确率记录
7. 控制台输出：包含每轮训练的详细指标（总准确率、每类准确率、损失值等）

## 注意事项
- 若数据集较大，建议将`cache_dataset`设为False，避免内存不足
- 可根据GPU内存大小调整`batch_size`，内存不足时适当减小
- 若未安装onnx库，将跳过ONNX模型验证，但仍会导出模型文件
- 训练前会自动检测CUDA设备，若无GPU将使用CPU训练（速度较慢）

## 扩展建议
- 可通过修改`build_model`函数添加新的模型架构
- 如需更多数据增强方式，可扩展`train_transform`和`val_transform`
- 可根据需求调整学习率调度策略或优化器类型