# FeatCls：基于特征匹配的图像分类工具

FeatCls 是一个基于 ResNet 模型的图像分类工具，通过提取图像特征并构建特征库，实现高效的图像类别匹配与预测。工具支持自定义训练、特征库构建和批量预测，所有输出文件自动添加时间戳，避免误操作覆盖。


## 项目功能

1. **模型训练**：使用自定义数据集训练 ResNet 模型（支持 ResNet18/34/50），支持多种损失函数（交叉熵、Focal Loss、Triplet Loss 等）。
2. **特征库构建**：自动基于最新训练的模型，提取训练集中每个类别的平均特征向量，构建特征库。
3. **图像预测**：通过比对待预测图像与特征库中类别的特征相似度，实现图像分类，并将结果按类别保存。


## 环境依赖

- Python 3.8+
- 核心依赖库：
  ```bash
  pip install numpy torch torchvision tqdm scikit-learn matplotlib pillow
  ```


## 目录结构

```
FeatCls/
├── main.py               # 主程序（包含所有核心功能）
├── models/               # 训练好的模型保存目录（自动生成）
├── feature_database/     # 特征库保存目录（自动生成）
├── prediction_results/   # 预测结果保存目录（自动生成）
└── dataset/              # 数据集目录（用户需自行准备）
    ├── train_data/       # 训练集（按类别分文件夹）
    └── test_data/        # 测试集（待预测图像）
```


## 快速开始

### 1. 准备数据集

- **训练集**：需按类别分文件夹存放，目录结构如下：
  ```
  train_data/
  ├── 类别1/
  │   ├── img1.jpg
  │   └── img2.jpg
  ├── 类别2/
  │   ├── img3.jpg
  │   └── ...
  └── ...
  ```
- **测试集**：可存放单张图像或混合类别的文件夹（预测时会自动分类）。


### 2. 配置参数

修改 `main.py` 中的 `Config` 类，设置核心参数（无需关注时间戳，自动生成）：

```python
class Config:
    # 模型与损失函数
    model = "resnet18"        # 可选：resnet18/resnet34/resnet50
    loss = "ce"               # 可选：ce/focal/triplet/arcface
    
    # 训练参数
    epochs = 30               # 训练轮次
    batch_size = 32           # 批处理大小
    learning_rate = 1e-4      # 学习率
    val_split_ratio = 0.2     # 训练集/验证集划分比例（验证集占比）
    
    # 路径配置（需修改为你的实际路径）
    train_data_dir = r"dataset/train_data"       # 训练集目录
    model_save_root = r"models"                  # 模型保存根目录
    db_save_root = r"feature_database"           # 特征库保存根目录
    predict_save_root = r"prediction_results"    # 预测结果保存根目录
    test_data_dir = r"dataset/test_data"         # 默认测试集目录
```


### 3. 运行方式

支持两种运行方式，推荐新手使用**右键直接运行**：

#### 方式1：右键直接运行（适合新手）

修改 `main.py` 末尾的 `preset_mode` 切换运行模式：

```python
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        preset_mode = 'train'  # 可选：'train'（训练）/'build'（构建特征库）/'predict'（预测）
        # ... 自动填充参数 ...
```

- 先设置 `preset_mode = 'train'` 运行，完成模型训练。
- 再设置 `preset_mode = 'build'` 运行，构建特征库。
- 最后设置 `preset_mode = 'predict'` 运行，进行预测。


#### 方式2：命令行运行（适合进阶用户）

```bash
# 训练模型
python main.py train

# 构建特征库（基于最新模型）
python main.py build

# 预测（使用默认测试集路径）
python main.py predict

# 预测（指定自定义路径）
python main.py predict --input "path/to/your/test/images"
```


## 输出文件说明

1. **训练输出**：
   - 模型文件：`models/resnet18_ce_20251029_155834.pth`（格式：模型名_损失函数_时间戳.pth）
   - 训练曲线：`models/train_curve_20251029_155834.png`（损失和准确率曲线）

2. **特征库输出**：
   - 特征库文件：`feature_database/resnet18_ce_20251029_155834_feature_db.npy`  
     （存储每个类别的平均特征向量，格式为 NumPy 字典）

3. **预测输出**：
   - 分类结果：`prediction_results/prediction_20251029_160000/`（按类别分文件夹保存图像）
   - 预测日志：`prediction_results/prediction_20251029_160000/prediction_log_20251029_160000.txt`  
     （记录图像路径、预测类别和相似度距离）


## 核心原理

1. **特征提取**：使用 ResNet 模型的 backbone（去除分类头）将图像转换为固定维度的特征向量（如 ResNet18 输出 512 维向量）。
2. **特征库构建**：对训练集中每个类别的所有图像特征取平均值，作为该类别的“标准特征”。
3. **预测匹配**：待预测图像的特征向量与特征库中各类别的“标准特征”计算余弦距离，距离越小则相似度越高，取最相似的类别作为预测结果。


## 常见问题

1. **Q：训练后构建特征库提示“未找到模型”？**  
   A：确保模型保存目录（`model_save_root`）正确，且训练过程正常完成（模型文件已生成）。

2. **Q：预测时提示“未找到特征库”？**  
   A：需先运行 `build` 模式生成特征库，特征库与最新模型自动关联。

3. **Q：如何更换模型或损失函数？**  
   A：修改 `Config` 类中的 `model` 和 `loss` 参数，重新训练即可。

4. **Q：训练集/测试集路径可以修改吗？**  
   A：可以，直接修改 `Config` 类中的 `train_data_dir` 和 `test_data_dir` 即可。


## 扩展建议

- 增加数据增强策略：在 `get_transforms` 函数中添加更多数据增强方法（如随机裁剪、高斯模糊）。
- 支持预训练模型：将 `models.resnet18(weights=None)` 改为 `models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)` 加载预训练权重。
- 增加预测阈值：在预测时设置最小相似度阈值，低于阈值的图像标记为“未知类别”。