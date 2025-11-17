import torch
import torch.nn as nn
from torchvista import trace_model

# 定义一个简单的线性模型
class LinearModel(nn.Module):   
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10,5)

    def forward(self,x):
        return self.linear(x)

# 实例化模型和输入
model = LinearModel()
input = torch.randn(2,10)


trace_model(model,input)  # 可视化计算图
