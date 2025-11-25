# 此文件夹下为torch的一些基础概念和核心组件
import torch 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import Dataset
import pandas as pd

#张量tensors 类似于numpy数组，但是支持gpu加速
x=torch.tensor([1,2,3]) #创建张量
y=torch.randn(3,4) #随机生成3×4张量
z=torch.zeros(2,3) #生成2×3的全零张量
print(x.device) #查看张量在cpu上还是cuda上

#自动求导
x=torch.tensor([2.0],requires_grad=True) 
#注意张量若需要求导元素必须是float类型，且只有一个元素（标量），且要写requires_grad=True
y=x**2+3*x+1
y.backward()
print(x.grad)

#构建神经网络
class FCN(nn.Module):
    def __init__(self):
        super(FCN,self).__init__()
        self.fc1=nn.Linear(784,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x

model=FCN()
print(model)

#训练流程示例

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(), # 将图像数据转换为PyTorch张量
    transforms.Normalize((0.5,), (0.5,)) # 标准化，前后两个0.5分别是均值和标准差，数据范围[0,1]->[-1,1]，这样梯度下降更稳定
])                                       # (0-0.5)/0.5=-1 (1-0.5)/0.5=1

# 加载数据（以MNIST为例）
#实例化dataset类
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
#创建dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# 定义模型、损失函数和优化器
criterion = nn.CrossEntropyLoss() # 损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) #优化器

# 训练循环
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train() #训练模式
    
    for epoch in range(epochs):
        running_loss = 0.0 
        for batch_idx, (data, target) in enumerate(train_loader):#从dataloader中取数据
            # 将图像展平
            data = data.view(data.size(0), -1)
            
            # 前向传播
            optimizer.zero_grad() #梯度清零，否则会梯度累积
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch+1} completed. Average Loss: {running_loss/len(train_loader):.4f}')

# 开始训练
train_model(model, train_loader, criterion, optimizer, epochs=5)