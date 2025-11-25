import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import datasets,transforms

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        '''
    初始输入：1x28x28
    经过conv1（6个滤波器，padding=1）: 6x28x28
    经过pool1: 6x14x14
    经过conv2（12个滤波器，padding=1）: 12x14x14
    经过pool2: 12x7x7
    展平后的维度：12*7*7 = 588，而不是784。
        '''
        self.fc1=nn.Linear(588,128)
        self.fc2=nn.Linear(128,10)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=x.view(-1,588)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)

        return x


transform = transforms.Compose([
    transforms.ToTensor(), # 将图像数据转换为PyTorch张量
    transforms.Normalize((0.5,), (0.5,)) # 标准化，前后两个0.5分别是均值和标准差，数据范围[0,1]->[-1,1]，这样梯度下降更稳定
])  
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


model=Net()
criterion=nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        loss_all=0
        for data,target in train_loader:
            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            loss_all+=loss.item()
        print(f'{epoch+1}--loss--:{loss_all}')
train(model,train_loader,criterion,optim,5)