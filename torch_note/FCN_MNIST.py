import numpy as np
from pathlib import Path 
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

class FCN(nn.Module):
    def __init__(self):
        super(FCN,self).__init__()
        self.fc1=nn.Linear(784,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax()
    
    def forward(self,x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x
    
model=FCN()

class MNIST_data(Dataset):
    def __init__(self):
        super(MNIST_data,self).__init__()
        self.data_dir=Path("./MNIST_IMG/TRAIN")
        self.data=[]
        self.labels=[]
        self.load_data()
    def load_data(self):
        for num in self.data_dir.iterdir():
            for item in num.iterdir():
                if item.suffix=='.jpg':
                    img=cv2.imread(item,cv2.IMREAD_GRAYSCALE)
                    # img=torch.from_numpy(img)
                    totensor=transforms.ToTensor()
                    img=totensor(img)
                    nor=transforms.Normalize((0.5,),(0.5,))
                    img=nor(img)
                    img=img.flatten()
                    self.data.append(img)
                    self.labels.append(int(num.stem))#注意直接num.stem是str类型
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        img=self.data[idx]
        label=self.labels[idx]
        return img,label

        
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
train_data=MNIST_data()
train_data_loader=DataLoader(train_data,batch_size=32,shuffle=True)
def train(epochs):
    model.train()
    for epoch in range(epochs):
        loss_sum=0.0
        for img,label in train_data_loader:
            optimizer.zero_grad()
            output=model(img)
            loss=criterion(output,label)
            loss.backward()
            optimizer.step()
            loss_sum=loss_sum+loss.item()
        print(f'Epoch{epoch} loss:{loss_sum/len(train_data_loader)}')

train(10)


