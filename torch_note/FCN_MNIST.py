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
    
class test_MNIST_data(Dataset):
    def __init__(self):
        super(test_MNIST_data,self).__init__()
        self.data_dir=Path(__file__).parent / "MNIST_IMG" / "TEST"
        self.data=[]
        self.labels=[]
        self.to_tensor=transforms.ToTensor()
        self.nor=transforms.Normalize((0.5,),(0.5,))
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))])
        self.load_data()
    def load_data(self):
        for num in self.data_dir.iterdir():
            for item in num.iterdir():
                if item.suffix=='.jpg':
                    img=cv2.imread(item,cv2.IMREAD_GRAYSCALE)
                    img=self.transform(img)   
                    img=img.flatten()
                    self.data.append(img)
                    self.labels.append(int(num.stem))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        img=self.data[index]
        label=self.labels[index]
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
        print(f'Epoch{epoch+1} loss:{loss_sum/len(train_data_loader)}')

train(1)

test_data=test_MNIST_data()
test_data_loader=DataLoader(test_data,batch_size=32,shuffle=True)
def test():
    model.eval()
    loss_sum=0.0
    correct_num=0
    with torch.no_grad():
        for img,label in test_data_loader:
            out=model(img)
            loss_sum+=criterion(out,label)
            max_value,max_pridict=torch.max(out,dim=1)
            #dim=1很常用，是在向量组内的每个向量（沿行）查找最大值
            '''
            tensor [[1,2,3]
                    [2,3,4]
                    [3,5,6]]
            '''
            max_pridict=torch.argmax(out,dim=1)#只需要索引可以用argmax
            correct_num+=(max_pridict==label).sum().item()
        
        accuracy=correct_num/(len(test_data_loader)*test_data_loader.batch_size)
        print(f'loss:{loss_sum/len(test_data_loader)} accuracy:{accuracy}')

test()