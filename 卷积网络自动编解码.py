import torchvision
import torch
import cv2
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

def dis(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2,mode="nearest"),
            torch.nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
        )
    def forward(self,input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])
dataset_train = datasets.MNIST(root="./data",transform=transform,train=True,download=True)

dataset_test = datasets.MNIST(root="./data",transform=transform,train=False)

train_load = torch.utils.data.DataLoader(dataset = dataset_train,batch_size = 1,shuffle = True)

test_load = torch.utils.data.DataLoader(dataset = dataset_test,batch_size = 1,shuffle = True)
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model = AutoEncoder()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_f = torch.nn.MSELoss()
    
    epoch_n = 5
    for epoch in range(epoch_n):
        running_loss = 0.0
        print("Epoch {}/{}".format(epoch, epoch_n))
        print("-"*10)
        for data in train_load:
            x_train,_ = data
            noisy_x_train = x_train + 0.5*torch.randn(x_train.shape)
            noisy_x_train = torch.clamp(noisy_x_train,0.,1.)
            #x_train , noisy_x_train = Variable(x_train),Variable(noisy_x_train)
            x_train = x_train.to(device)
            noisy_x_train = noisy_x_train.to(device)
            train_pre = model(noisy_x_train)
            #print("train_pre=",train_pre.shape)
            #print("x_train=",x_train.shape)
            print(train_pre)
            print(x_train)
            loss = loss_f(train_pre,x_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print("Loss is:{:.4f}".format(running_loss/len(dataset_train)))