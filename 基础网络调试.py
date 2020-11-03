import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class VGGNet(nn.Module):
    def __init__(self, num_classes=10):       #num_classes，此处为 二分类值为2
        super(VGGNet, self).__init__()
        net = torchvision.models.vgg16(pretrained=True)   #从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()    #将分类层置空，下面将改变我们的分类层
        self.features = net        #保留VGG16的特征层
        self.classifier = nn.Sequential(    #定义自己的分类层
                nn.Linear(512 * 7 * 7, 512),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        output = model(images)
        #print(output.max(1, keepdim=True)[1])
        #print(labels.data)
        #output = output.logits
        loss = cost(output, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        #scheduler.step(loss)
        optimizer.step()
        
        #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, lr: {:.8f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item(),scheduler.get_lr()[0]))
        #print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #output = output.logits
            test_loss += cost(output, target) # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    #model = VGGNet()
    model = torchvision.models.vgg16(pretrained=False)
    #print(model)
    #layers  = list(model.features)[:-1]
    #base_layers = nn.Sequential(*layers)
    #print(base_layers) 
    #model = torchvision.models.vgg19(pretrained=True)
    #model = torchvision.models.resnet101(pretrained=True)
    #model = torchvision.models.googlenet(pretrained=False)
    #model = torchvision.models.inception_v3(pretrained=True)
    model.to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        #transforms.RandomRotation(30),#随机旋转(-30,30)
        #transforms.RandomHorizontalFlip(),#随机水平翻转
        #transforms.RandomVerticalFlip(),#上下随机翻转
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),#饱和度当为a时，从[max(0, 1-a), 1+a]中随机选择当为(a, b)时，从[a, b]中随机选择
        #transforms.RandomResizedCrop(224),
        #transforms.Resize(299),
        #transforms.CenterCrop(224),#中心剪裁，如果size大于当前图片，则pading填充
        transforms.ToTensor(),
        normalize
    ])
    
    transform_test = transforms.Compose([
        #transforms.Resize(299),
        transforms.ToTensor(),
        normalize
    ])
    
    train_datasets = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    
    # pin_memory=True 只是用真实内存加载数据，而不会使用虚拟内存，即锁页内存
    train_loader = torch.utils.data.DataLoader(train_datasets,batch_size=512, shuffle=True,num_workers=8, pin_memory=True)
    train_loader = list(train_loader)
    #train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=20,shuffle=True, num_workers=1)
    
    test_datasets = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=512,shuffle=True, num_workers=8, pin_memory=True)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    num_epochs = 5000
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step(epoch+1)
