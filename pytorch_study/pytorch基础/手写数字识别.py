import torch
from torch import nn
from torch.nn import functional as tf
from torch import optim
from torchvision import datasets,transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader,dataset

batch_size = 200
learning_rate = 1e-3
epochs = 10

#train_loader

train_loader = DataLoader(
    datasets.MNIST('../data',train=True,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('../data',train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True
)
class Mynetwork(nn.Module):
    def __init__(self):
        super(Mynetwork, self).__init__()
        self.w1,self.b1 = torch.randn(200,784,requires_grad=True),\
                torch.zeros(200,requires_grad=True)
        self.w2,self.b2 = torch.randn(200,200,requires_grad=True),\
                torch.zeros(200,requires_grad=True)
        self.w3,self.b3 = torch.randn(10,200,requires_grad=True),\
                torch.zeros(10,requires_grad=True)
        #w init
        torch.nn.init.kaiming_normal(self.w1)
        torch.nn.init.kaiming_normal(self.w2)
        torch.nn.init.kaiming_normal(self.w3)

    def forward(self,x):
        x = x @ self.w1.t() + self.b1
        x = F.relu(x)
        x = x @ self.w2.t() + self.b2
        x = F.relu(x)
        x = x @ self.w3.t() + self.b3
        x = F.relu(x)

        return x

if __name__ == '__main__':
    model = Mynetwork()
    optimizer = optim.SGD([model.w1,model.b1,model.w2,model.b2,model.w3,model.b3],lr = 1e-3)
    criteon = nn.CrossEntropyLoss()

    for epochs in range(epochs):
        for batch_idx,(data,target) in enumerate(train_loader):
            data = data.view(-1,28*28)
            logits = model(data)
            loss = criteon(logits,target)#求误差，输出和标签

            optimizer.zero_grad()
            loss.backward()#误差反传求梯度
            optimizer.step()#使用梯度进行优化

            if batch_idx % 100 == 0:
                print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epochs,batch_idx*len(data),len(train_loader.dataset),
                                                                             100.*batch_idx/len(train_loader),loss.item()))

        test_loss = 0
        correct = 0
        for data ,target in test_loader:
            data = data.view(-1, 28*28)
            logits = model(data)
            test_loss += criteon(logits,target).item()

            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
