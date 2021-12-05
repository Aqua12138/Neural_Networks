import torch
from torch import nn
from torch.nn import functional as F
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        #因为尺寸可能会变小，如果前后输入尺寸相同，就不会维度减少，所以要加pool
        self.extra = nn.Sequential(nn.MaxPool2d(kernel_size=stride))
        #如果维度尺寸不一样就直接覆盖掉，新建立一个改变channel的网络层
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        '''

        :param x:
        :return:
        '''

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut [b, ch_in, h, w] with [b, ch_out, h, w]
        #
        out = self.extra(x) + out

        return out
class ResNet(nn.Module):
    def __init__(self, num_class):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm2d(16)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h, w]
        self.blk1 = ResBlk(16, 32, stride=3)
        self.blk2 = ResBlk(32, 64, stride=3)
        self.blk3 = ResBlk(64, 128, stride=2)
        self.blk4 = ResBlk(128, 256, stride=2)

        self.outlayer = nn.Linear(256*3*3, num_class)
    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # # [b, 512, 4, 4] => [b, 1024, 1, 1]
        # x = F.adaptive_max_pool2d(x, [1,1])
        x = x.view(-1, 256*3*3)
        x = self.outlayer(x)

        return x

def main():
    blk = ResBlk(64, 128, stride=2)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print(out.shape)

    x = torch.randn(2, 3, 224, 224)
    model = ResNet(10)
    out = model(x)
    print('resnet:', out.shape)
    #获得模型参数数量
    p = sum(map(lambda p:p.numel(), model.parameters()))
    print(p)
if __name__ == '__main__':
    main()