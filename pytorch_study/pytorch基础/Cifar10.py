from torch import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6, ]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),#channel 3->6 卷积框大小5，没次移动1，不设置填充
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # x: [b, 6, ]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        #flatten
        #fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

        # tep = torch.randn(2, 3, 32, 32)
        # out = self.conv_unit(tep)
        # # [b, 16, 5, 5]
        # print('conv out:', out.shape)

        # use Cross EntropyLoss 分类问题

        # self.criteon = nn.CrossEntropyLoss()#会自动对传入的logits进行softmax，并取出最大值对应的分类与label进行比对，所以传入的值应该是没有经过softmax的值
    def forward(self, x):
        '''

        :param x: [b, 3, 32, 32]
        :return:
        '''
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        # [b, 16, 5, 5] =? [b, 16 * 5 * 5]
        batchsz = x.size(0)
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16 * 5 * 5]
        x = x.view(batchsz, -1)
        # [b, 16 * 5 * 5] => [b, 10]
        logits = self.fc_unit(x)

        # [b, 10]
        # pred = F.softmax(logits, dim=1)
        # loss = self.criteon(logits, y)

        return logits
def main():
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)

if __name__ == '__main__':
    main()