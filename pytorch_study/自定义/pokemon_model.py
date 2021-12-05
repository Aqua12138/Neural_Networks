import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torch.utils.data import DataLoader

from Pokemon_data import Pokemon
from Pokemon_ResBlk import ResNet

batchsz = 32
lr = 1e-3
epochs = 10

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Pokemon('/home/zhx/文档/pokemon',(224,224),'train')
val_db = Pokemon('/home/zhx/文档/pokemon',(224,224),'val')
test_db = Pokemon('/home/zhx/文档/pokemon',(224,224),'test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)#num_workers 是多进程

val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
def evalute(model, loader):
    total = len(loader.dataset)
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim = 1)
        correct += torch.eq(pred, y).sum().float().item()#item表示tensor->scalar 当tensor只有一个值的时候才可以使用它
    return correct / total


def main():
    model = ResNet(4).to(device)
    optimizer = optim.Adam
    criteon = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:

            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'model/best.mdl')
    print('best acc', best_acc, 'best epoch:', best_epoch)
    model.load_state_dict(torch.load('model/best.mdl'))
    print('loaded from ckpt!')

    test_acc = evalute(model, test_loader)
    print('test acc:', test_acc)
if __name__ == '__main__':
    main()