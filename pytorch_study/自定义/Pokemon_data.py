import torch
import os,glob
import random,csv

import torchvision.datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cv2 import cv2
from PIL import Image
import visdom
import time
class Pokemon(Dataset):
    def __init__(self, root_dir, resize, mode):
        super(Pokemon, self).__init__()
        self.root_dir = root_dir
        self.resize = resize

        self.name2label = {} #标签编码
        for name in sorted(os.listdir(os.path.join(root_dir))):
            if not os.path.isdir(os.path.join(root_dir,name)):#判断rootdir中的文件是不是路径
                continue
            self.name2label[name] = len(self.name2label.keys())
        print(self.name2label)
        # image，label
        self.images, self.labels = self.load_csv('images.csv')

        if mode == 'train': # 60%
            self.images = self.images[:int(0.6*len(self.images))]
            self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode == 'val': #20%
            self.images = self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels =self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        elif mode == 'test': #10%
            self.images = self.images[int(0.8*len(self.images)):]
            self.labels =self.labels[int(0.8*len(self.labels)):]
        else:
            print('no'+mode+'type')
    def load_csv(self,filename):
        #如果没有先创建
        if not os.path.exists(os.path.join(self.root_dir,filename)):
            images = []
            #'pokemon\\mewtwo\\00001.png'
            for name in self.name2label.keys():
                #遍历，索引对应的路径
                images += glob.glob(os.path.join(self.root_dir, name, '*.png'))
                images += glob.glob(os.path.join(self.root_dir, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root_dir, name, '*.jpeg'))

            print(len(images),images)

            random.shuffle(images)#打乱
            #write
            with open(os.path.join(self.root_dir, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: #'pokemon\\mewtwo\\00001.png',3
                    name = img.split(os.sep)[-2]#按照分割符划分，取倒数第二个的名称
                    label = self.name2label[name]#按照名称取标签
                    writer.writerow([img, label])
                print('write to csv file',filename)

        # read
        images, labels = [], []
        with open(os.path.join(self.root_dir,filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)#保证长度一致，如果不一致会报错

        return images, labels

    def __len__(self):
        #返回样本长度
        return len(self.images)

    def denormalize(self, x_hat):
        #将标准化的图片转换成原来的数值
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean) / std
        # x = x_hat * std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x  = x_hat * std + mean

        return x
    def __getitem__(self, item):
        # index~[0~len(images)]
        #self.images self.labels
        #img : 'pokemon\\mewtwo\\00001.png'
        #label:3
        img, label = self.images[item], self.labels[item]
        # img = cv2.imread(img)
        # #resize
        # img = cv2.resize(img,self.resize,interpolation=cv2.INTER_CUBIC)#resize = [w,h]
        # img = torch.tensor(img)
        # label = torch.tensor(label)

        #transform处理

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),#图片转化为RGB格式
            transforms.Resize((int(self.resize[0]*1.25),int(self.resize[1]*1.25))),#重新resize图片像素大小，大一点方便裁剪
            transforms.RandomRotation(15),#随机旋转15度
            transforms.CenterCrop(self.resize),#中心裁剪
            transforms.ToTensor(),#转变为tensor
            transforms.Normalize(mean=[0.485 ,0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])#对每个通道进行正态标准化  -1~1
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

def main():
    # 方法一：
    db = Pokemon('/home/zhx/文档/pokemon',(224,224),'train')
    x,y = next(iter(db))

    viz = visdom.Visdom()
    print('sample:', x.shape, y.shape)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    #batch处理
    loader = DataLoader(db, batch_size=32, shuffle=True)




    for x, y in loader:
        viz.images(db.denormalize(x), nrow=16, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch'))

        time.sleep(10)

    # #方法二：
    # tf = transforms.Compose([
    #     lambda x: Image.open(x).convert('RGB'),  # 图片转化为RGB格式
    #     transforms.Resize((224, 224)),  # 重新resize图片像素大小，大一点方便裁剪
    #     transforms.ToTensor()# 转变为tensor
    # ])
    #
    # db = torchvision.datasets.ImageFolder(root='/home/zhx/文档/pokemon', transform=tf)#要把图片分成2级目录格式
    # loader = DataLoader(db, batch_size=32, shuffle=True)



if __name__ == '__main__':
    main()





