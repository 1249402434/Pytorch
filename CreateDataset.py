'''
本文件是读取自己的图像数据集
在train.txt和test.txt中分别存放了训练集的地址和标签，测试集的地址和标签
'''

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage

# 可以将Tensor转化为Image，方便可视化
show = ToPILImage()
def data_loader(path):
    return Image.open(path).convert('RGB')


# 继承Dataset类需要重写__getitem__方法和__len__方法
class MyDataset(Dataset):
    def __init__(self, root, loader, transform=None):
        super(MyDataset, self).__init__()

        imgs = []
        f = open(root, 'r')

        for line in f.readline():
            line = line.strip('\n')
            line = line.rstrip('\n')

            content = line.split()
            imgs.append((content[0], int(content[1])))

        self.imgs = imgs
        self.loader = loader
        self.transform = transform

    def __getitem__(self, item):
        item = self.imgs[item]
        path, label = item[0], item[1]
        img = self.loader(path)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


root_path = './data/'

train_data = MyDataset(root_path + 'train.txt', data_loader)
test_data = MyDataset(root_path + 'test.txt', data_loader)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
