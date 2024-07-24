import math
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

# 定义CIFAR-10数据集的均值和标准差
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# 获取CIFAR-10数据集
def get_cifar10(root, num_labeled, num_classes, batch_size, eval_step):
    # 有标签数据的转换
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect'),    # 随机裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)    # 标准化
    ])
    # 测试数据的转换
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    # 下载CIFAR-10基础数据集
    base_dataset = datasets.CIFAR10(root, train=True, download=True)
    # 分割有标签和无标签的数据索引
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(num_labeled, num_classes, batch_size, eval_step, base_dataset.targets)
    # 定义有标签和无标签的数据集、测试数据集
    train_labeled_dataset = CIFAR10SSL(root, train_labeled_idxs, train=True,transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(root, train_unlabeled_idxs, train=True,transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))
    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

# 分割有标签和无标签数据
def x_u_split(num_labeled, num_classes, batch_size, eval_step, labels):
    label_per_class = num_labeled // num_classes    # 每个类的标签数量
    labels = np.array(labels)
    labeled_idx = []    # 存储有标签数据的索引
    unlabeled_idx = np.array(range(len(labels)))    # 存储无标签数据的索引
    for i in range(num_classes):
        idx = np.where(labels == i)[0]  # 获取当前类的所有索引
        idx = np.random.choice(idx, label_per_class, False) # 随机选择指定数量的索引
        labeled_idx.extend(idx) # 添加到有标签索引列表
    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == num_labeled  # 确认标签数量正确

    num_expand_x = math.ceil(batch_size * eval_step / num_labeled)  # 计算扩展的数量
    labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)]) # 扩展有标签数据索引
    np.random.shuffle(labeled_idx)  # 随机打乱索引
    return labeled_idx, unlabeled_idx

# 定义TransformFixMatch类，包含弱增强和强增强
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect')])   # 随机裁剪
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomCrop(size=32,padding=int(32*0.125),padding_mode='reflect'),    # 随机裁剪
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),])    # 颜色抖动
        self.normalize = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=mean, std=std)])  # 标准化

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong) # 返回增强后的结果

# 定义CIFAR10SSL类，继承自CIFAR10
class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,transform=None, target_transform=None,download=False):
        super().__init__(root, train=train,transform=transform,target_transform=target_transform,download=download)
        if indexs is not None:  # 如果提供了索引
            self.data = self.data[indexs]   # 选择数据索引
            self.targets = np.array(self.targets)[indexs]   # 选择标签索引

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] # 获取数据和标签
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)   # 应用变换

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
