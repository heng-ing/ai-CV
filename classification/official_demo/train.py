import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 导入50000张训练图片
train_set = torchvision.datasets.CIFAR10(root='./data',  # 数据集存放目录
                                         train=True,  # 表示是数据集中的训练集
                                         download=False,  # 第一次运行时为True，下载数据集，下载完成后改为False
                                         transform=transform)  # 预处理过程

# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set,  # 导入的训练集
                                           batch_size=50,  # 每批训练的样本数
                                           shuffle=False,  # 是否打乱训练集
                                           num_workers=0)  # 使用线程数，在windows下设置为0

# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,  # 表示是数据集中的测试集
                                        download=False, transform=transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=10000,  # 每批用于验证的样本数
                                          shuffle=False, num_workers=0)
# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda")
print(device)
net = LeNet()
net.to(device) # 将网络分配到指定的device中
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(5):

    running_loss = 0.0
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.to(device))				  # 将inputs分配到指定的device中
        loss = loss_function(outputs, labels.to(device))  # 将labels分配到指定的device中
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 1000 == 999:
            with torch.no_grad():
                outputs = net(test_image.to(device)) # 将test_image分配到指定的device中
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0) # 将test_label分配到指定的device中

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 1000, accuracy))

                print('%f s' % (time.perf_counter() - time_start))
                running_loss = 0.0

print('Finished Training')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
