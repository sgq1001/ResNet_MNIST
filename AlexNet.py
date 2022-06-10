# -*- coding: utf-8 -*-
# @Time    : 6/10/22 1:09 PM
# @Author  : sunguoqing
# @File    : AlexNet.py
# @Software: PyCharm 
# @Comment :

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt

batch_size = 64

train_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='cifar10', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR10(root='cifar10', train=False, transform=test_transform, download=True)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def evaluate_accuracy(data_iter, model):
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_iter):
            outputs = model(images)
            _, predicts = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicts == labels).cpu().sum()
            img = images[0]
            plt.imshow(img)
            plt.axis('on')
            plt.title("$The picture in {} batch,predicted label={}$".format(i + 1, predicts[0]))
            plt.show()
    return correct / total


class AlexNet(nn.Module):
    def __init__(self, in_dim, num_class):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(True))

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer6 = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)
        output = self.layer6(x)
        return output


model = AlexNet(3, 10)
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
num_epoches = 4

for epoch in range(num_epoches):
    print('current epoch :{}'.format(epoch))
    for i, (images, labels) in enumerate(train_loader):
        train_accuracy_total = 0
        train_correct = 0
        train_loss_sum = 0
        model.train()  # start BN and Dropout
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        _, predicts = torch.max(outputs.data, 1)
        train_accuracy_total += labels.size(0)
        train_correct += (predicts == labels).cpu().sum().item()

    test_acc = evaluate_accuracy(test_loader, model)
    print('epoch %d, loss %.4f, train_accuracy %.3f, test accuracy %.3f'
          % (epoch, train_loss_sum, train_correct / train_accuracy_total, test_acc))

print("finish training")
