import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


train_dataset = datasets.MNIST(root='data/', train=True,
                               transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data/', train=False,
                              transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)


def evaluate_accuracy(data_iter, model):
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_iter:
            model.eval()
            outputs = model(images)
            _, predicts = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicts == labels).cpu().sum()

            img = images[0].resize(28, 28)
            plt.imshow(img)
            plt.axis('on')
            plt.title("$The picture in {} batch,predicted label={}$".format(i + 1, predicts[0]))
            plt.show()
    return correct/total


# model
class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            #print(residual.shape)
            residual = self.downsample(residual)
            #print(residual.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        #print("x: ", x.shape)
        #print("residual: ", residual.shape)
        x += residual

        out = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_class=10):
        super(ResNet, self).__init__()
        self.in_channel = 16

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, block_num[0])
        self.layer2 = self._make_layer(block, 32, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 64, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 128, block_num[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


model = resnet5 = ResNet(Bottleneck, [1, 1, 1, 0])
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
num_epoches = 10
batch_size = 10

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
          % (epoch, train_loss_sum/batch_size, train_correct/train_accuracy_total, test_acc))
print("finish training")





