import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 数据仓库
class DataStore:
    trainDataLoader: DataLoader = None
    testDataLoader: DataLoader = None
    batch_size: int = 1

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        self.loadData()

    def loadData(self):
        train_data = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
        test_data = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
        self.trainDataLoader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        self.testDataLoader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True)


# 卷积神经网络，10 分类模型，输入大小 28 x 28 x 1
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1  28x28x1 -> 28x28x16
        # 2  28x28x16 -> 14 x 14 x 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # 卷积
            nn.ReLU(),  # 激活
            nn.MaxPool2d(kernel_size=2),  # 池化
        )

        # 1 14 x 14 x 16 -> 14 x 14 x 32
        # 2 14 x 14 x 32 -> 7 x 7 x 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # 卷积
            nn.ReLU(),  # 激活
            nn.MaxPool2d(kernel_size=2),  # 池化
        )

        # 输出10维向量
        self.out = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 展开x
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output


# 测试
class Tester:
    function: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    model: CNNModel = None

    def __init__(self, model: CNNModel):
        self.model = model

    # 统计正确数量
    def accuracy(self, predictions: torch.tensor, labels: torch.tensor):
        # 取出预测结果的最大值坐标
        pred = torch.max(predictions, 1)[1].numpy()
        label_y = torch.max(labels, 1)[1].numpy()
        rights = (pred == label_y).sum()
        return rights, len(labels)

    def forward(self, data: DataStore):
        self.model.eval()
        loss_sum = 0.0
        total_num = 0
        crrect_num = 0
        for xd, yd in data.testDataLoader:
            predicted = self.model(xd)
            loss = self.function(predicted, yd)
            loss_f = loss.item()
            loss_sum += loss_f * len(xd)
            total_num += len(xd)
            crrect, num = self.accuracy(predictions=predicted, labels=yd)
            crrect_num += crrect

        loss = loss_sum / total_num
        print("测试损失 loss: {}, 正确率：{:.2f}%".format(loss, float(crrect_num) / float(total_num)))

    def __call__(self, data: DataStore):
        return self.forward(data=data)


#  训练
class Trainer(Tester):
    optimizer: optim.Adam = None
    lr: float = 0.001
    epochs: int = 10

    def __init__(self, model: CNNModel, epochs: int = 10):
        super().__init__(model)
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, data: DataStore):
        self.model.train()
        for epoch in range(self.epochs):
            epoch += 1
            loss_sum = 0.0
            num = 0
            for xd, yd in data.trainDataLoader:
                predicted = self.model(xd)
                loss = self.function(predicted, yd)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_f = loss.item()
                loss_sum += loss_f * len(xd)
                num += len(xd)

            print("当前 epoch:{}, 损失 loss: {}".format(epoch, loss_sum / num))


def main():
    data = DataStore(batch_size=1)
    model = CNNModel()

    trainer = Trainer(model=model, epochs=10)
    trainer(data=data)

    tester = Trainer(model=model)
    tester(data=data)


if __name__ == '__main__':
    main()
