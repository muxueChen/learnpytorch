import gzip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import requests
import pickle

# 数据仓库
class DataStore:
    trainLoader: DataLoader = None
    testLoader: DataLoader = None
    batch_size: int = 1

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size
        result = self._loadOriginData()
        if result is None:
            print("Error:加载数据失败")
            return
        x_train, y_train, x_test, y_test = map(torch.tensor, result)
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        print("train_dataset:{}", train_dataset)
        print("test_dataset:{}", test_dataset)
        self.trainLoader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.testLoader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

    def _loadOriginData(self):
        data_path = Path("data")
        path = data_path / "mnist"
        path.mkdir(parents=True, exist_ok=True)
        url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
        file_name = "mnist.pkl.gz"
        file_path = path / file_name
        if not file_path.exists():
            content = requests.get(url=url).content
            file_path.open("wb").write(content)

        with gzip.open(file_path, "rb") as f:
            try:
                ((x_train, y_train), (x_test, y_test), _) = pickle.load(f, encoding="latin-1")
                return (x_train, y_train, x_test, y_test)
            except EOFError:
                return None


# 10 分类模型，输入为一个通道，28*28的图像
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.liner1 = nn.Linear(784, 128)
        self.liner2 = nn.Linear(128, 256)
        self.liner3 = nn.Linear(256, 10)

    def forward(self, x):
        out = F.relu(self.liner1(x))
        out = F.relu(self.liner2(out))
        out = self.liner3(out)
        return out


# 训练器
class Trainer:
    # 损失函数
    lossfun: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    # 优化器
    optimer: optim.SGD = None
    # 模型
    model: Model = None
    # 学习率
    lr: float = 0.0001
    # 迭代次数
    epochs: int = 100

    def __init__(self, model: Model, epochs: int = 10):
        self.optimer = optim.SGD(model.parameters(), lr=self.lr)
        self.model = model
        self.epochs = epochs

    def forward(self, data: DataStore):
        self.model.train()
        for epoch in range(self.epochs):
            epoch += 1
            for xd, yd in data.trainLoader:
                predicted = self.model(xd)
                loss = self.lossfun(predicted, yd)
                self.optimer.zero_grad()
                loss.backward()
                self.optimer.step()
                loss_f = loss.item()
                print("trainer epoch:{}, loss:{}".format(epoch, loss_f))

    def __call__(self, data: DataStore):
        return self.forward(data)


# 验证员
class Tester:
    # 模型
    model: Model = None
    lossfun: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def __init__(self, model):
        self.model = model

    def forward(self, data: DataStore):
        self.model.eval()
        nums: int = 0
        loss_sum: float = 0.0
        for xd, yd in data.testLoader:
            predicted = self.model(xd)
            loss = self.lossfun(predicted, yd)

            l = len(xd)
            loss_sum += loss * l
            nums += l

        loss = loss_sum / nums
        print("tester loss: {}".format(loss))

    def __call__(self, data: DataStore):
        return self.forward(data)

def main():
    data = DataStore(batch_size=4)
    model = Model()
    trainer = Trainer(model=model, epochs=30)
    trainer(data=data)

    tester = Tester(model=model)
    tester(data=data)


if __name__ == '__main__':
    main()
