# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch as t
import torch.nn as nn
import torch.optim
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(LinearRegressionModel, self).__init__()
        self.liner1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y = self.liner1(x)
        return y


class Trainer:
    # 迭代次数
    epochs = 10000
    # 学习率
    learning_rate = 0.0001
    # 优化器
    optim: torch.optim.SGD = None
    # 损失函数
    lossfun: nn.MSELoss = nn.MSELoss()
    # 模型
    model: LinearRegressionModel = None
    train_x = None
    train_y = None

    def __init__(self, model: LinearRegressionModel):
        self.model = model
        self.optim = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        self.loaddata()

    def loaddata(self):
        x_value = [i for i in range(11)]
        y_value = [x * 2 + 1 for x in x_value]

        self.train_x = np.array(x_value, dtype=np.float32)
        self.train_x = self.train_x.reshape(-1, 1)
        self.train_x = torch.from_numpy(self.train_x)

        self.train_y = np.array(y_value, dtype=np.float32)
        self.train_y = self.train_y.reshape(-1, 1)
        self.train_y = torch.from_numpy(self.train_y)

    def startTrain(self):

        for epoch in range(self.epochs):
            epoch += 1
            # 预测结果
            predicted = self.model(self.train_x)
            # 计算损失
            loss: torch.tensor = self.lossfun(predicted, self.train_y)
            # 清空梯度
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if epoch % 50 == 0:
                print('epoch: {}, loss:{}'.format(epoch, loss.item()))

    def test(self, x):
        out = np.array([x], dtype=np.float32)
        out = out.reshape(-1, 1)
        out = torch.tensor(out)
        out = self.model(out)
        out = out.view(-1, 1)
        out = out.item()
        return out


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    model = LinearRegressionModel(1, 1)
    trainer = Trainer(model=model)
    trainer.startTrain()
    x = 10
    y = trainer.test(x=x)

    print('测试结果x：{}, y:{}'.format(x, y))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
