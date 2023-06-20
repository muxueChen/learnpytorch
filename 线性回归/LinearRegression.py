import torch
import torch.nn as nn
import numpy as np
from torch import optim

class Model(nn.Module):
    def __init__(self, in_n, out_n):
        super().__init__()
        self.liner = nn.Linear(in_n, out_n)

    def forward(self, x):
        y = self.liner(x)
        return y

class Trainer:
    model: Model = None
    epochs = 100000
    lr = 0.001
    # 均方误差
    lossfun = nn.MSELoss()
    optimer: optim.SGD = None
    _train_x = None
    _label = None
    def __init__(self, model: Model):
        self.model = model
        self.optimer = optim.SGD(model.parameters(),  lr=self.lr)
        self.loadData()

    def loadData(self):
        x = [i + 10 for i in range(11)]
        y = [i * 3 + 5 for i in x]
        print("x:{}".format(x))
        print("x:{}".format(y))
        x = np.array(x, dtype=np.float32)
        x = x.reshape(-1, 1)
        self._train_x = torch.from_numpy(x)



        y = np.array(y, dtype=np.float32)
        y = y.reshape(-1, 1)
        self._label = torch.from_numpy(y)

    def train(self):
        miniLoss = 10.0
        for epoch in range(self.epochs):
            epoch += 1
            predicted = self.model(self._train_x)
            loss: torch.tensor = self.lossfun(predicted, self._label)
            self.optimer.zero_grad()
            loss.backward()
            self.optimer.step()
            loss_f = loss.item()
            if epoch % 50 == 0:
                print("epoch:{}, loss:{}".format(epoch, loss_f))


    def test(self, x):
        in_x = [x]
        in_x = np.array(in_x, dtype=np.float32)
        in_x = in_x.reshape(-1, 1)
        in_x = torch.from_numpy(in_x)

        out_y = self.model(in_x)
        out_y = out_y.view(-1, 1)
        return out_y.item()

if __name__ == '__main__':
    model = Model(1, 1)
    trainer = Trainer(model=model)
    trainer.train()
    y = trainer.test(x=50)
    print("test x:{}, y:{}".format(10, y))