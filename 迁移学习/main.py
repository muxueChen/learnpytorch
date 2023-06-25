import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import transforms, datasets, models
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import os

from torchvision.models import ResNet


# 数据仓库
class DataStore:
    trainDataLoader: DataLoader = None
    validDataLoader: DataLoader = None
    _train_dir = "./flower_data/train"
    _valid_dir = "./flower_data/valid"
    train_class_names = []
    vaild_class_names = []
    batch_size = 8
    _cat_to_json = "cat_to_name.json"
    catdict = None

    def __init__(self):
        self.loadData()
        self.loadCat_to_json()

    def loadData(self):
        train_transform = transforms.Compose([transforms.RandomRotation(45),
                                              transforms.CenterCrop(224),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.RandomVerticalFlip(0.5),
                                              transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1,
                                                                     hue=0.1),
                                              transforms.RandomGrayscale(p=0.025),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.244, 0.255])])

        valid_transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.244, 0.255])])

        train_datasets = datasets.ImageFolder(self._train_dir)
        vaild_datasets = datasets.ImageFolder(self._valid_dir)
        self.trainDataLoader = DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True)
        self.validDataLoader = DataLoader(vaild_datasets, batch_size=self.batch_size, shuffle=True)
        self.train_class_names = train_datasets.classes
        self.vaild_class_names = vaild_datasets.classes

    def loadCat_to_json(self):
        with open(self._cat_to_json, "r") as f:
            self.catdict = json.load(f)


# 模型
class Model:
    input_size = 224
    model: ResNet = None
    # 是否使用别人训练号的参数
    feature_extract = True
    device: torch.device = None
    def __init__(self):
        self.model = models.resnet152(pretrained=True)
        if self.feature_extract:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        in_num = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_features=in_num, out_features=102), nn.LogSoftmax(dim=1))

    def parameters(self):
        params = []
        if self.feature_extract:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params.append(param)
        else:
            for name, param in self.model.named_parameters():
                param.requires_grad = True
                params.append(param)

        return params

    def is_available(self):
        return torch.cuda.is_available()

    def load(self, state_dict):
        self.model.load_state_dict(state_dict=state_dict)

    def to(self, device: torch.device):
        self.device = device
        self.model.to(device)

    def state_dict(self):
        return self.model.state_dict()

    def __call__(self, x):
        return self.model(x)


class Tester:
    model: Model = None
    lossfunc: nn.NLLLoss()
    device = None

    def __init__(self, model: Model):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, data: DataStore):
        running_loss = 0.0
        running_corrects = 0.0
        for xd, yd in data.trainDataLoader:
            xd = xd.to(self.device)
            yd = yd.to(self.device)
            predicted = self.model.model(xd)
            loss = self.lossfunc(predicted, yd)
            running_loss += xd.size(0) * loss.item()
            running_corrects += torch.sum(torch.max(predicted, 1)[1] == yd.data)

        epoch_loss = running_loss / len(data.trainDataLoader.dataset)
        epoch_acc = running_corrects / len(data.trainDataLoader.dataset)

        return running_loss, running_corrects

    def __call__(self, data: DataStore):
        return self.forward(data=data)


class Trainer(Tester):
    optimizer: optim.Adam = None
    scheduler: optim.lr_scheduler.StepLR = None
    lr = 0.001
    epochs = 100
    filename = 'checkpoint.pth'
    tester: Tester = None

    def __init__(self, model: Model):
        super().__init__(model=model)
        self.tester = Tester(model=model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 学习率每7个 epoch 衰减成原来的1/10
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def forward(self, data: DataStore):
        since = time.time()
        best_cc = 0
        self.model.to(self.device)
        best_modes_wts = copy.deepcopy(self.model.state_dict())

        train_acc_history = []  # 训练正确率历史
        train_losses = []  # 训练 损失数据
        test_losses = []  # 测试 损失数据
        test_acc_history = []  # 验证正确率历史
        lr_list = [self.optimizer.param_groups[0]['lr']]

        for epoch in range(self.epochs):
            epoch += 1
            # 统计一次迭代的平均损失
            running_loss = 0.0
            running_corrects = 0.0
            print("Epoch: {}/{}".format(epoch, self.epochs))
            for xd, yd in data.trainDataLoader:
                xd = xd.to(self.device)
                yd = yd.to(self.device)

                predicted = self.model.model(xd)
                loss = self.lossfunc(predicted, yd)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += xd.size(0) * loss.item()
                running_corrects += torch.sum(torch.max(predicted, 1)[1] == yd.data)

            train_loss = running_loss/len(data.trainDataLoader.dataset)
            train_acc = running_corrects/ len(data.trainDataLoader.dataset)

            test_loss, test_acc = self.tester(data=data)

            if test_acc > best_cc:
                best_cc = test_acc
                best_modes_wts = copy.deepcopy(self.model.state_dict())
                state = {
                    "state_dict": self.model.state_dict(),
                    "best_acc": best_cc,
                    "optimizer": self.optimizer.state_dict()
                }
                torch.save(state, self.filename)

            # 记录每一次迭代的正确率和损失值
            train_losses.append(train_loss)
            train_acc_history.append(train_acc)

            test_losses.append(test_loss)
            test_acc_history.append(test_acc)

            # 几率每一次迭代的学习率
            lr_list.append(self.optimizer.param_groups[0]['lr'])
            print("Optimizer learning rate : {:.7f}".format(self.optimizer.param_groups[0]['lr']))

        duration = time.time() - since
        print("Traing learning")
        self.model.load(best_modes_wts)
        return self.model


def test(model: Model, data: DataStore):

    data_iter = iter(data.validDataLoader)
    xd, yd = data_iter.__next__()

    model.model.eval()
    output = model(x=xd)

    preds_tensor = torch.max(output, 1)[1]
    preds = np.squeeze(preds_tensor.numpy()) if not model.is_available() else np.squeeze(preds_tensor.cpu().numpy())
    return 0


def main():
    model = Model()
    trainer = Trainer(model=model)

    data = DataStore()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_on_gpu = torch.cuda.is_available()
    model.to(device=device)
    model = trainer(data=data)

    test(model=model, data=data)


if __name__ == '__main__':
    main()
