import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import transforms, datasets
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


class DataStore:
    def __init__(self):
        pass


class Model(nn.Module):
    def __init__(self):
        super().__init__()


class Tester:
    model: Model = None

    def __init__(self, model: Model):
        self.model = model

    def forward(self, data: DataStore):
        pass

    def __call__(self, data: DataStore):
        return self.forward(data=data)


class Trainer(Tester):
    def __init__(self, model: Model):
        super().__init__(model=model)

    def forward(self, data: DataStore):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
