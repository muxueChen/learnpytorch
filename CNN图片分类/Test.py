import torch
import numpy as np

predicted = torch.tensor([[1, 5, 62, 54], [2, 6, 2, 6], [2, 65, 2, 6]])

labels = torch.tensor([2, 3, 1])

pred = torch.max(predicted, 1)[1]
print("pred: {}".format(pred))
pred = pred.numpy()
print("pred: {}".format(pred))
x = (pred == labels.numpy()).sum()
print("x :{}".format(x))
