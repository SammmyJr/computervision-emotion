import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchmetrics

from cnn import CNN

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Datasets
# Using ImageFolder because it automatically maps subfolders as classes! Handy!
train_dataset = datasets.ImageFolder(
    root="dataset/train", transform=transforms.ToTensor()
)

test_dataset = datasets.ImageFolder(
    root="dataset/test", transform=transforms.ToTensor()
)

# Dataloaders
batch_size = 60

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Load CNN
model = CNN(in_channels=1, num_classes=7).to(device)
print(model)
