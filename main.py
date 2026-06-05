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

batch_size = 60

# Using ImageFolder because it automatically maps subfolders as classes! Handy!
train_dataset = datasets.ImageFolder(
    root="datasets/train", transform=transforms.ToTensor()
)

test_dataset = datasets.ImageFolder(
    root="datasets/test", transform=transforms.ToTensor()
)
