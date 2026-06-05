import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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
model = CNN(in_channels=3, num_classes=7).to(device)

# Check if theres a file to load
if Path("models/EmotionCNN.pt").is_file():
    model.load_state_dict(torch.load("models/EmotionCNN.pt"))
# Otherwise, run training
else:
    # Loss function
    # Going with triplet margin loss as it's most efficient for facial recognition
    # Minimises distance to correct output while maximising distance to incorrect outputs,
    # This is good for emotion recognition because it's never binary, it's more on a spectrum
    # Will need to reimplement this, needs a CNN overhaul as well...
    criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")

        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    # Save model
    torch.save(model.state_dict(), "models/EmotionCNN.pt")

# Evaluation
acc = torchmetrics.Accuracy(task="multiclass", num_classes=7).to(device)
precision = torchmetrics.Precision(
    task="multiclass", num_classes=7, average="weighted"
).to(device)
recall = torchmetrics.Recall(task="multiclass", num_classes=7, average="weighted").to(
    device
)

# Iterate over batches, check accuracy
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        # Move to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Get predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        acc(preds, labels)
        precision(preds, labels)
        recall(preds, labels)

# Compute total accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")
