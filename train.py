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

from modules.cnn import CNN

# Hyperparameters
force_train = True
num_epochs = 10
batch_size = 60

# Transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Testing: No augmentation (use raw images)
test_transform = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Datasets
# Using ImageFolder because it automatically maps subfolders as classes! Handy!
train_dataset = datasets.ImageFolder(root="dataset/train", transform=train_transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=test_transform)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Load CNN
model = CNN(in_channels=3, num_classes=7).to(device)

# Check if theres a file to load
if Path("models/EmotionCNN.pt").is_file() and not force_train:
    print("Loading model...")
    model.load_state_dict(torch.load("models/EmotionCNN.pt"))
    print("Loaded!")
# Otherwise, run training
else:
    print("Training model...")
    # Loss function
    # Going with triplet margin loss as it's most efficient for facial recognition
    # Minimises distance to correct output while maximising distance to incorrect outputs,
    # This is good for emotion recognition because it's never binary, it's more on a spectrum
    # Will need to reimplement this, needs a CNN overhaul as well...
    criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # Training
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

    print("Trained!")
    #
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), "models/EmotionCNN.pt")
    print("Saved!")

# Evaluation
acc = torchmetrics.Accuracy(task="multiclass", num_classes=8).to(device)
precision = torchmetrics.Precision(
    task="multiclass", num_classes=8, average="weighted"
).to(device)
recall = torchmetrics.Recall(task="multiclass", num_classes=8, average="weighted").to(
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
