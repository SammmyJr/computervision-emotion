from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Deeper Convolutional Neural Network with Batch Normalization and Dropout.

        Parameters:
            * in_channels: Number of channels in the input image (3 for RGB)
            * num_classes: Number of emotion classes (8)
        """
        super(CNN, self).__init__()

        # Convolutional Block 1: 3 → 32 channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        # Convolutional Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        # Convolutional Block 4: 128 → 256 channels
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)

        # Adaptive Average Pooling to handle any spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers with Dropout
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 512)

        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor of shape (batch_size, 3, 48, 48)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1: 48×48 → 24×24
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Conv Block 2: 24×24 → 12×12
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Conv Block 3: 12×12 → 6×6
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Conv Block 4: 6×6 → 6×6 (no pooling)
        x = F.relu(self.bn4(self.conv4(x)))

        # Adaptive Average Pooling: 6×6 → 1×1
        x = self.adaptive_pool(x)

        # Flatten: (batch_size, 256, 1, 1) → (batch_size, 256)
        x = x.reshape(x.shape[0], -1)

        # Fully Connected Layers with Dropout and ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
