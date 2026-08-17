# models.py
"""
Model definitions for Sub-network 1.

This file contains:
- Identity: a pass-through module to replace classifier heads.
- EfficientNet: 3D EfficientNet-B1 backbone with multi-task heads
  for OS risk, PFS risk, age regression and clinical multi-label prediction.
"""

import torch
import torch.nn as nn
import monai


class Identity(nn.Module):
    """A simple identity layer to replace existing classifier heads."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class EfficientNet(nn.Module):
    """
    3D EfficientNet-B1 backbone with four task-specific heads:

    - fc1: OS risk prediction (scalar)
    - fc2: PFS risk prediction (scalar)
    - fc3: Age regression (scalar, sigmoid + MSE on log(age))
    - fc4: 9-dimensional multi-label logistic outputs
    """

    def __init__(self):
        super(EfficientNet, self).__init__()

        self.backbone = monai.networks.nets.EfficientNetBN(
            "efficientnet-b1",
            spatial_dims=3,
            in_channels=3,
            num_classes=1,
        )
        # Replace the final fully-connected layer with identity to use raw features.
        self.backbone._fc = Identity()

        # 1280 is the default feature dimension for EfficientNet-B1.
        self.fc1 = nn.Linear(1280, 1)   # OS risk
        self.fc2 = nn.Linear(1280, 1)   # PFS risk
        self.fc3 = nn.Linear(1280, 1)   # Age regression
        self.fc4 = nn.Linear(1280, 9)   # 9 clinical multi-label logits

    def forward(self, x):
        encoded = self.backbone(x)
        os = self.fc1(encoded)
        pfs = self.fc2(encoded)
        age = torch.sigmoid(self.fc3(encoded))
        label = self.fc4(encoded)
        return os, pfs, age, label
