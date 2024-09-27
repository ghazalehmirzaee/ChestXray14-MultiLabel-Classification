import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=14):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return torch.sigmoid(self.efficientnet(x))

