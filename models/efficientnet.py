import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetBackbone(nn.Module):
    def __init__(self, version='b0', pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(f'efficientnet-{version}') if pretrained else EfficientNet.from_name(f'efficientnet-{version}')
        self.num_features = self.efficientnet._fc.in_features

    def forward(self, x):
        return self.efficientnet.extract_features(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EfficientNetWithAttention(nn.Module):
    def __init__(self, version='b0', num_classes=14, pretrained=True):
        super(EfficientNetWithAttention, self).__init__()
        self.backbone = EfficientNetBackbone(version, pretrained)
        self.attention = SEBlock(self.backbone.num_features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        attended_features = self.attention(features)
        pooled_features = self.global_pool(attended_features).view(x.size(0), -1)
        output = self.fc(pooled_features)
        return output, pooled_features

