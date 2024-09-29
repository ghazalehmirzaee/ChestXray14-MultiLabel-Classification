import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

class EnsembleBinaryClassifiers(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EnsembleBinaryClassifiers, self).__init__()
        self.classifiers = nn.ModuleList([BinaryClassifier(input_dim) for _ in range(num_classes)])

    def forward(self, x):
        return torch.cat([classifier(x) for classifier in self.classifiers], dim=1)

