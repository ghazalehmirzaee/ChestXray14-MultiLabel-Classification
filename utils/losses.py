import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalWeightedCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        N, C = inputs.size()

        # Calculate class weights
        pos_weights = (targets.sum(dim=0) + 1e-5) / (N + 1e-5)
        neg_weights = 1 - pos_weights

        # Compute the focal weight
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Compute the weighted cross-entropy loss
        pos_loss = -self.alpha * focal_weight * targets * torch.log(probs + 1e-7)
        neg_loss = -(1 - self.alpha) * focal_weight * (1 - targets) * torch.log(1 - probs + 1e-7)

        loss = pos_weights * pos_loss + neg_weights * neg_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

