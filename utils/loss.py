import torch
import torch.nn as nn
import torch.nn.functional as F


class FWCELoss(nn.Module):
    def __init__(self, class_frequencies, alpha=0.25, gamma=2, lambda1=0.01, lambda2=0.01):
        super(FWCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.register_buffer('class_weights', self._calculate_weights(class_frequencies))

    def _calculate_weights(self, class_frequencies):
        total_samples = sum(class_frequencies.values())
        weights = {cls: total_samples / (freq + 1e-5) for cls, freq in class_frequencies.items()}
        max_weight = max(weights.values())
        normalized_weights = {cls: weight / max_weight for cls, weight in weights.items()}
        return torch.tensor([normalized_weights[i] for i in range(len(class_frequencies))])

    def forward(self, inputs, targets, model_params):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        weighted_focal_loss = self.class_weights.unsqueeze(0) * focal_loss

        loss = weighted_focal_loss.mean()

        # L1 and L2 regularization
        l1_reg = sum(p.abs().sum() for p in model_params)
        l2_reg = sum(p.pow(2.0).sum() for p in model_params)

        loss += self.lambda1 * l1_reg + self.lambda2 * l2_reg

        return loss

# Usage example:
# class_frequencies = {0: 1000, 1: 500, ...}  # Example frequencies for each class
# criterion = FWCELoss(class_frequencies)
# outputs = torch.randn(32, 14)  # Example outputs (logits) from the model
# targets = torch.randint(0, 2, (32, 14)).float()  # Example binary targets
# model_params = model.parameters()  # Get model parameters for regularization
# loss = criterion(outputs, targets, model_params)
