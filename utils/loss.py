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
        normalized_weights = torch.tensor([weights[i] / max_weight for i in range(len(class_frequencies))])
        return normalized_weights

    def forward(self, inputs, targets, model_params):
        device = inputs.device
        targets = targets.to(device)
        self.class_weights = self.class_weights.to(device)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        weighted_focal_loss = self.class_weights.unsqueeze(0) * focal_loss
        loss = weighted_focal_loss.mean()

        l1_reg = sum(p.abs().sum() for p in model_params)
        l2_reg = sum(p.pow(2.0).sum() for p in model_params)
        loss += self.lambda1 * l1_reg + self.lambda2 * l2_reg

        return loss

