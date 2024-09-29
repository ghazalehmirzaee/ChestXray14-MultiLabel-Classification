import torch
import torch.nn as nn

class CorrelationLearningModule(nn.Module):
    def __init__(self, num_classes):
        super(CorrelationLearningModule, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('w_pearson', torch.zeros(num_classes, num_classes))
        self.register_buffer('w_bio', torch.zeros(num_classes, num_classes))

    def forward(self, x):
        dynamic_weight = self.alpha * self.w_pearson + self.beta * self.w_bio
        correlation_adjusted = torch.matmul(x, dynamic_weight)
        return correlation_adjusted

    def update_pearson_matrix(self, pearson_matrix):
        self.w_pearson.copy_(torch.tensor(pearson_matrix, dtype=torch.float32))

    def update_bio_matrix(self, bio_matrix):
        self.w_bio.copy_(torch.tensor(bio_matrix, dtype=torch.float32))

