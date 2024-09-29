import torch
import torch.nn as nn

class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes, pearson_matrix, bio_matrix, use_correlation_learning=True):
        super(MultiLabelClassifier, self).__init__()
        self.backbone = EfficientNetAttention(num_classes)
        self.use_correlation_learning = use_correlation_learning
        if use_correlation_learning:
            self.correlation_adjustment = DiseaseCorrelationAdjustment(num_classes, pearson_matrix, bio_matrix)
        self.meta_learner = FullMetaLearner(num_classes, hidden_dim=64, num_layers=2, num_heads=4,
                                            num_classes=num_classes)

    def forward(self, x):
        initial_predictions, features = self.backbone(x)

        if self.use_correlation_learning:
            correlation_adjusted = self.correlation_adjustment(initial_predictions)
        else:
            correlation_adjusted = initial_predictions

        final_predictions = self.meta_learner(correlation_adjusted)

        return final_predictions, features

