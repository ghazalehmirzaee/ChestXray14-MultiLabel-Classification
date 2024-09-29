import torch
import torch.nn as nn

class MetaLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4):
        super(MetaLearner, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.multi_head_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        gru_out, _ = self.gru(x)
        attn_out, _ = self.multi_head_attention(gru_out, gru_out, gru_out)
        output = self.fc(attn_out.squeeze(1))
        return output

