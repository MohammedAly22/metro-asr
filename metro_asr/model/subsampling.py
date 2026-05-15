import torch.nn as nn
import torch.nn.functional as F

from metro_asr.model.attention import RMSNorm


class ConvSubsampling(nn.Module):
    def __init__(self, n_mels, d_model, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        freq_out = (n_mels + 1) // 2
        freq_out = (freq_out + 1) // 2
        self.linear = nn.Linear(d_model * freq_out, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        x = x.unsqueeze(1)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))

        B, C, T, F_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F_dim)
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)

        if lengths is not None:
            lengths = ((lengths - 1) // 2 + 1)
            lengths = ((lengths - 1) // 2 + 1)
            lengths = lengths.clamp(min=1)

        return x, lengths
