import torch
import torch.nn as nn
import torch.nn.functional as F

from metro_asr.model.attention import RMSNorm


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        mid = max(1, channels // ratio)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        scale = x.mean(dim=1, keepdim=True)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, expansion_factor=2, se_ratio=4, dropout=0.1):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.norm = RMSNorm(d_model)
        self.pointwise_conv1 = nn.Linear(d_model, inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size,
            padding=kernel_size // 2, groups=inner_dim, bias=True,
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.se = SqueezeExcitation(inner_dim, ratio=se_ratio)
        self.pointwise_conv2 = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=-1)

        x = x.transpose(1, 2)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)

        x = self.se(x)
        x = self.pointwise_conv2(x)
        return residual + self.dropout(x)
