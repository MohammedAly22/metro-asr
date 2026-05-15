import torch.nn as nn
import torch.nn.functional as F

from metro_asr.model.attention import RMSNorm


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, ff_multiplier=3, dropout=0.1):
        super().__init__()
        inner_dim = int(d_model * ff_multiplier)
        self.norm = RMSNorm(d_model)
        self.gate_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.up_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(self.dropout(gate * up))
        return residual + 0.5 * self.dropout(x)
