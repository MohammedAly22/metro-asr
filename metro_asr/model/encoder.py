import torch
import torch.nn as nn

from metro_asr.model.attention import MultiHeadSelfAttention, RMSNorm
from metro_asr.model.feed_forward import SwiGLUFeedForward
from metro_asr.model.convolution import ConvolutionModule


class MetroBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        ff_multiplier=3,
        conv_kernel_size=31,
        dropout=0.1,
        conv_expansion_factor=2,
        se_ratio=4,
        layer_drop_rate=0.0,
    ):
        super().__init__()
        self.ff1 = SwiGLUFeedForward(d_model, ff_multiplier, dropout)
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, conv_expansion_factor, se_ratio, dropout)
        self.ff2 = SwiGLUFeedForward(d_model, ff_multiplier, dropout)
        self.final_norm = RMSNorm(d_model)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, x, mask=None):
        if self.training and self.layer_drop_rate > 0.0:
            if torch.rand(1).item() < self.layer_drop_rate:
                return x

        x = self.ff1(x)
        residual = x
        x = self.attn_norm(x)
        x = residual + self.attn_dropout(self.attn(x, mask=mask))
        x = self.conv(x, mask=mask)
        x = self.ff2(x)
        x = self.final_norm(x)
        return x


class MetroEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_layers,
        n_heads,
        ff_multiplier=3,
        conv_kernel_size=31,
        dropout=0.1,
        conv_expansion_factor=2,
        se_ratio=4,
        stochastic_depth_rate=0.0,
        intermediate_ctc_layers=None,
        vocab_size=None,
    ):
        super().__init__()
        self.intermediate_ctc_layers = set(intermediate_ctc_layers or [])

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            drop_rate = stochastic_depth_rate * (i / max(n_layers - 1, 1))
            self.layers.append(
                MetroBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_multiplier=ff_multiplier,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    conv_expansion_factor=conv_expansion_factor,
                    se_ratio=se_ratio,
                    layer_drop_rate=drop_rate,
                )
            )

        self.intermediate_ctc_heads = nn.ModuleDict()
        if self.intermediate_ctc_layers and vocab_size:
            for layer_idx in self.intermediate_ctc_layers:
                head = nn.Linear(d_model, vocab_size)
                head.bias.data[0] = -3.0
                self.intermediate_ctc_heads[str(layer_idx)] = head

    def forward(self, x, mask=None):
        intermediate_logits = {}
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if i in self.intermediate_ctc_layers and str(i) in self.intermediate_ctc_heads:
                intermediate_logits[i] = self.intermediate_ctc_heads[str(i)](x)
        return x, intermediate_logits
