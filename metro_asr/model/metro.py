import torch
import torch.nn as nn
import torch.nn.functional as F

from metro_asr.model.subsampling import ConvSubsampling
from metro_asr.model.encoder import MetroEncoder


class MetroASR(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        audio_cfg = config["audio"]

        self.d_model = model_cfg["d_model"]
        self.vocab_size = config["tokenizer"]["vocab_size"]

        intermediate_ctc = config["training"].get("intermediate_ctc_layers", [])

        self.subsampling = ConvSubsampling(
            n_mels=audio_cfg["n_mels"],
            d_model=self.d_model,
            dropout=model_cfg["dropout"],
        )

        self.encoder = MetroEncoder(
            d_model=self.d_model,
            n_layers=model_cfg["n_layers"],
            n_heads=model_cfg["n_heads"],
            ff_multiplier=model_cfg["ff_multiplier"],
            conv_kernel_size=model_cfg["conv_kernel_size"],
            dropout=model_cfg["dropout"],
            conv_expansion_factor=model_cfg.get("conv_expansion_factor", 2),
            se_ratio=model_cfg.get("se_ratio", 4),
            stochastic_depth_rate=model_cfg.get("stochastic_depth_rate", 0.0),
            intermediate_ctc_layers=intermediate_ctc,
            vocab_size=self.vocab_size,
        )

        self.ctc_head = nn.Linear(self.d_model, self.vocab_size)
        self.ctc_head.bias.data[0] = -3.0

    def forward(self, features, feature_lengths):
        x, lengths = self.subsampling(features, feature_lengths)

        B, T, _ = x.shape
        if lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        else:
            mask = None

        x, intermediate_logits = self.encoder(x, mask=mask)
        logits = self.ctc_head(x)

        logits_f32 = logits.float()
        log_probs = F.log_softmax(logits_f32, dim=-1)

        intermediate_log_probs = {}
        for layer_idx, inter_logits in intermediate_logits.items():
            intermediate_log_probs[layer_idx] = F.log_softmax(inter_logits.float(), dim=-1)

        return log_probs, lengths, intermediate_log_probs

    def decode_greedy(self, log_probs, lengths=None):
        pred_ids = log_probs.argmax(dim=-1)
        decoded = []
        for i, seq in enumerate(pred_ids):
            max_t = lengths[i].item() if lengths is not None else seq.shape[0]
            tokens = []
            prev = -1
            for t in range(max_t):
                tok = seq[t].item()
                if tok != 0 and tok != prev:
                    tokens.append(tok)
                prev = tok
            decoded.append(tokens)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config):
        return cls(config)
