import torch
import torchaudio


class SpecAugment:
    def __init__(self, freq_mask_param=27, time_mask_param=100, n_freq_masks=2, n_time_masks=10):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, features):
        x = features.transpose(0, 1).unsqueeze(0)
        for _ in range(self.n_freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.n_time_masks):
            x = self.time_mask(x)
        return x.squeeze(0).transpose(0, 1)


class SpeedPerturb:
    def __init__(self, sample_rate=16000, factors=(0.9, 1.0, 1.1)):
        self.sample_rate = sample_rate
        self.factors = factors

    def __call__(self, waveform):
        factor = self.factors[torch.randint(0, len(self.factors), (1,)).item()]
        if factor == 1.0:
            return waveform

        if isinstance(waveform, torch.Tensor):
            wav = waveform
        else:
            wav = torch.tensor(waveform, dtype=torch.float32)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        effects = [["speed", str(factor)], ["rate", str(self.sample_rate)]]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(wav, self.sample_rate, effects)
        return augmented.squeeze(0)
