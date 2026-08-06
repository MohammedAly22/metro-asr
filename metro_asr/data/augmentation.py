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
    """
    Speed perturbation (tempo + pitch shift together, matching SoX's `speed` effect)
    implemented via resampling rather than `torchaudio.sox_effects`, which recent
    torchaudio releases no longer ship — `apply_effects_tensor` raises
    `AttributeError: module 'torchaudio' has no attribute 'sox_effects'`.

    The trick: treat the waveform as if it were recorded at `sample_rate * factor`
    and resample it down/up to `sample_rate`. That changes the sample count by
    `1/factor` — shortening or lengthening the clip exactly as a speed change
    would — while the resampling filter reproduces the accompanying pitch shift.
    """

    def __init__(self, sample_rate=16000, factors=(0.9, 1.0, 1.1)):
        self.sample_rate = sample_rate
        self.factors = factors
        self._resamplers = {}

    def _resampler_for(self, factor):
        if factor not in self._resamplers:
            src_rate = int(round(self.sample_rate * factor))
            self._resamplers[factor] = torchaudio.transforms.Resample(
                orig_freq=src_rate, new_freq=self.sample_rate
            )
        return self._resamplers[factor]

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

        augmented = self._resampler_for(factor)(wav)
        return augmented.squeeze(0)
