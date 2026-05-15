import torch
import torchaudio

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass


class LogMelFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=512, hop_length=160, win_length=400):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            normalized=False,
            mel_scale="slaney",
            norm="slaney",
        )

    def __call__(self, waveform):
        if isinstance(waveform, torch.Tensor):
            wav = waveform
        else:
            wav = torch.tensor(waveform, dtype=torch.float32)

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        mel = self.mel_transform(wav)
        log_mel = torch.clamp(mel, min=1e-10).log()
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel.squeeze(0).transpose(0, 1)

    def get_seq_length(self, audio_length):
        return audio_length // self.mel_transform.hop_length + 1


def resample_audio(waveform, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return waveform
    if isinstance(waveform, torch.Tensor):
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        return resampler(waveform)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(waveform)
    return resampled.squeeze(0).numpy()
