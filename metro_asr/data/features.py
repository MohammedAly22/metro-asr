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


def load_audio_file(path):
    """
    Decode an audio file to a mono float32 tensor, returning (waveform, sample_rate).

    soundfile is tried first: torchaudio >= 2.9 routes ``load`` through
    TorchCodec, which is a separate install and not a dependency of this package.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data).mean(dim=1), sr
    except Exception:
        pass

    try:
        waveform, sr = torchaudio.load(path)
        return waveform.mean(dim=0), sr
    except Exception:
        pass

    try:
        import librosa
        data, sr = librosa.load(path, sr=None, mono=True)
        return torch.from_numpy(data).float(), sr
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode '{path}'. Install a decoder for this format: "
            "`pip install soundfile` (wav/flac/ogg) or `pip install librosa` (mp3/m4a)."
        ) from exc


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
