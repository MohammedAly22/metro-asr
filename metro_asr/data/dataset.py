import io
import os
import re
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

from metro_asr.data.features import LogMelFeatureExtractor, resample_audio
from metro_asr.data.augmentation import SpecAugment, SpeedPerturb


def _patch_hf_audio_decoder():
    try:
        from datasets.features.audio import Audio as HFAudio
    except ImportError:
        return

    _original = getattr(HFAudio, "_original_decode_example", None)
    if _original is not None:
        return

    HFAudio._original_decode_example = HFAudio.decode_example

    def _soundfile_decode(self, value, token_per_repo_id=None):
        if not self.decode:
            return value

        if not isinstance(value, dict):
            return value

        path = value.get("path", "")
        audio_bytes = value.get("bytes", None)

        try:
            if audio_bytes is not None:
                waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            elif path and os.path.exists(path):
                waveform, sr = sf.read(path, dtype="float32")
            else:
                return value
        except Exception:
            try:
                import librosa
                src = io.BytesIO(audio_bytes) if audio_bytes else path
                waveform, sr = librosa.load(src, sr=None)
                waveform = waveform.astype(np.float32)
            except Exception:
                return value

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        target_sr = getattr(self, "sampling_rate", None)
        if target_sr and target_sr != sr:
            try:
                import torchaudio
                wav_t = torch.tensor(waveform).unsqueeze(0)
                wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
                waveform = wav_t.squeeze(0).numpy()
            except Exception:
                try:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                except Exception:
                    ratio = target_sr / sr
                    new_len = int(len(waveform) * ratio)
                    indices = np.linspace(0, len(waveform) - 1, new_len)
                    waveform = np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)
            sr = target_sr

        return {"path": path or "", "array": waveform, "sampling_rate": sr}

    HFAudio.decode_example = _soundfile_decode


_patch_hf_audio_decoder()


AUDIO_COLUMN_NAMES = ["audio", "speech", "wav", "recording", "input_values"]
TEXT_COLUMN_NAMES = [
    "text", "sentence", "transcription", "transcript",
    "label", "target_text", "normalized_text",
]

DATASET_COLUMN_OVERRIDES = {
    "AlaaSamir/custom-egy-tts": {"audio": "audio", "text": "text"},
    "OmarAhmedSobhy/egyption-with-emotion-dataset": {"audio": "audio_player", "text": "cohere_transcription"},
    "MightyStudent/Egyptian-ASR-MGB-3": {"audio": "audio", "text": "sentence"},
    "MAdel121/arabic-egy-cleaned": {"audio": "audio", "text": "text"},
    "MAdel121/Continuation-egy-for-ultravox-v1": {"audio": "audio", "text": "text"},
    "Raniahossam33/Egyptian_TTS3RS": {"audio": "audio", "text": "text"},
    "ahmedbasemdev/egyptain-tts-dataset": {"audio": "audio", "text": "text"},
}


def detect_columns(dataset, dataset_name=None):
    if dataset_name and dataset_name in DATASET_COLUMN_OVERRIDES:
        override = DATASET_COLUMN_OVERRIDES[dataset_name]
        return override["audio"], override["text"]

    columns = dataset.column_names if hasattr(dataset, "column_names") else []
    if isinstance(columns, dict):
        split = list(columns.keys())[0]
        columns = columns[split]

    audio_col = None
    for name in AUDIO_COLUMN_NAMES:
        if name in columns:
            audio_col = name
            break

    text_col = None
    for name in TEXT_COLUMN_NAMES:
        if name in columns:
            text_col = name
            break

    return audio_col, text_col


_ALLOWED_CHARS = re.compile(
    r"["
    r"؀-ۿ"   # Arabic block (letters, diacritics, digits)
    r"ݐ-ݿ"   # Arabic Supplement
    r"ﭐ-﷿"   # Arabic Presentation Forms-A
    r"ﹰ-﻿"   # Arabic Presentation Forms-B
    r"a-zA-Z"          # English
    r"0-9"             # Digits
    r"\s"              # Whitespace
    r".,!?;:\-'\"،؛؟"  # Punctuation
    r"]"
)


def normalize_arabic_text(text):
    if not text:
        return ""
    text = re.sub(r"[ـ]+", "", text)
    text = "".join(c for c in text if _ALLOWED_CHARS.match(c))
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class MetroASRDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        config,
        is_training=True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.is_training = is_training

        audio_cfg = config["audio"]
        train_cfg = config["training"]

        self.sample_rate = audio_cfg["sample_rate"]
        self.max_duration = train_cfg.get("max_audio_duration", 30.0)
        self.min_duration = train_cfg.get("min_audio_duration", 0.5)

        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=audio_cfg["sample_rate"],
            n_mels=audio_cfg["n_mels"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
        )

        self.spec_augment = None
        self.speed_perturb = None
        if is_training:
            if train_cfg.get("spec_augment", True):
                self.spec_augment = SpecAugment(
                    freq_mask_param=train_cfg.get("freq_mask_param", 27),
                    time_mask_param=train_cfg.get("time_mask_param", 100),
                    n_freq_masks=train_cfg.get("n_freq_masks", 2),
                    n_time_masks=train_cfg.get("n_time_masks", 10),
                )
            if train_cfg.get("speed_perturb", True):
                factors = train_cfg.get("speed_perturb_factors", [0.9, 1.0, 1.1])
                self.speed_perturb = SpeedPerturb(
                    sample_rate=audio_cfg["sample_rate"],
                    factors=tuple(factors),
                )

    def __len__(self):
        return len(self.data)

    def _decode_audio(self, audio_data):
        if isinstance(audio_data, dict):
            if "array" in audio_data:
                waveform = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data.get("sampling_rate", self.sample_rate)
                return waveform, sr
            if "bytes" in audio_data and audio_data["bytes"]:
                try:
                    waveform, sr = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
                except Exception:
                    import librosa
                    waveform, sr = librosa.load(io.BytesIO(audio_data["bytes"]), sr=None)
                    waveform = waveform.astype(np.float32)
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform, sr
            if "path" in audio_data and audio_data["path"]:
                try:
                    waveform, sr = sf.read(audio_data["path"], dtype="float32")
                except Exception:
                    import librosa
                    waveform, sr = librosa.load(audio_data["path"], sr=None)
                    waveform = waveform.astype(np.float32)
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                return waveform, sr
        if isinstance(audio_data, (np.ndarray, list)):
            return np.array(audio_data, dtype=np.float32), self.sample_rate
        return np.array(audio_data, dtype=np.float32), self.sample_rate

    def __getitem__(self, idx):
        item = self.data[idx]

        audio_data = item["audio"]
        waveform, sr = self._decode_audio(audio_data)

        if sr != self.sample_rate:
            waveform = resample_audio(waveform, sr, self.sample_rate)
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()

        if self.is_training and self.speed_perturb is not None:
            waveform = self.speed_perturb(waveform)
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()

        features = self.feature_extractor(waveform)

        if self.is_training and self.spec_augment is not None:
            features = self.spec_augment(features)

        text = item.get("text", "")
        text = normalize_arabic_text(text)
        token_ids = self.tokenizer.encode(text)

        return {
            "features": features,
            "feature_length": features.shape[0],
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "token_length": len(token_ids),
            "text": text,
        }


def load_hf_datasets(dataset_names, config, cache_dir=None, streaming=False):
    from datasets import load_dataset, concatenate_datasets, Audio

    target_sr = config["audio"]["sample_rate"]
    all_datasets = []

    for ds_name in dataset_names:
        try:
            ds = load_dataset(ds_name, cache_dir=cache_dir, trust_remote_code=True)
        except TypeError:
            ds = load_dataset(ds_name, cache_dir=cache_dir)
        except Exception:
            try:
                ds = load_dataset(ds_name, split="train", cache_dir=cache_dir, trust_remote_code=True)
                ds = {"train": ds}
            except TypeError:
                try:
                    ds = load_dataset(ds_name, split="train", cache_dir=cache_dir)
                    ds = {"train": ds}
                except Exception as e:
                    print(f"❌ Failed to load {ds_name}: {e}")
                    continue
            except Exception as e:
                print(f"❌ Failed to load {ds_name}: {e}")
                continue

        if isinstance(ds, dict):
            splits = list(ds.keys())
            combined = concatenate_datasets([ds[s] for s in splits])
        else:
            combined = ds

        audio_col, text_col = detect_columns(combined, ds_name)
        if audio_col is None or text_col is None:
            cols = combined.column_names
            print(f"⚠️ Could not detect columns for {ds_name}. Available: {cols}")
            for candidate in AUDIO_COLUMN_NAMES:
                if candidate in cols:
                    audio_col = candidate
                    break
            for candidate in TEXT_COLUMN_NAMES:
                if candidate in cols:
                    text_col = candidate
                    break
            if audio_col is None or text_col is None:
                print(f"❌ Skipping {ds_name}: no audio/text columns found")
                continue

        if audio_col != "audio":
            combined = combined.rename_column(audio_col, "audio")
        if text_col != "text":
            combined = combined.rename_column(text_col, "text")

        keep_cols = {"audio", "text"}
        remove_cols = [c for c in combined.column_names if c not in keep_cols]
        if remove_cols:
            combined = combined.remove_columns(remove_cols)

        try:
            combined = combined.cast_column("audio", Audio(sampling_rate=target_sr))
        except Exception:
            pass

        all_datasets.append(combined)
        print(f"✅ Loaded {ds_name}: {len(combined)} samples")

    if not all_datasets:
        raise RuntimeError("No datasets loaded successfully.")

    merged = concatenate_datasets(all_datasets)
    print(f"\n📊 Total samples: {len(merged)}")
    return merged
