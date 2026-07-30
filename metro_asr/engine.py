"""
Metro-ASR Inference Engine.

Provides the public API for end users:
    - MetroASREngine.from_pretrained("small")                  # HF alias
    - MetroASREngine.from_pretrained("MohammedAly22/metro-asr-small")
    - MetroASREngine.from_pretrained("checkpoints")            # local directory
    - engine.transcribe("audio.wav")
    - engine.transcribe_batch(["a.wav", "b.wav"])
    - engine.transcribe_stream(audio_chunk_generator)
"""

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union, Iterator, Generator

import torch
import torchaudio
import numpy as np

from metro_asr.utils.config import load_config
from metro_asr.model.metro import MetroASR as MetroASRModel
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.model.decoder import CTCBeamSearchDecoder
from metro_asr.data.features import (
    LogMelFeatureExtractor,
    resample_audio,
    load_audio_file as _read_audio_file,
)


HF_MODEL_REGISTRY = {
    "small": "MohammedAly22/metro-asr-small",
    "medium": "MohammedAly22/metro-asr-medium",
    "large": "MohammedAly22/metro-asr-large",
}

# Filenames looked up inside a model directory (local dir or downloaded snapshot).
CONFIG_FILENAMES = ("config.yaml", "config.yml")
CHECKPOINT_FILENAMES = ("model.pt", "best_model.pt", "final_model.pt")
LM_FILENAMES = ("lm_5gram.bin", "lm_4gram.bin", "lm.bin")

# Weights + config + tokenizer. The KenLM binary is multiple GB and is only
# fetched when the caller actually asks for it.
CORE_PATTERNS = ["config.yaml", "model.pt", "bpe.model", "bpe.vocab"]


def _first_existing(directory: str, filenames) -> Optional[str]:
    for name in filenames:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return None


@dataclass
class TranscriptionResult:
    text: str
    duration: float
    rtf: float
    inference_time: float
    decoding_time: float
    method: str


@dataclass
class StreamingChunk:
    text: str
    is_final: bool
    chunk_duration: float
    total_duration: float


class MetroASREngine:
    """
    Main inference engine for Metro-ASR.

    Usage:
        # From HuggingFace (auto-download)
        engine = MetroASREngine.from_pretrained("small")
        engine = MetroASREngine.from_pretrained("MohammedAly22/metro-asr-small")

        # From a directory you downloaded yourself — no network access
        engine = MetroASREngine.from_pretrained("checkpoints")
        engine = MetroASREngine.from_pretrained("checkpoints", lm_path="auto")

        # From individually placed files
        engine = MetroASREngine.from_local(
            config_path="configs/metro_small.yaml",
            checkpoint_path="checkpoints/model.pt",
            tokenizer_dir="checkpoints",
        )

        # Transcribe
        result = engine.transcribe("audio.wav")
        print(result.text)

        # Batch
        results = engine.transcribe_batch(["a.wav", "b.wav"])

        # Streaming
        for chunk in engine.transcribe_stream(audio_generator):
            print(chunk.text, end="", flush=True)
    """

    def __init__(
        self,
        model: MetroASRModel,
        tokenizer,
        feature_extractor: LogMelFeatureExtractor,
        device: torch.device,
        lm_path: Optional[str] = None,
        beam_width: int = 100,
        lm_alpha: float = 0.5,
        lm_beta: float = 5.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.device = device
        self.sample_rate = feature_extractor.sample_rate

        self._beam_decoder = None
        self._lm_path = lm_path
        self._beam_width = beam_width
        self._lm_alpha = lm_alpha
        self._lm_beta = lm_beta

        if lm_path and os.path.exists(lm_path):
            self._beam_decoder = CTCBeamSearchDecoder(
                tokenizer,
                lm_path=lm_path,
                beam_width=beam_width,
                alpha=lm_alpha,
                beta=lm_beta,
            )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        lm_path: Optional[str] = None,
        beam_width: int = 100,
        lm_alpha: float = 0.5,
        lm_beta: float = 5.0,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> "MetroASREngine":
        """
        Load a model, resolved in this order:

          1. A local directory holding ``config.yaml`` + ``model.pt`` + ``bpe.model``
             (e.g. the output of ``snapshot_download(...)``). Nothing is downloaded.
          2. A size alias — ``"small"``, ``"medium"``, ``"large"``.
          3. A HuggingFace repo id — ``"MohammedAly22/metro-asr-small"``.

        Pass ``lm_path="auto"`` to pick up a KenLM binary sitting next to the
        checkpoint (``lm_5gram.bin``); it is never downloaded implicitly.
        """
        if os.path.isdir(model_name_or_path):
            return cls.from_directory(
                model_name_or_path,
                device=device,
                lm_path=lm_path,
                beam_width=beam_width,
                lm_alpha=lm_alpha,
                lm_beta=lm_beta,
            )

        repo_id = HF_MODEL_REGISTRY.get(model_name_or_path, model_name_or_path)

        if "/" not in repo_id:
            raise ValueError(
                f"'{model_name_or_path}' is neither an existing directory, a known size "
                f"({', '.join(HF_MODEL_REGISTRY)}), nor a HuggingFace repo id of the form "
                "'owner/name'. To load a downloaded checkpoint, pass the path to the "
                "directory that contains config.yaml and model.pt."
            )

        model_dir = cls._download_snapshot(
            repo_id, cache_dir=cache_dir, revision=revision, with_lm=(lm_path == "auto")
        )
        return cls.from_directory(
            model_dir,
            device=device,
            lm_path=lm_path,
            beam_width=beam_width,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )

    @classmethod
    def from_directory(
        cls,
        model_dir: str,
        device: Optional[str] = None,
        lm_path: Optional[str] = None,
        beam_width: int = 100,
        lm_alpha: float = 0.5,
        lm_beta: float = 5.0,
    ) -> "MetroASREngine":
        """Load from a directory laid out like a released Metro-ASR checkpoint."""
        config_path = _first_existing(model_dir, CONFIG_FILENAMES)
        if config_path is None:
            raise FileNotFoundError(
                f"No model config in '{model_dir}' (looked for {', '.join(CONFIG_FILENAMES)}). "
                "Is this the directory you downloaded the checkpoint into?"
            )

        checkpoint_path = _first_existing(model_dir, CHECKPOINT_FILENAMES)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No model weights in '{model_dir}' (looked for {', '.join(CHECKPOINT_FILENAMES)})."
            )

        return cls.from_local(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            tokenizer_dir=model_dir,
            device=device,
            lm_path=lm_path,
            beam_width=beam_width,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )

    @staticmethod
    def _download_snapshot(
        repo_id: str,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
        with_lm: bool = False,
    ) -> str:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading models. "
                "Install it with: pip install huggingface_hub"
            )

        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "metro-asr")

        patterns = list(CORE_PATTERNS)
        if with_lm:
            patterns.append("*.bin")

        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=patterns,
        )

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        tokenizer_dir: str,
        device: Optional[str] = None,
        lm_path: Optional[str] = None,
        beam_width: int = 100,
        lm_alpha: float = 0.5,
        lm_beta: float = 5.0,
    ) -> "MetroASREngine":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        if lm_path == "auto":
            lm_path = _first_existing(os.path.dirname(checkpoint_path) or ".", LM_FILENAMES)

        config = load_config(config_path)
        tokenizer = build_tokenizer(config, tokenizer_dir)

        audio_cfg = config["audio"]
        feature_extractor = LogMelFeatureExtractor(
            sample_rate=audio_cfg["sample_rate"],
            n_mels=audio_cfg["n_mels"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
        )

        model = MetroASRModel.from_config(config)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
            lm_path=lm_path,
            beam_width=beam_width,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )

    def _load_audio(self, audio_input) -> torch.Tensor:
        if isinstance(audio_input, str):
            waveform, sr = _read_audio_file(audio_input)
        elif isinstance(audio_input, tuple):
            sr, waveform = audio_input
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
        elif isinstance(audio_input, np.ndarray):
            waveform = torch.tensor(audio_input, dtype=torch.float32)
            sr = self.sample_rate
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input
            sr = self.sample_rate
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
        else:
            raise TypeError(
                f"Unsupported audio input type: {type(audio_input)}. "
                "Pass a file path, (sample_rate, tensor) tuple, numpy array, or torch.Tensor."
            )

        if sr != self.sample_rate:
            waveform = resample_audio(waveform, sr, self.sample_rate)
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform

    def transcribe(
        self,
        audio_input: Union[str, torch.Tensor, np.ndarray, tuple],
        beam_search: bool = False,
        beam_width: Optional[int] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
        lm_path: Optional[str] = None,
    ) -> TranscriptionResult:
        waveform = self._load_audio(audio_input)
        duration = waveform.shape[0] / self.sample_rate

        features = self.feature_extractor(waveform)
        features = features.unsqueeze(0).to(self.device)
        feature_lengths = torch.tensor(
            [features.shape[1]], dtype=torch.long, device=self.device
        )

        t_start = time.time()
        with torch.no_grad():
            log_probs, out_lengths, _ = self.model(features, feature_lengths)
        inference_time = time.time() - t_start

        t_decode = time.time()
        text, method = self._decode(
            log_probs, out_lengths, beam_search,
            beam_width, lm_alpha, lm_beta, lm_path,
        )
        decoding_time = time.time() - t_decode

        total_time = inference_time + decoding_time
        rtf = total_time / duration if duration > 0 else 0.0

        return TranscriptionResult(
            text=text,
            duration=duration,
            rtf=rtf,
            inference_time=inference_time,
            decoding_time=decoding_time,
            method=method,
        )

    def transcribe_batch(
        self,
        audio_inputs: List[Union[str, torch.Tensor, np.ndarray, tuple]],
        beam_search: bool = False,
        beam_width: Optional[int] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
    ) -> List[TranscriptionResult]:
        waveforms = [self._load_audio(a) for a in audio_inputs]
        durations = [w.shape[0] / self.sample_rate for w in waveforms]

        features_list = [self.feature_extractor(w) for w in waveforms]
        max_len = max(f.shape[0] for f in features_list)
        n_mels = features_list[0].shape[1]

        padded = torch.zeros(len(features_list), max_len, n_mels)
        lengths = torch.zeros(len(features_list), dtype=torch.long)
        for i, f in enumerate(features_list):
            padded[i, :f.shape[0], :] = f
            lengths[i] = f.shape[0]

        padded = padded.to(self.device)
        lengths = lengths.to(self.device)

        t_start = time.time()
        with torch.no_grad():
            log_probs, out_lengths, _ = self.model(padded, lengths)
        inference_time = time.time() - t_start

        results = []
        for i in range(len(audio_inputs)):
            t_decode = time.time()
            single_log_probs = log_probs[i : i + 1]
            single_lengths = out_lengths[i : i + 1]

            text, method = self._decode(
                single_log_probs, single_lengths, beam_search,
                beam_width, lm_alpha, lm_beta,
            )
            decoding_time = time.time() - t_decode

            per_sample_inf = inference_time / len(audio_inputs)
            total = per_sample_inf + decoding_time
            rtf = total / durations[i] if durations[i] > 0 else 0.0

            results.append(TranscriptionResult(
                text=text,
                duration=durations[i],
                rtf=rtf,
                inference_time=per_sample_inf,
                decoding_time=decoding_time,
                method=method,
            ))

        return results

    def transcribe_stream(
        self,
        audio_generator: Generator,
        chunk_duration: float = 5.0,
        overlap_duration: float = 1.0,
        beam_search: bool = False,
    ) -> Iterator[StreamingChunk]:
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap_duration * self.sample_rate)
        buffer = torch.tensor([], dtype=torch.float32)
        total_duration = 0.0

        for chunk_data in audio_generator:
            if isinstance(chunk_data, np.ndarray):
                chunk_data = torch.tensor(chunk_data, dtype=torch.float32)
            elif isinstance(chunk_data, bytes):
                chunk_data = torch.frombuffer(chunk_data, dtype=torch.float32)

            if chunk_data.dim() > 1:
                chunk_data = chunk_data.mean(dim=0)

            buffer = torch.cat([buffer, chunk_data])

            while buffer.shape[0] >= chunk_samples:
                segment = buffer[:chunk_samples]
                buffer = buffer[chunk_samples - overlap_samples:]

                segment_duration = segment.shape[0] / self.sample_rate
                total_duration += segment_duration - (overlap_duration if total_duration > 0 else 0)

                features = self.feature_extractor(segment)
                features = features.unsqueeze(0).to(self.device)
                feature_lengths = torch.tensor(
                    [features.shape[1]], dtype=torch.long, device=self.device
                )

                with torch.no_grad():
                    log_probs, out_lengths, _ = self.model(features, feature_lengths)

                text, _ = self._decode(
                    log_probs, out_lengths, beam_search,
                )

                yield StreamingChunk(
                    text=text,
                    is_final=False,
                    chunk_duration=segment_duration,
                    total_duration=total_duration,
                )

        if buffer.shape[0] > 0:
            segment_duration = buffer.shape[0] / self.sample_rate
            total_duration += segment_duration

            features = self.feature_extractor(buffer)
            features = features.unsqueeze(0).to(self.device)
            feature_lengths = torch.tensor(
                [features.shape[1]], dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                log_probs, out_lengths, _ = self.model(features, feature_lengths)

            text, _ = self._decode(
                log_probs, out_lengths, beam_search,
            )

            yield StreamingChunk(
                text=text,
                is_final=True,
                chunk_duration=segment_duration,
                total_duration=total_duration,
            )

    def _decode(
        self,
        log_probs,
        out_lengths,
        beam_search=False,
        beam_width=None,
        lm_alpha=None,
        lm_beta=None,
        lm_path=None,
    ):
        if beam_search:
            decoder = self._get_beam_decoder(lm_path, beam_width, lm_alpha, lm_beta)
            if decoder is not None:
                results = decoder.decode(log_probs, out_lengths)
                bw = beam_width or self._beam_width
                a = lm_alpha if lm_alpha is not None else self._lm_alpha
                b = lm_beta if lm_beta is not None else self._lm_beta
                return results[0], f"beam_search+lm (beam={bw}, alpha={a}, beta={b})"

        decoded_ids = self.model.decode_greedy(log_probs, out_lengths)
        text = self.tokenizer.decode(decoded_ids[0])
        return text, "greedy"

    def _get_beam_decoder(self, lm_path=None, beam_width=None, lm_alpha=None, lm_beta=None):
        lm = lm_path or self._lm_path
        bw = beam_width or self._beam_width
        a = lm_alpha if lm_alpha is not None else self._lm_alpha
        b = lm_beta if lm_beta is not None else self._lm_beta

        if lm_path and lm_path != self._lm_path:
            return CTCBeamSearchDecoder(
                self.tokenizer, lm_path=lm, beam_width=bw, alpha=a, beta=b,
            )

        if self._beam_decoder is not None:
            return self._beam_decoder

        if lm and os.path.exists(lm):
            self._beam_decoder = CTCBeamSearchDecoder(
                self.tokenizer, lm_path=lm, beam_width=bw, alpha=a, beta=b,
            )
            return self._beam_decoder

        return None

    @property
    def param_count(self) -> int:
        return self.model.count_parameters()

    @property
    def has_lm(self) -> bool:
        """True when a language head is loaded and `beam_search=True` will use it."""
        return self._beam_decoder is not None

    @property
    def lm_path(self) -> Optional[str]:
        return self._lm_path if self.has_lm else None

    def load_lm(
        self,
        lm_path: str,
        beam_width: Optional[int] = None,
        lm_alpha: Optional[float] = None,
        lm_beta: Optional[float] = None,
    ) -> "MetroASREngine":
        """
        Attach (or swap) the n-gram language head at runtime.

        The acoustic model is untouched, so a domain-specific head trained on
        text alone can be plugged into an already-loaded engine.
        """
        if not os.path.exists(lm_path):
            raise FileNotFoundError(f"Language model not found: {lm_path}")

        self._lm_path = lm_path
        if beam_width is not None:
            self._beam_width = beam_width
        if lm_alpha is not None:
            self._lm_alpha = lm_alpha
        if lm_beta is not None:
            self._lm_beta = lm_beta

        self._beam_decoder = CTCBeamSearchDecoder(
            self.tokenizer,
            lm_path=self._lm_path,
            beam_width=self._beam_width,
            alpha=self._lm_alpha,
            beta=self._lm_beta,
        )
        return self

    def __repr__(self):
        return (
            f"MetroASREngine(\n"
            f"  params={self.param_count:,},\n"
            f"  device={self.device},\n"
            f"  sample_rate={self.sample_rate},\n"
            f"  lm={'loaded' if self.has_lm else 'none'},\n"
            f")"
        )

    def __call__(self, audio_input, **kwargs):
        return self.transcribe(audio_input, **kwargs)
