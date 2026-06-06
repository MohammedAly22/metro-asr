import os
import time
from dataclasses import dataclass

import torch
import torchaudio

from metro_asr.utils.config import load_config
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.model.decoder import CTCBeamSearchDecoder
from metro_asr.data.features import LogMelFeatureExtractor, resample_audio


MODELS = {
    "tiny": {
        "config": "configs/metro_tiny_char.yaml",
        "checkpoint": "checkpoints/metro-22m/best_model.pt",
        "tokenizer_dir": "tokenizer",
    },
    "tiny-bpe": {
        "config": "configs/metro_tiny.yaml",
        "checkpoint": "checkpoints/metro-tiny/best_model.pt",
        "tokenizer_dir": "tokenizer_final",
    },
    "small": {
        "config": "configs/metro_small.yaml",
        "checkpoint": "checkpoints/metro-small-v2/best_model.pt",
        "tokenizer_dir": "tokenizer_bpe5k_v2",
    },
    "medium": {
        "config": "configs/metro_medium.yaml",
        "checkpoint": "checkpoints/metro-medium/best_model.pt",
        "tokenizer_dir": "tokenizer_bpe2000",
    },
    "large": {
        "config": "configs/metro_large.yaml",
        "checkpoint": "checkpoints/metro-large/best_model.pt",
        "tokenizer_dir": "tokenizer_bpe4000",
    },
    "xlarge": {
        "config": "configs/metro_xlarge.yaml",
        "checkpoint": "checkpoints/metro-xlarge/best_model.pt",
        "tokenizer_dir": "tokenizer_bpe8000",
    },
}


@dataclass
class TranscriptionResult:
    text: str
    duration: float
    rtf: float
    inference_time: float
    decoding_time: float
    method: str


class MetroASRPipeline:
    """
    High-level inference pipeline for Metro-ASR.

    Usage:
        pipe = MetroASRPipeline("tiny")
        result = pipe("audio.wav")
        print(result.text)

        # With beam search + LM
        result = pipe("audio.wav", beam_search=True, lm_path="lm/lm_4gram.arpa")
        print(result.text)
    """

    def __init__(
        self,
        model_size="tiny",
        checkpoint=None,
        config_path=None,
        tokenizer_dir=None,
        device=None,
        lm_path=None,
        beam_width=100,
        lm_alpha=0.5,
        lm_beta=5.0,
    ):
        if model_size not in MODELS:
            sizes = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model size '{model_size}'. Choose from: {sizes}")

        defaults = MODELS[model_size]
        config_path = config_path or defaults["config"]
        checkpoint = checkpoint or defaults["checkpoint"]
        tokenizer_dir = tokenizer_dir or defaults["tokenizer_dir"]

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.config = load_config(config_path)
        self.tokenizer = build_tokenizer(self.config, tokenizer_dir)

        audio_cfg = self.config["audio"]
        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=audio_cfg["sample_rate"],
            n_mels=audio_cfg["n_mels"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
        )
        self.sample_rate = audio_cfg["sample_rate"]

        self.model = MetroASR.from_config(self.config)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device).eval()
        self.param_count = self.model.count_parameters()

        self._beam_decoder = None
        if lm_path and os.path.exists(lm_path):
            self._beam_decoder = CTCBeamSearchDecoder(
                self.tokenizer,
                lm_path=lm_path,
                beam_width=beam_width,
                alpha=lm_alpha,
                beta=lm_beta,
            )

        self._beam_width = beam_width
        self._lm_alpha = lm_alpha
        self._lm_beta = lm_beta
        self._lm_path = lm_path

    def _load_audio(self, audio_input):
        if isinstance(audio_input, str):
            waveform, sr = torchaudio.load(audio_input)
        elif isinstance(audio_input, tuple):
            sr, waveform = audio_input
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        elif isinstance(audio_input, torch.Tensor):
            waveform = audio_input
            sr = self.sample_rate
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            raise TypeError(
                f"Unsupported audio input type: {type(audio_input)}. "
                "Pass a file path (str), a (sample_rate, tensor) tuple, or a torch.Tensor."
            )

        waveform = waveform.mean(dim=0)

        if sr != self.sample_rate:
            waveform = resample_audio(waveform, sr, self.sample_rate)
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform, dtype=torch.float32)

        return waveform

    def __call__(self, audio_input, beam_search=False, **kwargs):
        return self.transcribe(audio_input, beam_search=beam_search, **kwargs)

    def transcribe(
        self,
        audio_input,
        beam_search=False,
        beam_width=None,
        lm_alpha=None,
        lm_beta=None,
        lm_path=None,
    ):
        waveform = self._load_audio(audio_input)
        duration = waveform.shape[0] / self.sample_rate

        features = self.feature_extractor(waveform)
        features = features.unsqueeze(0).to(self.device)
        feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=self.device)

        t_start = time.time()
        with torch.no_grad():
            log_probs, out_lengths, _ = self.model(features, feature_lengths)
        inference_time = time.time() - t_start

        t_decode = time.time()

        if beam_search:
            decoder = self._get_beam_decoder(
                lm_path=lm_path,
                beam_width=beam_width,
                lm_alpha=lm_alpha,
                lm_beta=lm_beta,
            )
            if decoder is not None:
                results = decoder.decode(log_probs, out_lengths)
                text = results[0]
                bw = beam_width or self._beam_width
                a = lm_alpha if lm_alpha is not None else self._lm_alpha
                b = lm_beta if lm_beta is not None else self._lm_beta
                method = f"beam_search+lm (beam={bw}, alpha={a}, beta={b})"
            else:
                decoded_ids = self.model.decode_greedy(log_probs, out_lengths)
                text = self.tokenizer.decode(decoded_ids[0])
                method = "greedy (lm not available)"
        else:
            decoded_ids = self.model.decode_greedy(log_probs, out_lengths)
            text = self.tokenizer.decode(decoded_ids[0])
            method = "greedy"

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

    def transcribe_batch(self, audio_paths, beam_search=False, **kwargs):
        return [self.transcribe(p, beam_search=beam_search, **kwargs) for p in audio_paths]

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

    def __repr__(self):
        return (
            f"MetroASRPipeline(\n"
            f"  model={self.config['model']['name']},\n"
            f"  params={self.param_count:,},\n"
            f"  device={self.device},\n"
            f"  tokenizer={self.config['tokenizer']['type']} "
            f"(vocab={self.tokenizer.vocab_size}),\n"
            f"  lm={'loaded' if self._beam_decoder else 'none'}\n"
            f")"
        )
