import os
import sys
import csv
import time
import glob
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils.config import load_config
from metro_asr.utils.logger import get_logger, print_banner
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.data.features import LogMelFeatureExtractor, resample_audio

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_PATH = "configs/metro_tiny.yaml"
TOKENIZER_DIR = "tokenizer"
CHECKPOINT_PATH = "checkpoints/metro-tiny/best_model.pt"
INPUT_DIR = "audio_input"
OUTPUT_CSV = "transcriptions.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
# ─────────────────────────────────────────────────────────────────────────────


def transcribe_file(model, feature_extractor, tokenizer, audio_path, device):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)

    if sr != feature_extractor.sample_rate:
        waveform = resample_audio(waveform, sr, feature_extractor.sample_rate)
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

    features = feature_extractor(waveform)
    features = features.unsqueeze(0).to(device)
    feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
        log_probs, out_lengths, _ = model(features, feature_lengths)

    decoded_ids = model.decode_greedy(log_probs, out_lengths)
    text = tokenizer.decode(decoded_ids[0])

    duration = waveform.shape[0] / feature_extractor.sample_rate if isinstance(waveform, torch.Tensor) else len(waveform) / feature_extractor.sample_rate
    return text, duration


def main():
    logger = get_logger("metro-asr")
    print_banner()

    config = load_config(CONFIG_PATH)
    tokenizer = build_tokenizer(config, TOKENIZER_DIR)

    feature_extractor = LogMelFeatureExtractor(
        sample_rate=config["audio"]["sample_rate"],
        n_mels=config["audio"]["n_mels"],
        n_fft=config["audio"]["n_fft"],
        hop_length=config["audio"]["hop_length"],
        win_length=config["audio"]["win_length"],
    )

    model = MetroASR.from_config(config)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"📦 Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        logger.warning(f"⚠️  No checkpoint found at {CHECKPOINT_PATH}")

    model = model.to(DEVICE).eval()
    logger.info(f"🏗️  Model: {model.count_parameters():,} params on {DEVICE}")

    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        audio_files.extend(glob.glob(os.path.join(INPUT_DIR, "**", f"*{ext}"), recursive=True))
    audio_files = sorted(set(audio_files))

    if not audio_files:
        logger.error(f"❌ No audio files found in {INPUT_DIR}")
        return

    logger.info(f"📂 Found {len(audio_files)} audio files")

    results = []
    total_time = 0.0
    total_duration = 0.0

    for i, audio_path in enumerate(audio_files, 1):
        try:
            start = time.time()
            text, duration = transcribe_file(model, feature_extractor, tokenizer, audio_path, DEVICE)
            elapsed = time.time() - start

            total_time += elapsed
            total_duration += duration

            results.append({
                "file": os.path.basename(audio_path),
                "path": audio_path,
                "transcription": text,
                "duration_s": f"{duration:.2f}",
                "inference_time_s": f"{elapsed:.3f}",
            })

            logger.info(f"[{i}/{len(audio_files)}] {os.path.basename(audio_path)}: {text[:80]}...")

        except Exception as e:
            logger.error(f"❌ Error processing {audio_path}: {e}")
            results.append({
                "file": os.path.basename(audio_path),
                "path": audio_path,
                "transcription": f"ERROR: {e}",
                "duration_s": "0",
                "inference_time_s": "0",
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "path", "transcription", "duration_s", "inference_time_s"])
        writer.writeheader()
        writer.writerows(results)

    rtf = total_time / max(total_duration, 1e-6)
    logger.info(f"\n✅ Done! {len(results)} files processed")
    logger.info(f"📊 Total audio: {total_duration:.1f}s | Time: {total_time:.1f}s | RTF: {rtf:.3f}")
    logger.info(f"💾 Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
