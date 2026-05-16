import os
import sys
import time
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils.config import load_config
from metro_asr.utils.logger import get_logger, print_banner
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.data.features import LogMelFeatureExtractor, resample_audio

# ========================= CONFIGURATION =========================
AUDIO_PATH = "test_samples/test_5.wav"
CONFIG_PATH = "configs/metro_tiny.yaml"
CHECKPOINT_PATH = "checkpoints/metro-tiny/best_model.pt"
TOKENIZER_DIR = "tokenizer_final"
DEVICE = "cpu"  # "auto", "cpu", or "cuda"

LM_PATH = "lm/lm_4gram.arpa"  # Path to KenLM .arpa or .bin file (None = greedy only)
BEAM_WIDTH = 100
LM_ALPHA = 0.5   # LM weight — keep moderate so LM doesn't kill English words
LM_BETA = 5.0    # Word insertion bonus — high to prevent word deletion

# Set to True to sweep alpha/beta and find optimal values
TUNE_MODE = False
TUNE_ALPHAS = [0.3, 0.5, 0.7, 1.0, 1.5]
TUNE_BETAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TUNE_REFERENCE = ""  # Ground truth text for WER calculation (optional)
# =================================================================


def run_single_inference(model, tokenizer, feature_extractor, beam_decoder, audio_path, device, logger):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0)
    duration = waveform.shape[0] / sr

    if sr != feature_extractor.sample_rate:
        waveform = resample_audio(waveform, sr, feature_extractor.sample_rate)
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

    features = feature_extractor(waveform)
    features = features.unsqueeze(0).to(device)
    feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
        log_probs, out_lengths, _ = model(features, feature_lengths)

    print()
    logger.info(f"🎵 Audio: {audio_path} ({duration:.1f}s)")
    print()

    start = time.time()
    greedy_ids = model.decode_greedy(log_probs, out_lengths)
    greedy_text = tokenizer.decode(greedy_ids[0])
    greedy_time = time.time() - start

    logger.info(f"🔤 Greedy decoding:")
    logger.info(f"   📝 {greedy_text}")
    logger.info(f"   ⏱️  {greedy_time:.3f}s | RTF: {greedy_time/duration:.4f}")

    if beam_decoder:
        print()
        start = time.time()
        beam_results = beam_decoder.decode(log_probs, out_lengths)
        beam_time = time.time() - start
        logger.info(f"🔍 Beam + LM decoding:")
        logger.info(f"   📝 {beam_results[0]}")
        logger.info(f"   ⏱️  {beam_time:.3f}s | RTF: {beam_time/duration:.4f}")

    print()
    logger.info("✅ Done!")
    return log_probs, out_lengths, duration


def run_tuning(log_probs, out_lengths, duration, tokenizer, greedy_text, logger):
    from metro_asr.model.decoder import CTCBeamSearchDecoder

    logger.info("🔧 Hyperparameter tuning mode")
    logger.info(f"   Sweeping alpha={TUNE_ALPHAS}, beta={TUNE_BETAS}")
    if TUNE_REFERENCE:
        logger.info(f"   📋 Reference: {TUNE_REFERENCE}")

    has_jiwer = False
    if TUNE_REFERENCE:
        try:
            from jiwer import wer as compute_wer
            has_jiwer = True
        except ImportError:
            pass

    print()
    logger.info(f"   🔤 Greedy: {greedy_text}")
    if has_jiwer:
        g_wer = compute_wer(TUNE_REFERENCE, greedy_text)
        logger.info(f"      WER: {g_wer:.2%}")
    print()

    best_wer = float("inf")
    best_alpha = 0
    best_beta = 0
    best_text = ""

    for alpha in TUNE_ALPHAS:
        for beta in TUNE_BETAS:
            decoder = CTCBeamSearchDecoder(
                tokenizer,
                lm_path=LM_PATH,
                beam_width=BEAM_WIDTH,
                alpha=alpha,
                beta=beta,
            )
            start = time.time()
            results = decoder.decode(log_probs, out_lengths)
            elapsed = time.time() - start
            text = results[0]

            line = f"   α={alpha:.1f} β={beta:.1f} | ⏱️ {elapsed:.3f}s | 📝 {text}"

            if has_jiwer:
                w = compute_wer(TUNE_REFERENCE, text)
                line += f" | WER: {w:.2%}"
                if w < best_wer:
                    best_wer = w
                    best_alpha = alpha
                    best_beta = beta
                    best_text = text

            logger.info(line)

    if has_jiwer and best_text:
        print()
        logger.info(f"🏆 Best: α={best_alpha:.1f} β={best_beta:.1f} | WER: {best_wer:.2%}")
        logger.info(f"   📝 {best_text}")


def main():
    logger = get_logger("metro-asr")
    print_banner()

    if DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE

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

    ckpt_path = CHECKPOINT_PATH
    if ckpt_path is None:
        ckpt_dir = config["training"].get("checkpoint_dir", "checkpoints")
        ckpt_path = os.path.join(ckpt_dir, "best_model.pt")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"📦 Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"⚠️  No checkpoint found at {ckpt_path}")

    model = model.to(device).eval()
    logger.info(f"🏗️  Model: {model.count_parameters():,} params on {device}")

    beam_decoder = None
    if LM_PATH and not TUNE_MODE:
        from metro_asr.model.decoder import CTCBeamSearchDecoder
        beam_decoder = CTCBeamSearchDecoder(
            tokenizer,
            lm_path=LM_PATH,
            beam_width=BEAM_WIDTH,
            alpha=LM_ALPHA,
            beta=LM_BETA,
        )
        logger.info(f"📖 LM decoder: beam={BEAM_WIDTH}, α={LM_ALPHA}, β={LM_BETA}")

    if not os.path.exists(AUDIO_PATH):
        logger.error(f"❌ Audio file not found: {AUDIO_PATH}")
        return

    log_probs, out_lengths, duration = run_single_inference(
        model, tokenizer, feature_extractor, beam_decoder, AUDIO_PATH, device, logger
    )

    if TUNE_MODE and LM_PATH:
        greedy_ids = model.decode_greedy(log_probs, out_lengths)
        greedy_text = tokenizer.decode(greedy_ids[0])
        run_tuning(log_probs, out_lengths, duration, tokenizer, greedy_text, logger)


if __name__ == "__main__":
    main()
