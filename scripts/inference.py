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

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_PATH = "configs/metro_22m.yaml"
TOKENIZER_DIR = "tokenizer"
CHECKPOINT_PATH = "checkpoints/metro-22m/best_model.pt"
AUDIO_PATH = "test_samples/test_1.wav"
"""
لق إني الحاجة اللي بعملها دي الكانت ممة تقولي أهم حاجة بستادي سكي امج اشاعزا في المدرسة

فعنا انبني مع بعض برنامك ننزيل اكلمضاي بلkش على المغضايل من غير ما نحتب ولا ساتركون مع بعات هتعمل دزاي عن طريق منة ربلت
منزرينا عندهم اgn فر

تعر بني مع بعض برنامك زل اجلك مباع لابlcton على المبيل من غير مانكتب ولا ساطركو مع بعض هتعمل ده ازاي عنطري منصت rبلد منازلن
عندهم اv fر

لإن الحاجة اللي بعملها دي آكانت مم تقولي اهم حاجة بستاني فكر مل اللنش زا في المدرسة

لأن الحاجة اللي بعملها دي آه كانت ماما تقولي أهم حاجة بسصتادي ذكي عمل اللي انت عايزاها في المدرسة

تعالوا نبني مع بعض برنامج تزيل أكل كام وباiيل apبلition على الموبايل من غير ما نكتب ولا ستر كود مع بعض هتعمل ده إزاي عن طريق
منصة رابلد منازلين عندهم gent f4r

rكانوا هما كلهم على بعضهم أربعة pages على فيسبوك أول عن آخر بيعملوا الحوار ده اه كنا احنا و ال manus و إnجeزnي وكان في بيتo
محمد حز                    

 ملك اتريندات يا سادة صائد الرتش في كل مكان وزمان اللي طلع نص البلد مسرح ومن النص التاني بيستخبي منه حتى الكلاب

 لأني الحاجة اللي بعملها دي آه كانت ماما تقولي أهم حاجة بستادي اذكي اعملي اللي انتش عايزاها في المدرسة
 """

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"  # Force CPU for testing
# ─────────────────────────────────────────────────────────────────────────────

def transcribe(model, feature_extractor, tokenizer, audio_path, device):
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
    return text


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

    if not os.path.exists(AUDIO_PATH):
        logger.error(f"❌ Audio file not found: {AUDIO_PATH}")
        return

    start = time.time()
    text = transcribe(model, feature_extractor, tokenizer, AUDIO_PATH, DEVICE)
    elapsed = time.time() - start

    waveform, sr = torchaudio.load(AUDIO_PATH)
    duration = waveform.shape[1] / sr
    rtf = elapsed / duration

    logger.info(f"🎤 Input:  {AUDIO_PATH} ({duration:.1f}s)")
    logger.info(f"📝 Output: {text}")
    logger.info(f"⏱️  Time:   {elapsed:.3f}s (RTF: {rtf:.3f})")


if __name__ == "__main__":
    main()
