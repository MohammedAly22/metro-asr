import os
import sys
import time
import csv
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils import enable_utf8_stdout

enable_utf8_stdout()

from metro_asr.utils.config import load_config
from metro_asr.utils.logger import print_banner
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.model.decoder import CTCBeamSearchDecoder
from metro_asr.data.features import LogMelFeatureExtractor, resample_audio

# ========================= CONFIGURATION =========================

# Models to evaluate — add as many as you want to compare
MODELS = [
    {
        "name": "Metro-Small",
        "config": "checkpoints/config.yaml",
        "checkpoint": "checkpoints/model.pt",
        "tokenizer_dir": "checkpoints",
    },
]

# Test data
TEST_DATA_PATH = "data_prepared/test"

# Language Model (set LM_PATH to None to disable)
LM_PATH = "checkpoints/lm_5gram.bin"  # None = greedy only
BEAM_WIDTH = 200
LM_ALPHA = 0.5
LM_BETA = 3.0

# Decoding modes to evaluate
USE_GREEDY = True
USE_BEAM_LM = True  # Only if LM_PATH is set

# Output
OUTPUT_DIR = "eval_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================


def print_section(title, icon="📋"):
    width = 60
    print()
    print(f"  {icon} {'─' * 3} {title} {'─' * (width - len(title) - 6)}")
    print()


def is_code_switching(text):
    has_arabic = any("؀" <= c <= "ۿ" for c in text)
    has_english = any("a" <= c.lower() <= "z" for c in text)
    return has_arabic and has_english


def is_pure_arabic(text):
    has_arabic = any("؀" <= c <= "ۿ" for c in text)
    has_english = any("a" <= c.lower() <= "z" for c in text)
    return has_arabic and not has_english


def load_test_data(test_path):
    from datasets import load_from_disk
    dataset = load_from_disk(test_path)
    return dataset


def decode_audio(audio_data, target_sr):
    import numpy as np
    import soundfile as sf
    import io

    if isinstance(audio_data, dict):
        if "array" in audio_data:
            waveform = np.array(audio_data["array"], dtype=np.float32)
            sr = audio_data.get("sampling_rate", target_sr)
            return waveform, sr
        if "bytes" in audio_data and audio_data["bytes"]:
            waveform, sr = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            return waveform, sr
        if "path" in audio_data and audio_data["path"] and os.path.exists(audio_data["path"]):
            waveform, sr = sf.read(audio_data["path"], dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            return waveform, sr
    return None, None


def compute_metrics(predictions, references):
    from jiwer import wer as compute_wer, cer as compute_cer
    filtered = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not filtered:
        return 1.0, 1.0
    preds, refs = zip(*filtered)
    w = compute_wer(list(refs), list(preds))
    c = compute_cer(list(refs), list(preds))
    return w, c


def build_beam_decoder(tokenizer):
    if USE_BEAM_LM and LM_PATH and os.path.exists(LM_PATH):
        return CTCBeamSearchDecoder(
            tokenizer,
            lm_path=LM_PATH,
            beam_width=BEAM_WIDTH,
            alpha=LM_ALPHA,
            beta=LM_BETA,
        )
    return None


def evaluate_model(model_info, test_data, feature_extractor_cache, lm_cache, device):
    name = model_info["name"]
    config = load_config(model_info["config"])
    tokenizer = build_tokenizer(config, model_info["tokenizer_dir"])

    audio_cfg = config["audio"]
    target_sr = audio_cfg["sample_rate"]

    cache_key = f"{audio_cfg['sample_rate']}_{audio_cfg['n_mels']}_{audio_cfg['n_fft']}"
    if cache_key not in feature_extractor_cache:
        feature_extractor_cache[cache_key] = LogMelFeatureExtractor(
            sample_rate=audio_cfg["sample_rate"],
            n_mels=audio_cfg["n_mels"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
        )
    feature_extractor = feature_extractor_cache[cache_key]

    model = MetroASR.from_config(config)
    ckpt = torch.load(model_info["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    param_count = model.count_parameters()

    # Reuse LM decoder per tokenizer (tokenizer identity determines vocab)
    tokenizer_key = model_info["tokenizer_dir"]
    if tokenizer_key not in lm_cache:
        lm_cache[tokenizer_key] = build_beam_decoder(tokenizer)
    beam_decoder = lm_cache[tokenizer_key]

    results = {
        "greedy": {"predictions": [], "references": [], "rtfs": [], "categories": []},
    }
    if beam_decoder:
        results["beam_lm"] = {"predictions": [], "references": [], "rtfs": [], "categories": []}

    skipped = 0
    total = len(test_data)

    print(f"  🏗️  Model: {name}")
    print(f"  📦 Checkpoint: {model_info['checkpoint']}")
    print(f"  🔢 Parameters: {param_count:,}")
    print(f"  🎯 Test samples: {total}")
    print()

    for i in tqdm(range(total), desc=f"  🔄 {name}", ncols=80, leave=True):
        item = test_data[i]
        audio_data = item["audio"]
        reference = item.get("text", "")

        if not reference or not reference.strip():
            skipped += 1
            continue

        waveform, sr = decode_audio(audio_data, target_sr)
        if waveform is None:
            skipped += 1
            continue

        if sr != target_sr:
            waveform = resample_audio(waveform, sr, target_sr)
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()

        duration = len(waveform) / target_sr
        if duration < 0.5 or duration > 30.0:
            skipped += 1
            continue

        if not isinstance(waveform, torch.Tensor):
            waveform_t = torch.tensor(waveform, dtype=torch.float32)
        else:
            waveform_t = waveform

        features = feature_extractor(waveform_t)
        features = features.unsqueeze(0).to(device)
        feature_lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=device)

        with torch.no_grad():
            log_probs, out_lengths, _ = model(features, feature_lengths)

        # Categorize sample
        if is_code_switching(reference):
            category = "cs"
        elif is_pure_arabic(reference):
            category = "arabic"
        else:
            category = "other"

        # Greedy
        if USE_GREEDY:
            t0 = time.time()
            decoded_ids = model.decode_greedy(log_probs, out_lengths)
            pred_text = tokenizer.decode(decoded_ids[0])
            greedy_time = time.time() - t0
            rtf = greedy_time / duration if duration > 0 else 0

            results["greedy"]["predictions"].append(pred_text)
            results["greedy"]["references"].append(reference)
            results["greedy"]["rtfs"].append(rtf)
            results["greedy"]["categories"].append(category)

        # Beam + LM
        if beam_decoder:
            t0 = time.time()
            beam_results = beam_decoder.decode(log_probs, out_lengths)
            beam_time = time.time() - t0
            rtf = beam_time / duration if duration > 0 else 0

            results["beam_lm"]["predictions"].append(beam_results[0])
            results["beam_lm"]["references"].append(reference)
            results["beam_lm"]["rtfs"].append(rtf)
            results["beam_lm"]["categories"].append(category)

    print(f"  ✅ Evaluated: {total - skipped} | ⏭️  Skipped: {skipped}")

    # Compute metrics overall + per category
    metrics = {}
    for method, data in results.items():
        if not data["predictions"]:
            continue

        wer_all, cer_all = compute_metrics(data["predictions"], data["references"])
        avg_rtf = sum(data["rtfs"]) / len(data["rtfs"]) if data["rtfs"] else 0

        # Split by category
        cs_preds = [p for p, c in zip(data["predictions"], data["categories"]) if c == "cs"]
        cs_refs = [r for r, c in zip(data["references"], data["categories"]) if c == "cs"]
        ar_preds = [p for p, c in zip(data["predictions"], data["categories"]) if c == "arabic"]
        ar_refs = [r for r, c in zip(data["references"], data["categories"]) if c == "arabic"]

        wer_cs, cer_cs = compute_metrics(cs_preds, cs_refs) if cs_preds else (0.0, 0.0)
        wer_ar, cer_ar = compute_metrics(ar_preds, ar_refs) if ar_preds else (0.0, 0.0)

        metrics[method] = {
            "wer_all": wer_all, "cer_all": cer_all, "avg_rtf": avg_rtf,
            "wer_cs": wer_cs, "cer_cs": cer_cs, "n_cs": len(cs_preds),
            "wer_ar": wer_ar, "cer_ar": cer_ar, "n_ar": len(ar_preds),
        }

    return results, metrics, param_count


def save_csv(all_results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        headers = ["Index", "Category", "Reference"]
        for model_name, methods in all_results.items():
            for method in methods.keys():
                headers.append(f"{model_name} ({method})")
        writer.writerow(headers)

        first_model = list(all_results.keys())[0]
        first_method = list(all_results[first_model].keys())[0]
        n_samples = len(all_results[first_model][first_method]["references"])

        for i in range(n_samples):
            first_data = all_results[first_model][first_method]
            category = first_data["categories"][i] if i < len(first_data["categories"]) else ""
            ref = first_data["references"][i] if i < len(first_data["references"]) else ""

            row = [i, category, ref]
            for model_name, methods in all_results.items():
                for method, data in methods.items():
                    if i < len(data["predictions"]):
                        row.append(data["predictions"][i])
                    else:
                        row.append("")
            writer.writerow(row)


def print_results_table(all_metrics):
    # Overall
    print_section("Overall Results", "🏆")
    header = f"  {'Model':<35} {'Method':<10} {'WER':<9} {'CER':<9} {'RTF':<8}"
    print(header)
    print(f"  {'─' * 35} {'─' * 10} {'─' * 9} {'─' * 9} {'─' * 8}")

    best_wer = float("inf")
    best_combo = ""
    for model_name, data in all_metrics.items():
        for method, m in data["metrics"].items():
            wer_s = f"{m['wer_all']:.2%}"
            cer_s = f"{m['cer_all']:.2%}"
            rtf_s = f"{m['avg_rtf']:.4f}"
            print(f"  {model_name:<35} {method:<10} {wer_s:<9} {cer_s:<9} {rtf_s:<8}")
            if m["wer_all"] < best_wer:
                best_wer = m["wer_all"]
                best_combo = f"{model_name} ({method})"
    print()
    print(f"  🥇 Best Overall: {best_combo} — WER: {best_wer:.2%}")

    # Pure Arabic
    print_section("Pure Arabic Results 🇪🇬", "📊")
    header = f"  {'Model':<35} {'Method':<10} {'WER':<9} {'CER':<9} {'Samples':<8}"
    print(header)
    print(f"  {'─' * 35} {'─' * 10} {'─' * 9} {'─' * 9} {'─' * 8}")

    best_wer_ar = float("inf")
    best_combo_ar = ""
    for model_name, data in all_metrics.items():
        for method, m in data["metrics"].items():
            if m["n_ar"] == 0:
                continue
            wer_s = f"{m['wer_ar']:.2%}"
            cer_s = f"{m['cer_ar']:.2%}"
            n_s = str(m["n_ar"])
            print(f"  {model_name:<35} {method:<10} {wer_s:<9} {cer_s:<9} {n_s:<8}")
            if m["wer_ar"] < best_wer_ar:
                best_wer_ar = m["wer_ar"]
                best_combo_ar = f"{model_name} ({method})"
    print()
    print(f"  🥇 Best Arabic: {best_combo_ar} — WER: {best_wer_ar:.2%}")

    # Code-Switching
    print_section("Code-Switching Results 🌍", "📊")
    header = f"  {'Model':<35} {'Method':<10} {'WER':<9} {'CER':<9} {'Samples':<8}"
    print(header)
    print(f"  {'─' * 35} {'─' * 10} {'─' * 9} {'─' * 9} {'─' * 8}")

    best_wer_cs = float("inf")
    best_combo_cs = ""
    for model_name, data in all_metrics.items():
        for method, m in data["metrics"].items():
            if m["n_cs"] == 0:
                continue
            wer_s = f"{m['wer_cs']:.2%}"
            cer_s = f"{m['cer_cs']:.2%}"
            n_s = str(m["n_cs"])
            print(f"  {model_name:<35} {method:<10} {wer_s:<9} {cer_s:<9} {n_s:<8}")
            if m["wer_cs"] < best_wer_cs:
                best_wer_cs = m["wer_cs"]
                best_combo_cs = f"{model_name} ({method})"
    print()
    print(f"  🥇 Best Code-Switching: {best_combo_cs} — WER: {best_wer_cs:.2%}")


def main():
    print_banner()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Config summary ──
    print_section("Configuration", "⚙️")
    print(f"  📂 Test data:    {TEST_DATA_PATH}")
    print(f"  🖥️  Device:       {DEVICE}")
    print(f"  📝 Greedy:       {'✅' if USE_GREEDY else '❌'}")
    print(f"  🔍 Beam + LM:    {'✅' if USE_BEAM_LM and LM_PATH else '❌'}")
    if USE_BEAM_LM and LM_PATH:
        print(f"  📖 LM path:      {LM_PATH}")
        print(f"  🔭 Beam width:   {BEAM_WIDTH}")
        print(f"  α  LM alpha:     {LM_ALPHA}")
        print(f"  β  LM beta:      {LM_BETA}")
    print(f"  📊 Models:       {len(MODELS)}")
    for m in MODELS:
        exists = "✅" if os.path.exists(m["checkpoint"]) else "❌"
        print(f"      {exists} {m['name']} → {m['checkpoint']}")

    # ── Validate ──
    if not os.path.exists(TEST_DATA_PATH):
        print(f"\n  ❌ Test data not found: {TEST_DATA_PATH}")
        print(f"     Run: python scripts/prepare_data.py")
        return

    valid_models = [m for m in MODELS if os.path.exists(m["checkpoint"])]
    if not valid_models:
        print(f"\n  ❌ No valid model checkpoints found!")
        return

    # ── Load test data ──
    print_section("Loading Test Data", "📂")
    test_data = load_test_data(TEST_DATA_PATH)

    # Count categories
    n_cs = 0
    n_ar = 0
    n_other = 0
    for i in range(len(test_data)):
        text = test_data[i].get("text", "")
        if is_code_switching(text):
            n_cs += 1
        elif is_pure_arabic(text):
            n_ar += 1
        else:
            n_other += 1

    print(f"  ✅ Loaded {len(test_data)} test samples")
    print(f"     🇪🇬 Pure Arabic:    {n_ar}")
    print(f"     🌍 Code-Switching: {n_cs}")
    if n_other > 0:
        print(f"     ❓ Other:          {n_other}")

    # ── Evaluate each model (LM loaded once per tokenizer) ──
    all_results = {}
    all_metrics = {}
    feature_extractor_cache = {}
    lm_cache = {}

    for model_info in valid_models:
        print_section(f"Evaluating: {model_info['name']}", "🧪")
        results, metrics, param_count = evaluate_model(
            model_info, test_data, feature_extractor_cache, lm_cache, DEVICE
        )
        all_results[model_info["name"]] = results
        all_metrics[model_info["name"]] = {"metrics": metrics, "params": param_count}

    # ── Results tables ──
    print_results_table(all_metrics)

    # ── Save CSV ──
    print_section("Saving Results", "💾")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"eval_{timestamp}.csv")
    save_csv(all_results, csv_path)
    print(f"  📄 CSV saved: {csv_path}")

    # ── Save summary ──
    summary_path = os.path.join(OUTPUT_DIR, f"eval_{timestamp}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Metro-ASR Evaluation Summary\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test data: {TEST_DATA_PATH} ({len(test_data)} samples)\n")
        f.write(f"  Pure Arabic: {n_ar} | Code-Switching: {n_cs}\n")
        f.write(f"Device: {DEVICE}\n\n")

        if USE_BEAM_LM and LM_PATH:
            f.write(f"LM: {LM_PATH}\n")
            f.write(f"Beam width: {BEAM_WIDTH}, Alpha: {LM_ALPHA}, Beta: {LM_BETA}\n\n")

        f.write(f"{'Model':<35} {'Method':<10} {'WER(All)':<10} {'WER(AR)':<10} {'WER(CS)':<10} {'CER':<8} {'Params':<12}\n")
        f.write(f"{'-' * 95}\n")

        for model_name, data in all_metrics.items():
            for method, m in data["metrics"].items():
                wer_all_s = f"{m['wer_all']:.2%}"
                wer_ar_s = f"{m['wer_ar']:.2%}" if m["n_ar"] > 0 else "N/A"
                wer_cs_s = f"{m['wer_cs']:.2%}" if m["n_cs"] > 0 else "N/A"
                cer_s = f"{m['cer_all']:.2%}"
                f.write(
                    f"{model_name:<35} {method:<10} "
                    f"{wer_all_s:<10} {wer_ar_s:<10} {wer_cs_s:<10} "
                    f"{cer_s:<8} {data['params']:,}\n"
                )

        f.write(f"\n{'=' * 80}\n")

    print(f"  📊 Summary saved: {summary_path}")

    # ── Sample predictions ──
    print_section("Sample Predictions (first 3)", "🔎")

    for model_name, methods in all_results.items():
        print(f"\n  📦 {model_name}")
        for method, data in methods.items():
            print(f"    🔹 {method}:")
            n_show = min(3, len(data["predictions"]))
            for i in range(n_show):
                cat = data["categories"][i] if i < len(data["categories"]) else "?"
                cat_icon = "🌍" if cat == "cs" else "🇪🇬" if cat == "arabic" else "❓"
                print(f"      {cat_icon} [{i}] ref:  {data['references'][i][:80]}")
                print(f"         [{i}] pred: {data['predictions'][i][:80]}")
            print()

    print_section("Done!", "✅")
    print(f"  📁 Results saved to: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
