"""
Decode every demo clip with greedy and with each available language head,
score against the human references, and emit JSON for the HTML report.

    python scripts/compare_lm_heads.py --out docs/results.json

The acoustic model is loaded once and the log-probability matrices are cached,
so swapping heads costs only the beam search — which is the whole point of a
detachable language head.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils import enable_utf8_stdout

enable_utf8_stdout()

import torch

from metro_asr.utils.config import load_config
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.model.decoder import CTCBeamSearchDecoder
from metro_asr.data.features import LogMelFeatureExtractor, resample_audio, load_audio_file

# ========================= CONFIGURATION =========================
MODEL_DIR = "checkpoints"
SAMPLES_DIR = "test_samples"
GROUND_TRUTH = "test_samples/ground_truth.json"

# name -> (path, description). Missing files are skipped.
LM_HEADS = {
    "general": ("checkpoints/lm_5gram.bin",
                "Shipped 5-gram: Egyptian corpus + ASR transcripts + code-switching + LibriSpeech"),
    "technical": ("lm/technical_4gram.arpa",
                  "4-gram: Egyptian Arabic Wikipedia (tech) + code-switching + synthesised tech terms"),
    "medical": ("lm/medical_4gram.arpa",
                "4-gram: Egyptian medical chat/QA + code-switching + synthesised medical terms"),
}

BEAM_WIDTH = 100
LM_ALPHA = 0.5
LM_BETA = 5.0
THREADS = 4
# =================================================================


def wer_cer(ref, hyp):
    try:
        from jiwer import wer, cer
    except ImportError:
        return None, None
    if not ref.strip():
        return None, None
    return wer(ref, hyp), cer(ref, hyp)


_PUNCT = "".join(["،", "؛", "؟", ".", ",", "!", "?", ";", ":", "-", "'", '"', "…"])
_ARABIC_DIACRITICS = "ًٌٍَُِّْ"


def normalize_for_scoring(text):
    """
    Standard ASR scoring normalization: lowercase, strip punctuation and
    diacritics, unify Alef/Ya/Ta-Marbuta orthographic variants, collapse
    whitespace.

    Punctuation must go — the model emits "..." for pauses while the human
    references do not, and counting those as tokens penalises every decoder
    equally but meaninglessly. Alef/Ya normalisation is standard for Arabic
    ASR: أ/إ/آ vs ا and ى vs ي are transcription conventions, not errors.
    """
    from metro_asr.data.dataset import normalize_arabic_text

    text = normalize_arabic_text(text).lower()
    text = text.translate(str.maketrans("", "", _PUNCT + _ARABIC_DIACRITICS))
    text = text.translate(str.maketrans("أإآىة", "ااايه"))
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results.json")
    ap.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    args = ap.parse_args()

    torch.set_num_threads(THREADS)

    with open(GROUND_TRUTH, encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    print(f"Loading acoustic model from {MODEL_DIR}/ ...")
    config = load_config(os.path.join(MODEL_DIR, "config.yaml"))
    tokenizer = build_tokenizer(config, MODEL_DIR)
    model = MetroASR.from_config(config)
    ckpt = torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()
    fe = LogMelFeatureExtractor(**config["audio"])
    print(f"  {model.count_parameters():,} parameters\n")

    # ---- encode every clip once ----
    cache = {}
    for s in samples:
        path = os.path.join(SAMPLES_DIR, s["file"])
        if not os.path.exists(path):
            print(f"  ! missing {path}")
            continue
        wav, sr = load_audio_file(path)
        if sr != fe.sample_rate:
            wav = resample_audio(wav, sr, fe.sample_rate)
            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav, dtype=torch.float32)
        duration = wav.shape[0] / fe.sample_rate
        feats = fe(wav).unsqueeze(0)
        lens = torch.tensor([feats.shape[1]])

        t0 = time.perf_counter()
        with torch.no_grad():
            log_probs, out_lengths, _ = model(feats, lens)
        enc_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        greedy = tokenizer.decode(model.decode_greedy(log_probs, out_lengths)[0])
        greedy_time = time.perf_counter() - t0

        cache[s["file"]] = dict(log_probs=log_probs, out_lengths=out_lengths,
                                duration=duration, enc_time=enc_time,
                                greedy=greedy, greedy_time=greedy_time)
        print(f"  encoded {s['file']:<18} {duration:6.2f}s  enc {enc_time*1000:6.1f}ms")

    # ---- decode with each head ----
    heads_meta = {}
    outputs = {f: {"greedy": cache[f]["greedy"]} for f in cache}
    timings = {f: {"greedy": cache[f]["greedy_time"]} for f in cache}

    for head_name, (lm_path, desc) in LM_HEADS.items():
        if not os.path.exists(lm_path):
            print(f"\n! skipping '{head_name}' — {lm_path} not found")
            continue
        print(f"\nLoading '{head_name}' head: {lm_path} "
              f"({os.path.getsize(lm_path)/1e6:.0f} MB)")
        t0 = time.perf_counter()
        decoder = CTCBeamSearchDecoder(tokenizer, lm_path=lm_path,
                                       beam_width=args.beam_width,
                                       alpha=LM_ALPHA, beta=LM_BETA)
        load_time = time.perf_counter() - t0
        heads_meta[head_name] = {
            "path": lm_path,
            "description": desc,
            "size_mb": round(os.path.getsize(lm_path) / 1e6, 1),
            "load_seconds": round(load_time, 2),
        }
        print(f"  loaded in {load_time:.1f}s")

        for f, c in cache.items():
            t0 = time.perf_counter()
            text = decoder.decode(c["log_probs"], c["out_lengths"])[0]
            dt = time.perf_counter() - t0
            outputs[f][head_name] = text
            timings[f][head_name] = dt
            print(f"    {f:<18} {dt*1000:7.1f}ms")

        del decoder

    # ---- score ----
    results = []
    for s in samples:
        f = s["file"]
        if f not in cache:
            continue
        ref = normalize_for_scoring(s["text"])
        entry = {
            "file": f,
            "domain": s["domain"],
            "duration": round(cache[f]["duration"], 2),
            "encoder_ms": round(cache[f]["enc_time"] * 1000, 1),
            "reference": s["text"],
            "reference_normalized": ref,
            "decodes": {},
        }
        for method, hyp in outputs[f].items():
            hyp_n = normalize_for_scoring(hyp)
            w, c = wer_cer(ref, hyp_n)
            entry["decodes"][method] = {
                "text": hyp,
                "text_normalized": hyp_n,
                "wer": round(w * 100, 2) if w is not None else None,
                "cer": round(c * 100, 2) if c is not None else None,
                "decode_ms": round(timings[f][method] * 1000, 1),
            }
        results.append(entry)

    # ---- aggregate ----
    methods = ["greedy"] + [h for h in LM_HEADS if h in heads_meta]
    summary = {}
    for scope in ["all", "general", "technical", "medical"]:
        rows = [r for r in results if scope == "all" or r["domain"] == scope]
        if not rows:
            continue
        summary[scope] = {}
        for m in methods:
            refs = [r["reference_normalized"] for r in rows if m in r["decodes"]]
            hyps = [r["decodes"][m]["text_normalized"] for r in rows if m in r["decodes"]]
            if not refs:
                continue
            try:
                from jiwer import wer, cer
                summary[scope][m] = {
                    "wer": round(wer(refs, hyps) * 100, 2),
                    "cer": round(cer(refs, hyps) * 100, 2),
                    "n": len(refs),
                }
            except ImportError:
                summary[scope][m] = {"n": len(refs)}

    payload = {
        "model": {
            "dir": MODEL_DIR,
            "parameters": model.count_parameters(),
            "vocab_size": tokenizer.vocab_size,
        },
        "decoding": {"beam_width": args.beam_width, "alpha": LM_ALPHA,
                     "beta": LM_BETA, "threads": THREADS},
        "heads": heads_meta,
        "samples": results,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"\n{'='*74}\nAggregate WER% / CER%\n{'='*74}")
    print(f"{'scope':<12}{'n':>4}  " + "".join(f"{m:>16}" for m in methods))
    for scope, per in summary.items():
        n = next(iter(per.values())).get("n", 0)
        cells = ""
        for m in methods:
            if m in per and "wer" in per[m]:
                cells += f"{per[m]['wer']:>9.1f}/{per[m]['cer']:<6.1f}"
            else:
                cells += f"{'-':>16}"
        print(f"{scope:<12}{n:>4}  {cells}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
