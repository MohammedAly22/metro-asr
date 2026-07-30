"""
Train an n-gram language head for CTC beam-search decoding.

Two backends:

  kenlm   — shells out to `lmplz` + `build_binary`. Fastest, produces a compact
            binary, and is what you want in production. Needs a C++ toolchain.
  python  — pure-Python interpolated Witten-Bell (metro_asr.lm). No compiler,
            no Boost, no Eigen; writes a standard ARPA that KenLM and
            pyctcdecode load identically. Slower and larger, but it always works.

`--backend auto` (the default) uses kenlm when `lmplz` is on PATH and falls
back to python otherwise.

Train from a corpus file you already have:

    python scripts/train_lm.py --corpus corpora/technical.txt \
                               --out lm/technical_4gram.arpa --order 4

Or let it assemble the general-purpose corpus from HuggingFace, using the
configuration block below:

    python scripts/train_lm.py
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils import enable_utf8_stdout

enable_utf8_stdout()

# ========================= CONFIGURATION =========================
# Used only when --corpus is not supplied.
ORDER = 5
OUTPUT_DIR = "lm"
DATA_DIR = "data_prepared/train"
MAX_SAMPLES = 500000

EGYPTIAN_CORPUS = "Prickly-Labs/1.9M-Egyptian-Corpus"
EGYPTIAN_TEXT_COL = "text"
MAX_EGYPTIAN_SAMPLES = 500000

CS_DATASET = "MohamedRashad/arabic-english-code-switching"
CS_TEXT_COL = "sentence"
CS_UPSAMPLE = 20

ENGLISH_DATASET = "librispeech_asr"
MAX_ENGLISH_SAMPLES = 200000

EXTRA_TEXT_FILE = None
CACHE_DIR = "data_cache"
# =================================================================


# ───────────────────────────── corpus assembly ──────────────────────────────

def load_egyptian_corpus():
    if not EGYPTIAN_CORPUS:
        return []
    from datasets import load_dataset
    print(f"Loading Egyptian corpus: {EGYPTIAN_CORPUS}...")
    ds = load_dataset(EGYPTIAN_CORPUS, split="train", cache_dir=CACHE_DIR)
    if EGYPTIAN_TEXT_COL in ds.column_names:
        ds = ds.select_columns([EGYPTIAN_TEXT_COL])
    texts = []
    for item in ds:
        t = item.get(EGYPTIAN_TEXT_COL, "")
        if t and len(t.strip()) >= 2:
            texts.append(t.lower().strip())
        if len(texts) >= MAX_EGYPTIAN_SAMPLES:
            break
    print(f"  Egyptian corpus: {len(texts)} sentences")
    return texts


def load_cs_texts():
    if not CS_DATASET:
        return []
    if os.path.isfile(CS_DATASET):
        with open(CS_DATASET, encoding="utf-8") as f:
            texts = [l.strip().lower() for l in f if l.strip()]
        print(f"  CS texts (file): {len(texts)}")
        return texts

    from datasets import load_dataset
    print(f"Loading CS dataset: {CS_DATASET}...")
    ds = load_dataset(CS_DATASET, split="train", cache_dir=CACHE_DIR)
    col = CS_TEXT_COL if CS_TEXT_COL in ds.column_names else "text"
    ds = ds.select_columns([col])
    texts = [i[col].lower().strip() for i in ds if i.get(col) and len(i[col].strip()) >= 2]
    print(f"  CS transcripts: {len(texts)}")
    return texts


def load_english_texts():
    if not ENGLISH_DATASET:
        return []
    from datasets import load_dataset
    texts = []
    if ENGLISH_DATASET == "librispeech_asr":
        print("Loading English from LibriSpeech transcripts...")
        for split in ["train.clean.100", "train.clean.360", "train.other.500"]:
            try:
                ds = load_dataset("librispeech_asr",
                                  "clean" if "clean" in split else "other",
                                  split=split, cache_dir=CACHE_DIR, trust_remote_code=True)
                for item in ds:
                    t = item["text"].strip()
                    if t and len(t) >= 5:
                        texts.append(t.lower())
                    if len(texts) >= MAX_ENGLISH_SAMPLES:
                        break
                print(f"  LibriSpeech {split}: total so far {len(texts)}")
            except Exception as e:
                print(f"  Could not load {split}: {e}")
            if len(texts) >= MAX_ENGLISH_SAMPLES:
                break
    else:
        ds = load_dataset(ENGLISH_DATASET, split="train", cache_dir=CACHE_DIR)
        for item in ds:
            t = item["text"].strip()
            if t and len(t) >= 10:
                texts.append(t.lower())
            if len(texts) >= MAX_ENGLISH_SAMPLES:
                break
    print(f"  English sentences: {len(texts)}")
    return texts


def assemble_default_corpus():
    from datasets import load_from_disk
    print(f"Loading Arabic transcripts from {DATA_DIR}...")
    dataset = load_from_disk(DATA_DIR)
    texts = []
    for t in dataset.remove_columns(["audio"])["text"]:
        if t and len(t.strip()) >= 2:
            texts.append(t.lower().strip())
        if len(texts) >= MAX_SAMPLES:
            break
    print(f"  Arabic transcripts: {len(texts)}")

    texts += load_egyptian_corpus()

    cs = load_cs_texts()
    if cs and CS_UPSAMPLE > 1:
        print(f"  CS upsampled: {len(cs)} x {CS_UPSAMPLE} = {len(cs) * CS_UPSAMPLE}")
        texts += cs * CS_UPSAMPLE
    else:
        texts += cs

    texts += load_english_texts()

    if EXTRA_TEXT_FILE and os.path.exists(EXTRA_TEXT_FILE):
        print(f"Loading extra text from {EXTRA_TEXT_FILE}...")
        with open(EXTRA_TEXT_FILE, encoding="utf-8") as f:
            texts += [l.strip().lower() for l in f if len(l.strip()) >= 2]

    return texts


# ───────────────────────────── kenlm backend ────────────────────────────────

def kenlm_available():
    return shutil.which("lmplz") is not None


def train_kenlm(corpus_path, arpa_path, order):
    lmplz = shutil.which("lmplz")
    build_binary = shutil.which("build_binary")

    print(f"Running lmplz (order={order})...")
    with open(corpus_path, encoding="utf-8") as inp, open(arpa_path, "w", encoding="utf-8") as out:
        subprocess.run([lmplz, "-o", str(order), "--discount_fallback"],
                       stdin=inp, stdout=out, check=True)

    if build_binary:
        binary_path = os.path.splitext(arpa_path)[0] + ".bin"
        print("Converting to binary format...")
        subprocess.run([build_binary, arpa_path, binary_path], check=True)
        return binary_path
    return arpa_path


# ───────────────────────────────── main ─────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="text file, one sentence per line (skips dataset assembly)")
    ap.add_argument("--out", help="output path (.arpa)")
    ap.add_argument("--order", type=int, default=ORDER)
    ap.add_argument("--backend", choices=["auto", "kenlm", "python"], default="auto")
    ap.add_argument("--min-count", type=int, default=1,
                    help="prune n-grams of order >= 2 seen fewer than this many times")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    arpa_path = args.out or os.path.join(OUTPUT_DIR, f"lm_{args.order}gram.arpa")
    os.makedirs(os.path.dirname(arpa_path) or ".", exist_ok=True)

    # ---- corpus ----
    if args.corpus:
        corpus_path = args.corpus
        n_lines = sum(1 for _ in open(corpus_path, encoding="utf-8"))
        print(f"Using corpus: {corpus_path} ({n_lines:,} lines)")
    else:
        texts = assemble_default_corpus()
        corpus_path = os.path.join(OUTPUT_DIR, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8", newline="\n") as f:
            for t in texts:
                f.write(t + "\n")
        n_lines = len(texts)
        print(f"\nTotal LM corpus: {n_lines:,} sentences -> {corpus_path}")

    # ---- backend ----
    backend = args.backend
    if backend == "auto":
        backend = "kenlm" if kenlm_available() else "python"
        print(f"Backend: {backend} (auto-selected)")
    if backend == "kenlm" and not kenlm_available():
        print("lmplz not found on PATH — falling back to the pure-Python backend.")
        backend = "python"

    # ---- train ----
    if backend == "kenlm":
        final = train_kenlm(corpus_path, arpa_path, args.order)
    else:
        from metro_asr.lm import build_arpa

        def sentences():
            with open(corpus_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

        stats = build_arpa(sentences(), arpa_path, order=args.order,
                           min_count=args.min_count, progress=lambda m: print("  " + m))
        print(f"  n-grams: {stats['ngrams']}")
        final = arpa_path

    size_mb = os.path.getsize(final) / 1e6
    print(f"\nLanguage head trained:")
    print(f"  Path:   {final} ({size_mb:.1f} MB)")
    print(f"  Order:  {args.order}-gram")
    print(f"  Corpus: {n_lines:,} sentences")
    print(f"\nUse it with:")
    print(f'  engine = MetroASREngine.from_pretrained("checkpoints", lm_path="{final}")')


if __name__ == "__main__":
    main()
