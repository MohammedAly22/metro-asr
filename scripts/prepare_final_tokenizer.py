"""
Train the final BPE tokenizer for the definitive Metro-22M model.

Combines:
  1. Arabic transcripts from pretraining data
  2. Egyptian Arabic corpus (1.9M from Prickly-Labs) for broader coverage
  3. Code-switching transcripts (Arabic-English)
  4. English text (from AG News) for English subwords

Usage:
    python scripts/prepare_final_tokenizer.py
"""
import os
import sys
import tempfile
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ========================= CONFIGURATION =========================
VOCAB_SIZE = 600
OUTPUT_DIR = "tokenizer_final"  # Used by metro_tiny config
DATA_DIR = "data_prepared/train"

EGYPTIAN_CORPUS = "Prickly-Labs/1.9M-Egyptian-Corpus"  # HF dataset name or None
EGYPTIAN_TEXT_COL = "text"
MAX_EGYPTIAN_SAMPLES = 800000

CS_DATASET = "MohamedRashad/arabic-english-code-switching"  # HF dataset or local file, or None
CS_TEXT_COL = "sentence"

MAX_ARABIC_SAMPLES = 500000
MAX_ENGLISH_SAMPLES = 50000
ENGLISH_RATIO = 0.08
CACHE_DIR = "data_cache"
# =================================================================


def load_cs_texts():
    if not CS_DATASET:
        return []
    if os.path.isfile(CS_DATASET):
        texts = []
        with open(CS_DATASET, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line.lower())
        return texts

    from datasets import load_dataset
    print(f"Loading CS dataset: {CS_DATASET}...")
    ds = load_dataset(CS_DATASET, split="train", cache_dir=CACHE_DIR)
    text_col = CS_TEXT_COL if CS_TEXT_COL in ds.column_names else "text"
    ds = ds.select_columns([text_col])
    texts = []
    for item in ds:
        t = item.get(text_col, "")
        if t and len(t.strip()) >= 2:
            texts.append(t.lower().strip())
    return texts


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
    return texts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Arabic transcripts from pretraining
    print(f"Loading Arabic transcripts from {DATA_DIR}...")
    from datasets import load_from_disk
    dataset = load_from_disk(DATA_DIR)
    all_texts = dataset.remove_columns(["audio"])["text"]

    arabic_texts = []
    for t in all_texts:
        if t and len(t.strip()) >= 2:
            arabic_texts.append(t.lower().strip())
        if len(arabic_texts) >= MAX_ARABIC_SAMPLES:
            break
    print(f"  Arabic transcripts: {len(arabic_texts)}")

    # 2. Egyptian corpus
    egyptian_texts = load_egyptian_corpus()
    print(f"  Egyptian corpus: {len(egyptian_texts)}")

    # 3. Code-switching transcripts
    cs_texts = load_cs_texts()
    print(f"  CS transcripts: {len(cs_texts)}")

    # 4. English text from AG News
    print(f"Loading English text corpus...")
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train", cache_dir=CACHE_DIR, trust_remote_code=True)
    english_texts = []
    for item in ds:
        t = item["text"].strip()
        if t and len(t) >= 10:
            english_texts.append(t.lower())
        if len(english_texts) >= MAX_ENGLISH_SAMPLES:
            break
    print(f"  English sentences: {len(english_texts)}")

    # Mix with controlled ratios
    n_arabic = len(arabic_texts) + len(egyptian_texts) + len(cs_texts)
    n_english = int(n_arabic * ENGLISH_RATIO / (1.0 - ENGLISH_RATIO))
    n_english = min(n_english, len(english_texts))

    mixed = arabic_texts + egyptian_texts + cs_texts + english_texts[:n_english]
    random.seed(42)
    random.shuffle(mixed)

    print(f"\nFinal corpus: {len(arabic_texts)} transcripts + {len(egyptian_texts)} Egyptian "
          f"+ {len(cs_texts)} CS + {n_english} English = {len(mixed)} total")
    print(f"Training BPE tokenizer, vocab_size={VOCAB_SIZE}...")

    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    for t in mixed:
        tmp_file.write(t + "\n")
    tmp_file.close()

    import sentencepiece as spm

    sp_vocab = VOCAB_SIZE - 3
    model_prefix = os.path.join(OUTPUT_DIR, "bpe")

    spm.SentencePieceTrainer.Train(
        input=tmp_file.name,
        model_prefix=model_prefix,
        vocab_size=sp_vocab,
        model_type="bpe",
        character_coverage=0.9999,
        pad_id=-1,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        max_sentence_length=16384,
        num_threads=4,
        byte_fallback=False,
    )

    os.unlink(tmp_file.name)

    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    print(f"\nBPE model saved to {model_prefix}.model")
    print(f"Total vocab (with blank/unk/pad): {sp.GetPieceSize() + 3}")

    n_arabic_tok = 0
    n_english_tok = 0
    n_other = 0
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        text_part = piece.replace("▁", "")
        if any("؀" <= c <= "ۿ" for c in text_part):
            n_arabic_tok += 1
        elif any("a" <= c <= "z" for c in text_part):
            n_english_tok += 1
        else:
            n_other += 1

    print(f"Vocab: {n_arabic_tok} Arabic, {n_english_tok} English, {n_other} other")

    print("\nSample tokenizations:")
    samples = [
        "ولا بسأله يا ابني أنت بتعمل إيه",
        "بس تقعدي تذاكري اعملي اللي انتي عايزاه",
        "this is a code switching example يعني كده",
        "تعالوا نبني mobile application على الموبايل",
        "منصة replit منزلين عندهم agent for",
        "فحطولي في ال comments تحت ال video",
        "هو ده اللي بيحصل في مصر دلوقتي",
        "الناس بتقول إن الأسعار غالية أوي",
    ]
    for s in samples:
        pieces = sp.EncodeAsPieces(s.lower())
        print(f"  '{s}'")
        print(f"    {pieces}")


if __name__ == "__main__":
    main()
