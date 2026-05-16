"""
Train a BPE tokenizer from prepared Arabic data + English + Egyptian corpus + CS data.

Usage:
    python scripts/train_bpe_tokenizer.py
"""
import os
import sys
import tempfile
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ========================= CONFIGURATION =========================
VOCAB_SIZE = 600
OUTPUT_DIR = "tokenizer_bpe600"
DATA_DIR = "data_prepared/train"

EGYPTIAN_CORPUS = "Prickly-Labs/1.9M-Egyptian-Corpus"  # HF dataset or None
EGYPTIAN_TEXT_COL = "text"
MAX_EGYPTIAN_SAMPLES = 300000

CS_DATASET = "MohamedRashad/arabic-english-code-switching"  # HF dataset or local file, or None
CS_TEXT_COL = "sentence"

MAX_ARABIC_SAMPLES = 200000
MAX_ENGLISH_SAMPLES = 50000
ENGLISH_RATIO = 0.08
CACHE_DIR = "data_cache"
# =================================================================


def load_english_texts():
    from datasets import load_dataset
    print(f"Loading English text corpus...")
    ds = load_dataset("ag_news", split="train", cache_dir=CACHE_DIR, trust_remote_code=True)
    texts = []
    for item in ds:
        t = item["text"].strip()
        if t and len(t) >= 10:
            texts.append(t.lower())
        if len(texts) >= MAX_ENGLISH_SAMPLES:
            break
    print(f"  English sentences: {len(texts)}")
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
    print(f"  Egyptian corpus: {len(texts)}")
    return texts


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
        print(f"  CS texts (file): {len(texts)}")
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
    print(f"  CS transcripts: {len(texts)}")
    return texts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Arabic transcripts
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

    # 3. Code-switching transcripts
    cs_texts = load_cs_texts()

    # 4. English text
    english_texts = load_english_texts()

    # Mix with controlled English ratio
    n_arabic_total = len(arabic_texts) + len(egyptian_texts) + len(cs_texts)
    n_english = int(n_arabic_total * ENGLISH_RATIO / (1.0 - ENGLISH_RATIO))
    n_english = min(n_english, len(english_texts))

    mixed = arabic_texts + egyptian_texts + cs_texts + english_texts[:n_english]
    random.seed(42)
    random.shuffle(mixed)

    print(f"\nMixing corpus: {len(arabic_texts)} transcripts + {len(egyptian_texts)} Egyptian "
          f"+ {len(cs_texts)} CS + {n_english} English = {len(mixed)} total")
    print(f"  Arabic ratio: {(n_arabic_total) / len(mixed) * 100:.1f}%")
    print(f"  English ratio: {n_english / len(mixed) * 100:.1f}%")
    print(f"\nTraining BPE tokenizer, vocab_size={VOCAB_SIZE}...")

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
    print(f"SentencePiece vocab size: {sp.GetPieceSize()}")
    print(f"Total vocab (with blank/unk/pad): {sp.GetPieceSize() + 3}")

    # Vocab breakdown
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

    print(f"Vocab breakdown: {n_arabic_tok} Arabic, {n_english_tok} English, {n_other} other/punctuation")

    print("\nSample tokenizations:")
    samples = [
        "ولا بسأله يا ابني أنت بتعمل إيه",
        "وبتخلي فيه منتصر واحد بس",
        "لأن نفسك دي هي أكتر ذات",
        "this is a code switching example يعني كده",
        "agent for replit",
        "فحطولي في الأسئلة تحت في ال comments تحت ال video عشان نعمل حلقة ال Q&A عن الأسئلة دي",
        "أسهل طريقة انك تعمله download ك zip",
        "the president of the united states announced a new policy",
        "هو ده اللي بيحصل في مصر دلوقتي",
        "الناس بتقول إن الأسعار غالية أوي",
    ]
    for s in samples:
        ids = sp.EncodeAsIds(s.lower())
        pieces = sp.EncodeAsPieces(s.lower())
        print(f"  '{s}'")
        print(f"    pieces: {pieces}")
        print(f"    ids+3:  {[i+3 for i in ids]}")
        print()

    print("First 30 vocab items:")
    for i in range(min(30, sp.GetPieceSize())):
        print(f"  [{i}] '{sp.IdToPiece(i)}'")

    print("\nEnglish subwords in vocab:")
    eng_pieces = []
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        text_part = piece.replace("▁", "")
        if len(text_part) >= 2 and all("a" <= c <= "z" for c in text_part):
            eng_pieces.append((i, piece))
    for idx, piece in eng_pieces[:50]:
        print(f"  [{idx}] '{piece}'")
    print(f"  ... total {len(eng_pieces)} English subwords (2+ chars)")


if __name__ == "__main__":
    main()
