"""
Train a high-quality bilingual BPE tokenizer for Arabic+English ASR.

Novel approach: Train on a MASSIVE balanced corpus from Wikipedia + ASR data,
with forced 50/50 Arabic/English balance and large vocab (5000).
This ensures common English words ("this", "download", "comments") are single tokens
while Arabic coverage remains excellent.

Usage:
    python scripts/train_bpe_tokenizer.py
"""
import os
import sys
import tempfile
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ========================= CONFIGURATION =========================
VOCAB_SIZE = 5000
OUTPUT_DIR = "tokenizer_bpe5k"
DATA_DIR = "data_prepared/train"        # Prepared Arabic ASR transcripts

# ---- Arabic sources ----
EGYPTIAN_CORPUS = "Prickly-Labs/1.9M-Egyptian-Corpus"
EGYPTIAN_TEXT_COL = "text"
MAX_EGYPTIAN_SAMPLES = 1000000          # 1M Egyptian sentences

ARABIC_WIKI = "wikimedia/wikipedia"     # Arabic Wikipedia for broad Arabic coverage
ARABIC_WIKI_CONFIG = "20231101.ar"
MAX_ARABIC_WIKI_SAMPLES = 500000        # 500K Wikipedia sentences

MAX_ARABIC_ASR_SAMPLES = 300000         # ASR transcripts from data_prepared

# ---- English sources ----
ENGLISH_WIKI = "wikimedia/wikipedia"    # English Wikipedia for broad English coverage
ENGLISH_WIKI_CONFIG = "20231101.en"
MAX_ENGLISH_WIKI_SAMPLES = 500000       # 500K Wikipedia sentences

MAX_LIBRISPEECH_SAMPLES = 300000        # LibriSpeech ASR transcripts

# ---- Code-switching ----
CS_DATASET = "MohamedRashad/arabic-english-code-switching"
CS_TEXT_COL = "sentence"

CACHE_DIR = "data_cache"
# =================================================================


def split_into_sentences(text, max_len=500):
    """Split long text into sentence-like chunks."""
    import re
    # Split on sentence boundaries
    sentences = re.split(r'[.!?؟،\n]+', text)
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) >= 5 and len(s) <= max_len:
            result.append(s.lower())
    return result


def load_arabic_wiki():
    """Load Arabic Wikipedia for broad Arabic vocabulary coverage."""
    from datasets import load_dataset
    print(f"Loading Arabic Wikipedia ({ARABIC_WIKI_CONFIG})...")
    try:
        ds = load_dataset(ARABIC_WIKI, ARABIC_WIKI_CONFIG,
                          split="train", cache_dir=CACHE_DIR,
                          trust_remote_code=True, streaming=True)
        texts = []
        for item in ds:
            raw = item.get("text", "")
            if not raw or len(raw) < 20:
                continue
            for sent in split_into_sentences(raw):
                texts.append(sent)
                if len(texts) >= MAX_ARABIC_WIKI_SAMPLES:
                    break
            if len(texts) >= MAX_ARABIC_WIKI_SAMPLES:
                break
        print(f"  Arabic Wikipedia: {len(texts)} sentences")
        return texts
    except Exception as e:
        print(f"  ⚠️ Could not load Arabic Wikipedia: {e}")
        return []


def load_english_wiki():
    """Load English Wikipedia for broad English vocabulary coverage."""
    from datasets import load_dataset
    print(f"Loading English Wikipedia ({ENGLISH_WIKI_CONFIG})...")
    try:
        ds = load_dataset(ENGLISH_WIKI, ENGLISH_WIKI_CONFIG,
                          split="train", cache_dir=CACHE_DIR,
                          trust_remote_code=True, streaming=True)
        texts = []
        for item in ds:
            raw = item.get("text", "")
            if not raw or len(raw) < 20:
                continue
            for sent in split_into_sentences(raw):
                texts.append(sent)
                if len(texts) >= MAX_ENGLISH_WIKI_SAMPLES:
                    break
            if len(texts) >= MAX_ENGLISH_WIKI_SAMPLES:
                break
        print(f"  English Wikipedia: {len(texts)} sentences")
        return texts
    except Exception as e:
        print(f"  ⚠️ Could not load English Wikipedia: {e}")
        return []


def load_librispeech_texts():
    """Load LibriSpeech ASR transcripts — real spoken English."""
    from datasets import load_dataset
    print("Loading LibriSpeech transcripts...")
    texts = []
    for split in ["train.clean.100", "train.clean.360", "train.other.500"]:
        try:
            config = "clean" if "clean" in split else "other"
            ds = load_dataset("librispeech_asr", config,
                              split=split, cache_dir=CACHE_DIR,
                              trust_remote_code=True)
            for item in ds:
                t = item["text"].strip()
                if t and len(t) >= 5:
                    texts.append(t.lower())
                if len(texts) >= MAX_LIBRISPEECH_SAMPLES:
                    break
            print(f"  LibriSpeech {split}: total so far {len(texts)}")
        except Exception as e:
            print(f"  ⚠️ Could not load {split}: {e}")
        if len(texts) >= MAX_LIBRISPEECH_SAMPLES:
            break
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


def load_arabic_asr_texts():
    """Load Arabic ASR transcripts from prepared data."""
    print(f"Loading Arabic ASR transcripts from {DATA_DIR}...")
    from datasets import load_from_disk
    dataset = load_from_disk(DATA_DIR)
    all_texts = dataset.remove_columns(["audio"])["text"]

    texts = []
    for t in all_texts:
        if t and len(t.strip()) >= 2:
            texts.append(t.lower().strip())
        if len(texts) >= MAX_ARABIC_ASR_SAMPLES:
            break
    print(f"  Arabic ASR transcripts: {len(texts)}")
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

    # ---- Collect all Arabic sources ----
    arabic_asr = load_arabic_asr_texts()
    egyptian = load_egyptian_corpus()
    arabic_wiki = load_arabic_wiki()
    cs_texts = load_cs_texts()

    all_arabic = arabic_asr + egyptian + arabic_wiki + cs_texts
    print(f"\n📊 Total Arabic sentences: {len(all_arabic)}")

    # ---- Collect all English sources ----
    librispeech = load_librispeech_texts()
    english_wiki = load_english_wiki()

    all_english = librispeech + english_wiki
    print(f"📊 Total English sentences: {len(all_english)}")

    # ---- Force 50/50 balance ----
    # Take the smaller side's count, then match the other
    target_per_lang = min(len(all_arabic), len(all_english))
    # Cap at 2M total (1M per language) to keep training fast
    target_per_lang = min(target_per_lang, 1000000)

    random.seed(42)
    random.shuffle(all_arabic)
    random.shuffle(all_english)

    balanced_arabic = all_arabic[:target_per_lang]
    balanced_english = all_english[:target_per_lang]

    # Add ALL CS texts (they bridge both languages)
    mixed = balanced_arabic + balanced_english + cs_texts
    random.shuffle(mixed)

    n_total = len(mixed)
    print(f"\n🔀 Balanced corpus:")
    print(f"  Arabic:  {len(balanced_arabic):,} sentences")
    print(f"  English: {len(balanced_english):,} sentences")
    print(f"  CS:      {len(cs_texts):,} sentences (added on top)")
    print(f"  Total:   {n_total:,} sentences")
    print(f"  Balance: {len(balanced_arabic)/n_total*100:.1f}% AR / {len(balanced_english)/n_total*100:.1f}% EN")
    print(f"\n🔤 Training BPE tokenizer, vocab_size={VOCAB_SIZE}...")

    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    for t in mixed:
        tmp_file.write(t + "\n")
    tmp_file.close()

    import sentencepiece as spm

    sp_vocab = VOCAB_SIZE - 3  # Reserve 3 for blank, unk, pad
    model_prefix = os.path.join(OUTPUT_DIR, "bpe")

    spm.SentencePieceTrainer.Train(
        input=tmp_file.name,
        model_prefix=model_prefix,
        vocab_size=sp_vocab,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=-1,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        max_sentence_length=16384,
        num_threads=8,
        byte_fallback=False,         # No byte tokens — saves 256 vocab slots for real subwords
        split_digits=True,           # Digits as separate tokens
        input_sentence_size=2000000, # Use up to 2M sentences for training
        shuffle_input_sentence=True, # Shuffle for better coverage
    )

    os.unlink(tmp_file.name)

    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    print(f"\n✅ BPE model saved to {model_prefix}.model")
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

    print(f"Vocab breakdown: {n_arabic_tok} Arabic, {n_english_tok} English, {n_other} other/special")

    print("\nSample tokenizations:")
    samples = [
        # Pure Arabic
        "ولا بسأله يا ابني أنت بتعمل إيه",
        "هو ده اللي بيحصل في مصر دلوقتي",
        "الناس بتقول إن الأسعار غالية أوي",
        # Pure English
        "this is a test",
        "the president of the united states announced a new policy",
        "download the file and install it",
        # Code-switching (the real test)
        "this is a code switching example يعني كده",
        "agent for replit",
        "فحطولي في الأسئلة تحت في ال comments تحت ال video عشان نعمل حلقة ال Q&A عن الأسئلة دي",
        "أسهل طريقة انك تعمله download ك zip",
        "هو عامل model كويس أوي for machine learning",
        "i need to make a presentation عن الموضوع ده",
    ]
    for s in samples:
        ids = sp.EncodeAsIds(s.lower())
        pieces = sp.EncodeAsPieces(s.lower())
        print(f"  '{s}'")
        print(f"    tokens: {len(ids)}")
        print(f"    pieces: {pieces}")
        print()

    print("First 50 vocab items:")
    for i in range(min(50, sp.GetPieceSize())):
        print(f"  [{i}] '{sp.IdToPiece(i)}'")

    print("\nEnglish whole words in vocab (▁ prefix = word start):")
    eng_words = []
    eng_subwords = []
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        text_part = piece.replace("▁", "")
        if len(text_part) >= 2 and all("a" <= c <= "z" for c in text_part):
            if piece.startswith("▁"):
                eng_words.append((i, piece))
            else:
                eng_subwords.append((i, piece))

    print(f"  Whole words (▁word): {len(eng_words)}")
    for idx, piece in eng_words[:80]:
        print(f"    [{idx}] '{piece}'")
    if len(eng_words) > 80:
        print(f"    ... and {len(eng_words) - 80} more")

    print(f"\n  Subword fragments: {len(eng_subwords)}")
    for idx, piece in eng_subwords[:30]:
        print(f"    [{idx}] '{piece}'")
    if len(eng_subwords) > 30:
        print(f"    ... and {len(eng_subwords) - 30} more")

    print(f"\n  Total English tokens: {len(eng_words) + len(eng_subwords)}")

    # Key English words check
    print("\n🔍 Key English words check:")
    key_words = ["this", "that", "the", "is", "are", "was", "for", "and",
                 "with", "from", "have", "will", "would", "could", "should",
                 "about", "make", "like", "just", "been", "more", "some",
                 "than", "them", "then", "when", "what", "which", "where",
                 "download", "video", "comments", "model", "code",
                 "machine", "learning", "python", "data", "file",
                 "computer", "network", "system", "example", "project",
                 "presentation", "application", "information", "technology"]
    for word in key_words:
        pieces = sp.EncodeAsPieces(word)
        status = "✅" if len(pieces) == 1 else f"❌ ({len(pieces)} tokens)"
        print(f"  {word:20s} → {str(pieces):40s} {status}")


if __name__ == "__main__":
    main()
