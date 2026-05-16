"""
Train a KenLM n-gram language model for CTC beam search decoding.

Supports:
  1. Arabic transcripts from pretraining data
  2. Egyptian Arabic corpus (1.9M sentences from Prickly-Labs)
  3. Code-switching transcripts
  4. Extra text files

Usage:
    python scripts/train_lm.py
"""
import os
import sys
import subprocess
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ========================= CONFIGURATION =========================
ORDER = 4                          # N-gram order (3, 4, or 5)
OUTPUT_DIR = "lm"
DATA_DIR = "data_prepared/train"    # Prepared Arabic transcripts
MAX_SAMPLES = 500000

EGYPTIAN_CORPUS = "Prickly-Labs/1.9M-Egyptian-Corpus"  # HF dataset name or None
EGYPTIAN_TEXT_COL = "text"
MAX_EGYPTIAN_SAMPLES = 500000

CS_DATASET = "MohamedRashad/arabic-english-code-switching"  # HF dataset or local file, or None
CS_TEXT_COL = "sentence"
CS_UPSAMPLE = 20                    # Repeat CS data N times (12k * 20 = 240k, balances vs 750k Arabic)

ENGLISH_DATASET = "ag_news"         # English text for LM to learn English n-grams (or None)
MAX_ENGLISH_SAMPLES = 50000

EXTRA_TEXT_FILE = None              # Path to additional text file (one per line)
CACHE_DIR = "data_cache"
# =================================================================


def ensure_kenlm_binaries():
    lmplz = shutil.which("lmplz")
    if lmplz:
        return os.path.dirname(lmplz)

    kenlm_build = os.path.join(os.path.dirname(__file__), "..", "kenlm_build")
    lmplz_local = os.path.join(kenlm_build, "bin", "lmplz")

    if os.path.exists(lmplz_local):
        return os.path.join(kenlm_build, "bin")

    print("KenLM binaries not found. Building from source...")
    os.makedirs(kenlm_build, exist_ok=True)

    src_dir = os.path.join(kenlm_build, "src")
    if not os.path.exists(src_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/kpu/kenlm.git", src_dir],
            check=True,
        )
    build_dir = os.path.join(src_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
    subprocess.run(["make", "-j4"], cwd=build_dir, check=True)

    bin_dir = os.path.join(kenlm_build, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    for binary in ["lmplz", "build_binary"]:
        src = os.path.join(build_dir, "bin", binary)
        dst = os.path.join(bin_dir, binary)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    return bin_dir


def find_system_libstdcpp():
    """Find a libstdc++ that has the GLIBCXX symbols the compiled binary needs."""
    candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/local/lib64",
    ]
    import glob
    for d in candidates:
        libs = glob.glob(os.path.join(d, "libstdc++.so.6*"))
        if libs:
            return d
    return None


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

    # 1. Arabic transcripts from pretraining
    print(f"Loading Arabic transcripts from {DATA_DIR}...")
    from datasets import load_from_disk
    dataset = load_from_disk(DATA_DIR)
    all_texts = dataset.remove_columns(["audio"])["text"]

    texts = []
    for t in all_texts:
        if t and len(t.strip()) >= 2:
            texts.append(t.lower().strip())
        if len(texts) >= MAX_SAMPLES:
            break
    print(f"  Arabic transcripts: {len(texts)}")

    # 2. Egyptian corpus
    egyptian_texts = load_egyptian_corpus()
    texts.extend(egyptian_texts)

    # 3. Code-switching transcripts (upsampled for balance)
    cs_texts = load_cs_texts()
    if cs_texts and CS_UPSAMPLE > 1:
        cs_upsampled = cs_texts * CS_UPSAMPLE
        print(f"  CS upsampled: {len(cs_texts)} x {CS_UPSAMPLE} = {len(cs_upsampled)}")
        texts.extend(cs_upsampled)
    else:
        texts.extend(cs_texts)

    # 4. English text (so LM knows English n-grams for code-switching)
    if ENGLISH_DATASET:
        from datasets import load_dataset
        print(f"Loading English text: {ENGLISH_DATASET}...")
        ds = load_dataset(ENGLISH_DATASET, split="train", cache_dir=CACHE_DIR)
        english_texts = []
        for item in ds:
            t = item["text"].strip()
            if t and len(t) >= 10:
                english_texts.append(t.lower())
            if len(english_texts) >= MAX_ENGLISH_SAMPLES:
                break
        print(f"  English sentences: {len(english_texts)}")
        texts.extend(english_texts)

    # 5. Extra text file
    if EXTRA_TEXT_FILE and os.path.exists(EXTRA_TEXT_FILE):
        print(f"Loading extra text from {EXTRA_TEXT_FILE}...")
        with open(EXTRA_TEXT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and len(line) >= 2:
                    texts.append(line)

    print(f"\nTotal LM corpus: {len(texts)} sentences")
    print(f"Training {ORDER}-gram LM...")

    text_file = os.path.join(OUTPUT_DIR, "corpus.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    bin_dir = ensure_kenlm_binaries()
    lmplz = os.path.join(bin_dir, "lmplz")
    build_binary = os.path.join(bin_dir, "build_binary")

    arpa_file = os.path.join(OUTPUT_DIR, f"lm_{ORDER}gram.arpa")
    binary_file = os.path.join(OUTPUT_DIR, f"lm_{ORDER}gram.bin")

    env = os.environ.copy()
    lib_dir = find_system_libstdcpp()
    if lib_dir:
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    print(f"Running lmplz (order={ORDER})...")
    with open(text_file, "r") as inp, open(arpa_file, "w") as out:
        subprocess.run(
            [lmplz, "-o", str(ORDER), "--discount_fallback"],
            stdin=inp, stdout=out, check=True, env=env,
        )

    print(f"Converting to binary format...")
    subprocess.run(
        [build_binary, arpa_file, binary_file],
        check=True, env=env,
    )

    arpa_size = os.path.getsize(arpa_file) / 1024 / 1024
    binary_size = os.path.getsize(binary_file) / 1024 / 1024
    print(f"\nLanguage model trained:")
    print(f"  ARPA:   {arpa_file} ({arpa_size:.1f} MB)")
    print(f"  Binary: {binary_file} ({binary_size:.1f} MB)")
    print(f"  Corpus: {len(texts)} sentences")
    print(f"  Order:  {ORDER}-gram")

    print(f"\nTo use with beam search decoder:")
    print(f"  Set LM_PATH = \"{binary_file}\" in scripts/inference.py")


if __name__ == "__main__":
    main()
