import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.data.dataset import load_hf_datasets, normalize_arabic_text  # noqa: patches Audio decoder
from metro_asr.utils.config import load_config
from metro_asr.utils.logger import get_logger, print_banner
from metro_asr.model.tokenizer import CharTokenizer, BPETokenizer

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_PATH = "configs/metro_22m.yaml"
TOKENIZER_DIR = "tokenizer"
OUTPUT_DIR = "data_prepared"
MAX_SAMPLES_FOR_TOKENIZER = 200000
# ─────────────────────────────────────────────────────────────────────────────


def main():
    logger = get_logger("metro-asr")
    print_banner()

    config = load_config(CONFIG_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    logger.info("📂 Loading and merging all datasets...")
    dataset = load_hf_datasets(
        config["data"]["datasets"],
        config,
        cache_dir=config["data"].get("cache_dir"),
    )

    # ── Text cleaning: isolate text to avoid audio overhead ──
    logger.info("🧹 Cleaning transcriptions...")
    original_len = len(dataset)

    text_only = dataset.remove_columns(["audio"])

    def clean_text(example):
        text = example.get("text", "")
        if text is None:
            text = ""
        text = normalize_arabic_text(text)
        return {"text": text}

    text_only = text_only.map(clean_text, num_proc=4, writer_batch_size=1000)

    all_texts = text_only["text"]
    valid_indices = [i for i, t in enumerate(all_texts) if t and len(t.strip()) >= 2]
    valid_texts = [all_texts[i] for i in valid_indices]

    dataset = dataset.select(valid_indices)
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.add_column("text", valid_texts)

    logger.info(f"   Kept {len(dataset)}/{original_len} samples after cleaning")

    # ── Tokenizer ──
    tok_type = config["tokenizer"].get("type", "char")
    tok_vocab_size = config["tokenizer"].get("vocab_size", 2000)

    logger.info("🔤 Building tokenizer...")
    if tok_type == "bpe":
        logger.info(f"   Training BPE tokenizer (vocab_size={tok_vocab_size})...")
        n_samples = min(MAX_SAMPLES_FOR_TOKENIZER, len(valid_texts))
        texts_for_tok = valid_texts[:n_samples]

        model_prefix = os.path.join(TOKENIZER_DIR, "bpe")
        BPETokenizer.train(texts_for_tok, model_prefix, vocab_size=tok_vocab_size)
        logger.info(f"   BPE model saved to {model_prefix}.model")
    else:
        char_tokenizer = CharTokenizer()
        char_path = os.path.join(TOKENIZER_DIR, "char_tokenizer.json")
        char_tokenizer.save(char_path)
        logger.info(f"   Char tokenizer saved to {char_path} (vocab_size={char_tokenizer.vocab_size})")

    # ── Save ──
    logger.info("💾 Saving prepared dataset...")
    eval_ratio = config["data"].get("eval_split_ratio", 0.02)
    split = dataset.train_test_split(test_size=eval_ratio, seed=42)

    train_path = os.path.join(OUTPUT_DIR, "train")
    eval_path = os.path.join(OUTPUT_DIR, "eval")
    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(eval_path)

    logger.info(f"   Train: {len(split['train'])} → {train_path}")
    logger.info(f"   Eval:  {len(split['test'])} → {eval_path}")

    # ── Statistics ──
    logger.info("\n📊 Dataset Statistics:")
    sample_texts = valid_texts[:min(10000, len(valid_texts))]

    avg_len = sum(len(t) for t in sample_texts) / max(len(sample_texts), 1)
    max_len = max(len(t) for t in sample_texts) if sample_texts else 0

    has_arabic = sum(1 for t in sample_texts if any("؀" <= c <= "ۿ" for c in t))
    has_english = sum(1 for t in sample_texts if any("a" <= c.lower() <= "z" for c in t))
    has_both = sum(
        1 for t in sample_texts
        if any("؀" <= c <= "ۿ" for c in t) and any("a" <= c.lower() <= "z" for c in t)
    )

    logger.info(f"   Avg text length: {avg_len:.1f} chars")
    logger.info(f"   Max text length: {max_len} chars")
    logger.info(f"   Arabic-only:     {has_arabic - has_both} ({(has_arabic - has_both) / len(sample_texts) * 100:.1f}%)")
    logger.info(f"   English-only:    {has_english - has_both} ({(has_english - has_both) / len(sample_texts) * 100:.1f}%)")
    logger.info(f"   Code-switching:  {has_both} ({has_both / len(sample_texts) * 100:.1f}%)")

    logger.info("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
