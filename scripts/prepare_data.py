import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.data.dataset import load_hf_datasets, normalize_arabic_text
from metro_asr.utils.config import load_config
from metro_asr.utils.logger import get_logger, print_banner

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_PATH = "configs/metro_tiny.yaml"
OUTPUT_DIR = "data_prepared"

# Test split: small curated set with guaranteed Arabic + CS samples for WER eval
TEST_SPLIT_SIZE = 500               # Total test samples
TEST_CS_GUARANTEED = 200            # Guaranteed code-switching samples in test
TEST_ARABIC_GUARANTEED = 300        # Guaranteed pure Egyptian Arabic samples in test
# ─────────────────────────────────────────────────────────────────────────────


def is_code_switching(text):
    has_arabic = any("؀" <= c <= "ۿ" for c in text)
    has_english = any("a" <= c.lower() <= "z" for c in text)
    return has_arabic and has_english


def is_pure_arabic(text):
    has_arabic = any("؀" <= c <= "ۿ" for c in text)
    has_english = any("a" <= c.lower() <= "z" for c in text)
    return has_arabic and not has_english


def main():
    logger = get_logger("metro-asr")
    print_banner()

    config = load_config(CONFIG_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("📂 Loading and merging all datasets...")
    dataset = load_hf_datasets(
        config["data"]["datasets"],
        config,
        cache_dir=config["data"].get("cache_dir"),
    )

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

    logger.info(f"   ✅ Kept {len(dataset)}/{original_len} samples after cleaning")

    # ── Build curated test split ──
    logger.info("🧪 Building curated test split...")

    cs_indices = []
    arabic_indices = []
    other_indices = []

    for i, t in enumerate(valid_texts):
        if is_code_switching(t):
            cs_indices.append(i)
        elif is_pure_arabic(t):
            arabic_indices.append(i)
        else:
            other_indices.append(i)

    logger.info(f"   🌍 Code-switching samples available: {len(cs_indices)}")
    logger.info(f"   🇪🇬 Pure Arabic samples available: {len(arabic_indices)}")

    random.seed(42)
    random.shuffle(cs_indices)
    random.shuffle(arabic_indices)

    n_cs = min(TEST_CS_GUARANTEED, len(cs_indices))
    n_arabic = min(TEST_ARABIC_GUARANTEED, len(arabic_indices))
    remaining = TEST_SPLIT_SIZE - n_cs - n_arabic

    test_indices = set()
    test_indices.update(cs_indices[:n_cs])
    test_indices.update(arabic_indices[:n_arabic])

    if remaining > 0:
        leftover = [i for i in range(len(valid_texts)) if i not in test_indices]
        random.shuffle(leftover)
        test_indices.update(leftover[:remaining])

    test_indices = sorted(test_indices)
    train_indices = sorted(set(range(len(valid_texts))) - set(test_indices))

    test_cs_count = sum(1 for i in test_indices if is_code_switching(valid_texts[i]))
    test_ar_count = sum(1 for i in test_indices if is_pure_arabic(valid_texts[i]))

    logger.info(f"   📊 Test split: {len(test_indices)} samples")
    logger.info(f"      🌍 Code-switching: {test_cs_count}")
    logger.info(f"      🇪🇬 Pure Arabic:    {test_ar_count}")

    # ── Train/eval/test split ──
    eval_ratio = config["data"].get("eval_split_ratio", 0.02)
    train_dataset = dataset.select(train_indices)

    eval_size = int(len(train_dataset) * eval_ratio)
    train_eval_split = train_dataset.train_test_split(test_size=eval_size, seed=42)

    test_dataset = dataset.select(test_indices)

    # ── Save ──
    logger.info("💾 Saving prepared dataset...")

    train_path = os.path.join(OUTPUT_DIR, "train")
    eval_path = os.path.join(OUTPUT_DIR, "eval")
    test_path = os.path.join(OUTPUT_DIR, "test")

    train_eval_split["train"].save_to_disk(train_path)
    train_eval_split["test"].save_to_disk(eval_path)
    test_dataset.save_to_disk(test_path)

    logger.info(f"   📁 Train: {len(train_eval_split['train']):,} → {train_path}")
    logger.info(f"   📁 Eval:  {len(train_eval_split['test']):,} → {eval_path}")
    logger.info(f"   📁 Test:  {len(test_dataset):,} → {test_path}")

    # ── Statistics ──
    logger.info("\n📊 Dataset Statistics:")
    sample_texts = valid_texts[:min(10000, len(valid_texts))]

    avg_len = sum(len(t) for t in sample_texts) / max(len(sample_texts), 1)
    max_len = max(len(t) for t in sample_texts) if sample_texts else 0

    has_arabic = sum(1 for t in sample_texts if any("؀" <= c <= "ۿ" for c in t))
    has_english = sum(1 for t in sample_texts if any("a" <= c.lower() <= "z" for c in t))
    has_both = sum(1 for t in sample_texts if is_code_switching(t))

    logger.info(f"   📏 Avg text length: {avg_len:.1f} chars")
    logger.info(f"   📏 Max text length: {max_len} chars")
    logger.info(f"   🇪🇬 Arabic-only:     {has_arabic - has_both} ({(has_arabic - has_both) / len(sample_texts) * 100:.1f}%)")
    logger.info(f"   🇬🇧 English-only:    {has_english - has_both} ({(has_english - has_both) / len(sample_texts) * 100:.1f}%)")
    logger.info(f"   🌍 Code-switching:  {has_both} ({has_both / len(sample_texts) * 100:.1f}%)")

    # ── Test split sample preview ──
    logger.info("\n🔎 Test split samples:")
    test_texts = [valid_texts[i] for i in test_indices]
    cs_samples = [t for t in test_texts if is_code_switching(t)]
    ar_samples = [t for t in test_texts if is_pure_arabic(t)]

    logger.info("   🌍 Code-switching examples:")
    for t in cs_samples[:5]:
        logger.info(f"      • {t}")

    logger.info("   🇪🇬 Pure Arabic examples:")
    for t in ar_samples[:5]:
        logger.info(f"      • {t}")

    logger.info("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
