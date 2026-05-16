import argparse
import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils.config import load_config
from metro_asr.utils.logger import get_logger, print_banner, print_config_summary
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.data.dataset import MetroASRDataset, load_hf_datasets
from metro_asr.training.trainer import MetroTrainer

# ========================= CONFIGURATION =========================
CONFIG_PATH = "configs/metro_tiny.yaml"
TOKENIZER_DIR = "tokenizer_final"
PREPARED_DATA_DIR = "data_prepared"
GPU = None  # GPU device index (None = auto), overridden by --gpu flag
# =================================================================


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=GPU)
    args = parser.parse_args()

    distributed = setup_distributed()
    rank = dist.get_rank() if distributed else 0

    if args.gpu is not None and not distributed:
        torch.cuda.set_device(args.gpu)

    logger = get_logger("metro-asr")

    if rank == 0:
        print_banner()

    config = load_config(CONFIG_PATH)

    if rank == 0:
        print_config_summary(config)

    tokenizer = build_tokenizer(config, TOKENIZER_DIR)
    logger.info(f"🔤 Tokenizer: {config['tokenizer']['type']} | Vocab: {tokenizer.vocab_size}")

    train_path = os.path.join(PREPARED_DATA_DIR, "train")
    eval_path = os.path.join(PREPARED_DATA_DIR, "eval")

    if os.path.exists(train_path) and os.path.exists(eval_path):
        from datasets import load_from_disk
        logger.info(f"📂 Loading prepared data from {PREPARED_DATA_DIR}...")
        train_data = load_from_disk(train_path)
        eval_data = load_from_disk(eval_path)
    else:
        logger.info("📂 No prepared data found, loading from HuggingFace...")
        dataset = load_hf_datasets(
            config["data"]["datasets"],
            config,
            cache_dir=config["data"].get("cache_dir"),
        )
        eval_ratio = config["data"].get("eval_split_ratio", 0.02)
        split = dataset.train_test_split(test_size=eval_ratio, seed=42)
        train_data = split["train"]
        eval_data = split["test"]

    logger.info(f"   📊 Train: {len(train_data):,} | Eval: {len(eval_data):,}")

    train_dataset = MetroASRDataset(train_data, tokenizer, config, is_training=True)
    eval_dataset = MetroASRDataset(eval_data, tokenizer, config, is_training=False)

    model = MetroASR.from_config(config)
    if rank == 0:
        logger.info(f"🏗️  Model: {model.count_parameters():,} params")

    trainer = MetroTrainer(model, train_dataset, eval_dataset, tokenizer, config, logger)
    trainer.train()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
