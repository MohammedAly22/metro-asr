import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metro_asr.utils import enable_utf8_stdout

enable_utf8_stdout()

from metro_asr.utils.config import load_config
from metro_asr.utils.logger import get_logger, print_banner, print_config_summary
from metro_asr.model.metro import MetroASR
from metro_asr.model.tokenizer import build_tokenizer
from metro_asr.data.dataset import MetroASRDataset, load_hf_datasets
from metro_asr.training.trainer import MetroTrainer

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG_PATH = "configs/metro_small.yaml"
TOKENIZER_DIR = "checkpoints"                        # directory holding bpe.model
PRETRAINED_CHECKPOINT = "checkpoints/model.pt"       # weights to start from

FINETUNE_DATASET = "MohamedRashad/arabic-english-code-switching"
FINETUNE_LR = 5e-5
FINETUNE_MAX_STEPS = 30000
FINETUNE_WARMUP_STEPS = 1000
FREEZE_ENCODER_STEPS = 3000

PREPARED_DATA_DIR = None  # Set to path if you pre-prepared CS data, else loads from HF
# ─────────────────────────────────────────────────────────────────────────────


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True
    return False


def load_finetune_data(config, logger):
    """Load the fine-tuning dataset (CS data from HuggingFace or prepared)."""
    if PREPARED_DATA_DIR and os.path.exists(PREPARED_DATA_DIR):
        from datasets import load_from_disk
        logger.info(f"Loading prepared finetune data from {PREPARED_DATA_DIR}...")
        train_path = os.path.join(PREPARED_DATA_DIR, "train")
        eval_path = os.path.join(PREPARED_DATA_DIR, "eval")
        return load_from_disk(train_path), load_from_disk(eval_path)

    logger.info(f"Loading finetune dataset: {FINETUNE_DATASET}...")
    dataset = load_hf_datasets(
        [FINETUNE_DATASET],
        config,
        cache_dir=config["data"].get("cache_dir"),
    )
    split = dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]


def main():
    distributed = setup_distributed()
    rank = dist.get_rank() if distributed else 0
    logger = get_logger("metro-asr")

    if rank == 0:
        print_banner()
        logger.info("Fine-tuning mode — CS data")

    config = load_config(CONFIG_PATH)

    config["training"]["learning_rate"] = FINETUNE_LR
    config["training"]["max_steps"] = FINETUNE_MAX_STEPS
    config["training"]["warmup_steps"] = FINETUNE_WARMUP_STEPS
    config["training"]["resume_from"] = None  # Don't resume — we load weights via PRETRAINED_CHECKPOINT
    config["training"]["checkpoint_dir"] = config["training"]["checkpoint_dir"].rstrip("/") + "-cs-finetune"
    config["training"]["wandb_run_name"] = (config["training"].get("wandb_run_name") or "metro") + "-cs-finetune"

    if rank == 0:
        print_config_summary(config)

    tokenizer = build_tokenizer(config, TOKENIZER_DIR)

    train_data, eval_data = load_finetune_data(config, logger)
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Eval:  {len(eval_data)} samples")

    train_dataset = MetroASRDataset(train_data, tokenizer, config, is_training=True)
    eval_dataset = MetroASRDataset(eval_data, tokenizer, config, is_training=False)

    model = MetroASR.from_config(config)

    if os.path.exists(PRETRAINED_CHECKPOINT):
        logger.info(f"Loading pretrained weights: {PRETRAINED_CHECKPOINT}")
        ckpt = torch.load(PRETRAINED_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        logger.warning(f"Pretrained checkpoint not found: {PRETRAINED_CHECKPOINT}")
        logger.warning("  Training from scratch instead.")

    if FREEZE_ENCODER_STEPS > 0:
        logger.info(f"Freezing encoder for first {FREEZE_ENCODER_STEPS} steps")
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.subsampling.parameters():
            param.requires_grad = False

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = model.count_parameters()
        logger.info(f"Trainable: {trainable:,} / {total:,}")

    trainer = MetroTrainer(model, train_dataset, eval_dataset, tokenizer, config, logger)

    original_train = trainer.train

    def train_with_unfreeze(resume_from=None):
        unfreeze_step = trainer.global_step + FREEZE_ENCODER_STEPS

        original_step = trainer._train_step

        def step_with_unfreeze(batch):
            if trainer.global_step == unfreeze_step:
                logger.info("Unfreezing encoder")
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.subsampling.parameters():
                    param.requires_grad = True
            return original_step(batch)

        trainer._train_step = step_with_unfreeze
        original_train(resume_from)

    if FREEZE_ENCODER_STEPS > 0:
        train_with_unfreeze()
    else:
        trainer.train()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
