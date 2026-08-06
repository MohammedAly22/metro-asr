import argparse
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

# ─── Defaults — override any of these with a CLI flag instead of editing this file ───
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    p.add_argument("--checkpoint", default=PRETRAINED_CHECKPOINT,
                   help="pretrained weights to start from")
    p.add_argument("--dataset", default=FINETUNE_DATASET,
                   help="HuggingFace dataset id, ignored if --prepared-data is set")
    p.add_argument("--audio-col", "--audio_col", dest="audio_col", default=None,
                   help="audio column name in --dataset (auto-detected if omitted)")
    p.add_argument("--text-col", "--text_col", dest="text_col", default=None,
                   help="transcript column name in --dataset (auto-detected if omitted)")
    p.add_argument("--split", default=None,
                   help="split to load from --dataset, e.g. 'train' "
                        "(default: load every split and concatenate them)")
    p.add_argument("--prepared-data", default=PREPARED_DATA_DIR,
                   help="directory with train/ and eval/ splits, skips HF download")
    p.add_argument("--lr", type=float, default=FINETUNE_LR)
    p.add_argument("--max-steps", type=int, default=FINETUNE_MAX_STEPS)
    p.add_argument("--warmup-steps", type=int, default=FINETUNE_WARMUP_STEPS)
    p.add_argument("--freeze-steps", type=int, default=FREEZE_ENCODER_STEPS,
                   help="0 disables the encoder freeze entirely")
    p.add_argument("--run-suffix", default="cs-finetune",
                   help="appended to checkpoint_dir and the wandb run name")
    return p.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True
    return False


def load_finetune_data(config, logger, dataset_id, prepared_data_dir,
                        audio_col=None, text_col=None, split=None):
    """Load the fine-tuning dataset (CS data from HuggingFace or prepared)."""
    if prepared_data_dir and os.path.exists(prepared_data_dir):
        from datasets import load_from_disk
        logger.info(f"Loading prepared finetune data from {prepared_data_dir}...")
        train_path = os.path.join(prepared_data_dir, "train")
        eval_path = os.path.join(prepared_data_dir, "eval")
        return load_from_disk(train_path), load_from_disk(eval_path)

    logger.info(f"Loading finetune dataset: {dataset_id}"
                f"{f' (split={split})' if split else ''}...")
    dataset = load_hf_datasets(
        [dataset_id],
        config,
        cache_dir=config["data"].get("cache_dir"),
        audio_col=audio_col,
        text_col=text_col,
        split=split,
    )
    split_data = dataset.train_test_split(test_size=0.05, seed=42)
    return split_data["train"], split_data["test"]


def main():
    args = parse_args()
    distributed = setup_distributed()
    rank = dist.get_rank() if distributed else 0
    logger = get_logger("metro-asr")

    if rank == 0:
        print_banner()
        logger.info("Fine-tuning mode — CS data")

    config = load_config(args.config)

    config["training"]["learning_rate"] = args.lr
    config["training"]["max_steps"] = args.max_steps
    config["training"]["warmup_steps"] = args.warmup_steps
    config["training"]["resume_from"] = None  # Don't resume — we load weights via --checkpoint
    config["training"]["checkpoint_dir"] = config["training"]["checkpoint_dir"].rstrip("/") + f"-{args.run_suffix}"
    config["training"]["wandb_run_name"] = (config["training"].get("wandb_run_name") or "metro") + f"-{args.run_suffix}"

    if rank == 0:
        print_config_summary(config)

    tokenizer = build_tokenizer(config, args.tokenizer_dir)

    train_data, eval_data = load_finetune_data(
        config, logger, args.dataset, args.prepared_data,
        audio_col=args.audio_col, text_col=args.text_col, split=args.split,
    )
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Eval:  {len(eval_data)} samples")

    train_dataset = MetroASRDataset(train_data, tokenizer, config, is_training=True)
    eval_dataset = MetroASRDataset(eval_data, tokenizer, config, is_training=False)

    model = MetroASR.from_config(config)

    if os.path.exists(args.checkpoint):
        logger.info(f"Loading pretrained weights: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_state = ckpt["model_state_dict"]
        model_state = model.state_dict()

        # strict=False only tolerates missing/unexpected *keys* — it still hard-errors
        # on shape mismatches. When --tokenizer-dir points at a different vocab_size
        # (adapting the tokenizer during fine-tuning), the vocab-sized CTC heads change
        # shape, so we drop just those tensors here and let them init fresh instead.
        compatible_state = {}
        skipped = []
        for key, tensor in checkpoint_state.items():
            if key in model_state and tensor.shape == model_state[key].shape:
                compatible_state[key] = tensor
            else:
                skipped.append(key)

        model.load_state_dict(compatible_state, strict=False)
        if skipped:
            logger.warning(
                f"Skipped {len(skipped)} tensor(s) with a different shape than the checkpoint "
                f"(new tokenizer/vocab_size) — these will train from scratch: {skipped}"
            )
    else:
        logger.warning(f"Pretrained checkpoint not found: {args.checkpoint}")
        logger.warning("  Training from scratch instead.")

    if args.freeze_steps > 0:
        logger.info(f"Freezing encoder for first {args.freeze_steps} steps")
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
        unfreeze_step = trainer.global_step + args.freeze_steps

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

    if args.freeze_steps > 0:
        train_with_unfreeze()
    else:
        trainer.train()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
