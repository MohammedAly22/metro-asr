import math
import torch


def build_optimizer(model, train_cfg):
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.01)

    no_decay = {
        "bias", "LayerNorm.weight", "layer_norm.weight",
        "batch_norm.weight", "batch_norm.bias",
        "norm.weight",
    }
    params_decay = []
    params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            params_no_decay.append(param)
        else:
            params_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": params_decay, "weight_decay": weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    return optimizer


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, min_lr_ratio=0.05, warmup_init_ratio=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_init_ratio = warmup_init_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)

        if step < self.warmup_steps:
            progress = step / max(self.warmup_steps, 1)
            scale = self.warmup_init_ratio + (1.0 - self.warmup_init_ratio) * progress
        else:
            progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return [base_lr * scale for base_lr in self.base_lrs]


def build_scheduler(optimizer, train_cfg):
    warmup_steps = train_cfg.get("warmup_steps", 10000)
    max_steps = train_cfg.get("max_steps", 500000)
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.05)
    warmup_init_ratio = train_cfg.get("warmup_init_lr_ratio", 0.0)

    return WarmupCosineScheduler(
        optimizer,
        warmup_steps=max(warmup_steps, 1),
        max_steps=max(max_steps, 1),
        min_lr_ratio=min_lr_ratio,
        warmup_init_ratio=warmup_init_ratio,
    )
