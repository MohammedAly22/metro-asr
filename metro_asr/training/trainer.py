import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from metro_asr.training.optimizer import build_optimizer, build_scheduler
from metro_asr.data.collator import MetroCollator


class MetroTrainer:
    def __init__(self, model, train_dataset, eval_dataset, tokenizer, config, logger):
        self.config = config
        self.logger = logger
        self.tokenizer = tokenizer
        train_cfg = config["training"]

        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main = self.rank == 0

        if torch.cuda.is_available():
            gpu_idx = torch.cuda.current_device() if not self.distributed else self.rank
            self.device = torch.device(f"cuda:{gpu_idx}")
        else:
            self.device = torch.device("cpu")
        self.model = model.to(self.device)

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.device.index], find_unused_parameters=False)

        self.raw_model = self.model.module if self.distributed else self.model

        self.optimizer = build_optimizer(self.raw_model, train_cfg)
        self.scheduler = build_scheduler(self.optimizer, train_cfg)

        self.use_bf16 = train_cfg.get("bf16", False)
        self.use_fp16 = train_cfg.get("fp16", False) and not self.use_bf16
        self.use_amp = self.use_bf16 or self.use_fp16
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16

        self.scaler = None
        if self.use_fp16:
            self.scaler = torch.amp.GradScaler("cuda")

        self.ctc_loss = nn.CTCLoss(
            blank=0,
            reduction=train_cfg.get("ctc_loss_reduction", "mean"),
            zero_infinity=True,
        )

        self.grad_accum_steps = train_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.max_steps = train_cfg.get("max_steps", 500000)
        self.save_every = train_cfg.get("save_every_n_steps", 5000)
        self.eval_every = train_cfg.get("eval_every_n_steps", 1000)
        self.log_every = train_cfg.get("log_every_n_steps", 10)
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        self.inter_ctc_weight = train_cfg.get("intermediate_ctc_weight", 0.3)

        batch_size = train_cfg.get("batch_size", 32)
        num_workers = config["data"].get("num_workers", 4)
        collator = MetroCollator(pad_id=tokenizer.pad_id)

        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = RandomSampler(train_dataset)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True,
            timeout=120,
            persistent_workers=True if num_workers > 0 else False,
        )

        self.eval_loader = None
        if eval_dataset is not None and len(eval_dataset) > 0:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collator,
                pin_memory=True,
            )

        self.global_step = 0
        self.epoch = 0
        self.best_wer = float("inf")
        self.wandb_run = None
        self._last_blank_prob = 0.0
        self._last_grad_norm = 0.0
        self._nan_count = 0
        self._consecutive_nans = 0

        if self.is_main:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._init_wandb(train_cfg)

    def _init_wandb(self, train_cfg):
        try:
            import wandb
            project = train_cfg.get("wandb_project", "metro-asr")
            run_name = train_cfg.get("wandb_run_name") or f"metro-{self.config['model']['name']}"
            self.wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=self.config,
                resume="allow",
            )
        except ImportError:
            self.logger.warning("wandb not installed, skipping logging")
        except Exception as e:
            self.logger.warning(f"wandb init failed: {e}")

    def train(self, resume_from=None):
        if resume_from:
            self._load_checkpoint(resume_from)
        elif self.config["training"].get("resume_from"):
            self._load_checkpoint(self.config["training"]["resume_from"])

        self.logger.info(f"Starting training from step {self.global_step}")
        self.logger.info(f"   Model parameters: {self.raw_model.count_parameters():,}")
        self.logger.info(f"   Device: {self.device} | World size: {self.world_size}")
        self.logger.info(f"   AMP: {'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'off'}")
        eff_batch = self.config['training']['batch_size'] * self.world_size * self.grad_accum_steps
        self.logger.info(f"   Effective batch size: {eff_batch}")

        self.model.train()
        accum_loss = 0.0
        micro_step = 0
        step_start = time.time()

        while self.global_step < self.max_steps:
            if self.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)

            for batch in self.train_loader:
                loss = self._train_step(batch)
                accum_loss += loss
                micro_step += 1

                if micro_step % self.grad_accum_steps == 0:
                    if self._has_nan_grads():
                        self.optimizer.zero_grad(set_to_none=True)
                        self._nan_count += 1
                        self._consecutive_nans += 1
                        if self.is_main and self._nan_count % 10 == 0:
                            self.logger.warning(f"⚠️  NaN gradients skipped: {self._nan_count} total ({self._consecutive_nans} consecutive)")
                        if self._consecutive_nans >= 50:
                            self.logger.error("❌ 50 consecutive NaN batches — saving checkpoint and stopping")
                            self._save_checkpoint("nan_recovery")
                            raise RuntimeError("Training diverged: 50 consecutive NaN gradients")
                        accum_loss = 0.0
                        continue
                    self._consecutive_nans = 0

                    if self.max_grad_norm > 0:
                        total_norm = nn.utils.clip_grad_norm_(
                            self.raw_model.parameters(), self.max_grad_norm
                        )
                        self._last_grad_norm = total_norm.item() if torch.is_tensor(total_norm) else total_norm

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if self.is_main and self.global_step % self.log_every == 0:
                        elapsed = time.time() - step_start
                        avg_loss = accum_loss / (self.log_every * self.grad_accum_steps)
                        lr = self.scheduler.get_last_lr()[0]
                        self.logger.info(
                            f"Step {self.global_step}/{self.max_steps} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                            f"P(blank): {self._last_blank_prob:.4f} | "
                            f"GradNorm: {self._last_grad_norm:.2f} | "
                            f"Time: {elapsed:.1f}s"
                        )
                        if self.wandb_run:
                            import wandb
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/lr": lr,
                                "train/blank_prob": self._last_blank_prob,
                                "train/grad_norm": self._last_grad_norm,
                                "train/step_time": elapsed / self.log_every,
                                "train/epoch": self.epoch,
                                "train/nan_skips": self._nan_count,
                            }, step=self.global_step)
                        accum_loss = 0.0
                        step_start = time.time()

                    if self.is_main and self.global_step % self.eval_every == 0 and self.eval_loader:
                        self._evaluate()
                        self.model.train()

                    if self.is_main and self.global_step % self.save_every == 0:
                        self._save_checkpoint()

                    if self.global_step >= self.max_steps:
                        break

            self.epoch += 1

        if self.is_main:
            self._save_checkpoint(final=True)
            self.logger.info("Training complete!")

    def _has_nan_grads(self):
        for p in self.raw_model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                return True
        return False

    def _train_step(self, batch):
        features = batch["features"].to(self.device, non_blocking=True)
        feature_lengths = batch["feature_lengths"].to(self.device, non_blocking=True)
        tokens = batch["tokens"].to(self.device, non_blocking=True)
        token_lengths = batch["token_lengths"].to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            log_probs, out_lengths, inter_log_probs = self.model(features, feature_lengths)

        if self.is_main and self.global_step % self.log_every == 0:
            with torch.no_grad():
                self._last_blank_prob = log_probs[:, :, 0].exp().mean().item()

        valid = out_lengths >= token_lengths
        if not valid.all():
            n_invalid = (~valid).sum().item()
            if self.is_main and self.global_step % self.log_every == 0:
                self.logger.warning(
                    f"   CTC skip: {n_invalid}/{features.shape[0]} samples "
                    f"(out_len < target_len)"
                )
            if valid.any():
                idx = valid.nonzero(as_tuple=True)[0]
                log_probs = log_probs[idx]
                out_lengths = out_lengths[idx]
                tokens = tokens[idx]
                token_lengths = token_lengths[idx]
                inter_log_probs = {
                    k: v[idx] for k, v in inter_log_probs.items()
                }
            else:
                return 0.0

        log_probs_ctc = log_probs.transpose(0, 1)
        loss = self.ctc_loss(log_probs_ctc, tokens, out_lengths, token_lengths)

        for _, inter_lp in inter_log_probs.items():
            inter_lp_ctc = inter_lp.transpose(0, 1)
            inter_loss = self.ctc_loss(inter_lp_ctc, tokens, out_lengths, token_lengths)
            loss = loss + self.inter_ctc_weight * inter_loss

        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0

        loss = loss / self.grad_accum_steps
        loss.backward()

        return loss.item() * self.grad_accum_steps

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_refs = []

        for batch in self.eval_loader:
            features = batch["features"].to(self.device)
            feature_lengths = batch["feature_lengths"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            token_lengths = batch["token_lengths"].to(self.device)
            texts = batch["texts"]

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                log_probs, out_lengths, _ = self.model(features, feature_lengths)

            valid = out_lengths >= token_lengths
            if valid.any():
                v_idx = valid.nonzero(as_tuple=True)[0]
                lp_ctc = log_probs[v_idx].transpose(0, 1)
                loss = self.ctc_loss(lp_ctc, tokens[v_idx], out_lengths[v_idx], token_lengths[v_idx])
                if not torch.isnan(loss):
                    total_loss += loss.item() * v_idx.shape[0]
                total_samples += v_idx.shape[0]
            else:
                total_samples += features.shape[0]

            eval_model = self.raw_model
            decoded_ids = eval_model.decode_greedy(log_probs, out_lengths)
            for ids, ref in zip(decoded_ids, texts):
                pred_text = self.tokenizer.decode(ids)
                all_preds.append(pred_text)
                all_refs.append(ref)

        avg_loss = total_loss / max(total_samples, 1)

        wer, cer = self._compute_metrics(all_preds, all_refs)

        self.logger.info(
            f"Eval @ step {self.global_step} | "
            f"Loss: {avg_loss:.4f} | WER: {wer:.2%} | CER: {cer:.2%}"
        )

        n_show = min(3, len(all_preds))
        for i in range(n_show):
            self.logger.info(f"   [{i}] pred: \"{all_preds[i][:80]}\"")
            self.logger.info(f"   [{i}] ref:  \"{all_refs[i][:80]}\"")

        if self.wandb_run:
            import wandb
            wandb.log({
                "eval/loss": avg_loss,
                "eval/wer": wer,
                "eval/cer": cer,
            }, step=self.global_step)

        if wer < self.best_wer:
            self.best_wer = wer
            self._save_checkpoint(best=True)
            self.logger.info(f"New best WER: {wer:.2%}")

    def _compute_metrics(self, predictions, references):
        try:
            from jiwer import wer as compute_wer, cer as compute_cer
            filtered_pairs = [
                (p, r) for p, r in zip(predictions, references) if r.strip()
            ]
            if not filtered_pairs:
                return 1.0, 1.0
            preds_f, refs_f = zip(*filtered_pairs)
            w = compute_wer(list(refs_f), list(preds_f))
            c = compute_cer(list(refs_f), list(preds_f))
            return w, c
        except ImportError:
            return 0.0, 0.0

    def _save_checkpoint(self, best=False, final=False):
        if best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
        elif final:
            path = os.path.join(self.checkpoint_dir, "final_model.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")

        checkpoint = {
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_wer": self.best_wer,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint not found: {path}")
            return

        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_wer = checkpoint.get("best_wer", float("inf"))

        self.logger.info(f"   Resumed at step {self.global_step}, epoch {self.epoch}")
