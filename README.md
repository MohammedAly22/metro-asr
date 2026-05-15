# Metro-ASR

**Non-Autoregressive CTC-based ASR for Egyptian Arabic + English Code-Switching**

Metro-ASR is a speech recognition system designed for Egyptian Arabic with English code-switching support. It uses a CTC-based non-autoregressive approach for fast, on-device inference with no decoder or beam search required.

## Current Results (Metro-22M, Character Tokenizer)

| Step | Train Loss | Eval Loss | WER | CER |
|------|-----------|-----------|-----|-----|
| 1,000 | 3.40 | 2.51 | 99.2% | 72.7% |
| 2,000 | 3.20 | 1.83 | 90.6% | 46.3% |
| 44,000 | 1.50 | 0.59 | 37.8% | 15.9% |
| 88,000 | 1.40 | 0.50 | 32.1% | 13.5% |

Sample predictions at step 88k:
```
pred: "ولا بس قاله يا ابني أنتبتعمل إيه ولا أي حاجة"
ref:  "ولا بسأله يا ابني أنتبت عمل إيه ولا أي حاجة"

pred: "وبتخلي فيه منتصر واحد بسا قوي من كل اللي في الحلبة"
ref:  "وبتخلي فيه منتصر واحد بساقو ي من كل اللي في الحلبة"
```

## Architecture

Metro-ASR builds on the Conformer foundation with modern improvements:

- **RoPE (Rotary Positional Embeddings)** for relative position encoding
- **SwiGLU Feed-Forward** modules (Macaron-style dual FF layout)
- **RMSNorm** pre-normalization
- **Squeeze-and-Excitation** enhanced convolution modules
- **Stochastic Depth** regularization
- **Intermediate CTC** losses for deep gradient flow
- **CTC head** for non-autoregressive, constant-time decoding

### Model Sizes

| Model | d_model | Layers | Heads | Params |
|-------|---------|--------|-------|--------|
| Metro-22M | 256 | 12 | 4 | ~25M |
| Metro-50M | 384 | 12 | 6 | ~50M |
| Metro-222M | 512 | 24 | 8 | ~222M |
| Metro-640M | 768 | 32 | 12 | ~640M |
| Metro-1B | 1024 | 36 | 16 | ~1B |

### Training Features

- **bf16 mixed precision** for stable, fast training on H100/A100
- **Gradient accumulation** for large effective batch sizes
- **NaN detection and gradient skipping** for training stability
- **WandB logging**: loss, WER, CER, blank probability, gradient norms
- **SpecAugment** + speed perturbation for data augmentation
- **Warmup cosine LR schedule** with configurable init ratio

## Installation

### 1. Create Environment

```bash
conda create -n metro-asr python=3.10 -y
conda activate metro-asr
```

### 2. Install PyTorch

```bash
# CUDA 12.1
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Login to Services

```bash
huggingface-cli login    # Dataset access
wandb login              # Training monitoring
```

## Quick Start

### Prepare Data

Downloads and merges 7 Egyptian Arabic datasets from HuggingFace. Builds the tokenizer.

```bash
python scripts/prepare_data.py
```

### Train

**Single GPU:**

```bash
python scripts/train.py --config configs/metro_22m.yaml
```

**Multi-GPU:**

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/metro_22m.yaml
```

### Inference

```bash
python scripts/inference.py
```

### Web App

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Project Structure

```
Metro-ASR/
├── configs/                    # YAML configs for each model size
│   ├── metro_22m.yaml
│   ├── metro_50m.yaml
│   ├── metro_222m.yaml
│   ├── metro_640m.yaml
│   └── metro_1b.yaml
├── metro_asr/                  # Core library
│   ├── model/                  # Architecture
│   │   ├── attention.py        # RoPE + Multi-Head Self-Attention
│   │   ├── feed_forward.py     # SwiGLU Feed-Forward
│   │   ├── convolution.py      # SE-Gated Convolution Module
│   │   ├── subsampling.py      # Conv2d 4x Subsampling
│   │   ├── encoder.py          # Metro Encoder (block stack + intermediate CTC)
│   │   ├── metro.py            # Full Metro-ASR model
│   │   └── tokenizer.py        # Character & BPE tokenizers
│   ├── data/                   # Data pipeline
│   │   ├── features.py         # Log-Mel feature extraction
│   │   ├── augmentation.py     # SpecAugment & speed perturbation
│   │   ├── dataset.py          # HuggingFace dataset loader
│   │   └── collator.py         # Batch padding & collation
│   ├── training/               # Training loop
│   │   ├── trainer.py          # DDP trainer with WandB + NaN protection
│   │   └── optimizer.py        # AdamW + warmup cosine scheduler
│   └── utils/
│       ├── config.py           # YAML config loader
│       └── logger.py           # Terminal logger
├── scripts/
│   ├── train.py                # Training entry point
│   ├── finetune.py             # Fine-tuning with frozen encoder
│   ├── inference.py            # Single-file inference
│   ├── batch_inference.py      # Batch inference to CSV
│   ├── prepare_data.py         # Dataset preparation
│   ├── diagnose.py             # Training diagnostic tool
│   ├── retrain_tokenizer.py    # Tokenizer retraining
│   └── export_onnx.py          # ONNX export
├── app.py                      # Gradio web interface
├── requirements.txt
└── README.md
```

## Training Configuration

Key settings in `configs/metro_22m.yaml`:

```yaml
tokenizer:
  type: "char"          # Character tokenizer (vocab ~102)
training:
  batch_size: 32
  grad_accumulation_steps: 4    # Effective batch = 128
  learning_rate: 0.001
  warmup_steps: 2500
  max_grad_norm: 1.0
  bf16: true
  intermediate_ctc_layers: [5]  # Auxiliary CTC at layer 5
  intermediate_ctc_weight: 0.3
```

### Resume Training

```yaml
training:
  resume_from: "checkpoints/metro-22m/checkpoint_step_50000.pt"
```

### WandB Metrics

- `train/loss`, `train/lr`, `train/blank_prob`, `train/grad_norm`
- `eval/loss`, `eval/wer`, `eval/cer`
- `train/nan_skips` (should stay at 0)

## Datasets

Trained on 7 Egyptian Arabic datasets from HuggingFace:

| Dataset | Description |
|---------|-------------|
| AlaaSamir/custom-egy-tts | Egyptian Arabic TTS |
| OmarAhmedSobhy/egyption-with-emotion-dataset | Emotional Egyptian speech |
| MightyStudent/Egyptian-ASR-MGB-3 | MGB-3 Egyptian ASR benchmark |
| MAdel121/arabic-egy-cleaned | Cleaned Egyptian Arabic |
| MAdel121/Continuation-egy-for-ultravox-v1 | Extended Egyptian data |
| Raniahossam33/Egyptian_TTS3RS | Egyptian TTS dataset |
| ahmedbasemdev/egyptain-tts-dataset | Egyptian TTS dataset |

## Fine-tuning

```bash
python scripts/finetune.py
```

Configure `PRETRAINED_CHECKPOINT`, `FINETUNE_LR`, and `FREEZE_ENCODER_STEPS` at the top of the script.

## ONNX Export

```bash
python scripts/export_onnx.py
```

## Roadmap

- [x] Core architecture (Conformer + RoPE + SwiGLU + Intermediate CTC)
- [x] Character tokenizer training with stable CTC convergence
- [x] Metro-22M baseline (32% WER, 13.5% CER at 88k steps)
- [ ] Complete 300k step pretraining
- [ ] Hybrid tokenizer for code-switching (Arabic chars + English BPE)
- [ ] Scale to Metro-50M, Metro-222M, Metro-1B
- [ ] Code-switching fine-tuning
- [ ] Knowledge distillation (1B -> smaller models)
- [ ] INT8 quantized ONNX for on-device deployment

## License

This project is for research purposes.
