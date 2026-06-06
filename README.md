<h1 align="center">🚇 Metro-ASR</h1>

<p align="center">
  <strong>Non-Autoregressive CTC Speech Recognition for Egyptian Arabic + Code-Switching</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/metro-asr/"><img src="https://img.shields.io/pypi/v/metro-asr?style=for-the-badge&logo=pypi&logoColor=white&color=blue" alt="PyPI"></a>
  <a href="https://huggingface.co/MohammedAly22"><img src="https://img.shields.io/badge/🤗_HuggingFace-Models-yellow?style=for-the-badge" alt="HuggingFace Models"></a>
  <a href="https://huggingface.co/spaces/MohammedAly22/metro-asr"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-orange?style=for-the-badge" alt="HuggingFace Space"></a>
  <a href="#"><img src="https://img.shields.io/badge/📜_License-MIT-green?style=for-the-badge" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/🔬_Paper-Coming%20Soon-red?style=for-the-badge" alt="Paper"></a>
</p>

<p align="center">
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-model-variants">Models</a> •
  <a href="#%EF%B8%8F-rest-api-server">Server</a> •
  <a href="#-streaming">Streaming</a> •
  <a href="#-fine-tuning">Fine-Tuning</a> •
  <a href="#-examples">Examples</a>
</p>

<p align="center">
  <img src="images/Metro-ASR Banner.jpeg" alt="Metro-ASR" width="100%">
</p>

---

## ✨ Features

| | Feature | Details |
|---|---|---|
| 🇪🇬 | **Egyptian Arabic Focus** | Trained specifically on Egyptian dialect, not MSA |
| 🔀 | **Code-Switching** | Handles Arabic-English mixing — "أنا رايح الـ meeting" |
| ⚡ | **Blazing Fast** | RTF < 0.002 on CPU — **500x faster** than real-time |
| 🎯 | **Non-Autoregressive** | Single forward pass via CTC, no autoregressive loop |
| 📖 | **Beam Search + LM** | KenLM n-gram for improved accuracy |
| 🔴 | **Real-Time Streaming** | Live microphone transcription with Gradio UI |
| 🌐 | **REST API** | Deploy as HTTP server — curl, Postman, Python |
| 📦 | **pip install** | `pip install metro-asr` — 3 lines to transcribe |
| 📐 | **Scales 26M → 1.4B** | From on-device to server-grade |

---

## 📦 Installation

### From PyPI (Recommended)

```bash
# CPU-only (lightweight, no CUDA needed)
pip install metro-asr

# With GPU support
pip install metro-asr
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# With beam search + language model
pip install metro-asr[lm]

# With REST API server
pip install metro-asr[server]

# With Gradio web demo
pip install metro-asr[demo]

# Everything
pip install metro-asr[all]
```

### From Source

```bash
git clone https://github.com/MohammedAly22/Metro-ASR.git
cd Metro-ASR
pip install -e .

# For development
pip install -e ".[dev]"
```

### Conda Environment (Training)

```bash
conda create -n metro-asr python=3.10 -y
conda activate metro-asr

# PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Metro-ASR with training dependencies
pip install -e ".[train]"
```

---

## 🚀 Quick Start

### 3 Lines to Transcribe

```python
from metro_asr import MetroASREngine

engine = MetroASREngine.from_pretrained("small")  # Auto-downloads from HuggingFace
result = engine.transcribe("audio.wav")
print(result.text)
# أنا رايح الـ meeting الساعة خمسة
```

### 🎯 Beam Search + Language Model

```python
engine = MetroASREngine.from_pretrained(
    "small",
    lm_path="lm/lm_5gram.bin",
    beam_width=100,
    lm_alpha=0.5,
    lm_beta=5.0,
)

# Greedy (instant)
result = engine.transcribe("audio.wav")
print(f"Greedy: {result.text}")

# Beam search (more accurate)
result = engine.transcribe("audio.wav", beam_search=True)
print(f"Beam+LM: {result.text}")
```

### 📊 Batch Transcription

```python
results = engine.transcribe_batch(["audio1.wav", "audio2.wav", "audio3.wav"])
for r in results:
    print(f"{r.text}  ({r.duration:.1f}s, RTF={r.rtf:.4f})")
```

### 📈 Detailed Results

```python
result = engine.transcribe("audio.wav")

print(f"Text:       {result.text}")
print(f"Duration:   {result.duration:.2f}s")
print(f"Inference:  {result.inference_time*1000:.1f}ms")
print(f"Decoding:   {result.decoding_time*1000:.1f}ms")
print(f"RTF:        {result.rtf:.6f}")
print(f"Speed:      {1/result.rtf:.0f}x real-time")
print(f"Method:     {result.method}")
```

### 🔧 Full Configuration

```python
from metro_asr import MetroASREngine

engine = MetroASREngine.from_pretrained(
    "small",                            # tiny, small, medium, large
    device="cuda",                      # cpu, cuda, cuda:0
    lm_path="lm/lm_5gram.bin",         # language model path
    beam_width=100,                     # beam search width
    lm_alpha=0.5,                       # LM weight
    lm_beta=5.0,                        # word insertion bonus
)
```

### 📁 From Local Checkpoint

```python
engine = MetroASREngine.from_local(
    config_path="configs/metro_small.yaml",
    checkpoint_path="checkpoints/metro-small-v2/best_model.pt",
    tokenizer_dir="tokenizer_bpe5k_v2",
    device="cpu",
)
```

> [!TIP]
> The engine accepts file paths, `(sample_rate, tensor)` tuples, numpy arrays, or raw `torch.Tensor` waveforms.

---

## 🏗️ Model Variants

Metro-ASR scales from edge devices to servers. All sizes use the same Conformer architecture:

```
Audio (16kHz) → Log-Mel (80-dim) → Conv2d 4x Subsampling → N × Conformer Blocks → CTC Head → Text
```

| Model | Params | d_model | Layers | Heads | Vocab | Status |
|-------|--------|---------|--------|-------|-------|--------|
| 🟢 **Metro-Tiny** | 26M | 256 | 12 | 4 | 600 | ✅ Available |
| 🟢 **Metro-Small** | 61M | 384 | 12 | 6 | 5,000 | ✅ Available |
| 🟡 **Metro-Medium** | ~200M | 512 | 24 | 8 | 2,000 | 📋 Planned |
| 🔴 **Metro-Large** | ~700M | 768 | 32 | 12 | 4,000 | 📋 Planned |

### ⚡ Performance

| Metric | Metro-Tiny (CPU) | Metro-Small (CPU) |
|--------|------------------|-------------------|
| 🏎️ Inference RTF | ~0.002 | ~0.004 |
| ⏱️ Latency (10s audio) | ~20ms | ~40ms |
| 💾 Memory | ~100MB | ~250MB |
| 🖥️ Device | CPU | CPU |

### 🧬 Architecture

| Component | Description |
|-----------|-------------|
| 🔄 **RoPE** | Rotary Position Embeddings — relative encoding without positional tokens |
| ⚡ **SwiGLU** | Gated feed-forward with SiLU activation (Macaron-style dual FFN) |
| 📏 **RMSNorm** | Pre-norm Root Mean Square — faster than LayerNorm |
| 🎛️ **SE-Conv** | Squeeze-and-Excitation gated depthwise separable convolution |
| 🎲 **Stochastic Depth** | Layer dropout increasing linearly with depth |
| 🎯 **Intermediate CTC** | Auxiliary CTC loss at middle layers for better gradient flow |

---

## 🖥️ REST API Server

Deploy Metro-ASR as an HTTP server for production use.

### Start Server

```bash
python scripts/serve.py
```

The server runs at `http://localhost:8000` by default. Configure via variables at the top of the script.

### curl

```bash
# Basic transcription
curl -X POST http://localhost:8000/transcribe \
    -F "audio=@audio.wav"

# With beam search
curl -X POST http://localhost:8000/transcribe \
    -F "audio=@audio.wav" \
    -F "beam_search=true"

# Batch transcription
curl -X POST http://localhost:8000/transcribe/batch \
    -F "audio=@audio1.wav" \
    -F "audio=@audio2.wav"

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/info
```

### Python Client

```python
import requests

# Single file
with open("audio.wav", "rb") as f:
    resp = requests.post("http://localhost:8000/transcribe", files={"audio": f})
print(resp.json()["text"])

# Batch
files = [
    ("audio", open("audio1.wav", "rb")),
    ("audio", open("audio2.wav", "rb")),
]
resp = requests.post("http://localhost:8000/transcribe/batch", files=files)
for r in resp.json()["results"]:
    print(r["text"])
```

### Postman

1. **Method**: `POST`
2. **URL**: `http://localhost:8000/transcribe`
3. **Body**: Select `form-data`
   - Key: `audio` (type: File) → Select your `.wav` file
   - Key: `beam_search` (type: Text) → `true` (optional)
4. Click **Send**

### API Response

```json
{
    "text": "أنا رايح الـ meeting",
    "duration": 3.5,
    "rtf": 0.0023,
    "inference_time": 0.006,
    "decoding_time": 0.002,
    "method": "greedy"
}
```

---

## 🔴 Streaming

### Python API — Real-Time Streaming

```python
import soundfile as sf
from metro_asr import MetroASREngine

engine = MetroASREngine.from_pretrained("small")

def audio_chunks(path, chunk_sec=5.0):
    data, sr = sf.read(path)
    chunk_size = int(sr * chunk_sec)
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

for chunk in engine.transcribe_stream(audio_chunks("long_audio.wav")):
    print(f"[{chunk.total_duration:.1f}s] {chunk.text}")
    if chunk.is_final:
        print("--- END ---")
```

### 🎮 Gradio Web Demo

```bash
python app.py
```

Opens `http://localhost:7860` with two tabs:
- **📁 Transcribe** — File upload/record with model selection, beam search controls, and metrics
- **🎙 Stream** — Live microphone streaming with real-time transcription

---

## 🎯 Fine-Tuning

### On HuggingFace Datasets

Edit variables in `scripts/finetune.py`:

```python
CONFIG_PATH = "configs/metro_small.yaml"
TOKENIZER_DIR = "tokenizer_bpe5k_v2"
PRETRAINED_CHECKPOINT = "checkpoints/metro-small-v2/best_model.pt"

FINETUNE_DATASET = "MohamedRashad/arabic-english-code-switching"
FINETUNE_LR = 5e-5
FINETUNE_MAX_STEPS = 30000
FREEZE_ENCODER_STEPS = 3000
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py
```

### On Local Data

```python
from datasets import Dataset, Audio

data = [
    {"audio": "path/to/audio1.wav", "text": "أنا رايح الـ meeting"},
    {"audio": "path/to/audio2.wav", "text": "الـ project ده محتاج update"},
]

dataset = Dataset.from_list(data)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset.save_to_disk("my_data/train")
```

Set `PREPARED_DATA_DIR = "my_data"` in `scripts/finetune.py`.

> [!NOTE]
> The encoder is frozen for the first N steps so the CTC head can adapt to the new data distribution before the full model is fine-tuned.

### 📊 Fine-Tuning Tips

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning rate | `1e-4` to `5e-5` | 10-20x lower than pretraining |
| Freeze encoder | 3,000 – 10,000 steps | Lets CTC head adapt first |
| Max steps | 20,000 – 50,000 | Depends on dataset size |
| Batch size | 16 – 32 | Same as pretraining |

---

## 📖 Beam Search & Language Model

### How It Works

CTC greedy decoding makes per-frame decisions independently. Beam search explores multiple hypotheses and a **KenLM n-gram language model** rescores them to produce valid words:

```
CTC Output (log probs) → Beam Search (top-K paths) → KenLM Rescoring → Best Transcript
```

### Train Your Own LM

```bash
python scripts/train_lm.py
```

### 🎛️ Parameter Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| **Beam Width** | 10 – 500 | More beams = better quality, slower. Default: `100` |
| **Alpha (α)** | 0.0 – 3.0 | LM influence. Higher → trusts LM more |
| **Beta (β)** | 0.0 – 10.0 | Word insertion bonus. Higher → keeps more words |

> [!TIP]
> **English words being deleted?** Increase Beta (try 5.0 – 8.0)
> **Too many insertions?** Decrease Beta
> **Arabic garbled?** Decrease Alpha

### Greedy vs Beam + LM

| Method | Speed (RTF) | When to Use |
|--------|-------------|-------------|
| ⚡ Greedy | 0.0001 | Real-time, streaming, pure Arabic |
| 🎯 Beam + LM | 0.0015 | Code-switching, offline, best quality |

---

## 🏋️ Training from Scratch

### 1. Train Tokenizer

```bash
python scripts/train_bpe_tokenizer.py
```

### 2. Train Language Model

```bash
python scripts/train_lm.py
```

### 3. Start Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --gpu 0

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 scripts/train.py
```

### 4. Monitor with WandB

| Metric | Healthy Range |
|--------|---------------|
| `train/loss` | Decreasing steadily |
| `train/blank_prob` | 0.3 – 0.7 |
| `eval/wer` | Decreasing |
| `eval/cer` | Decreasing |

> [!WARNING]
> If `blank_prob` stays above 0.85 and `eval/wer` is stuck at 99%, SpecAugment may be too aggressive. Reduce `n_time_masks` and `time_mask_param`.

---

## 📓 Examples

| Notebook | Description | |
|----------|-------------|---|
| 🚀 [Quick Start](examples/quick_start.ipynb) | Install, transcribe in 3 lines, batch mode | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Metro-ASR/blob/main/examples/quick_start.ipynb) |
| 🌐 [Streaming & Server](examples/streaming_server.ipynb) | REST API, curl, Postman, Python client, streaming | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Metro-ASR/blob/main/examples/streaming_server.ipynb) |
| 🎮 [Gradio App](examples/gradio_app.ipynb) | Web demo with live microphone & file upload | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Metro-ASR/blob/main/examples/gradio_app.ipynb) |
| 🎯 [Fine-Tuning](examples/fine_tuning.ipynb) | Adapt to your domain, train LM & tokenizer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohammedAly22/Metro-ASR/blob/main/examples/fine_tuning.ipynb) |

---

## 📊 Datasets

### 🎙️ Audio Training Data

| Dataset | Description |
|---------|-------------|
| [AlaaSamir/custom-egy-tts](https://huggingface.co/datasets/AlaaSamir/custom-egy-tts) | Egyptian Arabic TTS |
| [OmarAhmedSobhy/egyption-with-emotion-dataset](https://huggingface.co/datasets/OmarAhmedSobhy/egyption-with-emotion-dataset) | Emotional Egyptian speech |
| [MightyStudent/Egyptian-ASR-MGB-3](https://huggingface.co/datasets/MightyStudent/Egyptian-ASR-MGB-3) | MGB-3 broadcast data |
| [MAdel121/arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) | Cleaned Egyptian Arabic |
| [MAdel121/Continuation-egy-for-ultravox-v1](https://huggingface.co/datasets/MAdel121/Continuation-egy-for-ultravox-v1) | Extended Egyptian data |
| [Raniahossam33/Egyptian_TTS3RS](https://huggingface.co/datasets/Raniahossam33/Egyptian_TTS3RS) | Egyptian TTS |
| [ahmedbasemdev/egyptain-tts-dataset](https://huggingface.co/datasets/ahmedbasemdev/egyptain-tts-dataset) | Egyptian TTS |
| [MohamedRashad/arabic-english-code-switching](https://huggingface.co/datasets/MohamedRashad/arabic-english-code-switching) | Arabic-English CS (12K) |

### 📚 Text-Only Data (LM Training)

| Dataset | Description |
|---------|-------------|
| [Prickly-Labs/1.9M-Egyptian-Corpus](https://huggingface.co/datasets/Prickly-Labs/1.9M-Egyptian-Corpus) | 1.9M Egyptian sentences |

---

## 📁 Project Structure

```
Metro-ASR/
├── metro_asr/                       # 📦 Core library (pip install metro-asr)
│   ├── __init__.py                  #   Package exports
│   ├── engine.py                    #   🚀 Inference engine (transcribe, batch, stream)
│   ├── pipeline.py                  #   Legacy pipeline API
│   ├── model/
│   │   ├── metro.py                 #   Full model (encoder + CTC head)
│   │   ├── encoder.py               #   Conformer Encoder
│   │   ├── attention.py             #   RoPE Multi-Head Attention
│   │   ├── feed_forward.py          #   SwiGLU Feed-Forward
│   │   ├── convolution.py           #   SE-Gated Depthwise Conv
│   │   ├── subsampling.py           #   Conv2d 4x Subsampling
│   │   ├── decoder.py               #   CTC Beam Search + KenLM
│   │   └── tokenizer.py             #   Char & BPE Tokenizers
│   ├── data/
│   │   ├── dataset.py               #   Dataset loading
│   │   ├── features.py              #   Log-Mel extraction
│   │   ├── augmentation.py          #   SpecAugment & Speed Perturb
│   │   └── collator.py              #   Batch collation
│   ├── training/
│   │   ├── trainer.py               #   DDP Trainer
│   │   └── optimizer.py             #   AdamW + Cosine Schedule
│   └── utils/
│       ├── config.py                #   YAML config loader
│       └── logger.py                #   Logger
│
├── scripts/                         # 🔧 CLI tools
│   ├── train.py                     #   Training entry point
│   ├── finetune.py                  #   Fine-tuning
│   ├── inference.py                 #   Single-file inference
│   ├── serve.py                     #   🌐 REST API server
│   ├── evaluate.py                  #   Evaluation
│   ├── train_bpe_tokenizer.py       #   BPE tokenizer training
│   └── train_lm.py                  #   KenLM LM training
│
├── configs/                         # ⚙️ Model configurations
│   ├── metro_tiny.yaml              #   26M params
│   ├── metro_small.yaml             #   61M params
│   └── metro_medium.yaml            #   ~200M params
│
├── examples/                        # 📓 Colab notebooks
│   ├── quick_start.ipynb            #   3-line transcription
│   ├── streaming_server.ipynb       #   REST API & streaming
│   ├── gradio_app.ipynb             #   Web demo
│   └── fine_tuning.ipynb            #   Domain adaptation
│
├── app.py                           # 🎮 Gradio web demo (transcribe + stream tabs)
├── pyproject.toml                   # 📦 Package config
├── requirements.txt                 # Dependencies
└── README.md
```

---

## 🗺️ Roadmap

### ✅ Available
- [x] Metro-Tiny (26M) & Metro-Small (61M)
- [x] `MetroASREngine` — inference, batch, streaming
- [x] `pip install metro-asr` — PyPI package
- [x] REST API server (curl, Postman, Python)
- [x] Gradio web demo with streaming
- [x] HuggingFace auto-download
- [x] Code-switching fine-tuning pipeline
- [x] KenLM beam search decoding
- [x] Colab notebooks

### 🔄 In Progress
- [ ] Metro-Medium (~200M) — training on H100
- [ ] HuggingFace Hub model upload

### 📋 Planned
- [ ] Metro-Large (~700M)
- [ ] ONNX + INT8/INT4 quantization
- [ ] Knowledge distillation (large → small)
- [ ] Domain-specific language models
- [ ] Mobile deployment (iOS / Android)

---

## 📝 Citation

```bibtex
@software{metro-asr-2025,
  title  = {Metro-ASR: Non-Autoregressive CTC-based ASR for Egyptian Arabic and Code-Switching},
  author = {Mohammed Aly},
  year   = {2025},
  url    = {https://github.com/MohammedAly22/Metro-ASR}
}
```

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
