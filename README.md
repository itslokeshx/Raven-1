<div align="center">

# 🐦‍⬛ RAVEN-1

**A 30M parameter language model, built from scratch.**

No frameworks. No shortcuts. Just PyTorch.

---

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Parameters](https://img.shields.io/badge/Parameters-~30M-blueviolet?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## What is RAVEN-1?

RAVEN-1 is a GPT-style, decoder-only transformer language model with ~30 million parameters. Every component — tokenizer, model, training loop, inference engine, and web UI — is written from scratch in Python and PyTorch. No HuggingFace `transformers`, no pre-trained weights, no abstractions hiding the work.

It's designed to run on **Google Colab free tier** (T4 GPU, 12GB RAM) and can be trained end-to-end in a single session.

---

## Architecture

```
Input Tokens
     │
     ▼
┌─────────────────────────┐
│  Token Embedding (8192) │
│  + Position Embedding   │
│  + Dropout (0.1)        │
└────────────┬────────────┘
             │
             ▼
     ┌───────────────┐
     │  8× Transformer│ ◄── Pre-LayerNorm
     │     Block      │     Causal Self-Attention (8 heads)
     │                │     FlashAttention (auto-dispatched)
     │  ┌───────────┐ │     GELU Feed-Forward (4× expansion)
     │  │ LN → Attn │ │     Residual connections
     │  │ LN → FFN  │ │
     │  └───────────┘ │
     └───────┬───────┘
             │
             ▼
┌─────────────────────────┐
│    Final LayerNorm      │
│    LM Head (weight-tied)│
└─────────────────────────┘
             │
             ▼
        Logits (8192)
```

### Key Specs

| Parameter       | Value               |
|:----------------|:--------------------|
| Parameters      | ~30M (29.51M)       |
| Vocab Size      | 8,192 (byte-level BPE) |
| Context Length   | 256 tokens          |
| Embedding Dim   | 512                 |
| Attention Heads  | 8                   |
| Layers          | 8                   |
| FFN Dim         | 2,048 (4× embed)   |
| Head Size       | 64                  |
| Dropout         | 0.1                 |

---

## Project Structure

```
Raven-1/
├── config.py            # Central config — every hyperparameter lives here
├── model.py             # Transformer model (attention, FFN, blocks, generation)
├── tokenizer_train.py   # Train byte-level BPE tokenizer from corpus
├── data_prep.py         # Tokenize text → binary memmap files (chunked, RAM-safe)
├── train.py             # Pre-training loop (cosine LR, AMP, gradient accumulation)
├── posttrain.py         # Post-training / fine-tuning loop
├── inference.py         # Terminal chat interface with slash commands
├── app.py               # Gradio web UI for Hugging Face Spaces
├── colab_train.ipynb    # One-click Colab training notebook
├── requirements.txt     # Dependencies
├── data/
│   ├── corpus.txt       # Pre-training corpus (~389MB)
│   └── posttrain.txt    # Post-training conversational data
└── checkpoints/         # Generated at runtime
    ├── raven1_tokenizer.json
    ├── train.bin / val.bin
    ├── raven1_best.pt
    ├── raven1_posttrain_best.pt
    └── raven1_latest.pt
```

---

## Training Pipeline

The full pipeline has **4 stages**, each with its own script:

### 1. Train Tokenizer

```bash
python tokenizer_train.py
```

Trains a **byte-level BPE tokenizer** (8,192 tokens) on `data/corpus.txt` using HuggingFace's `tokenizers` library. Adds special tokens (`<|user|>`, `<|raven|>`, `<|eos|>`, `<|pad|>`) and saves to `checkpoints/raven1_tokenizer.json`.

### 2. Prepare Data

```bash
python data_prep.py
```

Tokenizes text corpora into binary `uint16` memmap files for fast training. Reads in **10MB chunks** to avoid OOM on Colab's 12GB RAM. Splits the pre-training corpus 95/5 into `train.bin` and `val.bin`.

### 3. Pre-train

```bash
python train.py
```

Full pre-training loop with production-grade features:

- ⚡ **Mixed precision** (AMP + GradScaler)
- 📈 **Cosine LR schedule** with linear warmup (300 steps → 10,000 steps)
- 🔄 **Gradient accumulation** (effective batch size = 64)
- ✂️ **Gradient clipping** (max norm 1.0)
- 💾 **Auto-resume** from latest checkpoint
- ⭐ **Best model tracking** (saves on val_loss improvement)
- ☁️ **Google Drive sync** (automatic backup on Colab)
- 🛑 **Clean interrupt** (Ctrl+C saves checkpoint before exit)

| Training Param        | Value     |
|:----------------------|:----------|
| Peak LR               | 3e-4      |
| Min LR (cosine floor) | 3e-5      |
| Warmup Steps           | 300       |
| Total Steps            | 10,000    |
| Batch Size (micro)     | 32        |
| Gradient Accum Steps   | 2         |
| Effective Batch Size   | 64        |
| Weight Decay           | 0.1       |
| Eval Interval          | 500 steps |

### 4. Post-train

```bash
python posttrain.py
```

Fine-tunes the pre-trained model on curated conversational data with a lower learning rate (5e-5). Runs before/after generation comparisons on test prompts. Uses the same training infrastructure (AMP, cosine LR, checkpointing).

| Post-training Param   | Value     |
|:----------------------|:----------|
| Peak LR               | 5e-5      |
| Min LR                | 5e-6      |
| Warmup Steps           | 100       |
| Total Steps            | 2,000     |

---

## Inference

### Terminal Chat

```bash
python inference.py
```

Interactive chat with conversation history (last 3 turns), multiple sampling modes, and slash commands:

| Command          | Action                     |
|:-----------------|:---------------------------|
| `/mode [name]`   | Switch sampling mode       |
| `/modes`         | List all available modes   |
| `/clear`         | Clear terminal screen      |
| `/reset`         | Clear conversation history |
| `/exit`          | Exit                       |

### Sampling Modes

| Mode      | Temperature | Top-K | Top-P | Rep Penalty |
|:----------|:------------|:------|:------|:------------|
| `default` | 0.65        | 25    | 0.85  | 1.10        |
| `sharp`   | 0.55        | 20    | 0.80  | 1.10        |
| `chaos`   | 0.85        | 45    | 0.92  | 1.05        |
| `cold`    | 0.45        | 12    | 0.75  | 1.08        |

### Web UI (Gradio)

```bash
python app.py
```

Dark monochrome Gradio interface at `localhost:7860`. Same generation logic as the terminal chat. Deployable to Hugging Face Spaces.

---

## Training on Google Colab

Open `colab_train.ipynb` and run all cells in order:

1. **Mount Drive & check GPU** — Verifies T4 GPU and VRAM
2. **Install dependencies** — `tokenizers`, `gradio`, `numpy`
3. **Create directories** — Upload project files via the Files panel
4. **Train tokenizer** — `python tokenizer_train.py`
5. **Prepare data** — `python data_prep.py` (chunked, RAM-safe)
6. **Pre-train** — `python train.py` (auto-resumes, syncs to Drive)
7. **Post-train** — `python posttrain.py`
8. **Test** — `python inference.py`
9. **Download weights** — Downloads `.pt` and tokenizer `.json`

> **Tip:** If the Colab session disconnects, just re-run from the pre-train cell — it auto-resumes from the latest checkpoint synced to Drive.

---

## Configuration

All hyperparameters live in [`config.py`](config.py). Zero hardcoded values anywhere else.

```python
# Model
vocab_size   = 8192      # BPE vocabulary size
block_size   = 256       # Context length (tokens)
n_embd       = 512       # Embedding dimension
n_head       = 8         # Attention heads
n_layer      = 8         # Transformer blocks
dropout      = 0.1       # Dropout rate

# Special Tokens
user_token   = "<|user|>"
raven_token  = "<|raven|>"
eos_token    = "<|eos|>"
pad_token    = "<|pad|>"
```

---

## Requirements

```
torch>=2.1.0
tokenizers>=0.15.0
numpy>=1.24.0
gradio>=4.0.0
```

```bash
pip install -r requirements.txt
```

---

## Design Decisions

- **Weight tying** — LM head shares weights with the token embedding, reducing parameter count and improving generalization.
- **Pre-LayerNorm** — LayerNorm before attention/FFN (not after), which stabilizes training for small models.
- **FlashAttention** — Uses `F.scaled_dot_product_attention` which auto-dispatches to FlashAttention on supported hardware.
- **Byte-level BPE** — Handles all Unicode without unknown tokens; 8,192 vocab is small enough for a 25M model.
- **Chunked tokenization** — Processes 10MB at a time during data prep to stay within Colab's RAM limits.
- **Memmap data** — Training data stored as memory-mapped `uint16` arrays for zero-copy random access.
- **Cosine schedule with warmup** — Standard LR schedule that prevents instability at start and enables graceful convergence.
- **Two-phase training** — Pre-train on raw text for language understanding, then post-train on conversational data for personality.

---

## License

MIT

---

<div align="center">

**Built from scratch. No shortcuts.**

</div>

