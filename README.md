<div align="center">

# 🐦‍⬛ RAVEN-1

**A 29.5M parameter language model, built entirely from scratch.**

No frameworks. No shortcuts. Just PyTorch, math, and stubbornness.

---

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Parameters](https://img.shields.io/badge/Parameters-29.51M-blueviolet?style=flat-square)
![Vocab](https://img.shields.io/badge/Vocab-8,192_BPE-ff6f61?style=flat-square)
![Context](https://img.shields.io/badge/Context-256_tokens-2ec4b6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## What is RAVEN-1?

RAVEN-1 is a **GPT-style decoder-only transformer** language model with **29.51 million parameters**. Every single component — tokenizer, model architecture, training loop, data pipeline, inference engine, and web UI — is written from scratch in Python and PyTorch.

No HuggingFace `transformers`. No pre-trained weights borrowed. No abstractions hiding the work.

### Why Build From Scratch?

- **Full understanding** — Every matrix multiply, every attention mask, every gradient step is explicit and visible.
- **Colab-friendly** — Designed to train end-to-end on Google Colab's free T4 GPU (15GB VRAM, 12GB RAM).
- **Two-phase training** — Pre-trained on a 389MB text corpus for language modeling, then post-trained on 977MB of curated conversational data for personality.
- **Production-grade tooling** — Resume-safe checkpoints, mixed precision, Google Drive sync, streaming data pipelines, and a Gradio web UI for deployment.

### Personality

RAVEN-1 was post-trained on conversational data that gives it a distinct personality — dry, sarcastic, concise. It's not trying to be helpful. It's trying to be honest. Sometimes annoyingly so.

> **User:** I'm bored  
> **Raven:** Good. Boredom is where ideas start.

> **User:** good morning  
> **Raven:** Morning. Let's keep it simple.

> **User:** thanks!  
> **Raven:** Thanks. I'll file that away.

---

## Architecture

```
Input Tokens (sequence of integers)
     │
     ▼
┌──────────────────────────────────────┐
│    Token Embedding (8,192 → 512)     │  Weight-tied with LM Head
│  + Learned Position Embedding (256)  │
│  + Dropout (0.1)                     │
└──────────────────┬───────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │                     │
        │   8× Transformer    │  ◄── Pre-LayerNorm design
        │       Block         │
        │                     │
        │  ┌───────────────┐  │
        │  │ LayerNorm     │  │
        │  │ ↓             │  │
        │  │ Multi-Head    │  │  8 heads × 64 dim = 512
        │  │ Causal Attn   │  │  FlashAttention (auto-dispatched)
        │  │ + Residual    │  │  Combined Q/K/V projection
        │  │               │  │
        │  │ LayerNorm     │  │
        │  │ ↓             │  │
        │  │ Feed-Forward  │  │  512 → 2048 → 512
        │  │ (GELU)        │  │  No bias terms
        │  │ + Residual    │  │
        │  └───────────────┘  │
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         Final LayerNorm              │
│         LM Head (512 → 8,192)       │  Weight-tied with Token Embedding
└──────────────────┬───────────────────┘
                   │
                   ▼
            Logits (8,192)
```

### Model Specifications

| Parameter | Value | Notes |
|:---|:---|:---|
| **Total Parameters** | 29.51M | Counted with weight tying |
| **Vocab Size** | 8,192 | Byte-level BPE |
| **Context Length** | 256 tokens | Maximum sequence length |
| **Embedding Dim** | 512 | `n_embd` |
| **Attention Heads** | 8 | `n_head` |
| **Head Dimension** | 64 | `n_embd / n_head` |
| **Transformer Layers** | 8 | `n_layer` |
| **FFN Inner Dim** | 2,048 | `4 × n_embd` |
| **Activation** | GELU | In feed-forward network |
| **Normalization** | Pre-LayerNorm | Before attention & FFN |
| **Dropout** | 0.1 | Embedding, attention, FFN residual |
| **Weight Tying** | Yes | LM head shares token embedding weights |
| **Attention** | FlashAttention | Auto-dispatched via `F.scaled_dot_product_attention` |
| **Bias Terms** | None | All Linear layers are bias-free |

### Special Tokens

| Token | ID | Purpose |
|:---|:---|:---|
| `<\|pad\|>` | 0 | Padding |
| `<\|user\|>` | 1 | User turn marker |
| `<\|raven\|>` | 2 | Model turn marker |
| `<\|eos\|>` | 3 | End of sequence |

---

## Project Structure

```
Raven-1/
│
├── config.py                  # Central configuration — ALL hyperparameters
├── model.py                   # Transformer architecture + generation
├── train.py                   # Pre-training loop
├── posttrain.py               # Post-training / fine-tuning loop
├── tokenizer_train.py         # Train byte-level BPE tokenizer
├── data_prep.py               # Tokenize text → binary memmap files
├── posttrain_data_prep.py     # Streaming post-train data pipeline
├── inference.py               # Terminal chat with slash commands
├── app.py                     # Gradio web UI (HF Spaces compatible)
│
├── eval/                      # Evaluation & benchmarks
│   ├── run_300_test.py        # 300-prompt benchmark suite
│   ├── stress_test.py         # Stress test with long inputs
│   ├── pretrain_results.txt   # Pre-train benchmark results
│   └── posttrain_results.txt  # Post-train benchmark results
│
├── notebooks/                 # Colab training notebooks
│   ├── colab_train.ipynb      # One-click pre-training notebook
│   └── colab_posttrain.ipynb  # One-click post-training notebook
│
├── data/                      # Training corpora (not in git)
│   ├── corpus.txt             # Pre-training corpus (389 MB)
│   └── posttrain.txt          # Post-training data (977 MB, ~13M lines)
│
├── checkpoints/               # Generated at runtime (not in git)
│   ├── raven1_tokenizer.json  # Trained BPE tokenizer (557 KB)
│   ├── raven1_best.pt         # Best pre-trained weights (338 MB)
│   ├── raven1_posttrain_best.pt  # Best post-trained weights (338 MB)
│   ├── train.bin / val.bin    # Pre-training binary data
│   └── posttrain_train.bin / posttrain_val.bin
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Total hand-written code: ~2,480 lines** across 11 Python files.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Inference (Terminal Chat)

```bash
python inference.py
```

The model automatically loads the best available weights:
1. **Post-train weights** (`raven1_posttrain_best.pt`) — tried first
2. **Pre-train weights** (`raven1_best.pt`) — fallback
3. **Random weights** — if no checkpoints found

### 3. Run Web UI

```bash
python app.py
# → Opens at http://localhost:7860
```

---

## Training Pipeline

The full pipeline has **5 stages**, each with a dedicated script. Every stage is independently runnable, resume-safe, and Colab-compatible.

### Stage 1 — Train Tokenizer

```bash
python tokenizer_train.py
```

Trains a **byte-level BPE tokenizer** on `data/corpus.txt` using HuggingFace's `tokenizers` library (Rust backend for speed).

| Detail | Value |
|:---|:---|
| Algorithm | Byte-level BPE |
| Vocab size | 8,192 |
| Min frequency | 2 |
| Special tokens | `<\|pad\|>`, `<\|user\|>`, `<\|raven\|>`, `<\|eos\|>` |
| Output | `checkpoints/raven1_tokenizer.json` |
| Corpus | `data/corpus.txt` (389 MB) |

### Stage 2 — Prepare Data

```bash
# Pre-training data
python data_prep.py

# Post-training data (streaming, handles 1GB+ without RAM issues)
python posttrain_data_prep.py
```

Tokenizes text corpora into binary `uint16` memmap files for zero-copy random access during training.

**Key features:**
- **Chunked processing** — reads 10MB (data_prep) or 8MB (posttrain_data_prep) at a time to stay within Colab's 12GB RAM
- **Streaming parser** — reconstructs `<|eos|>`-delimited samples across chunk boundaries
- **Batch tokenization** — uses `encode_batch()` with Rust multi-threading (4,096 samples per batch)
- **Metadata cache** — skips rebuild if source file and tokenizer haven't changed
- **Malformed sample detection** — warns if >1% of samples lack required special tokens

| Data Split | Pre-training | Post-training |
|:---|:---|:---|
| Split ratio | 95% / 5% | 90% / 10% |
| Source | `data/corpus.txt` (389 MB) | `data/posttrain.txt` (977 MB) |
| Output format | `uint16` memmap | `uint16` memmap |

### Stage 3 — Pre-train

```bash
python train.py
```

Full pre-training loop with production-grade features:

| Feature | Implementation |
|:---|:---|
| ⚡ Mixed precision | AMP + GradScaler (automatic loss scaling) |
| 📈 LR schedule | Cosine decay with linear warmup |
| 🔄 Gradient accumulation | Effective batch = micro_batch × accum_steps |
| ✂️ Gradient clipping | Max norm 1.0 |
| 💾 Auto-resume | Loads latest checkpoint automatically |
| ⭐ Best model tracking | Saves on val_loss improvement |
| ☁️ Drive sync | Auto-backup to Google Drive on Colab |
| 🛑 Clean interrupt | Ctrl+C saves checkpoint before exit |
| 📊 Parameter groups | Weight decay on 2D+ params only, none on biases/norms |

| Hyperparameter | Value |
|:---|:---|
| Peak learning rate | `3e-4` |
| Min learning rate | `3e-5` (cosine floor) |
| Warmup steps | 300 |
| Total steps | 10,000 |
| Micro batch size | 32 |
| Gradient accumulation | 2 steps |
| **Effective batch size** | **64** |
| Tokens per step | 64 × 256 = **16,384** |
| Weight decay | 0.1 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Eval interval | Every 500 steps |
| Eval batches | 100 |

**Pre-train results:**
- Best checkpoint: **step 9,000**
- Best val loss: **0.5906**

### Stage 4 — Post-train

```bash
python posttrain.py              # Full post-training
python posttrain.py --resume     # Resume from checkpoint
python posttrain.py --dry-run    # Test 20 steps only
```

Fine-tunes the pre-trained model on curated conversational data. Uses the same training infrastructure with lower learning rates and generates before/after sample comparisons.

| Hyperparameter | Value |
|:---|:---|
| Peak learning rate | `5e-5` |
| Min learning rate | `5e-6` |
| Warmup steps | 100 |
| Total steps | 2,000 |
| Eval interval | Every 250 steps |
| Log interval | Every 25 steps |

**Post-train results:**
- Best checkpoint: **step 2,000**
- Best val loss: **0.5950**

### Stage 5 — Test & Evaluate

```bash
python eval/run_300_test.py     # 300-prompt benchmark across 6 categories
python eval/stress_test.py      # Stress test with very long inputs
python inference.py             # Interactive terminal chat
```

---

## Inference

### Terminal Chat (`inference.py`)

```bash
python inference.py
```

Interactive chat with conversation history (last 3 turns), multiple sampling modes, and slash commands.

```
══════════════════════════════
  RAVEN-1  |  29.5M parameters
  Loaded: posttrain
  Device: cpu
══════════════════════════════

You: hey there
Raven: You found me. Now what.

You: I'm tired of everything
Raven: Everything is temporary. Including this conversation, hopefully.

You: /mode chaos
  Mode → chaos

You: /exit
Bye.
```

**Slash commands:**

| Command | Action |
|:---|:---|
| `/mode [name]` | Switch sampling mode (`default`, `sharp`, `chaos`, `cold`) |
| `/modes` | List all modes with current parameters |
| `/clear` | Clear terminal screen |
| `/reset` | Clear conversation history |
| `/exit` | Exit the chat |

### Web UI (`app.py`)

```bash
python app.py
# → http://localhost:7860
```

Premium dark monochrome Gradio interface. Same generation logic as the terminal chat. Features mode selection dropdown and conversation history. Deployable directly to **Hugging Face Spaces**.

**UI tech stack:** Gradio with custom dark theme, Inter font (Google Fonts), `#0a0a0a` background, 12px border-radius, 700px max-width container.

### Generation Pipeline

Both inference methods use the same pipeline:

1. **Build prompt** with conversation context (last 3 turns)
2. **Encode** with BPE tokenizer
3. **Crop** to `block_size` (256) if needed (keeps rightmost tokens)
4. **Generate** with autoregressive sampling:
   - Temperature scaling
   - Top-K filtering
   - Top-P (nucleus) filtering
   - Repetition penalty on previously generated tokens
5. **Decode** only new tokens
6. **Clean** leaked special tokens
7. **Fallback** to witty canned response if output is empty

---

## Training Results

### Pre-training

| Metric | Value |
|:---|:---|
| Corpus | 389 MB raw text |
| Steps completed | 10,000 |
| Best step | 9,000 |
| Best val loss | **0.5906** |
| Checkpoint size | 338 MB |

### Post-training

| Metric | Value |
|:---|:---|
| Corpus | 977 MB conversational data (~13M lines) |
| Steps completed | 2,000 |
| Best step | 2,000 |
| Best val loss | **0.5950** |
| Checkpoint size | 338 MB |

### Loss Trajectory

The model converges smoothly with cosine decay:

```
Pre-train:   random init → ~7.0 loss → 0.5906 val loss (9,000 steps)
Post-train:  0.5906 pretrain → 0.5950 posttrain val loss (2,000 steps)
```

---

## 300-Prompt Benchmark

Both the pre-trained and post-trained models were evaluated on 300 diverse prompts across 6 categories using identical sampling parameters (`default` mode).

### Benchmark Comparison

| Metric | Pre-train | Post-train | Δ |
|:---|:---|:---|:---|
| Total time | 358.02s | **181.56s** | **1.97× faster** |
| Avg latency | 1.19s | **0.61s** | **49% reduction** |
| Min latency | 0.08s | — | — |
| Max latency | 3.28s | — | — |

### Category Distribution

| Category | Count | Description |
|:---|:---|:---|
| `casual` | 53 | Greetings, small talk, emotions |
| `coding` | 52 | Code snippets, technical prompts |
| `creative` | 63 | Stories, poems, hypotheticals |
| `edge_cases` | 43 | Empty inputs, repeated chars, symbols |
| `math` | 36 | Arithmetic, algebra, number theory |
| `questions` | 53 | Factual knowledge queries |

### Sample Post-train Outputs

**Casual / Conversational** (strong improvement):

| Input | Output |
|:---|:---|
| hello | Hello. Make it quick. |
| I'm bored | Good. Boredom is where ideas start. |
| what's up? | Morning. Let's keep it simple. |
| good morning | The universe trembles. Or not. |
| good night | Night. The quiet suits you. |
| bye | Still here. The quiet suits you. |
| I am happy | Noted. Now use it. |
| class MyClass: | The body is being patient. Do not test that. |

**Personality-driven responses:**

| Input | Output |
|:---|:---|
| make me laugh | 1. You survived basic arithmetic. |
| who are you? | I was here before you typed that. Go ahead. |
| how does electricity work? | Electricity's just a phase. Probably. |
| what is gravity? | Mass pulling on mass. Keeps you grounded. |
| what is AI? | The answer existed before you asked. |

> **Note:** The post-train corpus included some competitive programming data, which occasionally causes reasoning-style leakage (e.g., `"To solve this problem..."`) on certain prompts. This is a data quality issue, not a model architecture issue.

---

## Sampling Modes

| Mode | Temperature | Top-K | Top-P | Repetition Penalty | Character |
|:---|:---|:---|:---|:---|:---|
| `default` | 0.65 | 25 | 0.85 | 1.10 | Balanced, reliable |
| `sharp` | 0.55 | 20 | 0.80 | 1.10 | More focused, deterministic |
| `chaos` | 0.85 | 45 | 0.92 | 1.05 | Creative, unpredictable |
| `cold` | 0.45 | 12 | 0.75 | 1.08 | Most conservative, factual |

All modes generate up to **60 new tokens** per response and stop on `<|eos|>`.

---

## Configuration Reference

All hyperparameters live in [`config.py`](config.py) — **zero hardcoded values** anywhere else in the codebase. Every script imports from this single source of truth.

```python
# ── Model Architecture ──
vocab_size   = 8192        # BPE vocabulary size
block_size   = 256         # Maximum context length (tokens)
n_embd       = 512         # Embedding dimension
n_head       = 8           # Number of attention heads
n_layer      = 8           # Number of transformer blocks
dropout      = 0.1         # Dropout rate during training

# ── Pre-training ──
batch_size   = 32          # Micro-batch size per step
gradient_accumulation_steps = 2   # Effective batch = 64
max_iters    = 10000       # Total pre-training steps
learning_rate = 3e-4       # Peak learning rate
min_lr       = 3e-5        # Cosine floor
warmup_iters = 300         # Linear warmup steps
weight_decay = 0.1         # AdamW weight decay
grad_clip    = 1.0         # Gradient norm clipping
eval_interval = 500        # Evaluate every N steps
eval_iters   = 100         # Batches per evaluation
use_amp      = True        # Mixed precision training

# ── Post-training ──
posttrain_max_iters       = 2000
posttrain_learning_rate   = 5e-5
posttrain_min_lr          = 5e-6
posttrain_warmup_iters    = 100

# ── Special Tokens ──
user_token   = "<|user|>"
raven_token  = "<|raven|>"
eos_token    = "<|eos|>"
pad_token    = "<|pad|>"

# ── Device ──
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Design Decisions

### Architecture Choices

| Decision | Rationale |
|:---|:---|
| **Pre-LayerNorm** | LayerNorm before attention/FFN (not after) stabilizes training for small models. Standard in GPT-2+ and modern transformers. |
| **Weight tying** | LM head shares weights with token embedding. Reduces parameter count by ~4M and improves generalization. |
| **FlashAttention** | `F.scaled_dot_product_attention` auto-dispatches to FlashAttention on supported hardware (A100, H100, T4). Falls back to standard attention elsewhere. |
| **No bias** | All `nn.Linear` layers use `bias=False`. Modern practice from LLaMA/PaLM — reduces parameters and doesn't hurt quality. |
| **GELU activation** | Smoother than ReLU, standard in transformer FFNs since GPT-2. |
| **Combined QKV projection** | Single `nn.Linear(n_embd, 3 * n_embd)` then chunk, instead of three separate projections. More efficient, same result. |

### Training Choices

| Decision | Rationale |
|:---|:---|
| **Cosine schedule + warmup** | Prevents instability at start (warmup) and enables graceful convergence (cosine decay). Industry standard. |
| **Two parameter groups** | Weight decay on 2D+ parameters (weight matrices) only. Biases and LayerNorm parameters get zero weight decay. Prevents regularization interference with normalization. |
| **AdamW (β₁=0.9, β₂=0.95)** | β₂=0.95 instead of default 0.999 — better for transformers, reduces sensitivity to gradient spikes. |
| **Gradient accumulation** | Achieves effective batch size of 64 with only 32 samples per micro-step. Essential for Colab's limited VRAM. |
| **Two-phase training** | Phase 1 (pre-train) teaches general language modeling. Phase 2 (post-train) teaches conversational structure and personality. Separate LR schedules for each phase. |

### Data Pipeline Choices

| Decision | Rationale |
|:---|:---|
| **Byte-level BPE** | Handles all Unicode without unknown tokens. 8,192 vocab is compact enough for a 29.5M model. |
| **Chunked tokenization** | Processes 8–10 MB chunks to stay within Colab's 12GB RAM. Never loads full corpus. |
| **Memmap data** | Training data stored as memory-mapped `uint16` arrays. Zero-copy random access, no RAM overhead. |
| **Streaming parser** | `posttrain_data_prep.py` reads samples across chunk boundaries without holding the full file in memory. Handles 1GB+ datasets. |
| **Metadata cache** | Saves source file hash/size/mtime. Skips rebuild if nothing changed. |

### Inference Choices

| Decision | Rationale |
|:---|:---|
| **60 token max** | Keeps responses concise. RAVEN-1 is designed for short, punchy responses, not essays. |
| **Repetition penalty** | Penalizes tokens that already appeared in context. Prevents degenerate repetition loops. |
| **Top-K + Top-P** | Combined filtering: Top-K removes long tail, Top-P (nucleus) adapts to probability distribution shape. Together they produce diverse but coherent text. |
| **Funny fallbacks** | If the model generates empty output, a random witty fallback is returned instead of an error. Keeps the UX clean. |
| **3-turn history** | Conversation context is limited to the last 3 user/raven exchanges. Keeps the prompt within context window and focuses on recent conversation. |

---

## Google Colab Guide

RAVEN-1 is designed to train end-to-end on **Google Colab's free tier** (T4 GPU, 15GB VRAM, 12GB RAM). Two notebooks are provided:

### Pre-training (`notebooks/colab_train.ipynb`)

1. Mount Drive & check GPU → verify T4
2. Install dependencies → `tokenizers`, `numpy`
3. Create directories → upload project files
4. Train tokenizer → `python tokenizer_train.py`
5. Prepare data → `python data_prep.py` (chunked, RAM-safe)
6. Pre-train → `python train.py` (auto-resumes, syncs to Drive)
7. Post-train → `python posttrain.py`
8. Test → `python inference.py`
9. Download weights → `.pt` + tokenizer `.json`

### Post-training (`notebooks/colab_posttrain.ipynb`)

1. Mount Drive & check GPU
2. Install dependencies
3. Upload code files (6 files)
4. Copy `posttrain.txt` from Drive
5. Copy pre-trained weights from Drive
6. Pre-flight check (validates all files)
7. Tokenize data → `python posttrain_data_prep.py` (streaming, ~1 min for 1GB)
8. Post-train → `python posttrain.py --resume` (resume-safe)
9. Test the model
10. Download weights

> **💡 Tip:** If the Colab session disconnects, just re-run from the training cell — it auto-resumes from the latest checkpoint synced to Google Drive. Checkpoints include optimizer state, scaler state, step count, and best val loss for exact continuation.

### Colab Resource Usage

| Resource | Pre-training | Post-training |
|:---|:---|:---|
| GPU | T4 (15GB VRAM) | T4 (15GB VRAM) |
| VRAM used | ~3-4 GB | ~3-4 GB |
| RAM used | < 6 GB | < 6 GB |
| Disk | ~2 GB | ~2 GB |
| Time | ~2-3 hours | ~30-45 minutes |

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

### Compatibility

- **Python:** 3.10+
- **PyTorch:** 2.1+ (for `F.scaled_dot_product_attention` / FlashAttention)
- **CUDA:** Optional. Runs on CPU (slower) or any CUDA-capable GPU
- **OS:** Linux, macOS, Windows
- **Hardware:** Colab T4 (minimum), any modern GPU, or CPU

---

## File Reference

| File | Lines | Purpose |
|:---|:---|:---|
| `config.py` | 65 | Central configuration — every hyperparameter |
| `model.py` | 299 | Transformer: attention, FFN, blocks, generation |
| `tokenizer_train.py` | 87 | Train byte-level BPE from corpus |
| `data_prep.py` | 124 | Tokenize corpus → binary memmap (pre-train) |
| `posttrain_data_prep.py` | 447 | Streaming tokenizer → binary memmap (post-train) |
| `train.py` | 329 | Pre-training loop |
| `posttrain.py` | 424 | Post-training / fine-tuning loop |
| `inference.py` | 240 | Terminal chat with slash commands |
| `app.py` | 269 | Gradio web UI |
| `eval/run_300_test.py` | 175 | 300-prompt benchmark suite |
| `eval/stress_test.py` | 51 | Stress test with long inputs |

---

## Checkpoint Format

Each `.pt` checkpoint contains:

```python
{
    "step": int,                 # Training step number
    "model_state": OrderedDict,  # Model weights (69 tensors)
    "optimizer_state": dict,     # AdamW state for exact resume
    "scaler_state": dict,        # AMP GradScaler state
    "best_val_loss": float,      # Best validation loss seen
    "config": {                  # Architecture config snapshot
        "vocab_size": 8192,
        "block_size": 256,
        "n_embd": 512,
        "n_head": 8,
        "n_layer": 8,
        "dropout": 0.1,
    }
}
```

Checkpoint size: **~338 MB** (includes optimizer momentum buffers).

---

## License

MIT

---

