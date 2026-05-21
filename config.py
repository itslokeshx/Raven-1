# config.py
# RAVEN-1 — Central configuration.
# Every hyperparameter, path, and special token lives here.
# No other file should hardcode any of these values.

import os
import torch

# ── Model Architecture ────────────────────────────────────────────────
vocab_size = 8192                    # BPE vocabulary size
block_size = 256                     # Maximum context length (tokens)
n_embd = 512                         # Embedding dimension
n_head = 8                           # Number of attention heads
n_layer = 8                          # Number of transformer blocks
dropout = 0.1                        # Dropout rate during training

# Derived (not configurable — just documented):
# head_size  = n_embd // n_head = 64
# ffn_dim    = 4 * n_embd       = 2048
# Parameters ≈ 25M

# ── Training — Phase 1: Pre-training ──────────────────────────────────
batch_size = 32                      # Micro-batch size per step
gradient_accumulation_steps = 2      # Effective batch = 64
max_iters = 10000                    # Total pre-training steps
learning_rate = 3e-4                 # Peak learning rate
min_lr = 3e-5                        # Minimum learning rate (cosine floor)
warmup_iters = 300                   # Linear warmup steps
weight_decay = 0.1                   # AdamW weight decay
grad_clip = 1.0                      # Gradient norm clipping
eval_interval = 500                  # Evaluate every N steps
eval_iters = 100                     # Batches per evaluation
use_amp = True                       # Mixed precision training

# ── Training — Phase 2: Post-training ─────────────────────────────────
# Note: posttrain.py auto-tunes step count based on dataset size.
# Set posttrain_max_iters > 2000 to force an override.
posttrain_max_iters = 2000           # Base post-training steps (auto-tuned up)
posttrain_learning_rate = 5e-5       # Lower LR for fine-tuning
posttrain_min_lr = 5e-6              # Cosine floor for post-training
posttrain_warmup_iters = 100         # Warmup for post-training (auto-tuned)

# ── Special Tokens ────────────────────────────────────────────────────
# These MUST match across tokenizer_train, data_prep, inference, and app.
user_token = "<|user|>"
raven_token = "<|raven|>"
eos_token = "<|eos|>"
pad_token = "<|pad|>"
special_tokens = [pad_token, user_token, raven_token, eos_token]

# ── File Paths ────────────────────────────────────────────────────────
checkpoint_dir = "checkpoints"
tokenizer_path = os.path.join(checkpoint_dir, "raven1_tokenizer.json")
weights_path = os.path.join(checkpoint_dir, "raven1_best.pt")
posttrain_weights_path = os.path.join(checkpoint_dir, "raven1_posttrain_best.pt")
latest_path = os.path.join(checkpoint_dir, "raven1_latest.pt")
train_bin = os.path.join(checkpoint_dir, "train.bin")
val_bin = os.path.join(checkpoint_dir, "val.bin")
posttrain_bin = os.path.join(checkpoint_dir, "posttrain.bin")

# ── Device ────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
