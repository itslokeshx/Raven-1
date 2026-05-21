# train.py
# RAVEN-1 — Pre-training loop.
#
# Complete production-quality training loop with:
#   - Cosine LR schedule with linear warmup
#   - Mixed precision (AMP) training
#   - Gradient accumulation
#   - Gradient clipping
#   - Automatic checkpoint resume
#   - Best model tracking
#   - Google Drive sync for Colab
#   - Clean KeyboardInterrupt handling

import os
import math
import time
import shutil
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from model import Raven1
import config


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_data(bin_path):
    """Load a binary token file as a numpy memmap (read-only)."""
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Data file not found: {bin_path}")
    return np.memmap(bin_path, dtype=np.uint16, mode='r')


def get_batch(data, batch_size=None, block_size=None):
    """Sample a random batch of (x, y) pairs from data.

    Args:
        data: numpy memmap of token IDs
        batch_size: number of sequences per batch
        block_size: length of each sequence

    Returns:
        x: (batch_size, block_size) input tokens
        y: (batch_size, block_size) target tokens (shifted by 1)
    """
    bs = batch_size or config.batch_size
    bl = block_size or config.block_size

    # Random starting indices
    ix = np.random.randint(0, len(data) - bl - 1, size=(bs,))

    x = torch.stack([torch.from_numpy(data[i:i + bl].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + bl + 1].astype(np.int64)) for i in ix])

    return x.to(config.device), y.to(config.device)


# ═══════════════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════════

def get_lr(step, max_iters=None, lr=None, min_lr_val=None, warmup=None):
    """Cosine decay with linear warmup.

    Args:
        step: current training step
        max_iters: total training steps
        lr: peak learning rate
        min_lr_val: minimum learning rate (cosine floor)
        warmup: number of warmup steps
    """
    max_it = max_iters or config.max_iters
    peak_lr = lr or config.learning_rate
    floor_lr = min_lr_val or config.min_lr
    warmup_steps = warmup or config.warmup_iters

    # Linear warmup
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # After training — clamp to min
    if step >= max_it:
        return floor_lr

    # Cosine decay
    decay = (step - warmup_steps) / (max_it - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return floor_lr + coeff * (peak_lr - floor_lr)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, data):
    """Estimate loss over eval_iters batches."""
    model.eval()
    losses = []
    for _ in range(config.eval_iters):
        x, y = get_batch(data)
        with autocast(device_type=config.device, enabled=config.use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ═══════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scaler, step, best_val_loss):
    """Save a training checkpoint."""
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout,
        }
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Load a training checkpoint. Returns (step, best_val_loss)."""
    ckpt = torch.load(path, map_location=config.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf"))


# ═══════════════════════════════════════════════════════════════════════
# Google Drive Sync
# ═══════════════════════════════════════════════════════════════════════

DRIVE_DIR = "/content/drive/MyDrive/raven1_checkpoints"


def sync_to_drive(path):
    """Copy a file to Google Drive if available. Silent on failure."""
    try:
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(DRIVE_DIR, exist_ok=True)
            dst = os.path.join(DRIVE_DIR, os.path.basename(path))
            shutil.copy2(path, dst)
            print(f"  → Drive synced: {os.path.basename(path)}")
    except Exception:
        pass  # Silent fail — Drive not available or read-only


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("RAVEN-1 — PRE-TRAINING")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading data...")
    train_data = load_data(config.train_bin)
    val_data = load_data(config.val_bin)
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens  : {len(val_data):,}")

    # ── Build model ───────────────────────────────────────────────────
    print("\nBuilding model...")
    model = Raven1().to(config.device)

    # ── Optimizer ─────────────────────────────────────────────────────
    # Separate weight-decay and no-decay parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95))

    scaler = GradScaler(enabled=config.use_amp)

    # ── Resume from checkpoint ────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")

    if os.path.exists(config.latest_path):
        print(f"\nResuming from {config.latest_path}...")
        start_step, best_val_loss = load_checkpoint(
            config.latest_path, model, optimizer, scaler
        )
        print(f"  Resumed at step {start_step} | best val_loss {best_val_loss:.4f}")
    else:
        print("\nStarting fresh training.")

    # ── Training ──────────────────────────────────────────────────────
    print(f"\nTraining for {config.max_iters} steps...")
    print(f"  Effective batch : {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  LR max → min    : {config.learning_rate} → {config.min_lr}")
    print(f"  AMP             : {config.use_amp}")
    print("-" * 60)

    model.train()
    t0 = time.time()

    try:
        for step in range(start_step, config.max_iters):
            # ── Update learning rate ──────────────────────────────
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Gradient accumulation ─────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(config.gradient_accumulation_steps):
                x, y = get_batch(train_data)

                with autocast(device_type=config.device, enabled=config.use_amp):
                    _, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            # ── Gradient clipping ─────────────────────────────────
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # ── Optimizer step ────────────────────────────────────
            scaler.step(optimizer)
            scaler.update()

            # ── Logging (every 50 steps) ──────────────────────────
            if step % 50 == 0 and step > 0:
                elapsed = time.time() - t0
                print(f"  step {step:>6d} | loss {accum_loss:.4f} | "
                      f"lr {lr:.2e} | {elapsed:.1f}s")
                t0 = time.time()

            # ── Evaluation & Checkpointing ────────────────────────
            if step > 0 and step % config.eval_interval == 0:
                val_loss = evaluate(model, val_data)
                train_loss_est = evaluate(model, train_data)

                print(f"\n{'─' * 60}")
                print(f"  Step {step:>6d}")
                print(f"  Train loss : {train_loss_est:.4f}")
                print(f"  Val loss   : {val_loss:.4f}")
                print(f"  LR         : {lr:.2e}")

                # Save latest
                save_checkpoint(config.latest_path, model, optimizer,
                                scaler, step, best_val_loss)
                sync_to_drive(config.latest_path)

                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(config.weights_path, model, optimizer,
                                    scaler, step, best_val_loss)
                    sync_to_drive(config.weights_path)
                    print(f"  ★ New best model saved! (val_loss={val_loss:.4f})")
                else:
                    print(f"  Best val_loss remains {best_val_loss:.4f}")

                print(f"{'─' * 60}\n")
                model.train()
                t0 = time.time()

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Saving checkpoint before exit...")
        current_step = step if 'step' in dir() else start_step
        save_checkpoint(config.latest_path, model, optimizer,
                        scaler, current_step, best_val_loss)
        sync_to_drive(config.latest_path)
        print("Checkpoint saved. You can resume training later.")
        return

    # ── Final evaluation ──────────────────────────────────────────────
    val_loss = evaluate(model, val_data)
    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Final val_loss : {val_loss:.4f}")
    print(f"  Best val_loss  : {best_val_loss:.4f}")
    print(f"{'=' * 60}")

    # Final save
    save_checkpoint(config.latest_path, model, optimizer,
                    scaler, config.max_iters, best_val_loss)
    sync_to_drive(config.latest_path)

    # Sync tokenizer to Drive too
    if os.path.exists(config.tokenizer_path):
        sync_to_drive(config.tokenizer_path)


if __name__ == "__main__":
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train()
