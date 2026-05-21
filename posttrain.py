# posttrain.py
# RAVEN-1 — Post-training (fine-tuning) loop.
#
# Loads the best pre-trained model and fine-tunes it on prepared
# post-training binary data with a lower learning rate.
#
# Mirrors train.py exactly in structure. No tokenization logic here.
# Run posttrain_data_prep.py FIRST.
#
# KEY COLAB-SAFE FEATURES:
#   1. Correct AMP: device_type is always "cuda" or "cpu", never "cuda:0"
#   2. Resume-safe: latest checkpoint saves optimizer + scaler + step
#   3. Drive sync: auto-syncs checkpoints to Drive if mounted
#   4. KeyboardInterrupt: saves checkpoint before exit
#   5. Before/after sample generation for visual comparison
#
# Usage:
#   python posttrain.py              # full post-training
#   python posttrain.py --resume     # resume from latest checkpoint
#   python posttrain.py --dry-run    # test only 20 steps then exit

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tokenizers import Tokenizer
from model import Raven1
from train import (load_data, get_batch, get_lr, evaluate,
                   save_checkpoint, load_checkpoint, sync_to_drive)
import config


# ═══════════════════════════════════════════════════════════════════════
# Safe Device Type for AMP
# ═══════════════════════════════════════════════════════════════════════
# torch.amp.autocast requires device_type to be "cuda" or "cpu",
# NOT "cuda:0". config.device may be "cuda" or "cpu", but we
# extract just the base type to be safe.

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════
# Post-Training Checkpoint Paths
# ═══════════════════════════════════════════════════════════════════════

PT_LATEST = os.path.join(config.checkpoint_dir, "raven1_posttrain_latest.pt")
PT_BEST = config.posttrain_weights_path


# ═══════════════════════════════════════════════════════════════════════
# Test Prompts for Before/After Comparison
# ═══════════════════════════════════════════════════════════════════════

TEST_PROMPTS = [
    "hi",
    "what is gravity",
    "I failed my exam.",
    "tell me a joke",
    "asdfghjkl",
    "I'll start tomorrow.",
    "what is the meaning of life",
]


def generate_samples(model, tokenizer, label=""):
    """Generate responses for test prompts. Used for before/after comparison."""
    model.eval()
    eos_id = tokenizer.token_to_id(config.eos_token)

    print(f"\n{'─' * 60}")
    print(f"  Sample Generations — {label}")
    print(f"{'─' * 60}")

    for prompt_text in TEST_PROMPTS:
        full_prompt = f"{config.user_token} {prompt_text}\n{config.raven_token}"

        encoded = tokenizer.encode(full_prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long,
                                 device=config.device)

        # Crop to block_size
        if input_ids.size(1) > config.block_size:
            input_ids = input_ids[:, -config.block_size:]

        # Generate
        try:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=60,
                temperature=0.65,
                top_k=25,
                top_p=0.85,
                repetition_penalty=1.1,
                stop_token_id=eos_id,
            )

            # Decode only new tokens
            new_tokens = output_ids[0, input_ids.size(1):].tolist()
            response = tokenizer.decode(new_tokens).strip()

            # Clean leaked special tokens
            for tok in config.special_tokens:
                response = response.replace(tok, "")
            response = response.strip()
        except Exception as e:
            response = f"(error: {e})"

        print(f"  User:  {prompt_text}")
        print(f"  Raven: {response if response else '(empty)'}")
        print()

    model.train()


# ═══════════════════════════════════════════════════════════════════════
# GPU Stats (safe for Colab)
# ═══════════════════════════════════════════════════════════════════════

def print_gpu_stats(label=""):
    """Print GPU memory usage. Silent on CPU."""
    if not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU [{label}]: {alloc:.1f}GB alloc / "
              f"{reserved:.1f}GB reserved / {total:.1f}GB total")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
# Main Post-Training Loop
# ═══════════════════════════════════════════════════════════════════════

def posttrain(resume=False, dry_run=False):
    print("=" * 60)
    print("RAVEN-1 — POST-TRAINING")
    print("=" * 60)

    if dry_run:
        print("  [DRY RUN] Will train for only 20 steps then exit.")

    # ── Validate prerequisites ────────────────────────────────────────
    if not os.path.exists(config.tokenizer_path):
        print(f"\n[ERROR] Tokenizer not found: {config.tokenizer_path}")
        sys.exit(1)

    if not os.path.exists(config.posttrain_train_bin):
        print(f"\n[ERROR] Training data not found: {config.posttrain_train_bin}")
        print("  Run posttrain_data_prep.py first.")
        sys.exit(1)

    if not os.path.exists(config.posttrain_val_bin):
        print(f"\n[ERROR] Validation data not found: {config.posttrain_val_bin}")
        print("  Run posttrain_data_prep.py first.")
        sys.exit(1)

    # ── Load tokenizer ────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    print(f"\nTokenizer loaded: {tokenizer.get_vocab_size()} tokens")

    # ── Load data (memory-mapped — no RAM cost) ───────────────────────
    print("\nLoading data (memory-mapped)...")
    train_data = load_data(config.posttrain_train_bin)
    val_data = load_data(config.posttrain_val_bin)
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens  : {len(val_data):,}")

    if len(train_data) < config.block_size + 1:
        print(f"\n[ERROR] Training data too small ({len(train_data)} tokens).")
        print(f"  Need at least {config.block_size + 1} tokens.")
        sys.exit(1)

    # ── Build model ───────────────────────────────────────────────────
    print("\nBuilding model...")
    model = Raven1().to(config.device)
    print_gpu_stats("after model load")

    # ── Load pre-trained weights ──────────────────────────────────────
    pretrain_loaded = False
    if not resume:
        if os.path.exists(config.weights_path):
            print(f"\nLoading pre-trained weights: {config.weights_path}")
            try:
                ckpt = torch.load(config.weights_path,
                                  map_location=config.device,
                                  weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                pretrain_step = ckpt.get("step", "?")
                pretrain_loss = ckpt.get("best_val_loss", "?")
                print(f"  Loaded from step {pretrain_step} | "
                      f"val_loss {pretrain_loss}")
                pretrain_loaded = True
            except Exception as e:
                print(f"  [WARN] Failed to load weights: {e}")
                print("  Proceeding with random initialization.")
        else:
            print(f"\n[WARN] No pre-trained weights at {config.weights_path}")
            print("  Post-training from random initialization.")

    # ── Before samples ────────────────────────────────────────────────
    if pretrain_loaded:
        generate_samples(model, tokenizer, label="BEFORE Post-Training")

    # ── Optimizer ─────────────────────────────────────────────────────
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
    ], lr=config.posttrain_learning_rate, betas=(0.9, 0.95))

    scaler = GradScaler(enabled=config.use_amp)

    # ── Training parameters ───────────────────────────────────────────
    max_iters = 20 if dry_run else config.posttrain_max_iters
    warmup_iters = min(config.posttrain_warmup_iters, max_iters // 5)
    eval_interval = min(config.eval_interval, 250)
    if dry_run:
        eval_interval = 10

    # ── Resume from checkpoint ────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")

    if resume and os.path.exists(PT_LATEST):
        print(f"\nResuming from {PT_LATEST}...")
        try:
            start_step, best_val_loss = load_checkpoint(
                PT_LATEST, model, optimizer, scaler)
            print(f"  Resumed at step {start_step} | "
                  f"best val_loss {best_val_loss:.4f}")
        except Exception as e:
            print(f"  [WARN] Failed to resume: {e}")
            print("  Starting fresh.")
            start_step = 0
            best_val_loss = float("inf")
    elif resume:
        print("\n  No checkpoint to resume from. Starting fresh.")

    # ── Print training config ─────────────────────────────────────────
    eff_batch = config.batch_size * config.gradient_accumulation_steps
    tokens_per_step = eff_batch * config.block_size

    print(f"\n{'─' * 60}")
    print(f"  Post-training for {max_iters} steps "
          f"(from step {start_step})")
    print(f"  Effective batch : {eff_batch}")
    print(f"  Tokens/step     : {tokens_per_step:,}")
    print(f"  LR max → min    : {config.posttrain_learning_rate} → "
          f"{config.posttrain_min_lr}")
    print(f"  Warmup steps    : {warmup_iters}")
    print(f"  Eval interval   : {eval_interval}")
    print(f"  AMP             : {config.use_amp}")
    print(f"  Device type     : {DEVICE_TYPE}")
    print(f"{'─' * 60}\n")

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    t_start = time.time()
    log_interval = 25  # Print every 25 steps for compact logs

    try:
        for step in range(start_step, max_iters):
            # ── Learning rate schedule ────────────────────────────
            lr = get_lr(
                step,
                max_iters=max_iters,
                lr=config.posttrain_learning_rate,
                min_lr_val=config.posttrain_min_lr,
                warmup=warmup_iters,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Gradient accumulation ─────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(config.gradient_accumulation_steps):
                x, y = get_batch(train_data)

                # IMPORTANT: use DEVICE_TYPE not config.device for autocast
                # config.device could be "cuda:0" which breaks autocast
                with autocast(device_type=DEVICE_TYPE,
                              enabled=config.use_amp):
                    _, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            # ── Gradient clipping ─────────────────────────────────
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.grad_clip)

            # ── Optimizer step ────────────────────────────────────
            scaler.step(optimizer)
            scaler.update()

            # ── Compact logging ───────────────────────────────────
            if step > 0 and step % log_interval == 0:
                elapsed = time.time() - t0
                total_elapsed = time.time() - t_start
                steps_done = step - start_step
                steps_remaining = max_iters - step
                if steps_done > 0:
                    eta = (total_elapsed / steps_done) * steps_remaining
                    eta_str = f"{eta / 60:.1f}min" if eta > 60 else f"{eta:.0f}s"
                else:
                    eta_str = "?"

                print(f"  step {step:>6d}/{max_iters} | "
                      f"loss {accum_loss:.4f} | "
                      f"lr {lr:.2e} | "
                      f"{elapsed:.1f}s | "
                      f"ETA {eta_str}")
                t0 = time.time()

            # ── Evaluation & checkpointing ────────────────────────
            if step > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, val_data)
                train_loss = evaluate(model, train_data)

                print(f"\n{'─' * 60}")
                print(f"  Step {step:>6d}/{max_iters}")
                print(f"  Train loss    : {train_loss:.4f}")
                print(f"  Val loss      : {val_loss:.4f}")
                print(f"  Learning rate : {lr:.2e}")
                print(f"  Best val_loss : {best_val_loss:.4f}")
                print_gpu_stats("eval")

                # Save latest checkpoint
                save_checkpoint(PT_LATEST, model, optimizer,
                                scaler, step, best_val_loss)
                sync_to_drive(PT_LATEST)

                # Save best if improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(PT_BEST, model, optimizer,
                                    scaler, step, best_val_loss)
                    sync_to_drive(PT_BEST)
                    print(f"  ★ New best model! (val_loss={val_loss:.4f})")
                else:
                    print(f"  Best val_loss remains {best_val_loss:.4f}")

                print(f"{'─' * 60}\n")
                model.train()
                t0 = time.time()

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Saving checkpoint before exit...")
        current_step = step if 'step' in dir() else start_step
        save_checkpoint(PT_LATEST, model, optimizer,
                        scaler, current_step, best_val_loss)
        sync_to_drive(PT_LATEST)
        print(f"  Saved at step {current_step}. Resume with --resume.")
        return

    # ── Final evaluation ──────────────────────────────────────────────
    total_time = time.time() - t_start

    val_loss = evaluate(model, val_data)

    print(f"\n{'=' * 60}")
    print(f"  Post-training complete!")
    print(f"  Final val_loss : {val_loss:.4f}")
    print(f"  Best val_loss  : {best_val_loss:.4f}")
    print(f"  Total time     : {total_time / 60:.1f} minutes")
    print(f"  Steps          : {max_iters - start_step}")
    print(f"{'=' * 60}")

    # Final best-model save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(PT_BEST, model, optimizer,
                        scaler, max_iters, best_val_loss)
        sync_to_drive(PT_BEST)
        print(f"  ★ Final model is new best! (val_loss={val_loss:.4f})")

    save_checkpoint(PT_LATEST, model, optimizer,
                    scaler, max_iters, best_val_loss)
    sync_to_drive(PT_LATEST)

    # ── After samples ─────────────────────────────────────────────────
    generate_samples(model, tokenizer, label="AFTER Post-Training")

    if dry_run:
        print("\n  [DRY RUN] Complete. Remove --dry-run for full training.")


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAVEN-1 Post-Training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest posttrain checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test: train only 20 steps")
    args = parser.parse_args()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    posttrain(resume=args.resume, dry_run=args.dry_run)
