# posttrain.py
# RAVEN-1 — Post-training (fine-tuning) loop.
#
# Loads the best pre-trained model and fine-tunes it on a curated
# post-training dataset with lower learning rate. Prints before/after
# comparison of sample generations.
#
# Optimized for Google Colab:
#   - Chunked tokenization (handles 1GB+ files without OOM)
#   - Memory-mapped data splits (no RAM duplication)
#   - Periodic VRAM monitoring
#   - Auto-tuned step count based on dataset size
#   - Mid-training sample generation
#   - Robust checkpoint resume

import os
import gc
import math
import time
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tokenizers import Tokenizer
from model import Raven1
from train import (get_batch, get_lr, evaluate, save_checkpoint,
                   load_checkpoint, sync_to_drive)
import config


# ═══════════════════════════════════════════════════════════════════════
# GPU Monitoring (Colab-friendly)
# ═══════════════════════════════════════════════════════════════════════

def print_gpu_stats(label=""):
    """Print current GPU memory usage. Silent on CPU."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"  GPU [{label}]: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.1f}GB total")


def flush_memory():
    """Force garbage collection and CUDA cache cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# Post-training Data Preparation (chunked, memory-safe)
# ═══════════════════════════════════════════════════════════════════════

def prepare_posttrain_data():
    """Tokenize posttrain.txt in chunks and save as binary.

    Uses 10MB chunked reading to handle 1GB+ files without OOM.
    Returns the total number of tokens, or 0 on failure.
    """
    corpus_path = "data/posttrain.txt"
    bin_path = config.posttrain_bin

    # If already prepared, just return the token count
    if os.path.exists(bin_path):
        existing = np.memmap(bin_path, dtype=np.uint16, mode='r')
        count = len(existing)
        del existing
        print(f"  Post-train data already prepared: {bin_path} ({count:,} tokens)")
        return count

    if not os.path.exists(corpus_path):
        print(f"  [ERROR] {corpus_path} not found.")
        return 0

    file_size = os.path.getsize(corpus_path)
    if file_size == 0:
        print(f"  [ERROR] {corpus_path} is empty.")
        return 0

    print(f"  Tokenizing {corpus_path} ({file_size / 1e6:.1f} MB)...")
    print(f"  Using chunked tokenization (10MB chunks) to stay memory-safe.")

    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    # ── Phase 1: Chunked tokenization ─────────────────────────────────
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per chunk
    all_ids = []
    total_chars = 0
    t0 = time.time()

    with open(corpus_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            ids = tokenizer.encode(chunk).ids
            all_ids.extend(ids)
            total_chars += len(chunk)

            elapsed = time.time() - t0
            pct = total_chars / file_size * 100
            speed = total_chars / 1e6 / max(elapsed, 0.01)
            eta = (file_size - total_chars) / 1e6 / max(speed, 0.01)

            print(f"\r  {pct:5.1f}% | {total_chars/1e6:.1f}/{file_size/1e6:.1f} MB | "
                  f"{len(all_ids):,} tokens | {speed:.1f} MB/s | ETA {eta:.0f}s", end="", flush=True)

    print(f"\n  Total: {len(all_ids):,} tokens in {time.time() - t0:.1f}s")

    # ── Phase 2: Save to memmap ───────────────────────────────────────
    token_ids = np.array(all_ids, dtype=np.uint16)
    del all_ids  # free RAM immediately
    gc.collect()

    mm = np.memmap(bin_path, dtype=np.uint16, mode='w+', shape=token_ids.shape)
    mm[:] = token_ids
    mm.flush()
    count = len(token_ids)
    del mm, token_ids
    gc.collect()

    print(f"  Saved: {bin_path} ({count:,} tokens)")
    return count


def create_posttrain_splits(total_tokens):
    """Create 90/10 train/val splits using memory-mapped files.

    Uses sequential reading to avoid loading both copies into RAM.
    Returns (train_data, val_data) as read-only memmaps.
    """
    pt_train_path = os.path.join(config.checkpoint_dir, "posttrain_train.bin")
    pt_val_path = os.path.join(config.checkpoint_dir, "posttrain_val.bin")

    split_idx = int(total_tokens * 0.9)
    val_count = total_tokens - split_idx

    # Only recreate if needed
    if os.path.exists(pt_train_path) and os.path.exists(pt_val_path):
        train_data = np.memmap(pt_train_path, dtype=np.uint16, mode='r')
        val_data = np.memmap(pt_val_path, dtype=np.uint16, mode='r')
        if len(train_data) == split_idx and len(val_data) == val_count:
            print(f"  Train/val splits already exist. Reusing.")
            return train_data, val_data
        # Size mismatch — recreate
        del train_data, val_data

    print(f"  Creating 90/10 train/val splits...")
    all_data = np.memmap(config.posttrain_bin, dtype=np.uint16, mode='r')

    # Write train split
    train_mm = np.memmap(pt_train_path, dtype=np.uint16, mode='w+',
                         shape=(split_idx,))
    # Copy in 50MB chunks to avoid RAM spike
    COPY_CHUNK = 25_000_000  # ~50MB in uint16
    for start in range(0, split_idx, COPY_CHUNK):
        end = min(start + COPY_CHUNK, split_idx)
        train_mm[start:end] = all_data[start:end]
    train_mm.flush()
    del train_mm

    # Write val split
    val_mm = np.memmap(pt_val_path, dtype=np.uint16, mode='w+',
                       shape=(val_count,))
    for start in range(0, val_count, COPY_CHUNK):
        end = min(start + COPY_CHUNK, val_count)
        val_mm[start:end] = all_data[split_idx + start:split_idx + end]
    val_mm.flush()
    del val_mm, all_data
    gc.collect()

    # Reopen as read-only
    train_data = np.memmap(pt_train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(pt_val_path, dtype=np.uint16, mode='r')

    return train_data, val_data


# ═══════════════════════════════════════════════════════════════════════
# Auto-tune Training Steps Based on Dataset Size
# ═══════════════════════════════════════════════════════════════════════

def compute_posttrain_steps(total_tokens):
    """Automatically compute appropriate number of posttrain steps.

    Aims for ~2 epochs over the dataset by default, with a floor of 2000
    steps and a ceiling of 20000 steps.

    Returns (max_iters, eval_interval, warmup_iters).
    """
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    tokens_per_step = effective_batch * config.block_size
    train_tokens = int(total_tokens * 0.9)

    steps_per_epoch = max(1, train_tokens // tokens_per_step)
    target_epochs = 2.0

    # Compute ideal step count
    ideal_steps = int(steps_per_epoch * target_epochs)

    # Clamp between floor and ceiling
    max_iters = max(2000, min(ideal_steps, 20000))

    # Eval interval: ~10 evals per run, min 100, max 1000
    eval_interval = max(100, min(max_iters // 10, 1000))

    # Warmup: 5% of total, min 50, max 500
    warmup = max(50, min(int(max_iters * 0.05), 500))

    print(f"\n  Auto-tuned training schedule:")
    print(f"    Dataset      : {total_tokens:,} total tokens ({train_tokens:,} train)")
    print(f"    Tokens/step  : {tokens_per_step:,}")
    print(f"    Steps/epoch  : {steps_per_epoch:,}")
    print(f"    Target epochs: {target_epochs:.1f}")
    print(f"    Max iters    : {max_iters:,}")
    print(f"    Eval interval: {eval_interval}")
    print(f"    Warmup iters : {warmup}")

    return max_iters, eval_interval, warmup


# ═══════════════════════════════════════════════════════════════════════
# Sample Generation for Comparison
# ═══════════════════════════════════════════════════════════════════════

TEST_PROMPTS = [
    f"{config.user_token} hi\n{config.raven_token}",
    f"{config.user_token} I'll start tomorrow.\n{config.raven_token}",
    f"{config.user_token} I failed my exam.\n{config.raven_token}",
    f"{config.user_token} what is gravity\n{config.raven_token}",
    f"{config.user_token} asdfghjkl\n{config.raven_token}",
]


def generate_samples(model, tokenizer, label=""):
    """Generate responses for all test prompts and print them."""
    model.eval()
    eos_id = tokenizer.token_to_id(config.eos_token)

    print(f"\n{'─' * 60}")
    print(f"  Sample Generations — {label}")
    print(f"{'─' * 60}")

    for prompt_text in TEST_PROMPTS:
        # Encode prompt
        encoded = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long,
                                 device=config.device)

        # Crop to block_size
        if input_ids.size(1) > config.block_size:
            input_ids = input_ids[:, -config.block_size:]

        # Generate
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

        # Extract user message for display
        user_msg = prompt_text.split(config.user_token)[-1]
        user_msg = user_msg.split(config.raven_token)[0].strip()

        print(f"  User:  {user_msg}")
        print(f"  Raven: {response if response else '(empty)'}")
        print()

    model.train()


# ═══════════════════════════════════════════════════════════════════════
# Main Post-Training Loop
# ═══════════════════════════════════════════════════════════════════════

def posttrain():
    print("=" * 60)
    print("RAVEN-1 — POST-TRAINING")
    print("=" * 60)

    print_gpu_stats("startup")

    # ── Load tokenizer ────────────────────────────────────────────────
    if not os.path.exists(config.tokenizer_path):
        print(f"[ERROR] Tokenizer not found: {config.tokenizer_path}")
        return

    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    print(f"Tokenizer loaded: {tokenizer.get_vocab_size()} tokens")

    # ── Prepare post-training data ────────────────────────────────────
    print("\nPreparing post-training data...")
    total_tokens = prepare_posttrain_data()
    if total_tokens == 0:
        return

    # ── Create train/val splits ───────────────────────────────────────
    train_data, val_data = create_posttrain_splits(total_tokens)
    print(f"\n  Train tokens : {len(train_data):,}")
    print(f"  Val tokens   : {len(val_data):,}")

    flush_memory()

    # ── Auto-tune step count ──────────────────────────────────────────
    max_iters, eval_interval, warmup_iters = compute_posttrain_steps(total_tokens)

    # Allow config override if user explicitly set a low value
    if config.posttrain_max_iters > 2000:
        max_iters = config.posttrain_max_iters
        print(f"  Using config override: {max_iters} steps")

    # ── Build model ───────────────────────────────────────────────────
    print("\nBuilding model...")
    model = Raven1().to(config.device)
    print_gpu_stats("model loaded")

    # ── Load pre-trained weights ──────────────────────────────────────
    if os.path.exists(config.weights_path):
        print(f"\nLoading pre-trained weights: {config.weights_path}")
        ckpt = torch.load(config.weights_path, map_location=config.device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        pretrain_step = ckpt.get("step", "?")
        pretrain_loss = ckpt.get("best_val_loss", "?")
        print(f"  Loaded from step {pretrain_step} | val_loss {pretrain_loss}")
        del ckpt
        flush_memory()
    else:
        print(f"\n[WARN] No pre-trained weights at {config.weights_path}")
        print("  Post-training from random initialization.")

    # ── Before comparison ─────────────────────────────────────────────
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

    # ── Check for post-train resume ───────────────────────────────────
    posttrain_latest = os.path.join(config.checkpoint_dir,
                                    "raven1_posttrain_latest.pt")
    start_step = 0
    best_val_loss = float("inf")

    if os.path.exists(posttrain_latest):
        print(f"\nResuming post-training from {posttrain_latest}...")
        start_step, best_val_loss = load_checkpoint(
            posttrain_latest, model, optimizer, scaler
        )
        print(f"  Resumed at step {start_step} | best val_loss {best_val_loss:.4f}")

    print_gpu_stats("pre-training-loop")

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nPost-training for {max_iters:,} steps...")
    print(f"  Effective batch : {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  LR max → min    : {config.posttrain_learning_rate} → {config.posttrain_min_lr}")
    print(f"  Eval interval   : {eval_interval}")
    print(f"  Warmup iters    : {warmup_iters}")
    print("-" * 60)

    model.train()
    t0 = time.time()
    run_start = time.time()

    # Mid-training sample generation interval
    sample_interval = max(eval_interval * 3, 1000)

    try:
        for step in range(start_step, max_iters):
            # Learning rate schedule (using posttrain params)
            lr = get_lr(
                step,
                max_iters=max_iters,
                lr=config.posttrain_learning_rate,
                min_lr_val=config.posttrain_min_lr,
                warmup=warmup_iters,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(config.gradient_accumulation_steps):
                x, y = get_batch(train_data)

                with autocast(device_type=config.device, enabled=config.use_amp):
                    _, loss = model(x, y)
                    loss = loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # Logging
            if step % 50 == 0 and step > 0:
                elapsed = time.time() - t0
                total_elapsed = time.time() - run_start
                steps_done = step - start_step
                steps_remaining = max_iters - step
                if steps_done > 0:
                    eta = steps_remaining * (total_elapsed / steps_done)
                    eta_str = f"{eta / 60:.1f}min" if eta > 60 else f"{eta:.0f}s"
                else:
                    eta_str = "?"

                print(f"  step {step:>5d}/{max_iters} | "
                      f"loss {accum_loss:.4f} | lr {lr:.2e} | "
                      f"{elapsed:.1f}s | ETA {eta_str}")
                t0 = time.time()

            # GPU stats every 500 steps
            if step % 500 == 0 and step > 0:
                print_gpu_stats(f"step {step}")

            # Evaluation & checkpointing
            if step > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, val_data)

                print(f"\n  Step {step:>5d} | val_loss {val_loss:.4f}")

                # Save latest
                save_checkpoint(posttrain_latest, model, optimizer,
                                scaler, step, best_val_loss)
                sync_to_drive(posttrain_latest)

                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(config.posttrain_weights_path, model,
                                    optimizer, scaler, step, best_val_loss)
                    sync_to_drive(config.posttrain_weights_path)
                    print(f"  ★ New best! val_loss={val_loss:.4f}")
                else:
                    print(f"  Best remains {best_val_loss:.4f}")

                print()
                model.train()
                t0 = time.time()

            # Mid-training sample generation
            if step > 0 and step % sample_interval == 0:
                generate_samples(model, tokenizer,
                                 label=f"Mid-Training (step {step})")
                model.train()
                t0 = time.time()

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Saving checkpoint before exit...")
        save_checkpoint(posttrain_latest, model, optimizer,
                        scaler, step, best_val_loss)
        sync_to_drive(posttrain_latest)
        print("Checkpoint saved. You can resume post-training later.")
        return

    # ── Final save ────────────────────────────────────────────────────
    val_loss = evaluate(model, val_data)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(config.posttrain_weights_path, model, optimizer,
                        scaler, max_iters, best_val_loss)
        sync_to_drive(config.posttrain_weights_path)

    total_time = time.time() - run_start
    print(f"\n{'=' * 60}")
    print(f"  Post-training complete!")
    print(f"  Final val_loss : {val_loss:.4f}")
    print(f"  Best val_loss  : {best_val_loss:.4f}")
    print(f"  Total time     : {total_time / 60:.1f} minutes")
    print(f"{'=' * 60}")

    # ── After comparison ──────────────────────────────────────────────
    generate_samples(model, tokenizer, label="AFTER Post-Training")

    # Cleanup temp splits
    pt_train_path = os.path.join(config.checkpoint_dir, "posttrain_train.bin")
    pt_val_path = os.path.join(config.checkpoint_dir, "posttrain_val.bin")
    for p in [pt_train_path, pt_val_path]:
        try:
            os.remove(p)
        except OSError:
            pass

    print_gpu_stats("final")


if __name__ == "__main__":
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    posttrain()
