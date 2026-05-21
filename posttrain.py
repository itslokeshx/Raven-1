# posttrain.py
# RAVEN-1 — Post-training (fine-tuning) loop.
#
# Loads the best pre-trained model and fine-tunes it on a curated
# post-training dataset with lower learning rate. Prints before/after
# comparison of sample generations.

import os
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
# Post-training Data Preparation
# ═══════════════════════════════════════════════════════════════════════

def prepare_posttrain_data():
    """Tokenize posttrain.txt and save as binary if not already done."""
    corpus_path = "data/posttrain.txt"
    bin_path = config.posttrain_bin

    if os.path.exists(bin_path):
        print(f"  Post-train data already prepared: {bin_path}")
        return True

    if not os.path.exists(corpus_path):
        print(f"  [ERROR] {corpus_path} not found.")
        return False

    if os.path.getsize(corpus_path) == 0:
        print(f"  [ERROR] {corpus_path} is empty.")
        return False

    print(f"  Tokenizing {corpus_path}...")
    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    encoded = tokenizer.encode(text)
    token_ids = np.array(encoded.ids, dtype=np.uint16)

    mm = np.memmap(bin_path, dtype=np.uint16, mode='w+', shape=token_ids.shape)
    mm[:] = token_ids
    mm.flush()

    print(f"  Tokenized: {len(token_ids):,} tokens → {bin_path}")
    return True


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

    # ── Load tokenizer ────────────────────────────────────────────────
    if not os.path.exists(config.tokenizer_path):
        print(f"[ERROR] Tokenizer not found: {config.tokenizer_path}")
        return

    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    print(f"Tokenizer loaded: {tokenizer.get_vocab_size()} tokens")

    # ── Prepare post-training data ────────────────────────────────────
    print("\nPreparing post-training data...")
    if not prepare_posttrain_data():
        return

    # Load full post-train data, then split 90/10
    all_data = np.memmap(config.posttrain_bin, dtype=np.uint16, mode='r')
    total = len(all_data)
    split_idx = int(total * 0.9)

    # Create temp train/val splits for post-training
    pt_train_path = os.path.join(config.checkpoint_dir, "posttrain_train.bin")
    pt_val_path = os.path.join(config.checkpoint_dir, "posttrain_val.bin")

    train_mm = np.memmap(pt_train_path, dtype=np.uint16, mode='w+',
                         shape=(split_idx,))
    train_mm[:] = all_data[:split_idx]
    train_mm.flush()

    val_mm = np.memmap(pt_val_path, dtype=np.uint16, mode='w+',
                       shape=(total - split_idx,))
    val_mm[:] = all_data[split_idx:]
    val_mm.flush()

    train_data = np.memmap(pt_train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(pt_val_path, dtype=np.uint16, mode='r')

    print(f"\n  Train tokens : {len(train_data):,}")
    print(f"  Val tokens   : {len(val_data):,}")

    # ── Build model ───────────────────────────────────────────────────
    print("\nBuilding model...")
    model = Raven1().to(config.device)

    # ── Load pre-trained weights ──────────────────────────────────────
    if os.path.exists(config.weights_path):
        print(f"\nLoading pre-trained weights: {config.weights_path}")
        ckpt = torch.load(config.weights_path, map_location=config.device,
                          weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        pretrain_step = ckpt.get("step", "?")
        pretrain_loss = ckpt.get("best_val_loss", "?")
        print(f"  Loaded from step {pretrain_step} | val_loss {pretrain_loss}")
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

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\nPost-training for {config.posttrain_max_iters} steps...")
    print(f"  Effective batch : {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  LR max → min    : {config.posttrain_learning_rate} → {config.posttrain_min_lr}")
    print("-" * 60)

    model.train()
    t0 = time.time()
    eval_interval = min(config.eval_interval, 250)  # Eval more often in posttrain

    try:
        for step in range(start_step, config.posttrain_max_iters):
            # Learning rate schedule (using posttrain params)
            lr = get_lr(
                step,
                max_iters=config.posttrain_max_iters,
                lr=config.posttrain_learning_rate,
                min_lr_val=config.posttrain_min_lr,
                warmup=config.posttrain_warmup_iters,
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
                print(f"  step {step:>5d}/{config.posttrain_max_iters} | "
                      f"loss {accum_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                t0 = time.time()

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
                        scaler, config.posttrain_max_iters, best_val_loss)
        sync_to_drive(config.posttrain_weights_path)

    print(f"\n{'=' * 60}")
    print(f"  Post-training complete!")
    print(f"  Final val_loss : {val_loss:.4f}")
    print(f"  Best val_loss  : {best_val_loss:.4f}")
    print(f"{'=' * 60}")

    # ── After comparison ──────────────────────────────────────────────
    generate_samples(model, tokenizer, label="AFTER Post-Training")

    # Cleanup temp splits
    for p in [pt_train_path, pt_val_path]:
        try:
            os.remove(p)
        except OSError:
            pass


if __name__ == "__main__":
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    posttrain()
