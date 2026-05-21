# data_prep.py
# RAVEN-1 — Tokenize corpus and save as binary files for fast training.
#
# Reads text corpora, tokenizes them with the trained BPE tokenizer,
# splits into train/val, and saves as numpy memmap files (uint16).
# Handles both pre-training corpus and post-training corpus.

import os
import numpy as np
from tokenizers import Tokenizer
import config


def tokenize_and_save(corpus_path, output_path, split_ratio=None,
                      train_path=None, val_path=None):
    """Tokenize a text file and save as binary.

    If split_ratio is provided, splits into train/val and saves to
    train_path and val_path. Otherwise saves entire corpus to output_path.

    Args:
        corpus_path: path to input text file
        output_path: path for single output (used when no split)
        split_ratio: float, fraction for training (e.g. 0.95)
        train_path: output path for train split
        val_path: output path for val split
    """
    if not os.path.exists(corpus_path):
        print(f"  [SKIP] {corpus_path} not found.")
        return False

    if os.path.getsize(corpus_path) == 0:
        print(f"  [SKIP] {corpus_path} is empty.")
        return False

    # Read corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"  Read {len(text):,} characters from {corpus_path}")

    # Tokenize
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    encoded = tokenizer.encode(text)
    token_ids = np.array(encoded.ids, dtype=np.uint16)

    print(f"  Tokenized to {len(token_ids):,} tokens")

    if split_ratio and train_path and val_path:
        # Split into train/val
        split_idx = int(len(token_ids) * split_ratio)
        train_tokens = token_ids[:split_idx]
        val_tokens = token_ids[split_idx:]

        # Save as memmap files
        train_mm = np.memmap(train_path, dtype=np.uint16, mode='w+',
                             shape=train_tokens.shape)
        train_mm[:] = train_tokens
        train_mm.flush()

        val_mm = np.memmap(val_path, dtype=np.uint16, mode='w+',
                           shape=val_tokens.shape)
        val_mm[:] = val_tokens
        val_mm.flush()

        print(f"  Train tokens : {len(train_tokens):,} → {train_path}")
        print(f"  Val tokens   : {len(val_tokens):,} → {val_path}")
    else:
        # Save entire corpus as single file
        mm = np.memmap(output_path, dtype=np.uint16, mode='w+',
                       shape=token_ids.shape)
        mm[:] = token_ids
        mm.flush()

        print(f"  Saved {len(token_ids):,} tokens → {output_path}")

    return True


def main():
    print("=" * 60)
    print("RAVEN-1 — Data Preparation")
    print("=" * 60)

    # Check tokenizer exists
    if not os.path.exists(config.tokenizer_path):
        print(f"\n[ERROR] Tokenizer not found at {config.tokenizer_path}")
        print("  Run tokenizer_train.py first.")
        return

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Pre-training corpus ───────────────────────────────────────────
    print("\n── Pre-training corpus ──")
    tokenize_and_save(
        corpus_path="data/corpus.txt",
        output_path=None,
        split_ratio=0.95,
        train_path=config.train_bin,
        val_path=config.val_bin,
    )

    # ── Post-training corpus ──────────────────────────────────────────
    print("\n── Post-training corpus ──")
    tokenize_and_save(
        corpus_path="data/posttrain.txt",
        output_path=config.posttrain_bin,
    )

    print("\n" + "=" * 60)
    print("Data preparation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
