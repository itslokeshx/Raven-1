# posttrain_data_prep.py
# RAVEN-1 — Streaming post-training data preparation.
#
# Converts data/posttrain.txt → posttrain_train.bin + posttrain_val.bin
#
# KEY DESIGN CHOICES FOR COLAB FREE TIER:
#   1. STREAMING PARSER: Never reads full file into RAM. Reads 8MB chunks,
#      reconstructs <|eos|>-delimited samples across chunk boundaries.
#   2. BATCH TOKENIZATION: Accumulates ~4096 samples, then tokenizes the
#      entire batch via encode_batch() which uses Rust multi-threading.
#   3. INCREMENTAL WRITES: Token IDs are appended to output files in batches.
#      Total RAM for tokens ≈ batch_size * avg_sample_length * 2 bytes.
#   4. METADATA CACHE: Saves source file hash/size/mtime so rebuilds only
#      happen when the source data or tokenizer actually changes.
#
# Usage:
#   python posttrain_data_prep.py                # normal run
#   python posttrain_data_prep.py --force        # force rebuild
#   python posttrain_data_prep.py --max-samples 1000  # debug mode

import os
import sys
import json
import time
import struct
import argparse
import numpy as np
from tokenizers import Tokenizer
import config


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

CORPUS_PATH = "data/posttrain.txt"
METADATA_PATH = os.path.join(config.checkpoint_dir, "posttrain_meta.json")

# Streaming chunk size — 8MB balances RAM and I/O
READ_CHUNK_SIZE = 8 * 1024 * 1024

# Tokenization batch — 4096 samples per encode_batch call
# Smaller = more Python overhead; larger = more RAM
TOKENIZE_BATCH = 4096

# Train/val split ratio
SPLIT_RATIO = 0.90

# Warn if more than 1% of samples are malformed
MALFORMED_WARN_THRESHOLD = 0.01


# ═══════════════════════════════════════════════════════════════════════
# Metadata — skip rebuild if nothing changed
# ═══════════════════════════════════════════════════════════════════════

def _get_file_info(path):
    """Return (size, mtime) for a file, or (0, 0) if missing."""
    if not os.path.exists(path):
        return 0, 0
    st = os.stat(path)
    return st.st_size, st.st_mtime


def _load_metadata():
    """Load cached metadata from previous successful build."""
    if not os.path.exists(METADATA_PATH):
        return None
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _save_metadata(meta):
    """Save build metadata for cache validation."""
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def _needs_rebuild(force=False):
    """Check if we need to rebuild the binary files.

    Returns True if:
      - force flag is set
      - output bins don't exist
      - source file or tokenizer changed since last build
    """
    if force:
        print("  Force rebuild requested.")
        return True

    # Check output files exist
    if not os.path.exists(config.posttrain_train_bin):
        print("  Train bin missing → rebuild needed.")
        return True
    if not os.path.exists(config.posttrain_val_bin):
        print("  Val bin missing → rebuild needed.")
        return True

    # Check metadata
    meta = _load_metadata()
    if meta is None:
        print("  No metadata cache → rebuild needed.")
        return True

    # Check source file
    src_size, src_mtime = _get_file_info(CORPUS_PATH)
    if src_size != meta.get("source_size") or src_mtime != meta.get("source_mtime"):
        print("  Source file changed → rebuild needed.")
        return True

    # Check tokenizer
    tok_size, tok_mtime = _get_file_info(config.tokenizer_path)
    if tok_size != meta.get("tokenizer_size") or tok_mtime != meta.get("tokenizer_mtime"):
        print("  Tokenizer changed → rebuild needed.")
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# Streaming Sample Parser
# ═══════════════════════════════════════════════════════════════════════

def stream_samples(corpus_path, eos_token, max_samples=None):
    """Yield complete samples from corpus without loading full file.

    Reads the file in 8MB chunks and reconstructs samples that are
    delimited by `eos_token`. Handles samples that span chunk boundaries.

    Yields:
        sample_text (str): complete sample text INCLUDING the eos_token
    """
    count = 0
    remainder = ""

    with open(corpus_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(READ_CHUNK_SIZE)
            if not chunk:
                break

            # Prepend any leftover text from previous chunk
            text = remainder + chunk
            remainder = ""

            # Split on the EOS delimiter
            parts = text.split(eos_token)

            # Last part is incomplete (no trailing eos_token) — save for next chunk
            remainder = parts[-1]

            # All parts except the last are complete samples
            for part in parts[:-1]:
                sample = part.strip()
                if not sample:
                    continue

                # Re-attach the EOS token so the model learns it
                yield sample + " " + eos_token

                count += 1
                if max_samples and count >= max_samples:
                    return

    # Handle any remaining text after the last EOS (usually empty/whitespace)
    if remainder.strip():
        yield remainder.strip() + " " + eos_token


# ═══════════════════════════════════════════════════════════════════════
# Incremental Binary Writer
# ═══════════════════════════════════════════════════════════════════════

class BinWriter:
    """Append token IDs to a binary file incrementally.

    Writes uint16 values in native byte order using buffered I/O.
    Never holds more than one batch of tokens in RAM.
    """

    def __init__(self, path):
        self.path = path
        self.f = open(path, "wb")
        self.count = 0

    def write_ids(self, ids_list):
        """Write a flat list of token IDs (ints) to the binary file."""
        arr = np.array(ids_list, dtype=np.uint16)
        arr.tofile(self.f)
        self.count += len(ids_list)

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def prepare(force=False, max_samples=None):
    print("=" * 60)
    print("RAVEN-1 — Post-Training Data Preparation")
    print("=" * 60)

    # ── Validate inputs ───────────────────────────────────────────────
    if not os.path.exists(CORPUS_PATH):
        print(f"\n[ERROR] Corpus not found: {CORPUS_PATH}")
        print("  Upload posttrain.txt to data/ first.")
        sys.exit(1)

    if not os.path.exists(config.tokenizer_path):
        print(f"\n[ERROR] Tokenizer not found: {config.tokenizer_path}")
        print("  Run tokenizer_train.py first, or upload the tokenizer.")
        sys.exit(1)

    src_size, src_mtime = _get_file_info(CORPUS_PATH)
    tok_size, tok_mtime = _get_file_info(config.tokenizer_path)

    print(f"\n  Source     : {CORPUS_PATH} ({src_size / 1e6:.1f} MB)")
    print(f"  Tokenizer  : {config.tokenizer_path}")
    print(f"  Train bin  : {config.posttrain_train_bin}")
    print(f"  Val bin    : {config.posttrain_val_bin}")
    if max_samples:
        print(f"  Debug mode : max {max_samples:,} samples")

    # ── Check cache ───────────────────────────────────────────────────
    print("\n── Cache check ──")
    if not _needs_rebuild(force=force):
        meta = _load_metadata()
        print(f"\n  ✅ Binaries up to date!")
        print(f"  Train tokens : {meta.get('train_tokens', '?'):,}")
        print(f"  Val tokens   : {meta.get('val_tokens', '?'):,}")
        print(f"  Total tokens : {meta.get('total_tokens', '?'):,}")
        print(f"\n  Skipping rebuild. Use --force to override.")
        return

    # ── Load tokenizer ────────────────────────────────────────────────
    print("\n── Loading tokenizer ──")
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")

    # ── Ensure output dirs ────────────────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ── Phase 1: Count samples (fast streaming scan) ──────────────────
    # We need the count to know where to split train/val.
    # This is a fast pass — just counts EOS tokens, no tokenization.
    print("\n── Phase 1: Counting samples ──")
    t0 = time.time()
    total_samples = 0
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            total_samples += chunk.count(config.eos_token)
    count_time = time.time() - t0
    print(f"  Found {total_samples:,} samples in {count_time:.1f}s")

    if total_samples == 0:
        print("\n[ERROR] No samples found! Check that your data contains "
              f"'{config.eos_token}' delimiters.")
        sys.exit(1)

    if max_samples and max_samples < total_samples:
        total_samples = max_samples
        print(f"  [Debug] Limiting to {max_samples:,} samples")

    train_count = int(total_samples * SPLIT_RATIO)
    val_count = total_samples - train_count
    print(f"  Train samples: {train_count:,}")
    print(f"  Val samples  : {val_count:,}")

    # ── Phase 2: Stream → Tokenize → Write ────────────────────────────
    # Reads samples in streaming fashion, tokenizes in Rust-threaded
    # batches, and writes token IDs directly to output binary files.
    # Peak RAM = ~TOKENIZE_BATCH * avg_tokens_per_sample * 2 bytes
    print("\n── Phase 2: Tokenize & write ──")
    t0 = time.time()

    samples_processed = 0
    tokens_written_train = 0
    tokens_written_val = 0
    malformed = 0

    # Accumulation buffer for batch tokenization
    batch_buffer = []

    with BinWriter(config.posttrain_train_bin) as train_writer, \
         BinWriter(config.posttrain_val_bin) as val_writer:

        def flush_batch():
            """Tokenize and write the accumulated batch."""
            nonlocal samples_processed, tokens_written_train, tokens_written_val
            nonlocal malformed

            if not batch_buffer:
                return

            # Multi-threaded Rust batch tokenization — the key speedup
            encodeds = tokenizer.encode_batch(batch_buffer)

            for enc in encodeds:
                ids = enc.ids

                if not ids:
                    malformed += 1
                    samples_processed += 1
                    continue

                # Route to train or val based on sample index
                if samples_processed < train_count:
                    train_writer.write_ids(ids)
                    tokens_written_train += len(ids)
                else:
                    val_writer.write_ids(ids)
                    tokens_written_val += len(ids)

                samples_processed += 1

            batch_buffer.clear()

        # Stream samples from disk
        for sample in stream_samples(CORPUS_PATH, config.eos_token,
                                     max_samples=max_samples):
            # Skip empty or whitespace-only
            if not sample or sample.isspace():
                malformed += 1
                continue

            # Basic validation: should contain user and raven tokens
            if config.user_token not in sample and config.raven_token not in sample:
                malformed += 1
                continue

            batch_buffer.append(sample)

            # When batch is full, tokenize and write
            if len(batch_buffer) >= TOKENIZE_BATCH:
                flush_batch()

                # Progress report every 50k samples
                if samples_processed % 50000 < TOKENIZE_BATCH:
                    elapsed = time.time() - t0
                    speed = samples_processed / max(elapsed, 0.01)
                    remaining = total_samples - samples_processed
                    eta = remaining / max(speed, 0.01)
                    pct = (samples_processed / total_samples) * 100

                    total_toks = tokens_written_train + tokens_written_val
                    print(f"\r  {pct:5.1f}% | "
                          f"{samples_processed:,}/{total_samples:,} samples | "
                          f"{total_toks:,} tokens | "
                          f"{speed:.0f} samples/s | "
                          f"ETA {eta:.0f}s",
                          end="", flush=True)

        # Flush remaining samples
        flush_batch()

    elapsed = time.time() - t0
    total_tokens = tokens_written_train + tokens_written_val

    print(f"\n\n── Results ──")
    print(f"  Samples processed : {samples_processed:,}")
    print(f"  Malformed/skipped : {malformed:,}")
    print(f"  Train tokens      : {tokens_written_train:,}")
    print(f"  Val tokens        : {tokens_written_val:,}")
    print(f"  Total tokens      : {total_tokens:,}")
    print(f"  Time              : {elapsed:.1f}s")
    print(f"  Speed             : {samples_processed / max(elapsed, 0.01):.0f} samples/s")
    print(f"  Throughput        : {total_tokens / max(elapsed, 0.01):.0f} tokens/s")

    # Malformed warning
    if samples_processed > 0:
        malformed_ratio = malformed / (samples_processed + malformed)
        if malformed_ratio > MALFORMED_WARN_THRESHOLD:
            print(f"\n  ⚠️  Warning: {malformed_ratio:.1%} of samples were malformed/skipped.")
            print(f"     Check your data format. Expected: "
                  f"{config.user_token}...{config.raven_token}...{config.eos_token}")

    # ── Save metadata ─────────────────────────────────────────────────
    meta = {
        "source_path": CORPUS_PATH,
        "source_size": src_size,
        "source_mtime": src_mtime,
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_size": tok_size,
        "tokenizer_mtime": tok_mtime,
        "train_tokens": tokens_written_train,
        "val_tokens": tokens_written_val,
        "total_tokens": total_tokens,
        "total_samples": samples_processed,
        "malformed_samples": malformed,
        "split_ratio": SPLIT_RATIO,
        "build_time_seconds": round(elapsed, 1),
    }
    _save_metadata(meta)

    # ── Verify output files ───────────────────────────────────────────
    train_file_tokens = os.path.getsize(config.posttrain_train_bin) // 2
    val_file_tokens = os.path.getsize(config.posttrain_val_bin) // 2

    print(f"\n── Verification ──")
    print(f"  {config.posttrain_train_bin}: {train_file_tokens:,} tokens "
          f"({os.path.getsize(config.posttrain_train_bin) / 1e6:.1f} MB)")
    print(f"  {config.posttrain_val_bin}: {val_file_tokens:,} tokens "
          f"({os.path.getsize(config.posttrain_val_bin) / 1e6:.1f} MB)")

    if train_file_tokens != tokens_written_train:
        print(f"  ⚠️  Train token mismatch! Expected {tokens_written_train:,}, "
              f"got {train_file_tokens:,}")
    if val_file_tokens != tokens_written_val:
        print(f"  ⚠️  Val token mismatch! Expected {tokens_written_val:,}, "
              f"got {val_file_tokens:,}")

    print(f"\n{'=' * 60}")
    print("Data preparation complete.")
    print("You can now run: python posttrain.py")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAVEN-1 Post-Training Data Preparation")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cache is valid")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()

    prepare(force=args.force, max_samples=args.max_samples)
