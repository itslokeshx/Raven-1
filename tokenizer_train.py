# tokenizer_train.py
# RAVEN-1 — Train a byte-level BPE tokenizer.
#
# Uses HuggingFace `tokenizers` library to build a BPE tokenizer
# from the pre-training corpus. Adds all special tokens defined
# in config.py and saves the trained tokenizer to disk.

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import config


def train_tokenizer():
    """Train a byte-level BPE tokenizer on data/corpus.txt."""

    corpus_path = "data/corpus.txt"

    # ── Verify corpus exists ──────────────────────────────────────────
    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus not found at {corpus_path}")
        print("  Place your training text in data/corpus.txt and re-run.")
        return

    file_size = os.path.getsize(corpus_path)
    if file_size == 0:
        print(f"[ERROR] {corpus_path} is empty. Add training data first.")
        return

    print(f"Training tokenizer on: {corpus_path}")
    print(f"  Corpus size: {file_size / 1024:.1f} KB")

    # ── Create checkpoint directory ───────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Build tokenizer ───────────────────────────────────────────────
    tokenizer = Tokenizer(models.BPE(unk_token=None))

    # Byte-level pre-tokenizer — handles all Unicode cleanly
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Byte-level decoder to reconstruct text from tokens
    tokenizer.decoder = decoders.ByteLevel()

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=config.special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    # ── Train ─────────────────────────────────────────────────────────
    print(f"  Target vocab size: {config.vocab_size}")
    print(f"  Special tokens: {config.special_tokens}")
    print("  Training...")

    tokenizer.train([corpus_path], trainer)

    # ── Save ──────────────────────────────────────────────────────────
    tokenizer.save(config.tokenizer_path)
    print(f"\nTokenizer saved to: {config.tokenizer_path}")

    # ── Verify ────────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"  Final vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Special token IDs:")
    for token in config.special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            print(f"    {token:12s} → {token_id}")
        else:
            print(f"    {token:12s} → [MISSING — ERROR!]")
    print(f"{'=' * 50}")

    # Quick encode/decode sanity check
    test_text = f"{config.user_token} hello\n{config.raven_token} sup"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n  Test encode: {test_text!r}")
    print(f"  Token IDs:   {encoded.ids[:20]}...")
    print(f"  Decoded:     {decoded!r}")
    print("\nDone.")


if __name__ == "__main__":
    train_tokenizer()
