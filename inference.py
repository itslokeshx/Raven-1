# inference.py
# RAVEN-1 — Terminal chat interface.
#
# Clean, crash-proof chat loop with multiple sampling modes,
# conversation history, and slash commands.

import os
import sys
import random
import torch
from tokenizers import Tokenizer
from model import Raven1
import config


# ═══════════════════════════════════════════════════════════════════════
# Sampling Modes
# ═══════════════════════════════════════════════════════════════════════

MODES = {
    "default": {"temperature": 0.65, "top_k": 25, "top_p": 0.85, "rep": 1.10},
    "sharp":   {"temperature": 0.55, "top_k": 20, "top_p": 0.80, "rep": 1.10},
    "chaos":   {"temperature": 0.85, "top_k": 45, "top_p": 0.92, "rep": 1.05},
    "cold":    {"temperature": 0.45, "top_k": 12, "top_p": 0.75, "rep": 1.08},
}

# Funny fallbacks when model returns empty
FALLBACKS = [
    "I had a thought, but it left without notice.",
    "Error 404: clever response not found.",
    "Even I can't make sense of what you just said.",
    "I stared at that for a while. Nothing happened.",
    "You broke me. Congratulations, that's actually impressive.",
]


# ═══════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    """Load the best available model weights.
    Tries posttrain weights first, falls back to pretrain weights."""
    model = Raven1()

    # Try posttrain first, then pretrain
    weights_path = None
    weights_name = None

    if os.path.exists(config.posttrain_weights_path):
        weights_path = config.posttrain_weights_path
        weights_name = "posttrain"
    elif os.path.exists(config.weights_path):
        weights_path = config.weights_path
        weights_name = "pretrain"
    else:
        print("[WARN] No checkpoint found. Using random weights.")
        return model.to(config.device).eval(), "random"

    ckpt = torch.load(weights_path, map_location=config.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    step = ckpt.get("step", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    if isinstance(val_loss, float):
        val_loss = f"{val_loss:.4f}"

    print(f"  Loaded {weights_name} from step {step} | val_loss {val_loss}")

    return model.to(config.device).eval(), weights_name


# ═══════════════════════════════════════════════════════════════════════
# Prompt Building
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(user_input, history):
    """Build a prompt with conversation history (last 3 turns)."""
    context = ""
    for turn in history[-3:]:
        context += f"{config.user_token} {turn['user']}\n"
        context += f"{config.raven_token} {turn['raven']}\n{config.eos_token}\n"
    context += f"{config.user_token} {user_input}\n{config.raven_token}"
    return context


# ═══════════════════════════════════════════════════════════════════════
# Response Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_response(model, tokenizer, user_input, history, mode_name="default"):
    """Generate a response. Never crashes — always returns something."""
    try:
        mode = MODES.get(mode_name, MODES["default"])
        eos_id = tokenizer.token_to_id(config.eos_token)

        # Build prompt
        prompt = build_prompt(user_input, history)

        # Encode
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long,
                                 device=config.device)

        # Crop to block_size (keep rightmost tokens for context)
        if input_ids.size(1) > config.block_size:
            input_ids = input_ids[:, -config.block_size:]

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=60,
            temperature=mode["temperature"],
            top_k=mode["top_k"],
            top_p=mode["top_p"],
            repetition_penalty=mode["rep"],
            stop_token_id=eos_id,
        )

        # Decode only new tokens
        new_tokens = output_ids[0, input_ids.size(1):].tolist()
        response = tokenizer.decode(new_tokens).strip()

        # Clean leaked special tokens
        for tok in config.special_tokens:
            response = response.replace(tok, "")
        response = response.strip()

        # Fallback if empty
        if not response:
            response = random.choice(FALLBACKS)

        return response

    except Exception:
        return random.choice(FALLBACKS)


# ═══════════════════════════════════════════════════════════════════════
# Chat Loop
# ═══════════════════════════════════════════════════════════════════════

def chat():
    """Main interactive chat loop."""

    # ── Load model ────────────────────────────────────────────────────
    print(f"\nLoading model...")
    model, weights_name = load_model()

    # ── Load tokenizer ────────────────────────────────────────────────
    if not os.path.exists(config.tokenizer_path):
        print(f"[ERROR] Tokenizer not found: {config.tokenizer_path}")
        return

    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    # ── Boot message ──────────────────────────────────────────────────
    param_count = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"\n══════════════════════════════")
    print(f"  RAVEN-1  |  {param_count:.0f}M parameters")
    print(f"  Loaded: {weights_name}")
    print(f"  Device: {config.device}")
    print(f"══════════════════════════════")
    print(f"\n  /mode [name]  → switch mode")
    print(f"  /modes        → list modes")
    print(f"  /clear        → clear screen")
    print(f"  /reset        → clear history")
    print(f"  /exit         → exit")
    print()

    # ── Chat state ────────────────────────────────────────────────────
    history = []
    current_mode = "default"

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye.")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()

            if cmd[0] == "/exit" or cmd[0] == "/quit":
                print("Bye.")
                break

            elif cmd[0] == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue

            elif cmd[0] == "/reset":
                history = []
                print("  History cleared.")
                continue

            elif cmd[0] == "/modes":
                print("\n  Available modes:")
                for name, params in MODES.items():
                    marker = " ←" if name == current_mode else ""
                    print(f"    {name:10s} temp={params['temperature']} "
                          f"top_k={params['top_k']} top_p={params['top_p']} "
                          f"rep={params['rep']}{marker}")
                print()
                continue

            elif cmd[0] == "/mode":
                if len(cmd) < 2:
                    print(f"  Current mode: {current_mode}")
                    print(f"  Usage: /mode [default|sharp|chaos|cold]")
                elif cmd[1] in MODES:
                    current_mode = cmd[1]
                    print(f"  Mode → {current_mode}")
                else:
                    print(f"  Unknown mode: {cmd[1]}")
                    print(f"  Available: {', '.join(MODES.keys())}")
                continue

            else:
                print(f"  Unknown command: {cmd[0]}")
                continue

        # ── Generate response ─────────────────────────────────────
        response = generate_response(model, tokenizer, user_input,
                                     history, current_mode)
        print(f"Raven: {response}\n")

        # Save to history
        history.append({"user": user_input, "raven": response})


if __name__ == "__main__":
    chat()
