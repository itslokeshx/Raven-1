"""
run_300_test.py
Run 300 diverse prompts through RAVEN-1 and save clean results to results.txt
"""

import os
import re
import time
import random
import torch
import config
from tokenizers import Tokenizer
from model import Raven1
from inference import generate_response, load_model


def clean_output(text):
    """Truncate output to the first actual response only.
    The model sometimes leaks multi-turn conversations."""
    # Cut at any special token boundary
    for tok in config.special_tokens:
        if tok in text:
            text = text.split(tok)[0]
    # Cut at newline followed by what looks like a new user turn
    text = re.split(r'\n\s*\n', text)[0]
    # Remove any remaining control chars
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_prompts(num=300):
    """Generate 300 diverse prompts across multiple categories."""
    bases = {
        "casual": [
            "hello", "how are you?", "what's up?", "good morning",
            "tell me a joke", "who are you?", "nice weather today",
            "I am happy", "I am sad", "What do you think?",
            "hey there", "good night", "thanks!", "bye",
            "what's your name?", "how's it going?", "tell me something fun",
            "I'm bored", "make me laugh", "say something cool",
        ],
        "questions": [
            "what is gravity?", "how do birds fly?",
            "what is the capital of france?", "why is the sky blue?",
            "who wrote hamlet?", "how far is the moon?",
            "when did WWII end?", "what is AI?",
            "how does electricity work?", "what is photosynthesis?",
            "what causes earthquakes?", "who invented the internet?",
            "what is DNA?", "how do computers work?",
            "what is quantum physics?",
        ],
        "edge_cases": [
            "", " ", "\n", "cat " * 50, "12345 " * 20,
            "!@#$%" * 10, "A" * 100, "\t\t\t\t", "none",
            "..." * 30, "??" * 50, "   spaces   ",
        ],
        "math": [
            "1+1=", "what is 2*2?", "solve for x: 2x = 4",
            "is 7 a prime number?", "pi equals", "100 / 2",
            "sqrt(16)", "what is 10 factorial?", "2^10=",
            "what is 15% of 200?",
        ],
        "coding": [
            "write a python script", "print('hello')",
            "def fibonacci():", "import sys", "html body",
            "int main()", "console.log", "for i in range(10):",
            "class MyClass:", "SELECT * FROM",
        ],
        "creative": [
            "write me a poem", "tell me a story",
            "describe the ocean", "what does love mean?",
            "imagine a world without gravity",
            "write a haiku", "describe a sunset",
            "what if cats could talk?",
        ],
    }

    prompts = []
    for i in range(num):
        cat = random.choice(list(bases.keys()))
        base = random.choice(bases[cat])
        if cat == "edge_cases":
            prompt = base
        else:
            prompt = f"{base} (test {i+1})"
        prompts.append((i + 1, cat, prompt))

    return prompts


def main():
    print("Loading RAVEN-1 model...")
    model, weights_name = load_model()
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {param_count:.0f}M params | Weights: {weights_name} | Device: {config.device}")

    random.seed(42)
    prompts = generate_prompts(300)
    print(f"\nRunning {len(prompts)} prompts...\n")

    results = []
    start_time = time.time()

    for idx, cat, prompt in prompts:
        t0 = time.time()
        raw_response = generate_response(model, tokenizer, prompt, [], "default")
        latency = time.time() - t0

        cleaned = clean_output(raw_response)

        results.append({
            "num": idx,
            "category": cat,
            "input": prompt,
            "output": cleaned,
            "latency": latency,
        })

        if idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  [{idx:3d}/300] done  ({elapsed:.1f}s elapsed)")

    total_time = time.time() - start_time

    # Write results.txt
    output_path = os.path.join(os.path.dirname(__file__), "results.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  RAVEN-1  —  300 Prompt Test Results\n")
        f.write(f"  Model: {param_count:.0f}M params | Weights: {weights_name}\n")
        f.write(f"  Device: {config.device}\n")
        f.write(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            display_input = r["input"] if r["input"].strip() else "(empty/whitespace)"
            f.write(f"[{r['num']:3d}] ({r['category']})\n")
            f.write(f"  Input : {display_input}\n")
            f.write(f"  Output: {r['output']}\n")
            f.write("-" * 80 + "\n")

        # Summary
        avg_latency = total_time / len(results)
        cat_counts = {}
        for r in results:
            cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

        f.write("\n" + "=" * 80 + "\n")
        f.write("  SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Total Prompts : {len(results)}\n")
        f.write(f"  Total Time    : {total_time:.2f}s\n")
        f.write(f"  Avg Latency   : {avg_latency:.2f}s per prompt\n\n")
        f.write(f"  Category Breakdown:\n")
        for cat, count in sorted(cat_counts.items()):
            f.write(f"    {cat:12s}: {count} prompts\n")
        f.write("=" * 80 + "\n")

    print(f"\nDone! 300 prompts in {total_time:.2f}s")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
