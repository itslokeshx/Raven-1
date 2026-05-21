import os
import torch
from tokenizers import Tokenizer
from model import Raven1
import config
from inference import generate_response, load_model

def run_stress_test():
    print("Loading model and tokenizer...")
    model, weights_name = load_model()
    
    if not os.path.exists(config.tokenizer_path):
        print(f"[ERROR] Tokenizer not found: {config.tokenizer_path}")
        return
        
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    # Generate 200 "cat" words as input
    user_input = "cat " * 200
    user_input = user_input.strip()
    
    print(f"\nTesting with long input (length: {len(user_input)} chars)...")
    print(f"Input preview: {user_input[:50]}...")
    
    history = []
    
    print("\nGenerating response (this might take a moment on CPU)...")
    response = generate_response(model, tokenizer, user_input, history, "default")
    
    print("\n" + "="*60)
    print("RAVEN-1 RESPONSE:")
    print("="*60)
    print(response)
    print("="*60)

if __name__ == "__main__":
    run_stress_test()
