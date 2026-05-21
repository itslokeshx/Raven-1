# app.py
# RAVEN-1 — Gradio web interface for Hugging Face Spaces.
#
# Premium dark monochrome chatbot UI. Runs on CPU.
# Same generation logic as inference.py.

import os
import random
import torch
import gradio as gr
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

FALLBACKS = [
    "I had a thought, but it left without notice.",
    "Error 404: clever response not found.",
    "Even I can't make sense of what you just said.",
    "I stared at that for a while. Nothing happened.",
    "You broke me. Congratulations, that's actually impressive.",
]


# ═══════════════════════════════════════════════════════════════════════
# Model Loading (runs once at startup)
# ═══════════════════════════════════════════════════════════════════════

def load_model():
    """Load model and tokenizer. Tries posttrain → pretrain → random."""
    model = Raven1()

    weights_name = "random"
    if os.path.exists(config.posttrain_weights_path):
        ckpt = torch.load(config.posttrain_weights_path,
                          map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        weights_name = "posttrain"
    elif os.path.exists(config.weights_path):
        ckpt = torch.load(config.weights_path,
                          map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        weights_name = "pretrain"

    model = model.to(config.device).eval()
    print(f"Model loaded: {weights_name} | device: {config.device}")
    return model, weights_name


def load_tokenizer():
    """Load the BPE tokenizer."""
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {config.tokenizer_path}")
    return Tokenizer.from_file(config.tokenizer_path)


# Load once at import time
print("Loading RAVEN-1...")
MODEL, WEIGHTS_NAME = load_model()
TOKENIZER = load_tokenizer()
EOS_ID = TOKENIZER.token_to_id(config.eos_token)


# ═══════════════════════════════════════════════════════════════════════
# Generation Logic
# ═══════════════════════════════════════════════════════════════════════

def build_prompt(user_input, history):
    """Build prompt with last 3 turns of conversation context."""
    context = ""
    for msg in history:
        if msg["role"] == "user":
            context += f"{config.user_token} {msg['content']}\n"
        elif msg["role"] == "assistant":
            context += f"{config.raven_token} {msg['content']}\n{config.eos_token}\n"
    context += f"{config.user_token} {user_input}\n{config.raven_token}"
    return context


def respond(user_input, history, mode_name="default"):
    """Generate a response given user input and chat history."""
    try:
        mode = MODES.get(mode_name, MODES["default"])

        # Keep last 6 messages (3 turns) for context
        recent_history = history[-6:] if history else []

        prompt = build_prompt(user_input, recent_history)

        # Encode and crop
        encoded = TOKENIZER.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long,
                                 device=config.device)
        if input_ids.size(1) > config.block_size:
            input_ids = input_ids[:, -config.block_size:]

        # Generate
        output_ids = MODEL.generate(
            input_ids,
            max_new_tokens=60,
            temperature=mode["temperature"],
            top_k=mode["top_k"],
            top_p=mode["top_p"],
            repetition_penalty=mode["rep"],
            stop_token_id=EOS_ID,
        )

        # Decode new tokens only
        new_tokens = output_ids[0, input_ids.size(1):].tolist()
        response = TOKENIZER.decode(new_tokens).strip()

        # Clean special tokens
        for tok in config.special_tokens:
            response = response.replace(tok, "")
        response = response.strip()

        if not response:
            response = random.choice(FALLBACKS)

        return response

    except Exception:
        return random.choice(FALLBACKS)


# ═══════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════

# Dark monochrome theme
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0a0a0a",
    body_background_fill_dark="#0a0a0a",
    body_text_color="#e5e5e5",
    body_text_color_dark="#e5e5e5",
    block_background_fill="#111111",
    block_background_fill_dark="#111111",
    block_border_width="1px",
    block_border_color="#222222",
    block_border_color_dark="#222222",
    block_label_text_color="#888888",
    block_label_text_color_dark="#888888",
    block_shadow="none",
    input_background_fill="#111111",
    input_background_fill_dark="#111111",
    input_border_color="#333333",
    input_border_color_dark="#333333",
    input_text_color="#e5e5e5",
    input_text_color_dark="#e5e5e5",
    button_primary_background_fill="#ffffff",
    button_primary_background_fill_dark="#ffffff",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    button_secondary_background_fill="#1a1a1a",
    button_secondary_background_fill_dark="#1a1a1a",
    button_secondary_text_color="#cccccc",
    button_secondary_text_color_dark="#cccccc",
)

# Custom CSS for polish
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

.gradio-container { max-width: 700px !important; margin: auto !important; }

/* Header */
.head { text-align: center; padding: 32px 0 16px; }
.head h1 { font-size: 28px; font-weight: 600; color: #ffffff;
           letter-spacing: -0.5px; margin: 0; }
.head p { font-size: 14px; color: #666; margin: 4px 0 0; font-weight: 300; }

/* Chatbot */
#chatbox { border: 1px solid #222 !important; border-radius: 12px !important; }

/* Input row */
#input-row { gap: 8px !important; }

/* Footer */
.foot { text-align: center; padding: 12px 0; }
.foot p { font-size: 11px; color: #444; margin: 0; }
"""

# Build the app
with gr.Blocks(theme=theme, css=custom_css, title="RAVEN-1") as demo:

    gr.HTML(f"""
    <div class="head">
        <h1>RAVEN-1</h1>
        <p>25M parameters. Zero patience.</p>
    </div>
    """)

    chatbot = gr.Chatbot(
        height=450,
        show_label=False,
        elem_id="chatbox",
    )

    with gr.Row(elem_id="input-row"):
        msg = gr.Textbox(
            placeholder="Say something.",
            show_label=False,
            scale=6,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1,
                             min_width=80)

    with gr.Row():
        mode_dropdown = gr.Dropdown(
            choices=list(MODES.keys()),
            value="default",
            label="Mode",
            scale=2,
        )
        clear_btn = gr.Button("Reset", variant="secondary", scale=1)

    gr.HTML('<div class="foot"><p>Built from scratch. No shortcuts.</p></div>')

    # ── Event handlers ────────────────────────────────────────────

    def submit(message, history, mode_name):
        """Handle user message submission."""
        if not message.strip():
            return "", history

        response = respond(message.strip(), history or [], mode_name)
        history = history or []
        history.append({"role": "user", "content": message.strip()})
        history.append({"role": "assistant", "content": response})
        return "", history

    def clear():
        """Clear chat history."""
        return None

    msg.submit(submit, [msg, chatbot, mode_dropdown], [msg, chatbot])
    send_btn.click(submit, [msg, chatbot, mode_dropdown], [msg, chatbot])
    clear_btn.click(clear, None, chatbot)


# ═══════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
