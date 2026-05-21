# model.py
# RAVEN-1 — GPT-style decoder-only transformer.
#
# Architecture:
#   - Pre-LayerNorm transformer blocks
#   - Causal self-attention via scaled_dot_product_attention (FlashAttention)
#   - GELU activations in feed-forward network
#   - Weight tying between token embedding and LM head
#   - ~25M parameters with default config
#
# All hyperparameters imported from config.py — zero hardcoded numbers.

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ═══════════════════════════════════════════════════════════════════════
# Causal Self-Attention
# ═══════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch's
    scaled_dot_product_attention, which automatically dispatches to
    FlashAttention when the hardware supports it."""

    def __init__(self):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"

        # Combined Q, K, V projection for efficiency
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Regularization
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V and split heads
        qkv = self.qkv_proj(x)                           # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)                   # each (B, T, C)

        # Reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        # This uses FlashAttention automatically when available
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # Recombine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection + dropout
        y = self.resid_dropout(self.out_proj(y))
        return y


# ═══════════════════════════════════════════════════════════════════════
# Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """Standard transformer FFN: Linear → GELU → Linear → Dropout.
    Inner dimension is 4x the embedding dimension."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
# Transformer Block (Pre-LayerNorm)
# ═══════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block:
        x → LN → Attention → + residual
        x → LN → FFN       → + residual
    """

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Attention with residual
        x = x + self.ffn(self.ln2(x))   # FFN with residual
        return x


# ═══════════════════════════════════════════════════════════════════════
# RAVEN-1 Model
# ═══════════════════════════════════════════════════════════════════════

class Raven1(nn.Module):
    """Full RAVEN-1 language model.

    Architecture:
        Token Embedding  →  Position Embedding  →  Dropout
        N × TransformerBlock (Pre-LN + Attention + FFN)
        Final LayerNorm
        LM Head (weight-tied with token embedding)
    """

    def __init__(self):
        super().__init__()

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(config.n_layer)])

        # Final layer norm (before LM head)
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language model head — tied to token embedding weights
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

        print(f"Raven-1 initialized")
        print(f"  Parameters : {self.get_num_params() / 1e6:.2f}M")
        print(f"  Layers     : {config.n_layer}")
        print(f"  Heads      : {config.n_head}")
        print(f"  Embed dim  : {config.n_embd}")
        print(f"  FFN dim    : {4 * config.n_embd}")
        print(f"  Vocab      : {config.vocab_size}")
        print(f"  Context    : {config.block_size}")

    def _init_weights(self, module):
        """Initialize weights using scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_num_params(self):
        """Return total number of parameters (excluding position embeddings
        from the count since they don't scale with vocab)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """Forward pass.

        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target token indices, or None

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= config.block_size, \
            f"Sequence length {T} exceeds block_size {config.block_size}"

        # Token + position embeddings
        tok_emb = self.token_embedding(idx)                         # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, C)
        x = self.emb_dropout(tok_emb + pos_emb)                    # (B, T, C)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm + LM head
        x = self.ln_f(x)                                           # (B, T, C)
        logits = self.lm_head(x)                                   # (B, T, V)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=0.65,
                 top_k=25, top_p=0.85, repetition_penalty=1.1,
                 stop_token_id=None):
        """Autoregressive generation with top-k, top-p, repetition penalty.

        Args:
            idx: (B, T) conditioning token indices
            max_new_tokens: max tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: keep only top-k logits (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            repetition_penalty: penalty for already-generated tokens (1.0 = off)
            stop_token_id: stop generation when this token is produced

        Returns:
            idx: (B, T + generated) full sequence including generated tokens
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to block_size if sequence is too long
            idx_cond = idx if idx.size(1) <= config.block_size \
                else idx[:, -config.block_size:]

            # Forward pass — get logits for last position only
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                               # (B, V)

            # ── Repetition penalty ────────────────────────────────
            if repetition_penalty != 1.0:
                # Gather scores for tokens that already appeared
                for b in range(idx.size(0)):
                    prev_tokens = idx[b].unique()
                    for token_id in prev_tokens:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            # ── Temperature ───────────────────────────────────────
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy: just take argmax
                next_token = logits.argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_token], dim=1)
                if stop_token_id is not None and next_token.item() == stop_token_id:
                    break
                continue

            # ── Top-K filtering ───────────────────────────────────
            if top_k > 0:
                top_k_clamped = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, top_k_clamped)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            # ── Top-P (nucleus) filtering ─────────────────────────
            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')

                # Scatter back to original ordering
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # ── Sample ────────────────────────────────────────────
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)    # (B, 1)

            # Append
            idx = torch.cat([idx, next_token], dim=1)

            # Stop on EOS
            if stop_token_id is not None and next_token.item() == stop_token_id:
                break

        return idx
