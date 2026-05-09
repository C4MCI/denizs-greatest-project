"""
PatchTST-style Transformer for Predictive Maintenance
------------------------------------------------------
Key ideas borrowed from PatchTST (Nie et al., 2023):
  1. Patch tokenisation  – divide the time series into non-overlapping patches
     so each token carries LOCAL temporal context (vs. point-wise tokens).
  2. Channel-independent  – each sensor is processed independently, then
     features are merged before the classification head.
  3. Learnable positional embedding – better than fixed sinusoidal for short
     industrial sequences.

Architecture:
  Input  (B, T, C)  →  patch  →  (B*C, N_patches, d_model)
  → Transformer encoder (L layers, H heads)
  → mean-pool patches
  → merge channels  →  MLP head  →  binary logit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Patch embedding ─────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split a 1-D sequence into non-overlapping patches and project."""
    def __init__(self, patch_len: int, d_model: int, seq_len: int):
        super().__init__()
        self.patch_len   = patch_len
        self.n_patches   = seq_len // patch_len
        self.proj        = nn.Linear(patch_len, d_model)
        # learnable positional embedding per patch position
        self.pos_emb     = nn.Embedding(self.n_patches, d_model)

    def forward(self, x):
        # x: (B, T)  — single channel
        B, T = x.shape
        # reshape into patches
        x = x[:, :self.n_patches * self.patch_len]          # trim if needed
        x = x.reshape(B, self.n_patches, self.patch_len)    # (B, N, P)
        x = self.proj(x)                                     # (B, N, d)
        pos = torch.arange(self.n_patches, device=x.device)
        x = x + self.pos_emb(pos)                           # broadcast
        return x   # (B, N_patches, d_model)


# ── Transformer encoder block ────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm (more stable for smaller datasets)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ── Full PatchTST Classifier ─────────────────────────────────────────────────

class PatchTSTClassifier(nn.Module):
    """
    Args
    ----
    seq_len    : input sequence length (T)
    n_channels : number of sensor channels (C)
    patch_len  : patch size (e.g. 12 → 5 patches for seq_len=60)
    d_model    : transformer hidden dim
    n_heads    : attention heads
    n_layers   : number of transformer blocks
    d_ff       : feed-forward hidden dim
    dropout    : dropout rate
    n_classes  : output classes (2 for binary)
    """
    def __init__(self,
                 seq_len:    int = 60,
                 n_channels: int = 4,
                 patch_len:  int = 12,
                 d_model:    int = 64,
                 n_heads:    int = 4,
                 n_layers:   int = 3,
                 d_ff:       int = 128,
                 dropout:   float = 0.1,
                 n_classes:  int = 2):
        super().__init__()
        self.n_channels = n_channels
        n_patches       = seq_len // patch_len

        # one shared patch embedding for all channels (channel-independent)
        self.patch_emb  = PatchEmbedding(patch_len, d_model, seq_len)

        self.encoder    = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # classification head: concat mean-pooled representations from all channels
        self.head = nn.Sequential(
            nn.Linear(d_model * n_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        channel_reps = []

        for c in range(C):
            xc = x[:, :, c]                          # (B, T)
            xc = self.patch_emb(xc)                  # (B, N_patches, d_model)
            xc = self.encoder(xc)                    # (B, N_patches, d_model)
            xc = xc.mean(dim=1)                      # (B, d_model) — mean pool
            channel_reps.append(xc)

        out = torch.cat(channel_reps, dim=-1)        # (B, d_model * C)
        return self.head(out)                        # (B, n_classes)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Baseline LSTM ────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Simple bidirectional LSTM baseline."""
    def __init__(self,
                 n_channels:  int = 4,
                 hidden_dim:  int = 64,
                 n_layers:    int = 2,
                 dropout:    float = 0.2,
                 n_classes:   int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_channels,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            dropout     = dropout if n_layers > 1 else 0.0,
            bidirectional = True,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.lstm(x)       # (B, T, 2*H)
        out = out[:, -1, :]         # last timestep
        return self.head(out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    B, T, C = 8, 60, 4
    x = torch.randn(B, T, C)

    model = PatchTSTClassifier(seq_len=T, n_channels=C)
    print(f'PatchTST params : {model.count_params():,}')
    print(f'Output shape    : {model(x).shape}')

    baseline = LSTMClassifier(n_channels=C)
    print(f'LSTM params     : {baseline.count_params():,}')
    print(f'Output shape    : {baseline(x).shape}')
