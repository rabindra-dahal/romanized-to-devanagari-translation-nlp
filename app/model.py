import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

# ----------------------------------------------------
# 1. SINUSOIDAL POSITIONAL ENCODING BLOCK
# ----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

# ----------------------------------------------------
# 2. BROADCAST-COMPATIBLE MULTI-HEAD ATTENTION BLOCK
# ----------------------------------------------------
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) # Regularization safeguard
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projection steps and split into multiple parallel heads
        Q = self.q_linear(q).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Calculate scaled attention energy cross-matrix metrics
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # FIX: Convert a 2D look-ahead matrix [seq, seq] to 4D [1, 1, seq, seq]
            # to safely broadcast over multi-head dimensions [batch, heads, seq, seq]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            # Mask out positions where mask coordinate values evaluate to 0
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        # Apply dropout directly to weight matrix maps to kill memorization shortcuts
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        # Re-assemble multi-head structural matrices back into unified d_model layout
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)

# ----------------------------------------------------
# 3. TRANSFORMER STRUCTURE MODULE LAYOUTS
# ----------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        attn_out = self.self_attn(src, src, src)
        x = self.norm1(src + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_out))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None):
        self_attn_out = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        x = self.norm1(tgt + self.dropout(self_attn_out))
        
        # Cross-attention isolates encoder state matrices smoothly
        cross_attn_out = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        ff_out = self.feed_forward(x)
        return self.norm3(x + self.dropout(ff_out))

# ----------------------------------------------------
# 4. FINAL SEQUENCE-TO-SEQUENCE SYSTEM WRAPPER
# ----------------------------------------------------
class NepaliTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, Config.D_MODEL)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, Config.D_MODEL)
        self.positional_encoding = PositionalEncoding(Config.D_MODEL)
        
        self.encoder_layer = TransformerEncoderLayer(Config.D_MODEL, Config.NHEAD, Config.DIM_FEEDFORWARD, Config.DROPOUT)
        self.decoder_layer = TransformerDecoderLayer(Config.D_MODEL, Config.NHEAD, Config.DIM_FEEDFORWARD, Config.DROPOUT)
        self.fc_out = nn.Linear(Config.D_MODEL, tgt_vocab_size)

    def _generate_causal_mask(self, sz, device):
        # Build strict lower triangular matrix (1 see, 0 ignore)
        return torch.tril(torch.ones(sz, sz, device=device))

    def forward(self, src, tgt):
        tgt_seq_len = tgt.size(1)
        tgt_mask = self._generate_causal_mask(tgt_seq_len, src.device)
        
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(Config.D_MODEL))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(Config.D_MODEL))
        
        memory = self.encoder_layer(src_emb)
        out = self.decoder_layer(tgt_emb, memory, tgt_mask=tgt_mask)
        
        return self.fc_out(out)
