import torch
import torch.nn as nn
import math
from config import Config

class PositionalEncoding(nn.Module):
    """Encodes positional context into word vectors using sinusoidal formulas."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class NepaliTransformer(nn.Module):
    """Sequence-to-Sequence Transformer Wrapper with correct dimension alignments."""
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, Config.D_MODEL)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, Config.D_MODEL)
        self.positional_encoding = PositionalEncoding(Config.D_MODEL)
        
        # Add normalization layers directly to keep character embeddings stable
        self.src_norm = nn.LayerNorm(Config.D_MODEL)
        self.tgt_norm = nn.LayerNorm(Config.D_MODEL)
        
        self.transformer = nn.Transformer(
            d_model=Config.D_MODEL, nhead=Config.NHEAD, 
            num_encoder_layers=Config.NUM_ENCODER_LAYERS, 
            num_decoder_layers=Config.NUM_DECODER_LAYERS, 
            dim_feedforward=Config.DIM_FEEDFORWARD, 
            dropout=Config.DROPOUT, batch_first=True
        )
        self.fc_out = nn.Linear(Config.D_MODEL, tgt_vocab_size)

    def _generate_causal_mask(self, sz, device):
        """Generates a proper Boolean causal mask where True blocks future tokens."""
        # Creates an upper triangular matrix of ones shifted by 1 diagonal step
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        # Convert directly to a Boolean matrix (1 becomes True, 0 becomes False)
        return mask.bool()


    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask):
        tgt_seq_len = tgt.size(1)
        # Generate target mask sequence
        tgt_mask = self._generate_causal_mask(tgt_seq_len, src.device)
        
        # Build embeddings and apply layer normalization to combat gradient explosions
        src_emb = self.src_norm(self.positional_encoding(self.src_embedding(src) * math.sqrt(Config.D_MODEL)))
        tgt_emb = self.tgt_norm(self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(Config.D_MODEL)))
        
        # Forward execution through core stack layers
        out = self.transformer(
            src_emb, tgt_emb, 
            tgt_mask=tgt_mask, # 2D square additive mask [tgt_seq_len, tgt_seq_len]
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.fc_out(out)
