import torch
import torch.nn as nn
import math
from config import Config

class PositionalEncoding(nn.Module):
    """Encodes positional context into the word vectors using sinusoidal formulas."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Instantiate a blank multi-dimensional matrix matching standard limits
        pe = torch.zeros(max_len, d_model)
        # Construct vertical layout vector capturing exact indices [0, 1, 2...max_len]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate log-scale steps for multi-frequency sine and cosine patterns
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Populate alternate columns with matching out-of-phase cyclical wave forms
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Freeze internal array state so parameter updating optimizations bypass this matrix
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Element-wise addition of spatial coordinates to sequence vectors
        return x + self.pe[:, :x.size(1)]

class NepaliTransformer(nn.Module):
    """Sequence-to-Sequence Transformer Wrapper built on top of native torch layers."""
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        # Convert Romanized integer tokens to continuous high-dimensional vector representations
        self.src_embedding = nn.Embedding(src_vocab_size, Config.D_MODEL)
        # Convert Devanagari integer tokens to continuous high-dimensional vector representations
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, Config.D_MODEL)
        # Instance spatial sequence tracker class
        self.positional_encoding = PositionalEncoding(Config.D_MODEL)
        
        # Core standard PyTorch Sequence-to-Sequence processing block
        self.transformer = nn.Transformer(
            d_model=Config.D_MODEL, nhead=Config.NHEAD, 
            num_encoder_layers=Config.NUM_ENCODER_LAYERS, 
            num_decoder_layers=Config.NUM_DECODER_LAYERS, 
            dim_feedforward=Config.DIM_FEEDFORWARD, 
            dropout=Config.DROPOUT, batch_first=True
        )
        # Linear layer mapping final attention spaces directly into raw vocabulary indexes
        self.fc_out = nn.Linear(Config.D_MODEL, tgt_vocab_size)

    def _generate_causal_mask(self, sz, device):
    	"""Builds a Boolean attention mask where True means BLOCKED (hidden future tokens)."""
    	# Generates a lower triangular matrix of ones, then flips it to get an upper triangular matrix
    	# where 1 indicates future tokens.
    	mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    	# Convert it directly to a Boolean tensor (True blocks, False allows)
    	return mask.bool()


    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask):
        # Generate the causal attention mask to prevent illegal future lookups during decoding
        tgt_mask = self._generate_causal_mask(tgt.size(1), src.device)
        
        # Convert characters to dense arrays and scale them to keep variances stable
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(Config.D_MODEL))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(Config.D_MODEL))
        
        # Send raw inputs into the stacked layer engine blocks
        out = self.transformer(
            src_emb, tgt_emb, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        # Project output vectors into probability distributions over the target vocabulary
        return self.fc_out(out)
