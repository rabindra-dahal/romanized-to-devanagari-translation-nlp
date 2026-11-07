import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


# ----------------------------------------------------
# 1. SINUSOIDAL POSITIONAL ENCODING BLOCK
# ----------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Injects deterministic sine and cosine wave signals into token embeddings
    so the Transformer knows the relative and absolute order of tokens in a sequence.
    """
    # Initialize the sinusoidal positional encoding table up to max_len positions
    def __init__(self, d_model, max_len=5000):
        # Call the parent nn.Module constructor to initialize PyTorch module tracking
        super().__init__()
        # Create an empty matrix of zeros with shape [max_len, d_model] to hold positional signals
        pe = torch.zeros(max_len, d_model)

        # Create a column vector of token positions [0, 1, 2, ..., max_len - 1] with shape [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the geometric frequency decay factor: 1 / (10000^(2i / d_model)) for even indices [0, 2, 4, ...]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Fill all even embedding dimensions (0, 2, 4, ...) with sine waves of varying wavelengths
        pe[:, 0::2] = torch.sin(position * div_term)

        # Fill all odd embedding dimensions (1, 3, 5, ...) with cosine waves of varying wavelengths
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a leading batch dimension [1, max_len, d_model] and register 'pe' as a non-trainable buffer
        # (so it saves inside model.pt and moves across devices without being updated by the optimizer)
        self.register_buffer('pe', pe.unsqueeze(0))

    # Forward pass that adds positional signals directly onto the token embeddings
    def forward(self, x):
        # Slice 'pe' to match the incoming sequence length x.size(1) and add element-wise: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


# ----------------------------------------------------
# 2. BROADCAST-COMPATIBLE MULTI-HEAD ATTENTION BLOCK
# ----------------------------------------------------
class CustomMultiHeadAttention(nn.Module):
    """
    Implements Scaled Dot-Product Multi-Head Attention in pure PyTorch.
    Bypasses native C++ nn.Transformer thread deadlocks on Windows CPUs and supports
    both 2D causal masks and 4D source padding masks.
    """
    # Initialize the Multi-Head Attention module with embedding size, head count, and dropout rate
    def __init__(self, d_model, nhead, dropout=0.1):
        # Call the parent nn.Module constructor to register layers and parameters properly
        super().__init__()
        # Store total embedding dimensionality (e.g., 128)
        self.d_model = d_model
        # Store number of parallel attention heads (e.g., 4)
        self.nhead = nhead
        # Calculate dimension per individual head (e.g., 128 // 4 = 32)
        self.head_dim = d_model // nhead

        # Define learnable linear projection layer for Query (Q) vectors
        self.q_linear = nn.Linear(d_model, d_model)
        # Define learnable linear projection layer for Key (K) vectors
        self.k_linear = nn.Linear(d_model, d_model)
        # Define learnable linear projection layer for Value (V) vectors
        self.v_linear = nn.Linear(d_model, d_model)
        # Define final learnable linear layer to mix concatenated multi-head outputs
        self.out_linear = nn.Linear(d_model, d_model)
        # Define dropout layer to regularize attention probability weights during training
        self.dropout = nn.Dropout(dropout)

    # Forward pass computing scaled dot-product attention across all heads in parallel
    def forward(self, q, k, v, mask=None):
        # Extract the current batch size from the first dimension of Query tensor [batch_size, q_len, d_model]
        batch_size = q.size(0)

        # Project Q, split d_model into [nhead, head_dim], and swap axes to get [batch_size, nhead, q_len, head_dim]
        Q = self.q_linear(q).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        # Project K, split d_model into [nhead, head_dim], and swap axes to get [batch_size, nhead, k_len, head_dim]
        K = self.k_linear(k).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        # Project V, split d_model into [nhead, head_dim], and swap axes to get [batch_size, nhead, v_len, head_dim]
        V = self.v_linear(v).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # Compute raw dot-product similarity between Q and K^T, scaled by sqrt(head_dim) for gradient stability
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Check if a causal look-ahead mask or source padding mask was passed
        if mask is not None:
            # Expand a 2D causal mask [seq_len, seq_len] to 4D [1, 1, seq_len, seq_len] so it broadcasts across batch and heads
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            # Replace masked positions (where mask == 0) with -infinity so softmax turns them into 0.0 probability
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax along the key sequence axis (dim=-1) to convert raw scores into attention probabilities
        attn_weights = F.softmax(scores, dim=-1)
        # Randomly zero out some attention connections during training to prevent memorization
        attn_weights = self.dropout(attn_weights)

        # Multiply attention probabilities by Value vectors to get weighted context vectors [batch_size, nhead, q_len, head_dim]
        context = torch.matmul(attn_weights, V)
        # Swap axes back to [batch_size, q_len, nhead, head_dim], make memory contiguous, and merge heads into [batch_size, q_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # Pass the merged context tensor through the final linear projection layer and return
        return self.out_linear(context)

# ----------------------------------------------------
# 3. TRANSFORMER ENCODER & DECODER LAYERS
# ----------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """
    Single Encoder block consisting of:
    1. Multi-Head Self-Attention (with source padding mask) + Residual + LayerNorm
    2. Position-wise Feed-Forward Network + Residual + LayerNorm
    """
    # Initialize a single Transformer Encoder block with self-attention, feed-forward network, and layer normalization
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        # Call the parent nn.Module constructor to register layers and parameters properly
        super().__init__()
        # Multi-head self-attention mechanism where queries, keys, and values all come from the source sequence
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        # Position-wise feed-forward network expanding d_model -> dim_feedforward -> d_model with ReLU non-linearity
        self.feed_forward = nn.Sequential(
            # First linear layer expanding embedding dimension (e.g., 128 -> 256)
            nn.Linear(d_model, dim_feedforward),
            # ReLU activation function introducing non-linearity between linear projections
            nn.ReLU(),
            # Dropout layer inside the feed-forward block to prevent overfitting
            nn.Dropout(dropout),
            # Second linear layer projecting back down to original model dimension (e.g., 256 -> 128)
            nn.Linear(dim_feedforward, d_model)
        )
        # First LayerNorm normalizing activations after the self-attention residual connection
        self.norm1 = nn.LayerNorm(d_model)
        # Second LayerNorm normalizing activations after the feed-forward residual connection
        self.norm2 = nn.LayerNorm(d_model)
        # Residual dropout layer applied to sub-layer outputs before adding back to the input
        self.dropout = nn.Dropout(dropout)

    # Forward pass processing the source sequence tensor [batch_size, src_len, d_model]
    def forward(self, src, src_mask=None):
        # Compute self-attention across all source tokens while masking out <pad> positions via src_mask
        attn_out = self.self_attn(src, src, src, mask=src_mask)
        # Apply dropout, add residual skip connection (src + attn_out), and normalize across d_model
        x = self.norm1(src + self.dropout(attn_out))
        # Pass normalized attention output through the position-wise feed-forward network
        ff_out = self.feed_forward(x)
        # Apply dropout, add residual skip connection (x + ff_out), normalize, and return [batch_size, src_len, d_model]
        return self.norm2(x + self.dropout(ff_out))


class TransformerDecoderLayer(nn.Module):
    """
    Single Decoder block consisting of:
    1. Masked Multi-Head Self-Attention (prevents peeking at future target tokens)
    2. Multi-Head Cross-Attention (attends to encoded Romanized memory while ignoring <pad>)
    3. Position-wise Feed-Forward Network + Residual + LayerNorm
    """
    # Initialize a single Transformer Decoder block with masked self-attention, cross-attention, FFN, and layer norms
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        # Call the parent nn.Module constructor to register layers and parameters properly
        super().__init__()
        # Masked multi-head self-attention allowing target tokens to attend only to earlier target tokens
        self.self_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        # Multi-head cross-attention allowing target queries to attend to source encoder memory keys and values
        self.cross_attn = CustomMultiHeadAttention(d_model, nhead, dropout)
        # Position-wise feed-forward network expanding d_model -> dim_feedforward -> d_model with ReLU non-linearity
        self.feed_forward = nn.Sequential(
            # First linear layer expanding embedding dimension (e.g., 128 -> 256)
            nn.Linear(d_model, dim_feedforward),
            # ReLU activation function introducing non-linearity between projections
            nn.ReLU(),
            # Dropout layer inside the feed-forward block to regularize hidden activations
            nn.Dropout(dropout),
            # Second linear layer projecting back down to original model dimension (e.g., 256 -> 128)
            nn.Linear(dim_feedforward, d_model)
        )
        # First LayerNorm normalizing activations after the masked self-attention residual connection
        self.norm1 = nn.LayerNorm(d_model)
        # Second LayerNorm normalizing activations after the encoder-decoder cross-attention residual connection
        self.norm2 = nn.LayerNorm(d_model)
        # Third LayerNorm normalizing activations after the feed-forward residual connection
        self.norm3 = nn.LayerNorm(d_model)
        # Residual dropout layer applied to each sub-layer output before adding back to the skip connection
        self.dropout = nn.Dropout(dropout)

    # Forward pass processing target embeddings [batch, tgt_len, d_model] conditioned on encoder memory [batch, src_len, d_model]
    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        # 1. Compute causal self-attention over target tokens so far (tgt_mask blocks future positions)
        self_attn_out = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        # Apply dropout, add residual skip connection (tgt + self_attn_out), and normalize
        x = self.norm1(tgt + self.dropout(self_attn_out))

        # 2. Compute cross-attention using target states (x) as Queries and encoder memory as Keys/Values (ignoring <pad> via src_mask)
        cross_attn_out = self.cross_attn(x, memory, memory, mask=src_mask)
        # Apply dropout, add residual skip connection (x + cross_attn_out), and normalize
        x = self.norm2(x + self.dropout(cross_attn_out))

        # 3. Pass combined representations through the position-wise feed-forward network
        ff_out = self.feed_forward(x)
        # Apply dropout, add final residual skip connection (x + ff_out), normalize, and return [batch, tgt_len, d_model]
        return self.norm3(x + self.dropout(ff_out))


# ----------------------------------------------------
# 4. 2-LAYER SEQ2SEQ TRANSFORMER WRAPPER
# ----------------------------------------------------
class NepaliTransformer(nn.Module):
    """
    Complete Sequence-to-Sequence Transformer for Romanized-to-Devanagari translation.
    Exposes standalone encode() and decode() methods so inference only encodes the
    source sentence once before running autoregressive decoding steps.
    """
    # Initialize the complete Sequence-to-Sequence Transformer with source and target vocabulary sizes
    def __init__(self, src_vocab_size, tgt_vocab_size):
        # Call the parent nn.Module constructor to register submodules and parameters
        super().__init__()
        # Embedding lookup table converting source Romanized token IDs into dense vectors [batch, src_len, d_model]
        self.src_embedding = nn.Embedding(src_vocab_size, Config.D_MODEL)
        # Embedding lookup table converting target Devanagari token IDs into dense vectors [batch, tgt_len, d_model]
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, Config.D_MODEL)
        # Sinusoidal positional encoder to inject token order information into both source and target embeddings
        self.positional_encoding = PositionalEncoding(Config.D_MODEL)
        # Dropout layer applied after adding positional encodings to regularize input representations
        self.emb_dropout = nn.Dropout(Config.DROPOUT)

        # Create a stack of N identical Encoder layers (e.g., 2 layers) stored in an iterable ModuleList
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(Config.D_MODEL, Config.NHEAD, Config.DIM_FEEDFORWARD, Config.DROPOUT)
            for _ in range(Config.NUM_ENCODER_LAYERS)
        ])

        # Create a stack of N identical Decoder layers (e.g., 2 layers) stored in an iterable ModuleList
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(Config.D_MODEL, Config.NHEAD, Config.DIM_FEEDFORWARD, Config.DROPOUT)
            for _ in range(Config.NUM_DECODER_LAYERS)
        ])

        # Final linear classifier projecting decoder hidden states [batch, tgt_len, d_model] to vocabulary logits [batch, tgt_len, tgt_vocab_size]
        self.fc_out = nn.Linear(Config.D_MODEL, tgt_vocab_size)

    # Generate a lower-triangular causal mask [sz, sz] so each target token can only attend to itself and earlier tokens
    def _generate_causal_mask(self, sz, device):
        # Return a square matrix with 1s on and below the main diagonal, and 0s above (blocking future tokens)
        return torch.tril(torch.ones(sz, sz, device=device))

    # Generate a 4D source padding mask [batch_size, 1, 1, src_len] to hide <pad> tokens from attention heads
    def _generate_src_mask(self, src):
        # Create boolean/int mask where real tokens = True (1) and PAD_IDX = False (0), expanded for multi-head broadcasting
        return (src != Config.PAD_IDX).unsqueeze(1).unsqueeze(2)

    # Encode the source Romanized token IDs into contextual memory representations (run once during inference)
    def encode(self, src):
        # Build the 4D source padding mask matching the current input batch
        src_mask = self._generate_src_mask(src)
        # Embed source tokens, scale by sqrt(d_model) for variance stability, add positional waves, and apply dropout
        memory = self.emb_dropout(
            self.positional_encoding(self.src_embedding(src) * math.sqrt(Config.D_MODEL))
        )
        # Pass the sequence sequentially through every stacked Encoder layer while ignoring <pad> positions
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask=src_mask)
        # Return the final encoded context tensor [batch, src_len, d_model] and its padding mask
        return memory, src_mask

    # Decode the target Devanagari sequence conditioned on encoder memory and causal look-ahead masks
    def decode(self, tgt, memory, src_mask):
        # Build the 2D lower-triangular causal mask matching the current target sequence length
        tgt_mask = self._generate_causal_mask(tgt.size(1), tgt.device)
        # Embed target tokens, scale by sqrt(d_model), add positional waves, and apply dropout
        out = self.emb_dropout(
            self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(Config.D_MODEL))
        )
        # Pass target representations sequentially through every stacked Decoder layer (self-attention + cross-attention + FFN)
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask=tgt_mask, src_mask=src_mask)
        # Return the final decoded hidden states [batch, tgt_len, d_model]
        return out

    # Full training forward pass combining encoding, decoding, and vocabulary projection
    def forward(self, src, tgt):
        # Encode the source Romanized batch into contextual memory and get its padding mask
        memory, src_mask = self.encode(src)
        # Decode the shifted target batch using the encoded memory and causal mask
        out = self.decode(tgt, memory, src_mask)
        # Project decoder hidden states into raw unnormalized vocabulary scores (logits) and return
        return self.fc_out(out)
