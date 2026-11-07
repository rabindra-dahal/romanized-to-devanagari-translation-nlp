import torch


class Config:
    """
    Centralized configuration for the 2-layer Nepali Sequence-to-Sequence Transformer.
    Stores model dimensions, sequence limits, device routing, and special token IDs.
    """

    # --- Transformer Architecture Dimensions ---
    D_MODEL = 128               # Dimensionality of token embeddings and hidden states
    NHEAD = 4                   # Number of parallel self/cross-attention heads (128 / 4 = 32 dims per head)
    NUM_ENCODER_LAYERS = 2      # Number of stacked Transformer encoder blocks
    NUM_DECODER_LAYERS = 2      # Number of stacked Transformer decoder blocks
    DIM_FEEDFORWARD = 256       # Inner hidden dimension of position-wise feed-forward networks
    DROPOUT = 0.05              # Light dropout rate to prevent overfitting while allowing 2-layer convergence

    # --- Sequence & Optimization Settings ---
    MAX_LEN = 36                # Maximum allowed token sequence length during encoding/decoding
    LR = 0.0015                 # Default learning rate for the Adam optimizer
    DEVICE = torch.device('cpu')  # Explicit CPU routing for stable execution on Windows laptops

    # --- Special Vocabulary Token IDs (dynamically synced by SubwordVocabulary) ---
    PAD_IDX = 0                 # Padding token ID used to equalize sequence lengths in a batch
    SOS_IDX = 1                 # Start-Of-Sequence (<sos>) token ID to trigger autoregressive decoding
    EOS_IDX = 2                 # End-Of-Sequence (<eos>) token ID to signal translation completion
    UNK_IDX = 3                 # Unknown (<unk>) token ID for out-of-vocabulary characters
