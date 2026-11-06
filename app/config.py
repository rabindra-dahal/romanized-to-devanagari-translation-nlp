import torch

class Config:
    # --- Architecture Dimensions ---
    D_MODEL = 256             # Hidden state size (dimensionality of token embeddings)
    NHEAD = 4                 # Number of Parallel Attention Heads in multi-head attention
    NUM_ENCODER_LAYERS = 3    # Number of stacked TransformerEncoder layers
    NUM_DECODER_LAYERS = 3    # Number of stacked TransformerDecoder layers
    DIM_FEEDFORWARD = 512     # Size of the hidden linear layer within feed-forward blocks
    DROPOUT = 0.1             # Regularization probability to randomly deactivate nodes
    MAX_LEN = 100             # Maximum sentence token limit allowed during prediction

    # --- Training Configuration ---
    LR = 0.001                # Optimization step multiplier (Adam Learning Rate)
    EPOCHS = 100              # Total complete passes through the dataset
    BATCH_SIZE = 2            # Number of parallel sample strings processed per step
    # Select hardware accelerator if available; fallback safely to CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Vocabulary Token Indices ---
    PAD_IDX = 0               # Index used to fill empty space in unequal sentences
    UNK_IDX = 1               # Index used when text contains an out-of-vocabulary item
    SOS_IDX = 2               # Index marking the literal absolute start of a sequence
    EOS_IDX = 3               # Index telling the system a sentence has finished
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<sos>', '<eos>'] # Literal vocabulary strings
