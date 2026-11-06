import torch

class Config:
    # --- Reduced Capacity to Stop Overfitting ---
    D_MODEL = 64              # Reduced embedding size for word vectors
    NHEAD = 2                 # Reduced parallel attention heads
    NUM_ENCODER_LAYERS = 1    # Streamlined encoder depth
    NUM_DECODER_LAYERS = 1    # Streamlined decoder depth
    DIM_FEEDFORWARD = 128     # Tight linear hidden dimension
    DROPOUT = 0.0
    MAX_LEN = 20              # Max word count allowed during generation

    # --- Optimized Learning Bounds ---
    LR = 0.005                # Slightly faster learning rate for tiny vocabulary
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Vocabulary Token Indices ---
    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<sos>', '<eos>']
