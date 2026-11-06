import torch

class Config:
    # --- Structural Layout Dimensions ---
    D_MODEL = 128             # Embedding state size
    NHEAD = 4                 # Parallel attention heads
    NUM_ENCODER_LAYERS = 2    # Balanced layer tracking bounds
    NUM_DECODER_LAYERS = 2    
    DIM_FEEDFORWARD = 256     
    DROPOUT = 0.1             
    
    # --- Tightened Sequence Bound (Eliminates CPU Overhead) ---
    MAX_LEN = 24              # Reduced from 50 to 24 to dramatically speed up CPU training
    LR = 0.001                
    DEVICE = torch.device('cpu') # Explicit CPU routing for platform safety
    
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
