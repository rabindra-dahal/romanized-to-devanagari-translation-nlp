import warnings
# Suppress the PyTorch prototype warning immediately before loading torch core
warnings.filterwarnings("ignore", category=UserWarning, message=".*nested tensors is in prototype stage.*")

import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from vocabulary import CharVocabulary, build_batches
from model import NepaliTransformer
from pipeline import train_one_epoch, evaluate_translation

def main():
    # --- 1. Robust Parallel Corpus ---
    # We expand the dataset so the model learns that 'ma' maps to 'म', 'manxe' maps to 'मान्छे', etc.
    src_data = [
        'ma', 'ma hu', 'manxe', 'manxe hu',
        'ma ramro manxe hu', 'ma kalo manxe hu', 'ma naramro manxe hu',
        'kalo manxe', 'ramro manxe',
        'ma khelna chahanxu', 'ma padhna chahanxu', 'ma garna chahanxu',
        'malai', 'malai dar lageko xa', 'malai thakai lageko xa',
        'dar lageko xa', 'thakai lageko xa'
    ]
    
    tgt_data = [
        'म', 'म हुँ', 'मान्छे', 'मान्छे हुँ',
        'म राम्रो मान्छे हुँ', 'म कालो मान्छे हुँ', 'म नराम्रो मान्छे हुँ',
        'कालो मान्छे', 'राम्रो मान्छे',
        'म खेल्न चाहन्छु', 'म पढ्न चाहन्छु', 'म गर्न चाहन्छु',
        'मलाई', 'मलाई डर लागेको छ', 'मलाई थकाइ लागेको छ',
        'डर लागेको छ', 'थकाइ लागेको छ'
    ]

    # Initialize generalized alphabetic vocabularies
    src_vocab = CharVocabulary(is_target=False)
    tgt_vocab = CharVocabulary(is_target=True)
    
    # --- 2. Data Preparation ---
    src_batch, tgt_batch, src_mask, tgt_mask = build_batches(src_data, tgt_data, src_vocab, tgt_vocab)
    src_batch, tgt_batch = src_batch.to(Config.DEVICE), tgt_batch.to(Config.DEVICE)
    src_mask, tgt_mask = src_mask.to(Config.DEVICE), tgt_mask.to(Config.DEVICE)

    # --- 3. Model Initialization ---
    model = NepaliTransformer(len(src_vocab), len(tgt_vocab)).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    
    # We bump epochs to 300 so it can fully digest the newly added grammatical patterns
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    epochs = 300

    # --- 4. Training ---
    print(f"Beginning training on device: {Config.DEVICE}")
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, src_batch, tgt_batch, src_mask, tgt_mask, criterion, optimizer)
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | CrossEntropyLoss: {loss:.4f}")

    # --- 5. True Inference Testing (Unseen Combinations) ---
    print("\n--- Running Inference Engine (Testing Generalization) ---")
    
    # We feed it both expressions it has seen, and new modular arrangements
    test_phrases = [
        'ma kalo manxe hu',      # Seen during training
        'malai dar lageko xa',   # Seen during training (spelled correctly based on data)
        'kalo manxe hu',         # UNSEEN combination! Assesses if it can synthesize fragments
        'ma padhna chahanxu'     # Structural pattern verification
    ]
    
    for text in test_phrases:
        prediction = evaluate_translation(model, text, src_vocab, tgt_vocab)
        print(f"Romanized Input : '{text}'")
        print(f"Devanagari Output: '{prediction}'\n")

if __name__ == '__main__':
    main()
