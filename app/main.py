import warnings
# Suppress the PyTorch prototype warning immediately before loading torch core
warnings.filterwarnings("ignore", category=UserWarning, message=".*nested tensors is in prototype stage.*")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import Config
from vocabulary import CharVocabulary, build_batches
from model import NepaliTransformer
from pipeline import train_one_epoch, evaluate_translation

class ParallelTextDataset(Dataset):
    """Custom Dataset wrapper to pass clean item indexes to the DataLoader."""
    def __init__(self, src_sentences, tgt_sentences):
        self.src = src_sentences
        self.tgt = tgt_sentences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def collate_fn(batch, src_vocab, tgt_vocab):
    """Dynamically structures strings from the DataLoader into padded mini-batches."""
    src_list, tgt_list = zip(*batch)
    # Reuse your existing build_batches function to create masks for this mini-batch only
    return build_batches(src_list, tgt_list, src_vocab, tgt_vocab)

def main():
    # --- 1. Expanded Structural Parallel Corpus ---
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
        'म खेल्न चाहन्छु', 'म पढ्न चाहन्छु', 'म करना चाहन्छु',
        'मलाई', 'मलाई डर लागेको छ', 'मलाई thakai लागेको छ',
        'डर लागेको छ', 'thakai लागेको छ'
    ]

    # Initialize complete character alphabets
    src_vocab = CharVocabulary(is_target=False)
    tgt_vocab = CharVocabulary(is_target=True)
    
    # --- 2. PyTorch DataLoader Pipeline ---
    dataset = ParallelTextDataset(src_data, tgt_data)
    # Using a small batch size forces the model to constantly adjust to shifting context layouts
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab)
    )

    # --- 3. Model Architecture Construction ---
    model = NepaliTransformer(len(src_vocab), len(tgt_vocab)).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # Since it's seeing smaller, moving batches, we run for 400 epochs to establish stable convergence
    epochs = 400

    # --- 4. Training Execution ---
    print(f"Beginning training on device: {Config.DEVICE}")
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer)
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | Average MiniBatch Loss: {avg_loss:.4f}")

    # --- 5. Validation Inference Testing ---
    print("\n--- Running Inference Engine (Testing Generalization) ---")
    test_phrases = [
        'ma kalo manxe hu',      # Seen during training
        'malai dar lageko xa',   # Seen during training
        'kalo manxe hu',         # UNSEEN combination - verifies feature synthesis
        'ma padhna chahanxu'     # Structural layout check
    ]
    
    for text in test_phrases:
        prediction = evaluate_translation(model, text, src_vocab, tgt_vocab)
        print(f"Romanized Input : '{text}'")
        print(f"Devanagari Output: '{prediction}'\n")

if __name__ == '__main__':
    main()
