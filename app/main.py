import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*nested tensors is in prototype stage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Support for mismatched key_padding_mask.*")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import Config
from vocabulary import WordVocabulary, build_batches
from model import NepaliTransformer
from pipeline import train_one_epoch, evaluate_translation

class ParallelTextDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences):
        self.src = src_sentences
        self.tgt = tgt_sentences
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def collate_fn(batch, src_vocab, tgt_vocab):
    src_list, tgt_list = zip(*batch)
    return build_batches(src_list, tgt_list, src_vocab, tgt_vocab)

def main():
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

    # Initialize word-level lookups built specifically from our translation dataset corpus
    src_vocab = WordVocabulary(src_data)
    tgt_vocab = WordVocabulary(tgt_data)
    
    dataset = ParallelTextDataset(src_data, tgt_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab)
    )

    model = NepaliTransformer(len(src_vocab), len(tgt_vocab)).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    epochs = 200

    print(f"Beginning training on device: {Config.DEVICE}")
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer)
        if epoch % 40 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | Average MiniBatch Loss: {avg_loss:.4f}")

    print("\n--- Running Inference Engine (Testing Generalization) ---")
    test_phrases = [
        'ma kalo manxe hu',
        'malai dar lageko xa',
        'kalo manxe hu',         # UNSEEN combination!
        'ma padhna chahanxu'
    ]
    
    for text in test_phrases:
        prediction = evaluate_translation(model, text, src_vocab, tgt_vocab)
        print(f"Romanized Input : '{text}'")
        print(f"Devanagari Output: '{prediction}'\n")

if __name__ == '__main__':
    main()
