import os
import warnings
import time

# 1. Prevent Hugging Face Hub from triggering administrative symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# 2. Suppress internal PyTorch prototype and canopy formatting messages
warnings.filterwarnings("ignore", category=UserWarning, message=".*nested tensors is in prototype stage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Support for mismatched key_padding_mask.*")
warnings.filterwarnings("ignore", category=UserWarning)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from config import Config
from vocabulary import SubwordVocabulary
from model import NepaliTransformer
from pipeline import train_one_epoch, evaluate_translation
import random

class KidLearningDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def build_augmented_pairs(hf_dataset):
    """Combines original pairs with synthetic noun/subject swaps to break template memorization."""
    base_pairs = [(str(row['romanized']), str(row['nepali'])) for row in hf_dataset]
    
    swaps = [
        (("Meri bahini", "मेरी बहिनी"), ("Meri aama", "मेरी आमा")),
        (("Meri aama", "मेरी आमा"), ("Meri didi", "मेरी दिदी")),
        (("naya chamcha", "नयाँ चम्चा"), ("chamcha", "चम्चा")),
        (("naya gainda", "नयाँ गैंडा"), ("gainda", "गैंडा")),
        (("Keti", "केटी"), ("Keta", "केटा")),
    ]
    
    augmented = list(base_pairs)
    random.seed(42)
    for rom, nep in base_pairs:
        for (r_old, n_old), (r_new, n_new) in swaps:
            if r_old in rom and n_old in nep and random.random() < 0.35:
                augmented.append((rom.replace(r_old, r_new, 1), nep.replace(n_old, n_new, 1)))
                
    return augmented

# Custom DataLoader collate function that converts a list of (romanized, nepali) pairs into equal-length padded tensors
def collate_fn(batch, vocab):
    # Import PyTorch's pad_sequence utility to pad variable-length 1D tensors to the longest sequence in the batch
    from torch.nn.utils.rnn import pad_sequence

    # Unzip the list of (source, target) tuples into two separate tuples: all Romanized strings and all Nepali strings
    src_list, tgt_list = zip(*batch)

    # Encode each Romanized source string into subword token IDs ([<sos>, ..., <eos>]) and wrap as a 1D PyTorch tensor
    src_tensors = [torch.tensor(vocab.encode(s)) for s in src_list]

    # Encode each Devanagari target string into subword token IDs ([<sos>, ..., <eos>]) and wrap as a 1D PyTorch tensor
    tgt_tensors = [torch.tensor(vocab.encode(t)) for t in tgt_list]

    # Pad all source tensors in the batch with PAD_IDX (0) to match the longest source sentence: [batch_size, max_src_len]
    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)

    # Pad all target tensors in the batch with PAD_IDX (0) to match the longest target sentence: [batch_size, max_tgt_len]
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)

    # Return the aligned 2D source and target batch tensors ready for the Transformer forward pass
    return src_batch, tgt_batch


def normalize_chat_input(text):
    slang_map = {
        "kto": "keta",
        "keto": "keta",
        "kti": "keti",
        "xa": "chha",
        "xaina": "chhaina",
        "xu": "chhu",
        "xau": "chhau",
        "xan": "chhan",
        "manxe": "manchhe"
    }
    words = text.strip().split()
    normalized_words = [slang_map.get(w.lower(), w) for w in words]
    clean_text = " ".join(normalized_words)
    if clean_text:
        clean_text = clean_text[0].upper() + clean_text[1:]
    if clean_text and clean_text[-1] not in {".", "?", "!"}:
        clean_text += "."
    return clean_text

def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(app_dir, "model.pt")
    tokenizer_path = os.path.join(app_dir, "tokenizer.json")

    # Tuned hyperparameters for 2-layer padded convergence
    Config.D_MODEL = 128
    Config.NHEAD = 4
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 256
    Config.DROPOUT = 0.05
    Config.MAX_LEN = 36
    epochs = 30

    raw_dataset = None
    if os.path.exists(tokenizer_path):
        print(f"[!] Loading saved tokenizer from '{tokenizer_path}' (Offline Mode)...")
        vocab = SubwordVocabulary.from_file(tokenizer_path)
    else:
        print("Loading Kid-Learning parallel corpus from HF to build syllable tokenizer...")
        raw_dataset = load_dataset("rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning", split="train")
        all_text_corpus = [str(row['romanized']) for row in raw_dataset] + [str(row['nepali']) for row in raw_dataset]
        
        vocab = SubwordVocabulary(all_text_corpus)
        vocab.save(tokenizer_path)
        print(f"[+] Tokenizer saved to '{tokenizer_path}' (Size: {len(vocab)} syllables).")

    model = NepaliTransformer(len(vocab), len(vocab)).to(Config.DEVICE)

    if os.path.exists(checkpoint_path):
        print(f"[!] Loading saved model weights from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        if raw_dataset is None:
            raw_dataset = load_dataset("rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning", split="train")
            
        pairs = build_augmented_pairs(raw_dataset)
        print(f"[+] Training set prepared with template-breaking augmentation: {len(pairs)} pairs.")
        
        dataset = KidLearningDataset(pairs)
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True, 
            collate_fn=lambda b: collate_fn(b, vocab),
            drop_last=True
        )

        # Lower label smoothing (0.05) + slightly higher lr (0.0015) for sharp 2-layer alignment
        criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX, label_smoothing=0.05)
        optimizer = optim.Adam(model.parameters(), lr=0.0015)

        print(f"\nBeginning 2-layer padded training on device: {Config.DEVICE}")
        total_start_time = time.time()
        interval_start_time = time.time()

        for epoch in range(1, epochs + 1):
            avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epochs=epochs)
            
            if epoch % 5 == 0:
                interval_elapsed = time.time() - interval_start_time
                time_str = f"{interval_elapsed / 60:.1f} min" if interval_elapsed >= 60 else f"{interval_elapsed:.1f} sec"
                print(f"Epoch {epoch:02d}/{epochs} | CrossEntropyLoss: {avg_loss:.4f} | Time Taken: {time_str}")
                interval_start_time = time.time()

        total_elapsed = time.time() - total_start_time
        print(f"\nTraining Complete! Total Session Duration: {total_elapsed / 60:.1f} min")

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"[+] Checkpoint saved to '{checkpoint_path}'!")

    print("\n--- Quick Sanity Check ---")
    sample_phrases = [
        "Uni ahile chamcha man parauchhin.",
        "Meri aama chamcha man parauchhin.",
        "Kto naya gainda herchha."
    ]
    for text in sample_phrases:
        formatted = normalize_chat_input(text)
        prediction = evaluate_translation(model, formatted, vocab, vocab)
        print(f"Romanized : {text}")
        print(f"Devanagari: {prediction}\n")

    print("=" * 55)
    print("LIVE ROMANIZED-TO-DEVANAGARI TRANSLATOR (OFFLINE)")
    print("Type a Romanized Nepali sentence and press Enter.")
    print("Type 'exit' or 'quit' to close.")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nRomanized > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Closing translator. Goodbye!")
                break
            formatted_input = normalize_chat_input(user_input)
            translation = evaluate_translation(model, formatted_input, vocab, vocab)
            print(f"Devanagari > {translation}")
        except (KeyboardInterrupt, EOFError):
            print("\nClosing translator. Goodbye!")
            break

if __name__ == '__main__':
    main()
