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
from datasets import load_dataset  # Reverted cleanly back to native HF data API
from config import Config
from vocabulary import SubwordVocabulary
from model import NepaliTransformer
from pipeline import train_one_epoch, evaluate_translation

class Compact20kDataset(Dataset):
    """Maps dataset rows cleanly from the official compact 20k Hugging Face structure."""
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Direct key access: 'romanized' for input Latin script, 'nepali' for target Devanagari
        return str(item['romanized']), str(item['nepali'])

def collate_fn(batch, vocab):
    from torch.nn.utils.rnn import pad_sequence
    src_list, tgt_list = zip(*batch)
    
    src_tensors = [torch.tensor(vocab.encode(s)) for s in src_list]
    tgt_tensors = [torch.tensor(vocab.encode(t)) for t in tgt_list]

    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    
    return src_batch, tgt_batch

def main():
    print("Loading compact 20K English-Nepali-Romanized parallel corpus natively from HF...")
    # Load dataset natively using the updated layout structure
    raw_dataset = load_dataset("rabindra-dahal/english-nepali-romanized-word-csv-compact-20k", split="train")
    print(f"[+] Dataset active! Total parallel lines loaded: {len(raw_dataset)}")

    # Extract text strings across both columns to build our lean 1,500-token custom subword map
    all_text_corpus = [str(row['romanized']) for row in raw_dataset] + [str(row['nepali']) for row in raw_dataset]
    
    print("Training custom local subword tokenizer...")
    vocab = SubwordVocabulary(all_text_corpus)
    print(f"Tokenizer optimized! Tight vocabulary size: {len(vocab)} items.")
    
    dataset = Compact20kDataset(raw_dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, # Balanced execution batch stride for Dell Latitude CPU calculations
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, vocab),
        drop_last=True
    )

    # Balanced hyperparameters tailored for CPU-bound attention calculation bounds
    Config.D_MODEL = 128
    Config.NHEAD = 4
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 256
    Config.MAX_LEN = 50
    epochs = 40

    model = NepaliTransformer(len(vocab), len(vocab)).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- DYNAMIC APP DIRECTORY PATH BINDING ---
    app_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(app_dir, "model.pt")

    # --- AUTOMATED WEIGHT DETECT ROUTINE ---
    if os.path.exists(checkpoint_path):
        print(f"\n[!] Pre-existing checkpoint found at '{checkpoint_path}'. Loading directly to inference engine...")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"\nBeginning training on device: {Config.DEVICE}")
        print("Epoch Training started...")
        total_start_time = time.time()
        interval_start_time = time.time()

        for epoch in range(1, epochs + 1):
            avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, epoch)
            
            if epoch % 5 == 0:
                interval_elapsed = time.time() - interval_start_time
                time_str = f"{interval_elapsed / 60:.1f} min" if interval_elapsed >= 60 else f"{interval_elapsed:.1f} sec"
                print(f"Epoch {epoch:02d}/{epochs} | CrossEntropyLoss: {avg_loss:.4f} | Time Taken: {time_str}")
                interval_start_time = time.time()

        total_elapsed = time.time() - total_start_time
        print(f"\nTraining Complete! Total Session Duration: {total_elapsed / 60:.1f} min")

        print(f"--> Exporting state dictionaries safely to '{checkpoint_path}'...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print("[+] Checkpoint file written successfully!")

    # --- 4. Running Inference Engine (Testing True Data Matches) ---
    print("\n--- Running Inference Engine (Testing Real Content Inside CSV Dataset) ---")
    
    # Extract 3 exact samples right out of your dataset memory lanes to check convergence
    sample_phrases = [
        "Kritrim buddhimattale aadhunik digital pranaliharulai teebra roopma roopantaran gardaichha.",
        "Cloud computingle tadhabaata kaam garna ra data pahunch garna sahaj banauchha.",
        "Cyber suraksha protocolle sambedansheel byaktigat datalai hackersbaata jogaauna maddat garcha."
    ]
    
    for text in sample_phrases:
        # Since model.pt was successfully written, this runs instantly via your loaded checkpoint weights!
        prediction = evaluate_translation(model, text, vocab, vocab)
        print(f"Romanized Input : '{text}'")
        print(f"Devanagari Output: '{prediction}'\n")

if __name__ == '__main__':
    main()
