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

class KidLearningDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return str(item['romanized']), str(item['nepali'])

def collate_fn(batch, vocab):
    from torch.nn.utils.rnn import pad_sequence
    src_list, tgt_list = zip(*batch)
    src_tensors = [torch.tensor(vocab.encode(s)) for s in src_list]
    tgt_tensors = [torch.tensor(vocab.encode(t)) for t in tgt_list]
    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    return src_batch, tgt_batch

def normalize_chat_input(text):
    """Normalizes casual chat typing to match the dataset's exact training format."""
    slang_map = {
        "kto": "keto",
        "kti": "keti",
        "x": "chh",
        "xa": "chha",
        "xaina": "chhaina",
        "xu": "chhu",
        "xau": "chhau",
        "xan": "chhan",
        "manxe": "manchhe"
    }
    words = text.strip().split()
    normalized_words = []
    for w in words:
        lower_w = w.lower()
        normalized_words.append(slang_map.get(lower_w, w))
        
    clean_text = " ".join(normalized_words)
    if clean_text:
        clean_text = clean_text.upper() + clean_text[1:]
    if clean_text and clean_text[-1] not in {".", "?", "!"}:
        clean_text += "."
    return clean_text

def main():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(app_dir, "model.pt")
    tokenizer_path = os.path.join(app_dir, "tokenizer.json")

    Config.D_MODEL = 128
    Config.NHEAD = 4
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 256
    Config.MAX_LEN = 36  # Slightly larger to hold syllable sequences cleanly
    epochs = 15          # Fewer epochs prevents sentence template memorization

    # --- 1. LOAD OR BUILD TOKENIZER ---
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

    # --- 2. INITIALIZE MODEL ---
    model = NepaliTransformer(len(vocab), len(vocab)).to(Config.DEVICE)

    # --- 3. LOAD OR TRAIN MODEL WEIGHTS ---
    if os.path.exists(checkpoint_path):
        print(f"[!] Loading saved model weights from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        if raw_dataset is None:
            raw_dataset = load_dataset("rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning", split="train")
            
        dataset = KidLearningDataset(raw_dataset)
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True, 
            collate_fn=lambda b: collate_fn(b, vocab),
            drop_last=True
        )

        # Label smoothing stops the network from becoming overconfident on exact templates
        criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX, label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"\nBeginning training on device: {Config.DEVICE}")
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

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"[+] Checkpoint saved to '{checkpoint_path}'!")

    # --- 4. SANITY CHECK SAMPLES ---
    print("\n--- Quick Sanity Check ---")
    sample_phrases = [
        "Uni ahile chamcha man parauchhin.",
        "Meri aama sanjha aanp khanuhunchha.",
        "Keti naya gainda herchhin."
    ]
    for text in sample_phrases:
        prediction = evaluate_translation(model, text, vocab, vocab)
        print(f"Romanized : {text}")
        print(f"Devanagari: {prediction}\n")

    # --- 5. INTERACTIVE LIVE TERMINAL PROMPT ---
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
