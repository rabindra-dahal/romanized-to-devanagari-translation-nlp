import torch
from torch.nn.utils.rnn import pad_sequence
from config import Config

class CharVocabulary:
    """A robust character-level vocabulary that pre-loads all necessary characters."""
    def __init__(self, is_target=False):
        # Base special tokens
        self.itos = list(Config.SPECIAL_TOKENS)
        
        if not is_target:
            # Source vocabulary: All lowercase, uppercase, numbers, and basic punctuation
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\""
            self.itos.extend(list(alphabet))
        else:
            # Target vocabulary: Comprehensive Devanagari alphabet (Vowels, Consonants, Modifiers)
            devanagari = (
                "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह" # Core letters
                "ािीुूृेैोौ्ंःँ" # Matras (vowel signs) and diacritics
                "०१२३४५६७८९ ।" # Numbers and Purna Viram
            )
            self.itos.extend(list(devanagari))
            
        # Ensure strict uniqueness and create the string-to-integer lookup map
        self.itos = sorted(list(set(self.itos)))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        
    def __len__(self):
        return len(self.itos)
        
    def encode(self, text):
        # Safely defaults to <unk> if an unexpected character enters the stream
        return [Config.SOS_IDX] + [self.stoi.get(ch, Config.UNK_IDX) for ch in text] + [Config.EOS_IDX]
        
    def decode(self, token_ids):
        return ''.join([self.itos[i] for i in token_ids if i not in [Config.SOS_IDX, Config.EOS_IDX, Config.PAD_IDX]])

def build_batches(src_data, tgt_data, src_vocab, tgt_vocab):
    src_tensors = [torch.tensor(src_vocab.encode(s)) for s in src_data]
    tgt_tensors = [torch.tensor(tgt_vocab.encode(t)) for t in tgt_data]

    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)

    src_padding_mask = (src_batch == Config.PAD_IDX)
    tgt_padding_mask = (tgt_batch == Config.PAD_IDX)
    
    return src_batch, tgt_batch, src_padding_mask, tgt_padding_mask
