import torch
from torch.nn.utils.rnn import pad_sequence
from config import Config

class WordVocabulary:
    """Tokenizes whole words instead of breaking them into characters."""
    def __init__(self, text_list):
        # Extract unique words by splitting sentences on space characters
        words = []
        for text in text_list:
            words.extend(text.split())
            
        self.itos = Config.SPECIAL_TOKENS + sorted(list(set(words)))
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        
    def __len__(self):
        return len(self.itos)
        
    def encode(self, text):
        # Split input sentence string into individual word components
        tokens = text.split()
        return [Config.SOS_IDX] + [self.stoi.get(w, Config.UNK_IDX) for w in tokens] + [Config.EOS_IDX]
        
    def decode(self, token_ids):
        # Re-assemble predicted token indexes into a space-separated string sentence
        words_list = []
        for i in token_ids:
            if i not in [Config.SOS_IDX, Config.EOS_IDX, Config.PAD_IDX]:
                words_list.append(self.itos[i])
        return ' '.join(words_list)

def build_batches(src_data, tgt_data, src_vocab, tgt_vocab):
    src_tensors = [torch.tensor(src_vocab.encode(s)) for s in src_data]
    tgt_tensors = [torch.tensor(tgt_vocab.encode(t)) for t in tgt_data]

    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)

    src_padding_mask = (src_batch == Config.PAD_IDX)
    tgt_padding_mask = (tgt_batch == Config.PAD_IDX)
    
    return src_batch, tgt_batch, src_padding_mask, tgt_padding_mask
