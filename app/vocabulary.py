import torch
from torch.nn.utils.rnn import pad_sequence
from config import Config
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit # Ensures words don't fragment into separated characters

class SubwordVocabulary:
    """Trains a tight local subword BPE tokenizer with word-boundary controls."""
    def __init__(self, all_sentences):
        # 1. Instantiate the BPE tracking model layout
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        # Using WhitespaceSplit forces the engine to respect standard word blocks
        self.tokenizer.pre_tokenizer = WhitespaceSplit()
        
        trainer = BpeTrainer(
            special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
            vocab_size=1500  # Tight capacity keeps output linear matrices lightning fast
        )
        
        # Train vocabulary mappings
        self.tokenizer.train_from_iterator(all_sentences, trainer)
        
        Config.PAD_IDX = self.tokenizer.token_to_id("<pad>")
        Config.SOS_IDX = self.tokenizer.token_to_id("<sos>")
        Config.EOS_IDX = self.tokenizer.token_to_id("<eos>")
        Config.UNK_IDX = self.tokenizer.token_to_id("<unk>")
        
    def __len__(self):
        return self.tokenizer.get_vocab_size()
        
    def encode(self, text):
        encoded = self.tokenizer.encode(text)
        return [Config.SOS_IDX] + encoded.ids + [Config.EOS_IDX]
        
    def decode(self, token_ids):
        clean_ids = [idx for idx in token_ids if idx not in [Config.SOS_IDX, Config.EOS_IDX, Config.PAD_IDX]]
        # Decodes token IDs back into unified, continuous space-separated strings
        return self.tokenizer.decode(clean_ids)

def build_batches(src_data, tgt_data, vocab):
    src_tensors = [torch.tensor(vocab.encode(s)) for s in src_data]
    tgt_tensors = [torch.tensor(vocab.encode(t)) for t in tgt_data]

    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)

    src_padding_mask = (src_batch == Config.PAD_IDX)
    tgt_padding_mask = (tgt_batch == Config.PAD_IDX)
    
    return src_batch, tgt_batch, src_padding_mask, tgt_padding_mask
