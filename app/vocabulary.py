import torch
import re
import unicodedata
from torch.nn.utils.rnn import pad_sequence
from config import Config
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace

class SubwordVocabulary:
    """Phonetic syllable BPE tokenizer using Metaspace for word-boundary preservation."""
    def __init__(self, all_sentences=None, tokenizer_obj=None):
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        else:
            cleaned_sentences = []
            for sent in all_sentences:
                if not isinstance(sent, str):
                    continue
                s = re.sub(r'[\u200b\u200c\u200d\xa0\ufeff]', '', sent)
                s = re.sub(r'\s*\u094d\s*', '\u094d', s)
                cleaned_sentences.append(unicodedata.normalize('NFC', s))

            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            # Metaspace preserves word spaces using '▁' so syllable tokens reassemble cleanly
            self.tokenizer.pre_tokenizer = Metaspace()
            
            trainer = BpeTrainer(
                special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
                vocab_size=600  # Forces phonetic syllable-level tokens instead of memorized whole words
            )
            self.tokenizer.train_from_iterator(cleaned_sentences, trainer)
        
        Config.PAD_IDX = self.tokenizer.token_to_id("<pad>")
        Config.SOS_IDX = self.tokenizer.token_to_id("<sos>")
        Config.EOS_IDX = self.tokenizer.token_to_id("<eos>")
        Config.UNK_IDX = self.tokenizer.token_to_id("<unk>")

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path):
        tok = Tokenizer.from_file(path)
        return cls(tokenizer_obj=tok)
        
    def __len__(self):
        return self.tokenizer.get_vocab_size()
        
    def encode(self, text):
        encoded = self.tokenizer.encode(text)
        return [Config.SOS_IDX] + encoded.ids + [Config.EOS_IDX]
        
    def decode(self, token_ids):
        # Join raw subword pieces directly and convert Metaspace '▁' markers back to real spaces
        pieces = []
        for idx in token_ids:
            if idx not in [Config.SOS_IDX, Config.EOS_IDX, Config.PAD_IDX]:
                tok = self.tokenizer.id_to_token(idx)
                if tok is not None:
                    pieces.append(tok)
                    
        text = "".join(pieces).replace("▁", " ")
        text = re.sub(r'[\u200b\u200c\u200d\xa0\ufeff]', '', text)
        
        # Clean up any stray spaces around Virama (Halant) or vowel matras
        text = re.sub(r'\s*\u094d\s*', '\u094d', text)
        text = re.sub(r'(?<=[\u0900-\u097F])\s+([\u093e-\u094c\u0901-\u0903])', r'\1', text)
        text = re.sub(r'\s+।', '।', text)
        
        text = unicodedata.normalize('NFC', text)
        return re.sub(r'\s+', ' ', text).strip()

def build_batches(src_data, tgt_data, vocab):
    src_tensors = [torch.tensor(vocab.encode(s)) for s in src_data]
    tgt_tensors = [torch.tensor(vocab.encode(t)) for t in tgt_data]

    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    
    src_padding_mask = (src_batch == Config.PAD_IDX)
    tgt_padding_mask = (tgt_batch == Config.PAD_IDX)
    
    return src_batch, tgt_batch, src_padding_mask, tgt_padding_mask
