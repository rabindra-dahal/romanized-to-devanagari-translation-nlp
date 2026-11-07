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
    """
    Manages Byte-Pair Encoding (BPE) subword tokenization, offline JSON persistence,
    and Devanagari script reconstruction (orthographic welding) during decoding.
    """

    def __init__(self, all_sentences=None, tokenizer_obj=None):
        # Case 1: Load directly from an existing Tokenizer instance (used by from_file)
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        # Case 2: Train a brand-new BPE tokenizer from raw text sentences
        else:
            cleaned_sentences = []
            for sent in all_sentences:
                if not isinstance(sent, str):
                    continue
                # Strip invisible Unicode control characters (zero-width spaces, BOM, non-breaking spaces)
                s = re.sub(r'[\u200b\u200c\u200d\xa0\ufeff]', '', sent)
                # Remove accidental spaces around Halant/Virama (\u094d) so conjunct letters stay fused
                s = re.sub(r'\s*\u094d\s*', '\u094d', s)
                # Normalize text into canonical composed Unicode (NFC) before training BPE
                cleaned_sentences.append(unicodedata.normalize('NFC', s))

            # Initialize an empty BPE model with <unk> for out-of-vocabulary characters
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))

            # Metaspace replaces spaces with '▁' so word boundaries survive subword splitting
            self.tokenizer.pre_tokenizer = Metaspace()

            # Configure BPE trainer: 600 tokens captures phonetic syllables (e.g., 'ke', 'to', 'के', 'टा')
            # rather than memorizing whole sentences
            trainer = BpeTrainer(
                special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
                vocab_size=600
            )
            self.tokenizer.train_from_iterator(cleaned_sentences, trainer)

        # Synchronize special token integer IDs with the global Config class
        Config.PAD_IDX = self.tokenizer.token_to_id("<pad>")
        Config.SOS_IDX = self.tokenizer.token_to_id("<sos>")
        Config.EOS_IDX = self.tokenizer.token_to_id("<eos>")
        Config.UNK_IDX = self.tokenizer.token_to_id("<unk>")

    def save(self, path):
        """Saves the trained BPE vocabulary and merge rules to a JSON file for offline use."""
        self.tokenizer.save(path)

    @classmethod
    def from_file(cls, path):
        """Loads a previously saved tokenizer.json file from disk."""
        tok = Tokenizer.from_file(path)
        return cls(tokenizer_obj=tok)

    def __len__(self):
        """Returns the total number of subword tokens in the vocabulary."""
        return self.tokenizer.get_vocab_size()

    def encode(self, text):
        """
        Converts a raw text string into a list of token IDs,
        wrapped with Start-Of-Sequence (<sos>) and End-Of-Sequence (<eos>) markers.
        """
        encoded = self.tokenizer.encode(text)
        return [Config.SOS_IDX] + encoded.ids + [Config.EOS_IDX]

    def decode(self, token_ids):
        """
        Converts predicted token IDs back into readable text and repairs
        any split Devanagari vowel signs or conjunct characters.
        """
        # Filter out special control tokens (<sos>, <eos>, <pad>)
        pieces = []
        for idx in token_ids:
            if idx not in [Config.SOS_IDX, Config.EOS_IDX, Config.PAD_IDX]:
                tok = self.tokenizer.id_to_token(idx)
                if tok is not None:
                    pieces.append(tok)

        # Join all subword pieces together and convert Metaspace '▁' markers back to real spaces
        text = "".join(pieces).replace("▁", " ")

        # Remove any hidden zero-width or non-breaking whitespace bytes
        text = re.sub(r'[\u200b\u200c\u200d\xa0\ufeff]', '', text)

        # Orthographic Welder:
        # 1. Remove spaces around Halant/Virama (\u094d) to form proper half-letters (e.g., ग + ् + या -> ग्या)
        text = re.sub(r'\s*\u094d\s*', '\u094d', text)

        # 2. Attach trailing vowel signs (matras \u093e-\u094c) and nasal modifiers (\u0901-\u0903)
        #    directly to the preceding Devanagari character
        text = re.sub(r'(?<=[\u0900-\u097F])\s+([\u093e-\u094c\u0901-\u0903])', r'\1', text)

        # 3. Remove any stray space before the Devanagari sentence-ending Purna Viram (।)
        text = re.sub(r'\s+।', '।', text)

        # Normalize composed Unicode characters and collapse multiple spaces into single spaces
        text = unicodedata.normalize('NFC', text)
        return re.sub(r'\s+', ' ', text).strip()


def build_batches(src_data, tgt_data, vocab):
    """
    Helper utility that encodes parallel source/target sentence lists into
    equal-length padded PyTorch tensors and generates boolean padding masks.
    """
    src_tensors = [torch.tensor(vocab.encode(s)) for s in src_data]
    tgt_tensors = [torch.tensor(vocab.encode(t)) for t in tgt_data]

    # Pad shorter sequences with PAD_IDX so all rows in a batch share the same width
    src_batch = pad_sequence(src_tensors, batch_first=True, padding_value=Config.PAD_IDX)
    tgt_batch = pad_sequence(tgt_tensors, batch_first=True, padding_value=Config.PAD_IDX)

    # True indicates padded positions that attention layers should ignore
    src_padding_mask = (src_batch == Config.PAD_IDX)
    tgt_padding_mask = (tgt_batch == Config.PAD_IDX)

    return src_batch, tgt_batch, src_padding_mask, tgt_padding_mask
