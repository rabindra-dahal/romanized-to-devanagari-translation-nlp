import os
import re
import torch
from config import Config
from vocabulary import SubwordVocabulary
from model import NepaliTransformer


class NepaliTranslator:
    """
    Reusable offline Romanized-to-Devanagari Nepali translation module.
    Synced with the 2-layer padded NepaliTransformer (model.encode / model.decode).
    """

    def __init__(self, model_path=None, tokenizer_path=None, device="cpu"):
        app_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(app_dir, "model.pt")
        self.tokenizer_path = tokenizer_path or os.path.join(app_dir, "tokenizer.json")
        self.device = torch.device(device)

        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at: {self.tokenizer_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.model_path}")

        # Match the exact 2-layer trained architecture hyperparameters
        Config.D_MODEL = 128
        Config.NHEAD = 4
        Config.NUM_ENCODER_LAYERS = 2
        Config.NUM_DECODER_LAYERS = 2
        Config.DIM_FEEDFORWARD = 256
        Config.DROPOUT = 0.05
        Config.MAX_LEN = 36
        Config.DEVICE = self.device

        # Load tokenizer and model weights into memory once
        self.vocab = SubwordVocabulary.from_file(self.tokenizer_path)
        self.model = NepaliTransformer(len(self.vocab), len(self.vocab)).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _normalize_input(self, text: str) -> str:
        """Normalizes casual Romanized chat typing to match training format."""
        slang_map = {
            "kto": "keta",
            "keto": "keta",
            "kti": "keti",
            "xa": "chha",
            "xaina": "chhaina",
            "xu": "chhu",
            "xau": "chhau",
            "xan": "chhan",
            "manxe": "manchhe",
        }
        words = text.strip().split()
        normalized_words = [slang_map.get(w.lower(), w) for w in words]
        clean_text = " ".join(normalized_words)

        if clean_text:
            clean_text = clean_text[0].upper() + clean_text[1:]
        if clean_text and clean_text[-1] not in {".", "?", "!"}:
            clean_text += "."
        return clean_text

    def translate(self, text: str) -> str:
        """Translates a single Romanized Nepali string into Devanagari."""
        if not text or not text.strip():
            return ""

        formatted_text = self._normalize_input(text)
        src_tokens = torch.tensor([self.vocab.encode(formatted_text)], device=self.device)
        tgt_tokens = [Config.SOS_IDX]
        max_target_tokens = min(Config.MAX_LEN, max(4, int(src_tokens.size(1) * 1.1) + 2))

        with torch.no_grad():
            memory, src_mask = self.model.encode(src_tokens)

            for _ in range(max_target_tokens):
                tgt_tensor = torch.tensor([tgt_tokens], device=self.device)
                out = self.model.decode(tgt_tensor, memory, src_mask)

                next_token = self.model.fc_out(out)[0, -1].argmax().item()
                if next_token == Config.EOS_IDX or len(tgt_tokens) > max_target_tokens:
                    break
                tgt_tokens.append(next_token)

        raw_output = self.vocab.decode(tgt_tokens)
        clean_output = re.sub(r"\s+।", "।", raw_output)
        return re.sub(r"\s+", " ", clean_output).strip()

    def translate_batch(self, sentences: list[str]) -> list[str]:
        """Translates a list of Romanized Nepali sentences."""
        return [self.translate(s) for s in sentences]


if __name__ == "__main__":
    translator = NepaliTranslator()
    samples = [
        "Uni ahile chamcha man parauchhin",
        "Meri aama chamcha man parauchhin",
        "kto naya gainda herchha",
    ]
    for s in samples:
        print(f"{s}  -->  {translator.translate(s)}")
