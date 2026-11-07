# Romanized Nepali to Devanagari Translation (Custom PyTorch Transformer)

A lightweight, CPU-friendly Sequence-to-Sequence Transformer built from scratch in PyTorch to translate Romanized Nepali into standard Devanagari script. Designed for fast offline inference using saved checkpoints (`model.pt`) and a phonetic syllable BPE tokenizer (`tokenizer.json`).

---

## Features

- **Custom 2-Layer PyTorch Transformer:** Implements sinusoidal positional encoding, multi-head self/cross-attention, 4D source padding masks (`src_mask`), and causal look-ahead masks without external sequence-to-sequence wrappers.
- **Phonetic Syllable BPE Tokenizer:** Uses `Metaspace` subword tokenization (`vocab_size=600`) paired with Unicode NFC normalization and Devanagari orthographic welding to preserve conjunct characters and vowel matras.
- **100% Offline Inference:** Automatically saves and loads `model.pt` and `tokenizer.json` locally for sub-second translations without internet access.
- **Reusable Translation Module:** Includes `NepaliTranslator` (`app/translator.py`) with built-in casual chat/slang normalization (`kto` → `keta`, `xa` → `chha`) and batch translation support.
- **Automated Evaluation Suite:** Includes `app/evaluate.py` to benchmark Exact Match Accuracy, Corpus BLEU-4, and chrF scores via `sacrebleu`.

---

## Benchmark Results

Evaluated on **200 samples** using `app/evaluate.py` (`sacrebleu`):

| Metric | Score |
| :--- | :--- |
| **Exact Match Accuracy** | **100.0% (200 / 200)** |
| **Corpus BLEU-4** | **100.0 / 100** |
| **chrF Score** | **100.0 / 100** |

**Sample Predictions:**
- `Shikshak aaja sneha man paraunuhunchha.` → `शिक्षक आज स्नेह मन पराउनुहुन्छ।`
- `Mero sathi harek din maha khanchha.` → `मेरो साथी हरेक दिन मह खान्छ।`
- `kto naya gainda herchha` → `केटा नयाँ गैंडा हेर्छ।`

---

## Project Structure

```text
romanized-to-devanagari-translation-nlp/
├── app/
│   ├── config.py          # Model hyperparameters, special tokens, and CPU device routing
│   ├── model.py           # Sinusoidal Positional Encoding, Custom Multi-Head Attention & 2-Layer Transformer
│   ├── vocabulary.py      # Metaspace BPE subword tokenizer & Devanagari orthographic welder
│   ├── pipeline.py        # Training epoch loop & greedy inference decoding
│   ├── main.py            # Offline checkpoint loader, augmented training entry point & interactive CLI
│   ├── translator.py      # Standalone reusable Python module (NepaliTranslator)
│   ├── evaluate.py        # Automated Exact Match, BLEU-4 & chrF evaluation script
│   ├── model.pt           # Saved trained model weights
│   └── tokenizer.json     # Saved BPE tokenizer vocabulary
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
└── README.md
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd romanized-to-devanagari-translation-nlp
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Interactive CLI Translator (Training + Live Chat)
Run `main.py` inside the `app/` directory. If `model.pt` and `tokenizer.json` already exist, it boots straight into offline interactive mode; otherwise, it downloads the dataset and trains automatically.

```bash
python app\main.py
```

**Example Session:**
```text
Romanized > kto naya gainda herchha
Devanagari > केटा नयाँ गैंडा हेर्छ।

```

### 2. Using the Reusable Python Module (`translator.py`)
Import `NepaliTranslator` into any Python script, API, or web app:

```python
from app.translator import NepaliTranslator

translator = NepaliTranslator()

# Single sentence translation
result = translator.translate("kto naya gainda herchha")
print(result)  # केटा नयाँ गैंडा हेर्छ।

# Batch translation
sentences = [
    "Meri aama chamcha man parauchhin",
    "Keta sanjha aanp khanchha"
]
print(translator.translate_batch(sentences))
```

### 3. Running Automated Evaluation
Benchmark the saved checkpoint on 200 samples:
```bash
python app\evaluate.py
```

---

## Model & Dataset Details

- **Dataset:** [`rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning`](https://huggingface.co) (enhanced with template-breaking subject/object swap augmentation)
- **Architecture:**
  - `d_model`: `128` | `nhead`: `4` | `dim_feedforward`: `256`
  - `num_encoder_layers`: `2` | `num_decoder_layers`: `2`
  - `max_len`: `36` | `dropout`: `0.05`
  - **Loss:** Cross-Entropy Loss with `label_smoothing=0.05`
  - **Training:** `30` epochs (`batch_size=32`, Adam optimizer `lr=0.0015`)

---
## Benchmark Results

Evaluated on **200 samples** using `app/evaluate.py` (`sacrebleu`):

| Metric | Score |
| :--- | :--- |
| **Exact Match Accuracy** | **100.0% (200 / 200)** |
| **Corpus BLEU-4 Score** | **100.0 / 100** |
| **chrF Score** | **100.0 / 100** |

### Sample Evaluation Predictions

| Status | Romanized Input | Target Reference | Model Prediction |
| :---   | :--- | :--- | :--- |
| `[MATCH]` | `Shikshak aaja sneha man paraunuhunchha.` | शिक्षक आज स्नेह मन पराउनुहुन्छ। | शिक्षक आज स्नेह मन पराउनुहुन्छ। |
| `[MATCH]` | `Mero sathi harek din maha khanchha.` | मेरो साथी हरेक दिन मह खान्छ। | मेरो साथी हरेक दिन मह खान्छ। |
| `[MATCH]` | `Bachcha sanjha dahi man parauchha.` | बच्चा साँझ दही मन पराउँछ। | बच्चा साँझ दही मन पराउँछ। |
| `[MATCH]` | `Hajurbuba harek din kharayo man paraunuhunchha.` | हजुरबुबा हरेक दिन खरायो मन पराउनुहुन्छ। | हजुरबुबा हरेक दिन खरायो मन पराउनुहुन्छ। |
| `[MATCH]` | `Meri aama sadhain roti man paraunuhunchha.` | मेरी आमा सधैं रोटी मन पराउनुहुन्छ। | मेरी आमा सधैं रोटी मन पराउनुहुन्छ। |

---

## Known Limitations

- **Domain Scope:** Trained on a kid-learning corpus; performs best on simple, everyday conversational sentences and known vocabulary rather than complex technical or literary prose.
- **Retraining:** To retrain from scratch on updated data, delete `app/model.pt` and `app/tokenizer.json` before running `python app\main.py`.
