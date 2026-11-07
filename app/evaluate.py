import random
import re
import unicodedata
from collections import Counter
import math
from datasets import load_dataset
from translator import NepaliTranslator


def normalize_nepali(text: str) -> str:
    """Normalizes Devanagari text so minor whitespace differences don't penalize scores."""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+।", "।", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_corpus_bleu(references: list[str], hypotheses: list[str], max_n: int = 4) -> float:
    """Computes standard Corpus BLEU-4 score (0 to 100) without external dependencies."""
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
            hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
            for ng, count in hyp_ngrams.items():
                clipped_counts[n - 1] += min(count, ref_ngrams[ng])
                total_counts[n - 1] += count

    if hyp_len == 0 or any(c == 0 for c in clipped_counts):
        return 0.0

    bp = 1.0 if hyp_len > ref_len else math.exp(1 - (ref_len / hyp_len))
    log_precisions = sum(math.log(clipped_counts[i] / total_counts[i]) for i in range(max_n)) / max_n
    return round(bp * math.exp(log_precisions) * 100, 2)


def compute_chrf(references: list[str], hypotheses: list[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Computes character n-gram F-score (chrF, 0 to 100), ideal for Devanagari conjuncts/matras."""
    total_f = 0.0
    valid_pairs = 0

    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            continue
        precisions, recalls = [], []
        for n in range(1, max_n + 1):
            ref_ng = Counter(ref[i : i + n] for i in range(max(0, len(ref) - n + 1)))
            hyp_ng = Counter(hyp[i : i + n] for i in range(max(0, len(hyp) - n + 1)))
            overlap = sum(min(count, ref_ng[ng]) for ng, count in hyp_ng.items())
            p = overlap / max(1, sum(hyp_ng.values()))
            r = overlap / max(1, sum(ref_ng.values()))
            precisions.append(p)
            recalls.append(r)

        avg_p = sum(precisions) / max_n
        avg_r = sum(recalls) / max_n
        if avg_p + avg_r > 0:
            f_score = (1 + beta**2) * (avg_p * avg_r) / ((beta**2 * avg_p) + avg_r)
            total_f += f_score
        valid_pairs += 1

    return round((total_f / max(1, valid_pairs)) * 100, 2)


def main(num_samples: int = 200):
    print("Loading saved NepaliTranslator...")
    translator = NepaliTranslator()

    print("Loading dataset for evaluation...")
    dataset = load_dataset(
        "rabindra-dahal/english-nepali-romanized-csv-text-for-kid-learning",
        split="train",
    )

    # Sample a deterministic validation slice
    random.seed(123)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    eval_rows = [dataset[i] for i in indices]

    sources = [str(row["romanized"]) for row in eval_rows]
    references = [normalize_nepali(row["nepali"]) for row in eval_rows]

    print(f"Translating {len(sources)} test sentences on CPU...")
    predictions = [normalize_nepali(pred) for pred in translator.translate_batch(sources)]

    # Compute metrics
    exact_matches = sum(1 for r, p in zip(references, predictions) if r == p)
    exact_acc = round((exact_matches / len(references)) * 100, 2)

    try:
        import sacrebleu
        bleu_score = round(sacrebleu.corpus_bleu(predictions, [references]).score, 2)
        chrf_score = round(sacrebleu.corpus_chrf(predictions, [references]).score, 2)
        metric_engine = "sacrebleu"
    except ImportError:
        bleu_score = compute_corpus_bleu(references, predictions)
        chrf_score = compute_chrf(references, predictions)
        metric_engine = "built-in"

    print("\n" + "=" * 55)
    print(f"EVALUATION REPORT ({len(sources)} Samples | Engine: {metric_engine})")
    print("=" * 55)
    print(f"Exact Match Accuracy : {exact_acc}% ({exact_matches}/{len(sources)})")
    print(f"Corpus BLEU-4 Score  : {bleu_score} / 100")
    print(f"chrF Score           : {chrf_score} / 100")
    print("=" * 55)

    # Show 5 sample comparisons (including any mismatches for error analysis)
    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(sources))):
        status = "MATCH" if references[i] == predictions[i] else "DIFF "
        print(f"[{status}] Input : {sources[i]}")
        print(f"        Target: {references[i]}")
        print(f"        Pred  : {predictions[i]}\n")


if __name__ == "__main__":
    main(num_samples=200)
