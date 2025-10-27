# Robust evaluator for Citation Hallucination Detection
# Handles Unicode normalization, fuzzy matching, and prints summary metrics
# Usage:
#   python evaluate_verdicts.py --pred examples.verdicts.jsonl --gold gold.jsonl

import json
import re
import unicodedata
import collections
import argparse
from rapidfuzz import fuzz, process

def normalize_title(s: str) -> str:
    """Clean, lowercase, and normalize Unicode for reliable matching."""
    s = unicodedata.normalize("NFKC", s)  # normalize Unicode
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def evaluate(pred_file, gold_file, threshold=85):
    # Load gold dataset
    gold = {}
    with open(gold_file, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            gold[normalize_title(j["title"])] = j["label"]

    # Load predictions
    preds = []
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            preds.append((normalize_title(j["title"]), j["label"]))

    cats = ["valid", "partially_valid", "hallucinated"]
    cm = {c: collections.Counter() for c in cats}

    gold_titles = list(gold.keys())
    matched = 0

    for title_pred, pred_label in preds:
        match = process.extractOne(title_pred, gold_titles, scorer=fuzz.token_set_ratio)
        if match and match[1] >= threshold:
            best_match, score = match[0], match[1]
            gold_label = gold[best_match]
            cm[gold_label][pred_label] += 1
            matched += 1
        else:
            # unmatched prediction
            pass

    print(f"\nMatched {matched}/{len(preds)} predictions ≥ fuzzy threshold {threshold}\n")

    # Calculate and print metrics
    macro_p, macro_r, macro_f1 = 0, 0, 0
    total_tp, total_fp, total_fn = 0, 0, 0

    print("Per-class metrics:")
    for c in cats:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in cats if g != c)
        fn = sum(cm[c][g] for g in cats if g != c)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        print(f"{c:17s} P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}")
        macro_p += prec
        macro_r += rec
        macro_f1 += f1
        total_tp += tp
        total_fp += fp
        total_fn += fn

    n = len(cats)
    macro_p /= n
    macro_r /= n
    macro_f1 /= n

    micro_p = total_tp / (total_tp + total_fp + 1e-9)
    micro_r = total_tp / (total_tp + total_fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    print("\nMacro-average:  P={:.2f}  R={:.2f}  F1={:.2f}".format(macro_p, macro_r, macro_f1))
    print("Micro-average:  P={:.2f}  R={:.2f}  F1={:.2f}".format(micro_p, micro_r, micro_f1))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    header = " " * 18 + " ".join(f"{c[:10]:>12}" for c in cats)
    print(header)
    for g in cats:
        row = f"{g[:15]:>15} | " + " ".join(f"{cm[g][p]:12d}" for p in cats)
        print(row)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Predictions JSONL file")
    ap.add_argument("--gold", required=True, help="Gold labels JSONL file")
    ap.add_argument("--threshold", type=int, default=85, help="Fuzzy match threshold (0-100)")
    args = ap.parse_args()
    evaluate(args.pred, args.gold, args.threshold)
