# evaluate_verdicts.py
import json, collections, argparse
from rapidfuzz import fuzz, process

def evaluate(pred_file, gold_file, threshold=85):
    gold = {}
    with open(gold_file, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            gold[j["title"].strip().lower()] = j["label"]

    preds = []
    with open(pred_file, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            preds.append((j["title"].strip().lower(), j["label"]))

    cats = ["valid", "partially_valid", "hallucinated"]
    cm = {c: collections.Counter() for c in cats}

    gold_titles = list(gold.keys())
    for title_pred, pred_label in preds:
        # Find closest gold title using fuzzy matching
        best_match, score, _ = process.extractOne(title_pred, gold_titles, scorer=fuzz.token_set_ratio)
        if score >= threshold:
            gold_label = gold[best_match]
            cm[gold_label][pred_label] += 1
        else:
            # treat as unmatched (ignore or count separately)
            pass

    for c in cats:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in cats if g != c)
        fn = sum(cm[c][g] for g in cats if g != c)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2*prec*rec / (prec + rec + 1e-9)
        print(f"{c:17s} P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="examples.verdicts.jsonl")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--threshold", type=int, default=85)
    args = ap.parse_args()
    evaluate(args.pred, args.gold, args.threshold)
