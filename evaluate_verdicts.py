# evaluate_verdicts.py
import json, collections, argparse

def evaluate(pred_file, gold_file):
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

    for title, pred in preds:
        g = gold.get(title)
        if g:
            cm[g][pred] += 1

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
    args = ap.parse_args()
    evaluate(args.pred, args.gold)
