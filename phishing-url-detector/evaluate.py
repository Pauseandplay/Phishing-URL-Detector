"""
evaluate.py
===========
Evaluate a trained checkpoint on a labelled CSV dataset and
produce a detailed report with confusion matrix, ROC curve data,
and per-threshold metrics.

Usage
-----
python evaluate.py --data data/urls.csv --checkpoint checkpoints/best.pt
python evaluate.py --sample --checkpoint checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from model.model import PhishingDetector
from utils.dataset import load_csv_dataset, load_sample_dataset, make_loaders
from utils.features import extract_features, feature_risk_score


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        probs = torch.sigmoid(model(x)).cpu().tolist()
        all_probs.extend(probs)
        all_labels.extend(y.tolist())
    return all_probs, all_labels


def threshold_sweep(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]
    rows = []
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        acc       = (tp + tn) / len(labels)
        rows.append({"threshold": round(t, 2), "precision": round(precision, 4),
                     "recall": round(recall, 4), "f1": round(f1, 4), "accuracy": round(acc, 4)})
    return rows


def print_confusion_matrix(labels, preds):
    cm   = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print("  Confusion Matrix")
    print(f"  {'':18} Pred: Legit   Pred: Phish")
    print(f"  {'True: Legit':18} {tn:^12} {fp:^12}")
    print(f"  {'True: Phish':18} {fn:^12} {tp:^12}")
    print(f"\n  False Positive Rate : {fp / (fp + tn):.3%}")
    print(f"  False Negative Rate : {fn / (fn + tp):.3%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phishing URL Detector")
    parser.add_argument("--data",       type=str,   default=None)
    parser.add_argument("--sample",     action="store_true")
    parser.add_argument("--url-col",    type=str,   default="url")
    parser.add_argument("--label-col",  type=str,   default="label")
    parser.add_argument("--checkpoint", type=str,   default="checkpoints/best.pt")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--device",     type=str,   default=None)
    parser.add_argument("--save-json",  type=str,   default=None, help="Save full results to JSON")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    model = PhishingDetector().to(device)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("⚠  Checkpoint not found — random weights (results meaningless)")

    # Load dataset
    if args.sample:
        dataset = load_sample_dataset()
    elif args.data:
        dataset = load_csv_dataset(args.data, url_col=args.url_col, label_col=args.label_col)
    else:
        parser.error("Provide --data or --sample")

    _, _, test_loader = make_loaders(dataset, batch_size=128)

    # Evaluate
    probs, labels = run_evaluation(model, test_loader, device)
    preds = [1 if p >= args.threshold else 0 for p in probs]

    auc_roc = roc_auc_score(labels, probs)
    auc_pr  = average_precision_score(labels, probs)

    print(f"\n{'═'*58}")
    print("  Neural Model Evaluation")
    print(f"{'═'*58}\n")
    print(f"  ROC-AUC  : {auc_roc:.4f}")
    print(f"  PR-AUC   : {auc_pr:.4f}")
    print(f"  Threshold: {args.threshold}\n")
    print_confusion_matrix(labels, preds)
    print()
    print(classification_report(labels, preds, target_names=["legitimate", "phishing"]))

    # Threshold sweep
    sweep = threshold_sweep(probs, labels)
    best  = max(sweep, key=lambda r: r["f1"])
    print(f"\n  Best threshold by F1: {best['threshold']}  "
          f"(F1={best['f1']:.4f}, P={best['precision']:.4f}, R={best['recall']:.4f})\n")

    print(f"  {'Threshold':>9}  {'Precision':>9}  {'Recall':>7}  {'F1':>6}  {'Accuracy':>8}")
    print("  " + "─" * 50)
    for row in sweep:
        marker = " ◀ best F1" if row == best else ""
        print(f"  {row['threshold']:>9.2f}  {row['precision']:>9.4f}  "
              f"{row['recall']:>7.4f}  {row['f1']:>6.4f}  {row['accuracy']:>8.4f}{marker}")

    # Optional JSON save
    if args.save_json:
        output = {
            "auc_roc": auc_roc, "auc_pr": auc_pr,
            "best_threshold": best,
            "threshold_sweep": sweep,
        }
        with open(args.save_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved → {args.save_json}")


if __name__ == "__main__":
    main()
