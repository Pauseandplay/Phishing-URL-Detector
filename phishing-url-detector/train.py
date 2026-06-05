"""
train.py
========
Train the PhishingDetector on a URL dataset.

Quick start
-----------
# Train on the built-in sample (CPU, ~5 seconds)
python train.py --sample

# Train on a real CSV dataset
python train.py --data data/urls.csv --epochs 20 --batch-size 128

# Resume from a checkpoint
python train.py --data data/urls.csv --resume checkpoints/best.pt
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score

from model.model import PhishingDetector
from utils.dataset import load_csv_dataset, load_sample_dataset, make_loaders


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds       = (torch.sigmoid(logits) >= 0.5).float()
        correct    += (preds == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= 0.5).float()
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    n   = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    return total_loss / n, acc, auc, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Phishing URL Detector")
    parser.add_argument("--data",       type=str,   default=None,           help="Path to CSV dataset")
    parser.add_argument("--sample",     action="store_true",                help="Use built-in sample data")
    parser.add_argument("--url-col",    type=str,   default="url",          help="CSV column name for URLs")
    parser.add_argument("--label-col",  type=str,   default="label",        help="CSV column name for labels")
    parser.add_argument("--epochs",     type=int,   default=15,             help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=64,             help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,           help="Learning rate")
    parser.add_argument("--dropout",    type=float, default=0.4,            help="Dropout rate")
    parser.add_argument("--resume",     type=str,   default=None,           help="Checkpoint path to resume from")
    parser.add_argument("--out-dir",    type=str,   default="checkpoints",  help="Directory to save checkpoints")
    parser.add_argument("--device",     type=str,   default=None,           help="cpu / cuda / mps")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Dataset
    if args.sample:
        dataset = load_sample_dataset()
        print(f"Using built-in sample dataset ({len(dataset)} URLs)")
    elif args.data:
        dataset = load_csv_dataset(args.data, url_col=args.url_col, label_col=args.label_col)
    else:
        parser.error("Provide --data <path> or --sample")

    train_loader, val_loader, test_loader = make_loaders(dataset, batch_size=args.batch_size)

    # Model
    model = PhishingDetector(dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Optionally resume
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1})")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'Val AUC':>8}  {'Time':>6}")
    print("─" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        elapsed = time.time() - t0

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.3%}  {val_loss:>9.4f}  {val_acc:>8.3%}  {val_auc:>8.4f}  {elapsed:>5.1f}s")

        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt_path = Path(args.out_dir) / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_auc": val_auc}, ckpt_path)
            print(f"         ✓ Saved best checkpoint (AUC={val_auc:.4f}) → {ckpt_path}")

    # Final evaluation on test set
    print("\n── Test set evaluation ──────────────────────────────────────────")
    ckpt = torch.load(Path(args.out_dir) / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, test_acc, test_auc, preds, labels = evaluate(model, test_loader, criterion, device)
    print(f"Accuracy : {test_acc:.3%}")
    print(f"ROC-AUC  : {test_auc:.4f}")
    print()
    print(classification_report(labels, preds, target_names=["legitimate", "phishing"]))


if __name__ == "__main__":
    main()
