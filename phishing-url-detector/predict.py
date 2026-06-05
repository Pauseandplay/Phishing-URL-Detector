"""
predict.py
==========
Run inference on one or many URLs using a trained checkpoint.

Examples
--------
# Single URL
python predict.py --url "http://paypal-secure-login.xyz/verify"

# Batch from a text file (one URL per line)
python predict.py --file urls.txt

# Load a specific checkpoint
python predict.py --url "https://github.com" --checkpoint checkpoints/best.pt

# JSON output
python predict.py --file urls.txt --json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from model.model import PhishingDetector
from utils.features import extract_features, feature_risk_score, top_indicators


RISK_COLOR = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[92m"}
RESET = "\033[0m"
BOLD  = "\033[1m"


def load_model(checkpoint: str | None, device: torch.device) -> PhishingDetector:
    model = PhishingDetector().to(device)
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {checkpoint}\n")
    else:
        print("⚠  No checkpoint found — using untrained model (random weights).\n"
              "   Train first with:  python train.py --sample\n")
    return model


def analyse_url(model: PhishingDetector, url: str, device: torch.device) -> dict:
    neural   = model.predict_url(url, device=str(device))
    features = extract_features(url)
    heuristic_score = feature_risk_score(features)
    indicators      = top_indicators(features, n=5)

    # Ensemble: blend neural confidence with heuristic score
    ensemble_score = 0.7 * neural["confidence"] + 0.3 * heuristic_score
    label     = "phishing"  if ensemble_score >= 0.5  else "legitimate"
    risk      = "HIGH"      if ensemble_score >= 0.75 else "MEDIUM" if ensemble_score >= 0.4 else "LOW"

    return {
        "url":              url,
        "label":            label,
        "risk":             risk,
        "ensemble_score":   round(ensemble_score, 4),
        "neural_confidence":neural["confidence"],
        "heuristic_score":  round(heuristic_score, 4),
        "top_indicators":   indicators,
    }


def print_result(result: dict, verbose: bool = True):
    color = RISK_COLOR.get(result["risk"], "")
    label_str = f"{color}{BOLD}{result['label'].upper()} [{result['risk']}]{RESET}"
    print(f"  URL     : {result['url'][:80]}")
    print(f"  Result  : {label_str}")
    print(f"  Score   : {result['ensemble_score']:.3f}  "
          f"(neural={result['neural_confidence']:.3f}, "
          f"heuristic={result['heuristic_score']:.3f})")
    if verbose and result["top_indicators"]:
        print("  Signals :")
        for ind in result["top_indicators"]:
            bar = "█" * int(ind["value"] * 12)
            print(f"    · {ind['feature']:<28} {bar:<12} {ind['value']:.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Phishing URL Detector — inference")
    parser.add_argument("--url",        type=str, default=None, help="Single URL to analyse")
    parser.add_argument("--file",       type=str, default=None, help="File with one URL per line")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--json",       action="store_true", help="Output as JSON")
    parser.add_argument("--quiet",      action="store_true", help="Suppress indicator breakdown")
    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error("Provide --url or --file")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_model(args.checkpoint, device)

    urls = []
    if args.url:
        urls.append(args.url.strip())
    if args.file:
        with open(args.file) as f:
            urls.extend(line.strip() for line in f if line.strip())

    results = [analyse_url(model, url, device) for url in urls]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Summary header
    phish_count = sum(1 for r in results if r["label"] == "phishing")
    print(f"\n{'─'*60}")
    print(f"  Phishing URL Detector  |  {len(results)} URL(s) analysed")
    print(f"  Phishing: {phish_count}  Legitimate: {len(results) - phish_count}")
    print(f"{'─'*60}\n")

    for r in results:
        print_result(r, verbose=not args.quiet)


if __name__ == "__main__":
    main()
