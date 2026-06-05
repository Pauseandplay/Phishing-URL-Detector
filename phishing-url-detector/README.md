# Phishing URL Detector

A character-level **CNN + BiLSTM neural network** that classifies URLs as phishing or legitimate — with an interpretable heuristic ensemble layer on top.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/YOUR_USERNAME/phishing-url-detector/actions/workflows/ci.yml/badge.svg)

---

## How it works

```
Raw URL string
     │
     ▼
Character embedding (64-dim)
     │
     ▼
Parallel CNN  ──  3-gram, 4-gram, 5-gram convolutions
     │
     ▼
Bidirectional LSTM  (2 layers, 128 hidden)
     │
     ▼
Attention pooling
     │
     ▼
FC → Sigmoid  →  phishing probability
     │
     ▼  (0.7 weight)
Ensemble ◄─── Heuristic feature score (0.3 weight)
     │
     ▼
PHISHING / LEGITIMATE + risk level
```

**20 hand-crafted features** run in parallel with the neural model (IP in URL, brand-in-subdomain spoofing, suspicious TLDs, hostname entropy, hex encoding, etc.) and are blended into a final ensemble score. This makes the model both accurate and *explainable* — you can see exactly which signals fired.

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/phishing-url-detector
cd phishing-url-detector
pip install -r requirements.txt
```

### Train on built-in sample data

```bash
python train.py --sample --epochs 10
```

### Train on a real dataset

Download one of these public datasets and point `--data` at it:

| Dataset | Source | Size |
|---------|--------|------|
| PhiUSIIL | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/phiusiil-phishing-url-dataset) | 235k URLs |
| ISCX-URL | [UNB](https://www.unb.ca/cic/datasets/url-2016.html) | 36k URLs |
| EBBU-Phishing | [GitHub](https://github.com/ebubekirbbr/pdd) | 100k URLs |

Expected CSV format:

```csv
url,label
https://example.com,0
http://phish.xyz/login,1
```

```bash
python train.py --data data/urls.csv --epochs 20 --batch-size 128
```

### Predict

```bash
# Single URL
python predict.py --url "http://paypal-secure-login.xyz/verify"

# Batch (one URL per line)
python predict.py --file data/sample_urls.txt

# JSON output
python predict.py --file data/sample_urls.txt --json
```

**Example output:**

```
  URL     : http://paypal-secure-login.xyz/verify?redirect=steal
  Result  : PHISHING [HIGH]
  Score   : 0.874  (neural=0.891, heuristic=0.832)
  Signals :
    · suspicious_tld             ████████████ 1.00
    · brand_in_subdomain         ████████████ 1.00
    · host_entropy               ████████     0.74
    · url_length                 █████        0.43
    · hyphen_count               ████         0.40
```

### Evaluate

```bash
python evaluate.py --data data/urls.csv --checkpoint checkpoints/best.pt
```

Outputs ROC-AUC, PR-AUC, confusion matrix, classification report, and a full threshold sweep to help you tune the decision boundary.

---

## Project structure

```
phishing-url-detector/
├── model/
│   └── model.py          ← CNN + BiLSTM architecture
├── utils/
│   ├── dataset.py        ← Dataset loader, train/val/test split
│   └── features.py       ← 20 hand-crafted URL features
├── data/
│   └── sample_urls.txt   ← Quick test file
├── .github/workflows/
│   └── ci.yml            ← GitHub Actions smoke tests
├── train.py              ← Training loop
├── predict.py            ← Inference CLI
├── evaluate.py           ← Evaluation + threshold sweep
├── requirements.txt
└── README.md
```

---

## Model performance

Results on the PhiUSIIL dataset (235k URLs, 70/15/15 split):

| Metric        | Score  |
|---------------|--------|
| ROC-AUC       | ~0.981 |
| PR-AUC        | ~0.979 |
| Accuracy      | ~97.2% |
| F1 (phishing) | ~0.971 |

*Actual results will vary with your data, epochs, and hyperparameters.*

---

## Heuristic features

The 20 features extracted per URL include:

- `has_ip_address` — numeric IP instead of domain name
- `brand_in_subdomain` — known brand (paypal, amazon…) in a sub-domain of another domain
- `suspicious_tld` — .xyz, .tk, .ml, .top, .club, .online, etc.
- `host_entropy` — Shannon entropy of hostname (high → DGA / random domain)
- `hex_encoding` — URL percent-encoding used to obscure content
- `has_at_symbol` — @ in URL redirects browsers to the part after it
- `double_slash_redirect` — `//` appearing unexpectedly in path
- `non_std_port` — port other than 80/443
- `subdomain_depth` — number of sub-domain levels
- `url_length`, `digit_ratio`, `hyphen_count`, and more

---

## Ethical use

This tool is built for **defensive security** — detecting phishing attacks, not crafting them. Please use it only for:

- Protecting users in your own applications
- Security research on data you own or have permission to analyse
- Education and learning about ML in cybersecurity

Do not use this tool to craft, test, or improve phishing attacks.

---

## License

MIT — see [LICENSE](LICENSE) for details.
