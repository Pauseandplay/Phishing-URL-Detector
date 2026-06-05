"""
utils/dataset.py
================
Dataset loading, preprocessing, and train/val/test splitting.

Supports two public datasets out of the box:
  - PhiUSIIL  (Kaggle: shashwatwork/phiusiil-phishing-url-dataset)
  - ISCX-URL  (University of New Brunswick)

For quick experimentation a tiny built-in sample is included.
"""

import csv
import random
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from model.model import url_to_tensor, MAX_URL_LEN


# ── Built-in sample URLs (for unit tests / CI) ─────────────────────────────

SAMPLE_LEGIT = [
    "https://www.google.com/search?q=machine+learning",
    "https://github.com/pytorch/pytorch",
    "https://stackoverflow.com/questions/tagged/python",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://news.ycombinator.com/",
    "https://www.reddit.com/r/netsec/",
    "https://arxiv.org/abs/1706.03762",
]

SAMPLE_PHISHING = [
    "http://paypal-secure-update.xyz/login?redirect=verify",
    "http://192.168.1.254/admin/steal-credentials",
    "https://accounts.google.com.login-verify.tk/signin",
    "http://amazon-prize-winner.club/claim?user=victim%40email.com",
    "http://secure-banking.online/wellsfargo/update-now",
    "https://appleid-account-locked.site/verify",
    "http://netflix-payment-failed.info/update-billing",
    "http://microsoft-security-alert.work/fix-now",
    "https://chase-bank-secure.xyz/login?token=abc123",
    "http://instagram-verify-account.ml/confirm",
]


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class URLDataset(Dataset):
    """
    Parameters
    ----------
    urls    : list of URL strings
    labels  : list of int (1 = phishing, 0 = legitimate)
    """

    def __init__(self, urls: List[str], labels: List[int]):
        assert len(urls) == len(labels)
        self.urls   = urls
        self.labels = labels

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        x = url_to_tensor(self.urls[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sample_dataset() -> URLDataset:
    """Return the tiny built-in sample dataset (20 URLs)."""
    urls   = SAMPLE_LEGIT + SAMPLE_PHISHING
    labels = [0] * len(SAMPLE_LEGIT) + [1] * len(SAMPLE_PHISHING)
    combined = list(zip(urls, labels))
    random.shuffle(combined)
    urls, labels = zip(*combined)
    return URLDataset(list(urls), list(labels))


def load_csv_dataset(path: str, url_col: str = "url", label_col: str = "label") -> URLDataset:
    """
    Load a CSV file with URL and label columns.

    Expected CSV format
    -------------------
    url,label
    https://example.com,0
    http://phish.xyz/login,1

    Parameters
    ----------
    path      : path to CSV file
    url_col   : column name for URLs
    label_col : column name for labels (0=legit, 1=phishing)
    """
    urls, labels = [], []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url   = row.get(url_col, "").strip()
            label = row.get(label_col, "").strip()
            if url and label in ("0", "1"):
                urls.append(url)
                labels.append(int(label))
    print(f"Loaded {len(urls):,} URLs from {path}  "
          f"(legit={labels.count(0):,}, phish={labels.count(1):,})")
    return URLDataset(urls, labels)


def split_dataset(
    dataset: URLDataset,
    train_frac: float = 0.7,
    val_frac: float   = 0.15,
    seed: int         = 42,
) -> Tuple[URLDataset, URLDataset, URLDataset]:
    """Split dataset into train / val / test subsets."""
    n       = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val
    gen     = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=gen)


def make_loaders(
    dataset: URLDataset,
    batch_size: int  = 64,
    num_workers: int = 0,
    **split_kwargs,
):
    """
    Convenience wrapper: split + create DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds, val_ds, test_ds = split_dataset(dataset, **split_kwargs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Splits — train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader
