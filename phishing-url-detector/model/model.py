"""
Phishing URL Detector — Neural Network Model
============================================
A character-level CNN + LSTM hybrid that classifies URLs as
phishing or legitimate without requiring external lookups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Character Vocabulary ─────────────────────────────────────────────────────

VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".-_~:/?#[]@!$&'()*+,;=%"
)
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}  # 0 = padding
VOCAB_SIZE = len(VOCAB) + 1
MAX_URL_LEN = 200


def url_to_tensor(url: str, max_len: int = MAX_URL_LEN) -> torch.Tensor:
    """Convert a raw URL string to a padded integer tensor."""
    indices = [CHAR2IDX.get(c, 0) for c in url[:max_len]]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class PhishingDetector(nn.Module):
    """
    Character-level CNN + Bidirectional LSTM classifier.

    Architecture
    ────────────
    1. Embedding layer  → dense character representations
    2. Parallel CNN     → extract local n-gram patterns (3-, 4-, 5-grams)
    3. BiLSTM           → capture sequential / positional context
    4. Attention        → focus on the most suspicious sub-sequences
    5. Fully-connected  → binary classification head
    """

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        embed_dim: int    = 64,
        num_filters: int  = 128,
        lstm_hidden: int  = 128,
        lstm_layers: int  = 2,
        dropout: float    = 0.4,
    ):
        super().__init__()

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. Parallel convolutions (3-, 4-, 5-gram)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in (3, 4, 5)
        ])
        self.conv_bn = nn.BatchNorm1d(num_filters * 3)

        # 3. BiLSTM
        self.lstm = nn.LSTM(
            input_size=num_filters * 3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 4. Attention
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # 5. Classifier head
        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(lstm_hidden * 2, 128)
        self.fc2      = nn.Linear(128, 1)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len) int64
        returns : (batch,) float logits
        """
        # Embedding  →  (B, L, E)
        emb = self.embedding(x)

        # CNN  →  (B, F*3, L)
        emb_t = emb.permute(0, 2, 1)                          # (B, E, L)
        conv_outs = [F.relu(conv(emb_t)) for conv in self.convs]

        # Align lengths (conv padding may shift by 1)
        min_len = min(c.size(2) for c in conv_outs)
        conv_outs = [c[:, :, :min_len] for c in conv_outs]

        cnn_out = torch.cat(conv_outs, dim=1)                  # (B, F*3, L)
        cnn_out = self.conv_bn(cnn_out)
        cnn_out = cnn_out.permute(0, 2, 1)                     # (B, L, F*3)

        # BiLSTM  →  (B, L, H*2)
        lstm_out, _ = self.lstm(cnn_out)

        # Attention  →  (B, H*2)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, L, 1)
        context = (attn_weights * lstm_out).sum(dim=1)                 # (B, H*2)

        # Classifier
        out = self.dropout(F.relu(self.fc1(context)))
        logits = self.fc2(out).squeeze(-1)                     # (B,)
        return logits

    # ── Convenience methods ───────────────────────────────────────────────────

    def predict_url(self, url: str, device: str = "cpu") -> dict:
        """
        Predict a single URL.

        Returns
        -------
        dict with keys:
            url         : original URL
            label       : 'phishing' | 'legitimate'
            confidence  : float 0-1 (probability of phishing)
            risk        : 'HIGH' | 'MEDIUM' | 'LOW'
        """
        self.eval()
        with torch.no_grad():
            tensor = url_to_tensor(url).unsqueeze(0).to(device)
            logit  = self(tensor)
            prob   = torch.sigmoid(logit).item()

        label = "phishing" if prob >= 0.5 else "legitimate"
        risk  = "HIGH" if prob >= 0.75 else "MEDIUM" if prob >= 0.4 else "LOW"

        return {
            "url":        url,
            "label":      label,
            "confidence": round(prob, 4),
            "risk":       risk,
        }

    def predict_batch(self, urls: list[str], device: str = "cpu") -> list[dict]:
        """Predict a list of URLs efficiently in one forward pass."""
        self.eval()
        with torch.no_grad():
            tensors = torch.stack([url_to_tensor(u) for u in urls]).to(device)
            logits  = self(tensors)
            probs   = torch.sigmoid(logits).cpu().tolist()

        results = []
        for url, prob in zip(urls, probs):
            label = "phishing" if prob >= 0.5 else "legitimate"
            risk  = "HIGH" if prob >= 0.75 else "MEDIUM" if prob >= 0.4 else "LOW"
            results.append({
                "url":        url,
                "label":      label,
                "confidence": round(prob, 4),
                "risk":       risk,
            })
        return results


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = PhishingDetector()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total:,}")

    sample = url_to_tensor("http://paypal-secure-login.xyz/verify").unsqueeze(0)
    out    = model(sample)
    print(f"Logit (untrained): {out.item():.4f}")
