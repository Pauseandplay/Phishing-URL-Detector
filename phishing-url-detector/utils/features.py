"""
utils/features.py
=================
Hand-crafted URL features used alongside the neural model for
interpretable analysis and feature-importance visualisation.
"""

import re
import math
from urllib.parse import urlparse
from typing import Dict, List


# ── Suspicious patterns ───────────────────────────────────────────────────────

BRAND_KEYWORDS = [
    "paypal", "amazon", "google", "microsoft", "apple", "facebook",
    "instagram", "netflix", "linkedin", "twitter", "bank", "secure",
    "account", "login", "signin", "verify", "update", "confirm",
    "password", "ebay", "wellsfargo", "chase", "citibank", "dropbox",
]

SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".club",
    ".online", ".site", ".info", ".biz", ".work", ".click", ".link",
}

IP_PATTERN    = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
HEX_PATTERN   = re.compile(r"%[0-9a-fA-F]{2}")
AT_SYMBOL     = re.compile(r"@")
DOUBLE_SLASH  = re.compile(r"//.*//")


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(url: str) -> Dict[str, float]:
    """
    Extract 20 interpretable numerical features from a URL.

    Returns a dict mapping feature name → float value (all normalised 0-1
    where possible so they can be displayed as bar lengths).
    """
    parsed  = urlparse(url if "://" in url else "http://" + url)
    host    = parsed.hostname or ""
    path    = parsed.path or ""
    query   = parsed.query or ""
    full    = url.lower()

    # ── Length features
    url_len          = len(url)
    host_len         = len(host)
    path_len         = len(path)

    # ── Structural features
    dot_count        = host.count(".")
    hyphen_count     = host.count("-")
    digit_count      = sum(c.isdigit() for c in host)
    subdomain_depth  = max(0, host.count(".") - 1)
    path_depth       = path.count("/")
    query_params     = query.count("&") + (1 if query else 0)

    # ── Suspicious signals
    has_ip           = int(bool(IP_PATTERN.match(host)))
    has_at           = int(bool(AT_SYMBOL.search(url)))
    hex_encoding     = len(HEX_PATTERN.findall(url))
    double_slash     = int(bool(DOUBLE_SLASH.search(url)))
    brand_in_subdomain = int(any(kw in host.split(".")[0] for kw in BRAND_KEYWORDS)
                              and host.count(".") >= 2)
    brand_in_path    = int(any(kw in path.lower() for kw in BRAND_KEYWORDS))
    suspicious_tld   = int(any(full.endswith(tld) or f"{tld}/" in full
                               for tld in SUSPICIOUS_TLDS))
    https            = int(parsed.scheme == "https")
    port_in_url      = int(parsed.port is not None and parsed.port not in (80, 443))

    # ── Entropy of hostname (high entropy → random/DGA domain)
    entropy          = _shannon_entropy(host)

    # ── Ratio: digits in full URL
    digit_ratio      = sum(c.isdigit() for c in url) / max(len(url), 1)

    return {
        "url_length":           min(url_len / 200, 1.0),
        "host_length":          min(host_len / 50, 1.0),
        "path_length":          min(path_len / 100, 1.0),
        "dot_count":            min(dot_count / 8, 1.0),
        "hyphen_count":         min(hyphen_count / 5, 1.0),
        "digit_in_host":        min(digit_count / 10, 1.0),
        "subdomain_depth":      min(subdomain_depth / 4, 1.0),
        "path_depth":           min(path_depth / 8, 1.0),
        "query_params":         min(query_params / 10, 1.0),
        "has_ip_address":       float(has_ip),
        "has_at_symbol":        float(has_at),
        "hex_encoding":         min(hex_encoding / 5, 1.0),
        "double_slash_redirect":float(double_slash),
        "brand_in_subdomain":   float(brand_in_subdomain),
        "brand_in_path":        float(brand_in_path),
        "suspicious_tld":       float(suspicious_tld),
        "uses_https":           float(https),
        "non_std_port":         float(port_in_url),
        "host_entropy":         min(entropy / 4.5, 1.0),
        "digit_ratio":          digit_ratio,
    }


def feature_risk_score(features: Dict[str, float]) -> float:
    """
    Weighted sum of hand-crafted features → 0-1 risk score.
    Used as a fast pre-filter and for explainability.
    """
    WEIGHTS = {
        "has_ip_address":        0.15,
        "has_at_symbol":         0.10,
        "brand_in_subdomain":    0.12,
        "suspicious_tld":        0.10,
        "double_slash_redirect": 0.08,
        "non_std_port":          0.07,
        "hex_encoding":          0.06,
        "host_entropy":          0.08,
        "subdomain_depth":       0.05,
        "hyphen_count":          0.05,
        "url_length":            0.04,
        "digit_in_host":         0.04,
        "brand_in_path":         0.06,
    }
    score = sum(features.get(k, 0.0) * w for k, w in WEIGHTS.items())
    return min(score, 1.0)


def top_indicators(features: Dict[str, float], n: int = 5) -> List[Dict]:
    """Return the n most suspicious features as a ranked list."""
    WEIGHTS = {
        "has_ip_address":        0.15,
        "has_at_symbol":         0.10,
        "brand_in_subdomain":    0.12,
        "suspicious_tld":        0.10,
        "double_slash_redirect": 0.08,
        "non_std_port":          0.07,
        "hex_encoding":          0.06,
        "host_entropy":          0.08,
        "subdomain_depth":       0.05,
        "hyphen_count":          0.05,
        "url_length":            0.04,
        "digit_in_host":         0.04,
        "brand_in_path":         0.06,
        "path_depth":            0.03,
        "query_params":          0.03,
        "digit_ratio":           0.03,
    }
    scored = [
        {
            "feature": k,
            "value":   round(features.get(k, 0.0), 3),
            "weight":  w,
            "impact":  round(features.get(k, 0.0) * w, 4),
        }
        for k, w in WEIGHTS.items()
        if features.get(k, 0.0) > 0
    ]
    scored.sort(key=lambda x: x["impact"], reverse=True)
    return scored[:n]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_urls = [
        "https://www.google.com/search?q=hello",
        "http://paypal-secure-login.xyz/verify?user=victim@email.com",
        "http://192.168.1.1/admin/login",
        "https://accounts.google.com.phishing-site.tk/signin",
    ]
    for url in test_urls:
        feats = extract_features(url)
        score = feature_risk_score(feats)
        inds  = top_indicators(feats, n=3)
        print(f"\n{url[:60]}")
        print(f"  Feature risk: {score:.3f}")
        for ind in inds:
            print(f"  · {ind['feature']:<26} val={ind['value']:.2f}  impact={ind['impact']:.3f}")
