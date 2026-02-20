"""
Dataset link extractor and verifier.

Scans paper text for URLs (especially dataset links), then verifies each
link's availability via HTTP HEAD requests. Returns structured info about
each dataset found.
"""

from __future__ import annotations

import re
import json
from typing import Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# ── Known dataset hosting domains (boost confidence for these) ───────────────
_DATASET_DOMAINS = {
    "kaggle.com", "huggingface.co", "github.com", "zenodo.org",
    "drive.google.com", "archive.org", "paperswithcode.com",
    "data.gov", "figshare.com", "datadryad.org", "dataverse.harvard.edu",
    "openml.org", "tensorflow.org", "pytorch.org", "aws.amazon.com",
    "storage.googleapis.com", "data.world", "ieee-dataport.org",
    "sci-hub.se", "arxiv.org",
}

# ── URL regex ────────────────────────────────────────────────────────────────
_URL_PATTERN = re.compile(
    r'https?://[^\s\)\]\},;\"\'<>]+',
    re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    """Extract all URLs from the given text, deduplicated and cleaned."""
    raw = _URL_PATTERN.findall(text)
    seen = set()
    urls = []
    for url in raw:
        # Strip trailing punctuation
        url = url.rstrip(".,;:!?")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _classify_url(url: str) -> str:
    """Classify a URL as 'dataset', 'code', 'paper', or 'other'."""
    lower = url.lower()
    parsed = urlparse(lower)
    domain = parsed.netloc.replace("www.", "")

    # Dataset indicators
    if any(d in domain for d in ["kaggle.com", "huggingface.co/datasets",
                                  "zenodo.org", "figshare.com", "datadryad.org",
                                  "openml.org", "data.world", "data.gov"]):
        return "dataset"
    if any(kw in lower for kw in ["/dataset", "/data/", "download", "benchmark"]):
        return "dataset"

    # Code indicators
    if "github.com" in domain and "/datasets" not in lower:
        return "code"

    # Paper indicators
    if any(d in domain for d in ["arxiv.org", "doi.org", "semanticscholar.org"]):
        return "paper"

    return "other"


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes < 0:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def verify_url(url: str, timeout: int = 10) -> dict:
    """
    Verify a single URL via HTTP HEAD/GET.

    Returns a dict with:
        url, status, status_code, content_type, size, size_human, category
    """
    result = {
        "url": url,
        "status": "unknown",
        "status_code": None,
        "content_type": None,
        "size": -1,
        "size_human": "Unknown",
        "category": _classify_url(url),
    }

    try:
        # Try HEAD first (lightweight)
        resp = requests.head(url, timeout=timeout, allow_redirects=True,
                             headers={"User-Agent": "ResearchPaperAnalyzer/1.0"})

        result["status_code"] = resp.status_code
        result["content_type"] = resp.headers.get("Content-Type", "unknown")

        if resp.status_code < 400:
            result["status"] = "available"
            content_length = resp.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                result["size"] = int(content_length)
                result["size_human"] = _format_size(int(content_length))
        elif resp.status_code == 403:
            result["status"] = "restricted"
        elif resp.status_code == 404:
            result["status"] = "not_found"
        else:
            result["status"] = "error"

    except requests.Timeout:
        result["status"] = "timeout"
    except requests.ConnectionError:
        result["status"] = "unreachable"
    except Exception:
        result["status"] = "error"

    return result


def extract_and_verify(text: str, max_urls: int = 30, timeout: int = 8) -> list[dict]:
    """
    Extract URLs from paper text and verify them in parallel.

    Returns a list of dicts sorted by category (datasets first).
    """
    urls = extract_urls(text)[:max_urls]

    if not urls:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(verify_url, url, timeout): url for url in urls}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                results.append({
                    "url": futures[future],
                    "status": "error",
                    "status_code": None,
                    "content_type": None,
                    "size": -1,
                    "size_human": "Unknown",
                    "category": "other",
                })

    # Sort: datasets first, then code, then papers, then other
    priority = {"dataset": 0, "code": 1, "paper": 2, "other": 3}
    results.sort(key=lambda r: priority.get(r["category"], 99))

    return results


def results_to_json(results: list[dict]) -> str:
    """Serialize results to JSON string for DB storage."""
    return json.dumps(results, default=str)


def results_from_json(json_str: str) -> list[dict]:
    """Deserialize results from JSON string."""
    if not json_str:
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []
