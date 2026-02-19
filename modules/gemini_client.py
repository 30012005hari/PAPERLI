"""
Google Gemini API client.

Provides streaming generation via the Gemini REST API.
Requires a GEMINI_API_KEY set in the Streamlit sidebar or environment.
"""

from __future__ import annotations

import json
import os
from typing import Generator

import requests


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Models available via Gemini
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def generate_streaming(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.4,
) -> Generator[str, None, None]:
    """
    Call Gemini's streamGenerateContent endpoint.

    Yields text chunks as they arrive.
    """
    url = (
        f"{GEMINI_API_BASE}/models/{model}:streamGenerateContent"
        f"?alt=sse&key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=(10, 600))
    except requests.ConnectionError as exc:
        raise ConnectionError("Cannot reach Gemini API. Check your internet.") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Gemini returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        # Extract text from Gemini response structure
        candidates = chunk.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                text = part.get("text", "")
                if text:
                    yield text


def generate_full(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.4,
) -> str:
    """Non-streaming convenience wrapper."""
    tokens: list[str] = []
    for tok in generate_streaming(prompt, model, api_key, temperature):
        tokens.append(tok)
    return "".join(tokens)
