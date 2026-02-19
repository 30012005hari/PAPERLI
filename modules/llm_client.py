"""
Ollama LLM client.

Provides streaming generation and model listing via the Ollama REST API.
Fully offline — no external network calls.
"""

from __future__ import annotations

import json
from typing import Generator

import requests

from config import (
    DEFAULT_NUM_CTX,
    DEFAULT_TEMPERATURE,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
)


def list_models() -> list[str]:
    """
    Return a list of model names available in the local Ollama instance.

    Returns an empty list if Ollama is unreachable.
    """
    try:
        resp = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def list_models_ranked() -> list[dict]:
    """
    Return models sorted by size (largest first) — bigger models are better.

    Each entry: {"name": str, "size": int}
    Cloud/tiny models are pushed to the bottom.
    """
    try:
        resp = requests.get(OLLAMA_TAGS_ENDPOINT, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [
            {"name": m["name"], "size": m.get("size", 0)}
            for m in data.get("models", [])
        ]
        # Sort by size descending — largest (best) first
        models.sort(key=lambda m: m["size"], reverse=True)
        return models
    except Exception:
        return []



def is_ollama_running() -> bool:
    """Quick health check — can we reach Ollama?"""
    try:
        requests.get(OLLAMA_TAGS_ENDPOINT, timeout=3)
        return True
    except Exception:
        return False


def generate_streaming(
    prompt: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> Generator[str, None, None]:
    """
    Call Ollama's ``/api/generate`` endpoint with streaming enabled.

    Yields text tokens one-by-one.

    Raises
    ------
    ConnectionError
        If Ollama is not reachable.
    RuntimeError
        If the API returns a non-200 status.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    try:
        resp = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json=payload,
            stream=True,
            timeout=(10, 900),  # (connect, read) — large models need time
        )
    except requests.ConnectionError as exc:
        raise ConnectionError(
            "Cannot reach Ollama. Is `ollama serve` running?"
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        token = chunk.get("response", "")
        if token:
            yield token
        if chunk.get("done", False):
            break


def generate_full(
    prompt: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """Non-streaming convenience wrapper — returns the full response text."""
    tokens: list[str] = []
    for tok in generate_streaming(prompt, model, temperature, num_ctx):
        tokens.append(tok)
    return "".join(tokens)
