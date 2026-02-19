"""
Section segmentation module.

Identifies standard research-paper sections (Abstract, Methods, etc.)
from extracted text and returns a structured dictionary.
"""

from __future__ import annotations

import re
from typing import Optional

from config import SECTION_KEYWORDS, MAX_TEXT_CHARS


def parse_sections(full_text: str) -> dict[str, str]:
    """
    Split *full_text* into labelled sections.

    Strategy
    --------
    1. Build a combined regex from SECTION_KEYWORDS.
    2. Find all heading-like lines (numbered or capitalised).
    3. Map content between headings to canonical section names.
    4. Fall back to a simple ``{"full_text": ...}`` if nothing matched.
    """
    # ── build heading pattern ────────────────────────────────────────────
    all_keywords: list[str] = []
    keyword_to_canonical: dict[str, str] = {}
    for canonical, variants in SECTION_KEYWORDS.items():
        for v in variants:
            all_keywords.append(re.escape(v))
            keyword_to_canonical[v.lower()] = canonical

    # Match lines like  "3. Methodology"  or  "METHODOLOGY"  or  "## Methods"
    heading_pattern = re.compile(
        r"^(?:\#{1,4}\s*)?(?:\d{1,2}[\.\)]\s*)?"  # optional # or number prefix
        r"(" + "|".join(all_keywords) + r")"
        r"\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(full_text))

    if not matches:
        return {"full_text": _truncate(full_text)}

    sections: dict[str, str] = {}

    for idx, match in enumerate(matches):
        keyword = match.group(1).strip().lower()
        canonical = keyword_to_canonical.get(keyword, keyword)

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        if canonical in sections:
            sections[canonical] += "\n\n" + content
        else:
            sections[canonical] = content

    # Always keep a full_text key as fallback context
    sections["full_text"] = _truncate(full_text)
    return sections


def _truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Truncate text to *max_chars*, appending an ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[… truncated …]"
