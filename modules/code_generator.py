"""
Code generation orchestrator.

Sends the code-generation prompt to the LLM, then parses the response
to extract individual source files (model.py, train.py, etc.).
"""

from __future__ import annotations

import re
from typing import Generator

from modules.llm_client import generate_streaming


# ── response parser ──────────────────────────────────────────────────────────

def parse_code_blocks(response: str) -> dict[str, str]:
    """
    Extract fenced code blocks tagged with filenames from *response*.

    Expected LLM output format::

        ```python  # model.py
        import torch
        ...
        ```

    Also accepts:
        ```python
        # filename: model.py
        ...
        ```

    Returns
    -------
    files : dict[str, str]
        Mapping of filename → source code.
    """
    # Pattern: ```<lang>  # <filename>\n<code>\n```
    pattern = re.compile(
        r"```[\w]*\s*(?:#|//|filename:?)\s*"      # opening fence + comment marker
        r"([\w\-.]+\.[\w]+)\s*\n"                   # captured filename
        r"(.*?)"                                    # captured code body
        r"\n```",                                   # closing fence
        re.DOTALL,
    )

    files: dict[str, str] = {}
    for match in pattern.finditer(response):
        fname = match.group(1).strip()
        code = match.group(2).strip()
        files[fname] = code

    # Fallback: if nothing matched, try a looser pattern
    if not files:
        files = _fallback_parse(response)

    return files


def _fallback_parse(response: str) -> dict[str, str]:
    """
    Looser parser — looks for any fenced block and tries to guess the
    filename from the first comment line inside the block.
    """
    block_pattern = re.compile(
        r"```[\w]*\n(.*?)\n```",
        re.DOTALL,
    )
    files: dict[str, str] = {}
    counter = 1
    for match in block_pattern.finditer(response):
        code = match.group(1).strip()
        # Try to find a filename hint
        first_line = code.split("\n", 1)[0]
        fname_match = re.search(r"([\w\-]+\.[\w]+)", first_line)
        if fname_match and "." in fname_match.group(1):
            fname = fname_match.group(1)
            # Remove the comment line from code
            code = code.split("\n", 1)[1].strip() if "\n" in code else code
        else:
            fname = f"generated_{counter}.py"
            counter += 1
        files[fname] = code

    return files


# ── streaming code generation ────────────────────────────────────────────────

def generate_code(
    prompt: str,
    model: str,
    temperature: float = 0.3,
) -> tuple[str, dict[str, str]]:
    """
    Generate implementation code by streaming from the LLM, then parse files.

    Returns
    -------
    raw_response : str
        The full LLM output.
    files : dict[str, str]
        Parsed filename → code mapping.
    """
    tokens: list[str] = []
    for tok in generate_streaming(prompt, model, temperature):
        tokens.append(tok)

    raw_response = "".join(tokens)
    files = parse_code_blocks(raw_response)
    return raw_response, files


def stream_code_generation(
    prompt: str,
    model: str,
    temperature: float = 0.3,
) -> Generator[str, None, None]:
    """
    Yield tokens as they arrive from the LLM.

    The caller should collect them and later call ``parse_code_blocks``
    on the accumulated text.
    """
    yield from generate_streaming(prompt, model, temperature)
