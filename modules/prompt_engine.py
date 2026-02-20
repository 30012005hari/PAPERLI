"""
Prompt construction engine.

Loads template files from ``prompts/`` and fills them with extracted
research-paper sections to build structured prompts for the LLM.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_template(name: str) -> str:
    """Read a prompt template from the prompts/ folder."""
    path = _PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _sections_block(sections: dict[str, str], keys: list[str] | None = None) -> str:
    """
    Format selected *sections* into a labelled text block.

    If *keys* is ``None``, all sections except ``full_text`` are included.
    """
    if keys is None:
        keys = [k for k in sections if k != "full_text"]
    parts: list[str] = []
    for key in keys:
        content = sections.get(key, "")
        if content:
            parts.append(f"### {key.replace('_', ' ').title()}\n{content}")
    return "\n\n".join(parts) if parts else sections.get("full_text", "")


# ── public builders ──────────────────────────────────────────────────────────

def build_explanation_prompt(sections: dict[str, str]) -> str:
    """Return a prompt asking the LLM to explain the paper simply."""
    template = _load_template("explain.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_architecture_prompt(sections: dict[str, str]) -> str:
    """Return a prompt asking for an architecture breakdown."""
    template = _load_template("architecture.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_code_prompt(sections: dict[str, str]) -> str:
    """Return a prompt asking the LLM to generate implementation code."""
    template = _load_template("code_gen.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_deployment_prompt(sections: dict[str, str]) -> str:
    """Return a prompt asking for deployment & optimisation instructions."""
    template = _load_template("deployment.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_flashcards_prompt(sections: dict[str, str]) -> str:
    """Return a prompt asking the LLM to create study flashcards."""
    template = _load_template("flashcards.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_equations_prompt(sections: dict[str, str]) -> str:
    """Return a prompt asking the LLM to extract key equations."""
    template = _load_template("equations.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_critique_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for a peer-review style critique."""
    template = _load_template("critique.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_chat_prompt(sections: dict[str, str], question: str) -> str:
    """Return a prompt for answering a user question about the paper."""
    template = _load_template("chat.txt")
    return (
        template
        .replace("{{PAPER_CONTENT}}", _sections_block(sections))
        .replace("{{USER_QUESTION}}", question)
    )


def build_comparison_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for comparing with the most similar paper."""
    template = _load_template("comparison.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_datasets_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for identifying datasets and suggesting alternatives."""
    template = _load_template("datasets.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


# ── New section-based builders (research engine v2) ──────────────────────────

def build_overview_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for paper overview: metadata + problem + contribution."""
    template = _load_template("overview.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_technical_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for technical breakdown: architecture, math, algorithm."""
    template = _load_template("technical_breakdown.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_dataset_eval_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for dataset analysis and evaluation metrics."""
    template = _load_template("dataset_eval.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_analysis_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for critical analysis: results, strengths, weaknesses."""
    template = _load_template("analysis.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))


def build_verdict_prompt(sections: dict[str, str]) -> str:
    """Return a prompt for final technical verdict with scores."""
    template = _load_template("verdict.txt")
    return template.replace("{{PAPER_CONTENT}}", _sections_block(sections))
