"""
Global configuration constants for the Research Paper Explainer.
"""

# ── Ollama API ───────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

# ── LLM defaults ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "deepseek-coder"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_NUM_CTX = 4096          # context window tokens (smaller = faster)

# ── PDF constraints ──────────────────────────────────────────────────────────
MAX_PDF_SIZE_MB = 50
MAX_TEXT_CHARS = 15_000         # truncate papers to keep prompts fast

# ── Section keywords used by the parser ──────────────────────────────────────
SECTION_KEYWORDS: dict[str, list[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "related_work": ["related work", "literature review", "background"],
    "methodology": [
        "methodology", "methods", "method", "proposed method",
        "proposed approach", "approach", "framework",
    ],
    "architecture": [
        "architecture", "model architecture", "network architecture",
        "model design", "system design", "model",
    ],
    "dataset": ["dataset", "data", "data collection", "datasets"],
    "experiments": [
        "experiments", "experimental setup", "experimental results",
        "results", "evaluation", "results and discussion",
    ],
    "hyperparameters": [
        "hyperparameters", "hyper-parameters", "training details",
        "implementation details", "training setup",
    ],
    "conclusion": ["conclusion", "conclusions", "summary", "future work"],
}

# ── File markers used to split generated code blocks ─────────────────────────
CODE_FILE_MARKER_START = "```"
CODE_FILE_MARKER_END = "```"
