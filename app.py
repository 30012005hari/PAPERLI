"""
Research Paper Explainer & Implementation Generator
====================================================
A fully offline Streamlit application that analyses AI research papers
and generates explanations, architecture breakdowns, runnable code,
downloadable project folders, and deployment instructions.

Powered by local LLMs via Ollama.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path for imports ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, MAX_PDF_SIZE_MB
from modules.pdf_extractor import extract_text
from modules.section_parser import parse_sections
from modules.prompt_engine import (
    build_architecture_prompt,
    build_code_prompt,
    build_deployment_prompt,
    build_explanation_prompt,
    build_flashcards_prompt,
    build_equations_prompt,
    build_critique_prompt,
    build_chat_prompt,
    build_comparison_prompt,
)
from modules.llm_client import generate_streaming as ollama_streaming, is_ollama_running, list_models, list_models_ranked
from modules.gemini_client import generate_streaming as gemini_streaming, GEMINI_MODELS
from modules.code_generator import parse_code_blocks
from modules.project_builder import build_project_zip, file_tree_string


# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Research Paper Explainer",
    page_icon="RP",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inject clean white theme ─────────────────────────────────────────────
_force_light = """
<style>
    html, body, [data-testid="stAppViewContainer"], .stApp,
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    .main, .main .block-container {
        background-color: #f5f5f7 !important;
        color: #111111 !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: transparent !important;
    }
</style>
"""
st.markdown(_force_light, unsafe_allow_html=True)

_css_path = PROJECT_ROOT / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Startup splash overlay ───────────────────────────────────────────────
_splash = """
<div class="startup-overlay" id="splash">
    <span class="splash-logo">Research Paper Explainer</span>
    <span class="splash-sub">Analyse. Understand. Build.</span>
    <div class="splash-bar"></div>
</div>
"""
st.markdown(_splash, unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════
# Session-state defaults
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    "extracted_text": None,
    "pages": None,
    "sections": None,
    "explanation": None,
    "architecture": None,
    "code_raw": None,
    "code_files": None,
    "deployment": None,
    "flashcards": None,
    "equations": None,
    "critique": None,
    "comparison": None,
    "chat_history": [],
    "zip_bytes": None,
    "analysis_done": False,
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


with st.sidebar:
    st.markdown("### Settings")
    st.markdown("")

    # Provider selector
    provider = st.radio(
        "Provider",
        ["Ollama (Offline)", "Gemini (Cloud)"],
        horizontal=True,
    )
    use_gemini = "Gemini" in provider

    if not use_gemini:
        ollama_ok = is_ollama_running()
        if ollama_ok:
            st.success("Ollama is running")
        else:
            st.error("Ollama not reachable. Run `ollama serve`.")

        ranked_models = list_models_ranked() if ollama_ok else []
        available_models = [m["name"] for m in ranked_models]
        if available_models:
            model_labels = []
            for m in ranked_models:
                size_gb = m["size"] / 1e9
                if size_gb > 0.1:
                    model_labels.append(f"{m['name']}  ({size_gb:.1f} GB)")
                else:
                    model_labels.append(f"{m['name']}  (cloud)")
            st.caption(f"Recommended: **{available_models[0]}**")
            selected_label = st.selectbox("Model", model_labels, index=0)
            selected_model = available_models[model_labels.index(selected_label)]
        else:
            selected_model = st.text_input("Model name", value=DEFAULT_MODEL)
        provider_ready = ollama_ok
        gemini_key = None
    else:
        ollama_ok = False
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your key at https://aistudio.google.com/apikey",
        )
        if gemini_key:
            st.success("API key set")
        else:
            st.warning("Enter your Gemini API key to proceed.")

        selected_model = st.selectbox("Model", GEMINI_MODELS, index=0)
        provider_ready = bool(gemini_key)

    st.markdown("")

    # Detail level — replaces temperature
    _DETAIL_LEVELS = {
        "Concise": 0.1,
        "Balanced": 0.4,
        "Detailed": 0.7,
        "Creative": 0.95,
    }
    detail_level = st.radio(
        "Detail Level",
        list(_DETAIL_LEVELS.keys()),
        index=1,
        horizontal=True,
        help="How detailed and creative the analysis should be.",
    )
    temperature = _DETAIL_LEVELS[detail_level]

    st.markdown("")
    st.divider()
    if use_gemini:
        st.caption("Cloud mode — requires internet and API key.")
    else:
        st.caption("Offline mode — all processing stays on your machine.")

# Header — editorial, centered
st.markdown("")
st.markdown("")
st.markdown('<h1 style="text-align:center;">Research Paper Explainer</h1>', unsafe_allow_html=True)
st.caption(
    "Upload a PDF. Get explanations, architecture, code, and deployment instructions."
)
st.markdown("")
st.markdown("")


# Upload
uploaded_file = st.file_uploader(
    "Drop your research paper here",
    type=["pdf"],
    help=f"Max {MAX_PDF_SIZE_MB} MB",
)

if uploaded_file is not None:
    # Size check
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        st.error(f"File is {size_mb:.1f} MB — max allowed is {MAX_PDF_SIZE_MB} MB.")
        st.stop()

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("File", uploaded_file.name)
    col_info2.metric("Size", f"{size_mb:.2f} MB")
    col_info3.metric("Status", "Ready" if provider_ready else "Not ready")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: stream LLM & show live tokens (dispatches to Ollama or Gemini)
# ═══════════════════════════════════════════════════════════════════════════
def _stream_and_collect(prompt: str, model: str, temp: float, placeholder) -> str:
    """Stream LLM tokens into a Streamlit placeholder and return full text."""
    try:
        if use_gemini:
            stream = gemini_streaming(prompt, model, gemini_key, temp)
        else:
            stream = ollama_streaming(prompt, model, temp)

        tokens: list[str] = []
        for tok in stream:
            tokens.append(tok)
            if len(tokens) % 3 == 0:
                placeholder.markdown("".join(tokens) + "▌")
        full = "".join(tokens)
        placeholder.markdown(full)
        return full
    except (RuntimeError, ConnectionError) as e:
        error_msg = str(e)
        if "403" in error_msg:
            placeholder.error(
                "**API Error (403):** Your Gemini API key doesn't have access. "
                "Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
            )
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            placeholder.error(
                "**Timeout:** The model took too long. Try a smaller model."
            )
        else:
            placeholder.error(f"**Error:** {error_msg[:300]}")
        return ""
    except Exception as e:
        placeholder.error(f"**Error:** {str(e)[:300]}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Analyse button
# ═══════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    st.markdown("")
    analyse_clicked = st.button(
        "Analyse Paper",
        use_container_width=True,
        disabled=not provider_ready,
    )

    if analyse_clicked:
        # Reset previous results
        for key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]

        with st.status("Analysing paper...", expanded=True) as status:

            st.write("Extracting text from PDF...")
            full_text, pages = extract_text(uploaded_file)
            st.session_state["extracted_text"] = full_text
            st.session_state["pages"] = pages
            st.write(f"Extracted {len(pages)} pages, {len(full_text):,} characters.")

            st.write("Identifying paper sections...")
            sections = parse_sections(full_text)
            st.session_state["sections"] = sections
            detected = [k for k in sections if k != "full_text"]
            if detected:
                st.write(f"Detected: {', '.join(s.title() for s in detected)}")
            else:
                st.write("No named sections found. Using full text.")

            st.write("Generating explanation...")
            prompt_explain = build_explanation_prompt(sections)
            ph_explain = st.empty()
            explanation = _stream_and_collect(prompt_explain, selected_model, temperature, ph_explain)
            st.session_state["explanation"] = explanation

            st.write("Breaking down architecture...")
            prompt_arch = build_architecture_prompt(sections)
            ph_arch = st.empty()
            architecture = _stream_and_collect(prompt_arch, selected_model, temperature, ph_arch)
            st.session_state["architecture"] = architecture

            st.write("Generating implementation code...")
            prompt_code = build_code_prompt(sections)
            ph_code = st.empty()
            code_raw = _stream_and_collect(prompt_code, selected_model, temperature, ph_code)
            code_files = parse_code_blocks(code_raw)
            st.session_state["code_raw"] = code_raw
            st.session_state["code_files"] = code_files
            st.write(f"Generated {len(code_files)} files.")

            st.write("Generating deployment instructions...")
            prompt_deploy = build_deployment_prompt(sections)
            ph_deploy = st.empty()
            deployment = _stream_and_collect(prompt_deploy, selected_model, temperature, ph_deploy)
            st.session_state["deployment"] = deployment

            st.write("Creating flashcards...")
            prompt_flash = build_flashcards_prompt(sections)
            ph_flash = st.empty()
            flashcards = _stream_and_collect(prompt_flash, selected_model, temperature, ph_flash)
            st.session_state["flashcards"] = flashcards

            st.write("Extracting key equations...")
            prompt_eq = build_equations_prompt(sections)
            ph_eq = st.empty()
            equations = _stream_and_collect(prompt_eq, selected_model, temperature, ph_eq)
            st.session_state["equations"] = equations

            st.write("Writing critique...")
            prompt_crit = build_critique_prompt(sections)
            ph_crit = st.empty()
            critique = _stream_and_collect(prompt_crit, selected_model, temperature, ph_crit)
            st.session_state["critique"] = critique

            st.write("Comparing with similar research...")
            prompt_comp = build_comparison_prompt(sections)
            ph_comp = st.empty()
            comparison = _stream_and_collect(prompt_comp, selected_model, temperature, ph_comp)
            st.session_state["comparison"] = comparison

            if code_files:
                st.write("Building project archive...")
                zip_bytes = build_project_zip(code_files)
                st.session_state["zip_bytes"] = zip_bytes

            st.session_state["analysis_done"] = True
            status.update(label="Analysis complete", state="complete", expanded=False)


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════
def _detect_language(filename: str) -> str:
    """Map file extension to Streamlit code-block language."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "py": "python",
        "yaml": "yaml",
        "yml": "yaml",
        "txt": "text",
        "md": "markdown",
        "json": "json",
        "sh": "bash",
        "dockerfile": "dockerfile",
    }.get(ext, "text")


# ═══════════════════════════════════════════════════════════════════════════
# Results tabs — shown when analysis is done
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.get("analysis_done"):
    st.markdown("---")
    st.markdown("## Results")

    tab_explain, tab_arch, tab_flash, tab_eq, tab_crit, tab_comp, tab_code, tab_download, tab_deploy = st.tabs([
        "Explanation",
        "Architecture",
        "Flashcards",
        "Equations",
        "Critique",
        "Compare",
        "Code",
        "Download",
        "Deployment",
    ])

    # -- Explanation tab
    with tab_explain:
        st.markdown(st.session_state["explanation"])

    # -- Architecture tab
    with tab_arch:
        st.markdown(st.session_state["architecture"])

    # -- Flashcards tab
    with tab_flash:
        st.markdown(st.session_state["flashcards"])

    # -- Equations tab
    with tab_eq:
        st.markdown(st.session_state["equations"])

    # -- Critique tab
    with tab_crit:
        st.markdown(st.session_state["critique"])

    # -- Compare tab
    with tab_comp:
        st.markdown(st.session_state["comparison"])

    # -- Code tab
    with tab_code:
        code_files: dict[str, str] = st.session_state.get("code_files", {})
        if code_files:
            for fname, code in code_files.items():
                lang = _detect_language(fname)
                with st.expander(fname, expanded=True):
                    st.code(code, language=lang, line_numbers=True)
        else:
            st.info("No code files were generated. The LLM response may not have used the expected format.")
            with st.expander("Raw LLM output"):
                st.text(st.session_state.get("code_raw", ""))

    # -- Download tab
    with tab_download:
        zip_bytes = st.session_state.get("zip_bytes")
        code_files = st.session_state.get("code_files", {})

        if zip_bytes:
            st.markdown("### Project Structure")
            st.code(file_tree_string(code_files), language="text")

            st.download_button(
                label="Download Project (.zip)",
                data=zip_bytes,
                file_name="research_implementation.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.warning("No project files to download.")

    # -- Deployment tab
    with tab_deploy:
        st.markdown(st.session_state["deployment"])

    # ═══════════════════════════════════════════════════════════════════════
    # Chat — ask questions about the paper
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## Ask about this paper")
    st.caption("Ask any question about the paper — methods, results, implications, anything.")

    # Display chat history
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_q := st.chat_input("Ask a question about the paper..."):
        # Show user message
        st.session_state["chat_history"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Generate answer
        sections = st.session_state["sections"]
        prompt = build_chat_prompt(sections, user_q)
        with st.chat_message("assistant"):
            ph = st.empty()
            answer = _stream_and_collect(prompt, selected_model, temperature, ph)
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
