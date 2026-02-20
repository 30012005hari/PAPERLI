"""
Research Paper Explainer & Implementation Generator
====================================================
A fully offline Streamlit application that analyses AI research papers
and generates explanations, architecture breakdowns, runnable code,
downloadable project folders, and deployment instructions.

Powered by local LLMs via Ollama.
"""

from __future__ import annotations

import json
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
    build_overview_prompt,
    build_architecture_prompt,
    build_technical_prompt,
    build_equations_prompt,
    build_analysis_prompt,
    build_verdict_prompt,
    build_dataset_eval_prompt,
    build_code_prompt,
    build_deployment_prompt,
    build_chat_prompt,
)
from modules.dataset_extractor import extract_and_verify, results_to_json, results_from_json
from modules.llm_client import generate_streaming as ollama_streaming, is_ollama_running, is_ollama_running_cached, list_models, list_models_ranked, list_models_ranked_cached
from modules.gemini_client import generate_streaming as gemini_streaming, GEMINI_MODELS
from modules.code_generator import parse_code_blocks
from modules.project_builder import build_project_zip, file_tree_string
from backend.database import SessionLocal, Paper, Analysis, ChatMessage


# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Paperli — AI Research Paper Analyzer",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# (Theme is enforced via assets/style.css — no inline override needed)


def _load_css() -> str:
    """Read CSS file (uncached so edits apply immediately)."""
    css_path = PROJECT_ROOT / "assets" / "style.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""

_css_content = _load_css()
if _css_content:
    st.markdown(f"<style>{_css_content}</style>", unsafe_allow_html=True)

# ── Startup splash overlay (only on first load) ─────────────────────────
if "_splash_shown" not in st.session_state:
    st.session_state["_splash_shown"] = True
    _splash = """
    <div class="startup-overlay" id="splash">
        <span class="splash-logo">Paperli</span>
        <span class="splash-sub">Start Your Research</span>
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
    # Section-based analysis (research engine v2)
    "overview": None,
    "technical": None,
    "dataset_eval": None,
    "analysis": None,
    "verdict": None,
    # Utility sections
    "code_raw": None,
    "code_files": None,
    "deployment": None,
    "datasets_json": None,
    # Legacy keys (for backward compat with old DB analyses)
    "explanation": None,
    "architecture": None,
    "flashcards": None,
    "equations": None,
    "critique": None,
    "comparison": None,
    "datasets_llm": None,
    # Shared state
    "chat_history": [],
    "zip_bytes": None,
    "analysis_done": False,
    "current_paper_id": None,
    "current_analysis_id": None,
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val.copy() if isinstance(val, (list, dict)) else val


# ── DB helper ────────────────────────────────────────────────────────────────
def _get_db():
    return SessionLocal()


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
        ollama_ok = is_ollama_running_cached()
        if ollama_ok:
            st.success("Ollama is running")
        else:
            st.error("Ollama not reachable. Run `ollama serve`.")

        ranked_models = list_models_ranked_cached() if ollama_ok else []
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

    # ── Paper History ────────────────────────────────────────────────────
    st.markdown("")
    st.divider()
    st.markdown("### Paper History")
    db = _get_db()
    try:
        past_papers = db.query(Paper).order_by(Paper.created_at.desc()).limit(20).all()
        # Pre-load analysis counts to avoid N+1 queries
        if past_papers:
            for pp in past_papers:
                analysis_count = len(pp.analyses)
                label = f"{pp.filename}  ({analysis_count} analysis{'es' if analysis_count != 1 else ''})"
                if st.button(label, key=f"hist_{pp.id}", use_container_width=True):
                    latest = (
                        db.query(Analysis)
                        .filter(Analysis.paper_id == pp.id)
                        .order_by(Analysis.created_at.desc())
                        .first()
                    )
                    if latest:
                        # Map DB columns → new section keys
                        st.session_state["overview"] = latest.explanation
                        st.session_state["technical"] = latest.architecture
                        st.session_state["dataset_eval"] = latest.datasets_llm
                        st.session_state["analysis"] = latest.critique
                        st.session_state["verdict"] = latest.comparison
                        # Also set legacy keys for backward compat
                        st.session_state["explanation"] = latest.explanation
                        st.session_state["architecture"] = latest.architecture
                        st.session_state["code_raw"] = latest.code_raw
                        st.session_state["code_files"] = parse_code_blocks(latest.code_raw or "")
                        st.session_state["deployment"] = latest.deployment
                        st.session_state["flashcards"] = latest.flashcards
                        st.session_state["equations"] = latest.equations
                        st.session_state["critique"] = latest.critique
                        st.session_state["comparison"] = latest.comparison
                        st.session_state["datasets_json"] = latest.datasets_json
                        st.session_state["datasets_llm"] = latest.datasets_llm
                        st.session_state["zip_bytes"] = latest.zip_bytes
                        st.session_state["sections"] = json.loads(pp.sections_json)
                        st.session_state["extracted_text"] = pp.extracted_text
                        st.session_state["pages"] = None
                        st.session_state["analysis_done"] = True
                        st.session_state["current_paper_id"] = pp.id
                        st.session_state["current_analysis_id"] = latest.id
                        chat_msgs = (
                            db.query(ChatMessage)
                            .filter(ChatMessage.analysis_id == latest.id)
                            .order_by(ChatMessage.created_at.asc())
                            .all()
                        )
                        st.session_state["chat_history"] = [
                            {"role": m.role, "content": m.content} for m in chat_msgs
                        ]
                        st.rerun()
                    else:
                        st.info("No analysis found. Upload and analyse again.")
        else:
            st.caption("No papers yet. Upload one above!")
    finally:
        db.close()

# ═══════════════════════════════════════════════════════════════════════════
# Show hero + upload ONLY when no analysis is active
# ═══════════════════════════════════════════════════════════════════════════
uploaded_file = None  # default; set inside the upload block

if not st.session_state.get("analysis_done"):
    # Header — animated hero with text logo
    import base64 as _b64
    _logo_svg = PROJECT_ROOT / "assets" / "logo.svg"
    _logo_b64 = _b64.b64encode(_logo_svg.read_bytes()).decode() if _logo_svg.exists() else ""
    _logo_tag = f'<img src="data:image/svg+xml;base64,{_logo_b64}" class="hero-logo-img" alt="Paperli"/>' if _logo_b64 else ""
    _hero_html = f'<div class="hero-container"><div class="hero-dots"><span></span><span></span><span></span><span></span><span></span><span></span></div><div class="hero-logo-wrap">{_logo_tag}</div><h1 class="hero-brand">Start Your Research</h1><p class="hero-subtitle">Drop a paper below — AI analyzes every section, equation, and insight.</p><div class="hero-tags"><span class="hero-tag">Overview</span><span class="hero-tag">Architecture</span><span class="hero-tag">Technical</span><span class="hero-tag">Equations</span><span class="hero-tag">Analysis</span><span class="hero-tag">Verdict</span><span class="hero-tag">Datasets</span><span class="hero-tag">Code</span><span class="hero-tag">Deploy</span></div></div>'
    st.markdown(_hero_html, unsafe_allow_html=True)
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
            if len(tokens) % 6 == 0:
                placeholder.markdown("".join(tokens) + "▌")
        full = "".join(tokens)
        placeholder.markdown(full)
        print(f"[LLM] Generated {len(full)} chars")
        return full
    except (RuntimeError, ConnectionError) as e:
        error_msg = str(e)
        print(f"[LLM ERROR] {error_msg[:300]}")
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
        print(f"[LLM EXCEPTION] {str(e)[:300]}")
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
        type="primary",
    )

    if analyse_clicked:
        # Reset previous results (copy mutable values to avoid sharing refs)
        for key, val in _DEFAULTS.items():
            st.session_state[key] = val.copy() if isinstance(val, (list, dict)) else val

        # ── Quick extraction pass (no LLM, just PDF parsing) ─────
        with st.spinner("Extracting text from PDF..."):
            full_text, pages = extract_text(uploaded_file)
            st.session_state["extracted_text"] = full_text
            st.session_state["pages"] = pages

            sections = parse_sections(full_text)
            st.session_state["sections"] = sections

            # ── Create Paper record upfront for incremental saves ────
            db = _get_db()
            paper_rec = None
            analysis_rec = None
            try:
                paper_rec = Paper(
                    filename=uploaded_file.name,
                    size_mb=round(size_mb, 2),
                    extracted_text=full_text,
                    page_count=len(pages),
                    sections_json=json.dumps(sections),
                )
                db.add(paper_rec)
                db.commit()
                db.refresh(paper_rec)
                st.session_state["current_paper_id"] = paper_rec.id

                analysis_rec = Analysis(
                    paper_id=paper_rec.id,
                    model_used=selected_model,
                    provider="gemini" if use_gemini else "ollama",
                    detail_level=detail_level,
                )
                db.add(analysis_rec)
                db.commit()
                db.refresh(analysis_rec)
                st.session_state["current_analysis_id"] = analysis_rec.id
                print(f"[DB] Created Paper #{paper_rec.id}, Analysis #{analysis_rec.id}")
            except Exception as db_err:
                print(f"[DB ERROR] Initial save failed: {db_err}")
                paper_rec = None
                analysis_rec = None
            finally:
                try:
                    db.close()
                except Exception:
                    pass

        # Mark tabs ready, flag that generation is needed
        st.session_state["analysis_done"] = True
        st.session_state["_needs_generation"] = True
        st.rerun()



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
# Helper: save a section to DB lazily
# ═══════════════════════════════════════════════════════════════════════════
def _save_section_to_db(db_column: str, value: str):
    """Save a lazily-generated section to the existing analysis record."""
    analysis_id = st.session_state.get("current_analysis_id")
    if not analysis_id or not value:
        return
    db = _get_db()
    try:
        rec = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if rec:
            setattr(rec, db_column, value)
            db.commit()
    except Exception:
        pass
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Helper: render mermaid diagrams
# ═══════════════════════════════════════════════════════════════════════════
import re as _re
import streamlit.components.v1 as _components

def _render_mermaid(mermaid_code: str, height: int = 500):
    """Render a mermaid diagram using the Mermaid JS CDN (UMD).
    Falls back to showing raw code if rendering fails."""
    # Clean up common LLM-generated mermaid issues
    clean_code = mermaid_code.strip()
    # Remove trailing semicolons from lines (common LLM mistake)
    clean_lines = []
    for line in clean_code.split('\n'):
        stripped = line.rstrip()
        if stripped.endswith(';') and not stripped.startswith('%%'):
            stripped = stripped[:-1]
        clean_lines.append(stripped)
    clean_code = '\n'.join(clean_lines)
    # Escape for safe embedding in HTML
    import html as _html_mod
    escaped_code = _html_mod.escape(clean_code)

    html = f"""<!DOCTYPE html>
<html><head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  body {{ margin:0; padding:16px; background:transparent; font-family:sans-serif; }}
  .mermaid {{ text-align:center; }}
  .fallback {{ display:none; padding:12px; background:#f8f8f8; border-radius:8px;
               font-family:monospace; font-size:12px; white-space:pre-wrap;
               color:#333; overflow-x:auto; }}
  .fallback-msg {{ display:none; color:#888; font-size:12px; margin-bottom:8px; }}
</style>
</head><body>
<div class="fallback-msg" id="errmsg">⚠ Diagram had syntax issues — showing raw code:</div>
<pre class="fallback" id="fallback">{escaped_code}</pre>
<div class="mermaid" id="diagram">
{clean_code}
</div>
<script>
  mermaid.initialize({{ startOnLoad: false, theme: 'default', securityLevel: 'loose' }});
  mermaid.run({{ querySelector: '#diagram' }}).catch(function(err) {{
    document.getElementById('diagram').style.display = 'none';
    document.getElementById('fallback').style.display = 'block';
    document.getElementById('errmsg').style.display = 'block';
  }});
</script>
</body></html>"""
    _components.html(html, height=height, scrolling=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: render a section with one-time fade-in + mermaid support
# ═══════════════════════════════════════════════════════════════════════════
_MERMAID_BLOCK_RE = _re.compile(r"```mermaid\r?\n(.*?)```", _re.DOTALL)

def _render_section(content: str, section_key: str):
    """Render markdown content, auto-converting ```mermaid blocks to diagrams."""
    reveal_key = f"_revealed_{section_key}"
    if reveal_key not in st.session_state:
        st.session_state[reveal_key] = True
        st.markdown(
            f'<div class="section-reveal">{""}</div>',
            unsafe_allow_html=True,
        )

    # Split content on mermaid blocks
    parts = _MERMAID_BLOCK_RE.split(content)
    # parts alternates: [text, mermaid_code, text, mermaid_code, ...]
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i % 2 == 0:
            # Regular markdown
            st.markdown(part)
        else:
            # Mermaid diagram code
            _render_mermaid(part)





# ═══════════════════════════════════════════════════════════════════════════
# Results tabs — shown when analysis is done
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.get("analysis_done"):
    st.markdown("---")
    _res_col1, _res_col2 = st.columns([4, 1])
    _res_col1.markdown("## Results")
    if _res_col2.button("↩ New Analysis", use_container_width=True):
        for key, val in _DEFAULTS.items():
            st.session_state[key] = val.copy() if isinstance(val, (list, dict)) else val
        st.rerun()

    tab_overview, tab_arch, tab_technical, tab_equations, tab_analysis, tab_verdict, tab_dataset, tab_code, tab_download, tab_deploy = st.tabs([
        "Overview",
        "Architecture",
        "Technical",
        "Equations",
        "Analysis",
        "Verdict",
        "Datasets",
        "Code",
        "Download",
        "Deploy",
    ])

    # ── If sections need generating, stream directly into tabs ──────
    if st.session_state.get("_needs_generation"):
        sections = st.session_state.get("sections", {})
        full_text = st.session_state.get("extracted_text", "")

        # Create placeholders inside EVERY tab upfront
        with tab_overview:
            ph_overview = st.empty()
        with tab_arch:
            ph_arch = st.empty()
        with tab_technical:
            ph_tech = st.empty()
        with tab_equations:
            ph_equations = st.empty()
        with tab_analysis:
            ph_analysis = st.empty()
        with tab_verdict:
            ph_verdict = st.empty()
        with tab_dataset:
            ph_dseval = st.empty()
        with tab_code:
            ph_code = st.empty()
        with tab_deploy:
            ph_deploy = st.empty()

        # DB helper
        def _save_field(field_name: str, value: str):
            analysis_id = st.session_state.get("current_analysis_id")
            if not analysis_id or not value:
                return
            db = _get_db()
            try:
                rec = db.query(Analysis).filter(Analysis.id == analysis_id).first()
                if rec:
                    setattr(rec, field_name, value)
                    db.commit()
            except Exception:
                pass
            finally:
                db.close()

        # ── Generate in TAB ORDER so user reads as content arrives ──

        # 1. Overview
        overview = _stream_and_collect(
            build_overview_prompt(sections), selected_model, temperature, ph_overview
        )
        st.session_state["overview"] = overview
        _save_field("explanation", overview)

        # 2. Architecture
        arch = _stream_and_collect(
            build_architecture_prompt(sections), selected_model, temperature, ph_arch
        )
        st.session_state["architecture"] = arch
        _save_field("architecture", arch)

        # 3. Technical
        tech = _stream_and_collect(
            build_technical_prompt(sections), selected_model, temperature, ph_tech
        )
        st.session_state["technical"] = tech
        _save_field("flashcards", tech)

        # 4. Equations
        equations = _stream_and_collect(
            build_equations_prompt(sections), selected_model, temperature, ph_equations
        )
        st.session_state["equations"] = equations
        _save_field("equations", equations)

        # 5. Analysis
        analysis_text = _stream_and_collect(
            build_analysis_prompt(sections), selected_model, temperature, ph_analysis
        )
        st.session_state["analysis"] = analysis_text
        _save_field("critique", analysis_text)

        # 6. Verdict
        verdict = _stream_and_collect(
            build_verdict_prompt(sections), selected_model, temperature, ph_verdict
        )
        st.session_state["verdict"] = verdict
        _save_field("comparison", verdict)

        # 6. Datasets (fast link extraction + LLM eval)
        datasets_json_str = "[]"
        try:
            dataset_results = extract_and_verify(full_text)
            datasets_json_str = results_to_json(dataset_results)
            st.session_state["datasets_json"] = datasets_json_str
        except Exception:
            st.session_state["datasets_json"] = datasets_json_str
        _save_field("datasets_json", datasets_json_str)

        dseval = _stream_and_collect(
            build_dataset_eval_prompt(sections), selected_model, temperature, ph_dseval
        )
        st.session_state["dataset_eval"] = dseval
        _save_field("datasets_llm", dseval)

        # 7. Code + project archive
        code_raw = _stream_and_collect(
            build_code_prompt(sections), selected_model, temperature, ph_code
        )
        code_files = parse_code_blocks(code_raw)
        st.session_state["code_raw"] = code_raw
        st.session_state["code_files"] = code_files
        _save_field("code_raw", code_raw)

        if code_files:
            try:
                zip_buf = build_project_zip(code_files)
                st.session_state["zip_bytes"] = zip_buf.getvalue()
                aid = st.session_state.get("current_analysis_id")
                if aid:
                    db = _get_db()
                    try:
                        rec = db.query(Analysis).filter(Analysis.id == aid).first()
                        if rec:
                            rec.zip_bytes = st.session_state["zip_bytes"]
                            db.commit()
                    except Exception:
                        pass
                    finally:
                        db.close()
            except Exception:
                pass

        # 8. Deploy
        deploy = _stream_and_collect(
            build_deployment_prompt(sections), selected_model, temperature, ph_deploy
        )
        st.session_state["deployment"] = deploy
        _save_field("deployment", deploy)

        # ── Done ──────────────────────────────────────────────
        has_content = bool(overview or arch or tech)
        st.session_state["analysis_done"] = has_content
        st.session_state["_needs_generation"] = False

        if has_content:
            print("[ANALYSIS] Complete — all sections generated")
        else:
            print("[ANALYSIS] FAILED — all LLM calls returned empty")

        st.rerun()  # final rerun to show clean rendered state

    else:
        # ── Normal display — all content already generated ──────
        with tab_overview:
            _render_section(
                st.session_state.get("overview")
                or st.session_state.get("explanation")
                or "_No overview generated._",
                "overview",
            )

        with tab_arch:
            _render_section(
                st.session_state.get("architecture")
                or "_No architecture generated._",
                "architecture",
            )

        with tab_technical:
            _render_section(
                st.session_state.get("technical")
                or st.session_state.get("flashcards")
                or "_No technical breakdown generated._",
                "technical",
            )

        with tab_equations:
            _render_section(
                st.session_state.get("equations")
                or "_No equations extracted._",
                "equations",
            )

        with tab_analysis:
            _render_section(
                st.session_state.get("analysis")
                or st.session_state.get("critique")
                or "_No analysis generated._",
                "analysis",
            )

        with tab_verdict:
            _render_section(
                st.session_state.get("verdict")
                or st.session_state.get("comparison")
                or "_No verdict generated._",
                "verdict",
            )

    # ── Datasets (extracted links + lazy LLM analysis)
    with tab_dataset:
        _ds_json_str = st.session_state.get("datasets_json") or ""
        _ds_results = results_from_json(_ds_json_str) if _ds_json_str else []

        if _ds_results:
            _total = len(_ds_results)
            _datasets = [r for r in _ds_results if r["category"] == "dataset"]
            _available = [r for r in _ds_results if r["status"] == "available"]
            _unavailable = [r for r in _ds_results if r["status"] in ("not_found", "unreachable", "error")]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Links", _total)
            m2.metric("Datasets", len(_datasets))
            m3.metric("Available", len(_available))
            m4.metric("Unavailable", len(_unavailable))

            st.markdown("")
            st.markdown("### Extracted Links")

            _STATUS_STYLES = {
                "available":   ("#22c55e", "#f0fdf4", "Available"),
                "restricted":  ("#f59e0b", "#fffbeb", "Restricted"),
                "not_found":   ("#ef4444", "#fef2f2", "Not Found"),
                "timeout":     ("#f97316", "#fff7ed", "Timeout"),
                "unreachable": ("#ef4444", "#fef2f2", "Unreachable"),
                "error":       ("#ef4444", "#fef2f2", "Error"),
                "unknown":     ("#94a3b8", "#f8fafc", "Unknown"),
            }
            _CAT_EMOJI = {"dataset": "📊", "code": "💻", "paper": "📄", "other": "🔗"}

            for r in _ds_results:
                _sc, _bg, _label = _STATUS_STYLES.get(r["status"], ("#94a3b8", "#f8fafc", r["status"]))
                _emoji = _CAT_EMOJI.get(r["category"], "🔗")
                _size = r.get("size_human", "Unknown")
                _url = r["url"]
                _short_url = _url[:80] + "..." if len(_url) > 80 else _url

                st.markdown(
                    f'<div class="dataset-card">'
                    f'<span class="ds-icon">{_emoji}</span>'
                    f'<div class="ds-info">'
                    f'<a href="{_url}" target="_blank" class="ds-url">{_short_url}</a>'
                    f'<div class="ds-meta">Size: {_size} · Type: {r.get("content_type", "N/A")}</div>'
                    f'</div>'
                    f'<span class="dataset-status-pill" style="'
                    f'color: {_sc};'
                    f'background: {_bg};'
                    f'border: 1px solid {_sc}30;'
                    f'">{_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")
        st.markdown("### AI Dataset & Evaluation Analysis")
        _render_section(
            st.session_state.get("dataset_eval")
            or st.session_state.get("datasets_llm")
            or "_No dataset evaluation generated._",
            "dataset_eval",
        )

    # ── Code (auto — IDE editor + terminal)
    with tab_code:
        code_files: dict[str, str] = st.session_state.get("code_files", {})
        if code_files:
            file_names = list(code_files.keys())
            selected_file = st.selectbox(
                "Select file",
                file_names,
                key="ide_file_select",
                label_visibility="collapsed",
            )

            lang = _detect_language(selected_file)

            editor_key = f"editor_{selected_file}"
            if editor_key not in st.session_state:
                st.session_state[editor_key] = code_files[selected_file]

            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="font-family:var(--mono, monospace);font-size:0.82rem;color:var(--text-muted,#888);">{selected_file}</span>'
                f'<span style="font-family:var(--mono, monospace);font-size:0.72rem;color:var(--text-faint,#bbb);margin-left:auto;">{lang}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            edited_code = st.text_area(
                "Code Editor",
                value=st.session_state[editor_key],
                height=400,
                key=f"code_input_{selected_file}",
                label_visibility="collapsed",
            )
            st.session_state[editor_key] = edited_code

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
            with btn_col1:
                run_clicked = st.button("▶ Run", key="run_code", use_container_width=True, type="primary")
            with btn_col2:
                reset_clicked = st.button("↺ Reset", key="reset_code", use_container_width=True)

            if reset_clicked:
                st.session_state[editor_key] = code_files.get(selected_file, "")
                st.rerun()

            with st.expander("Syntax-highlighted view", expanded=False):
                st.code(edited_code, language=lang, line_numbers=True)

            st.markdown(
                '<p style="font-family:var(--mono, monospace);font-size:0.82rem;color:var(--text-muted,#888);margin:16px 0 4px 0;">Terminal</p>',
                unsafe_allow_html=True,
            )

            if run_clicked:
                if lang == "python":
                    import subprocess, tempfile, os

                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".py", delete=False, encoding="utf-8"
                    ) as tmp:
                        tmp.write(edited_code)
                        tmp_path = tmp.name

                    try:
                        result = subprocess.run(
                            [sys.executable, tmp_path],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            cwd=str(PROJECT_ROOT),
                        )
                        output = ""
                        if result.stdout:
                            output += result.stdout
                        if result.stderr:
                            output += ("\n" if output else "") + result.stderr
                        exit_code = result.returncode
                    except subprocess.TimeoutExpired:
                        output = "⏰ Execution timed out (30s limit)."
                        exit_code = -1
                    except Exception as e:
                        output = f"Error: {str(e)}"
                        exit_code = -1
                    finally:
                        os.unlink(tmp_path)

                    if exit_code == 0:
                        status_badge = '<span class="term-ok">✓ exit 0</span>'
                    else:
                        status_badge = f'<span class="term-err">✗ exit {exit_code}</span>'

                    terminal_html = f"""
                    <div class="terminal-block">
                        <div class="term-header">
                            <span class="term-cmd">$ python {selected_file}</span>
                            {status_badge}
                        </div>
                        <pre>{output if output else '<span style="color:#585b70;">(no output)</span>'}</pre>
                    </div>
                    """
                    st.markdown(terminal_html, unsafe_allow_html=True)
                else:
                    st.warning(f"Running **{lang}** files is not supported yet. Only Python files can be executed.")
            else:
                st.markdown(
                    """
                    <div class="terminal-idle">
                        <span class="term-cmd">$</span> Press ▶ Run to execute the code...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No code files were generated. The LLM response may not have used the expected format.")
            with st.expander("Raw LLM output"):
                st.text(st.session_state.get("code_raw", ""))

    # ── Download
    with tab_download:
        zip_bytes = st.session_state.get("zip_bytes")
        code_files_dl: dict[str, str] = st.session_state.get("code_files", {})

        if zip_bytes:
            st.markdown("### Project Structure")
            st.code(file_tree_string(code_files_dl), language="text")

            st.download_button(
                label="Download Project (.zip)",
                data=zip_bytes,
                file_name="research_implementation.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.info("No project files to download. Generate code first.")

    # ── Deploy (auto)
    with tab_deploy:
        _render_section(
            st.session_state.get("deployment")
            or "_No deployment instructions generated._",
            "deployment",
        )

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

        # Persist chat messages to DB
        analysis_id = st.session_state.get("current_analysis_id")
        if analysis_id:
            db = _get_db()
            try:
                db.add(ChatMessage(analysis_id=analysis_id, role="user", content=user_q))
                db.add(ChatMessage(analysis_id=analysis_id, role="assistant", content=answer))
                db.commit()
            except Exception:
                pass  # Don't crash chat if DB write fails
            finally:
                db.close()
