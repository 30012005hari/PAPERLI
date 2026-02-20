"""
FastAPI application — REST API for the Research Paper Analyzer.

Endpoints
---------
POST   /api/upload                  Upload a PDF, extract text, save to DB
POST   /api/analyse/{paper_id}      Run full LLM analysis and persist results
GET    /api/papers                  List all uploaded papers
GET    /api/papers/{paper_id}       Get paper details
GET    /api/analyses/{analysis_id}  Get full analysis results
GET    /api/papers/{paper_id}/analyses  List analyses for a paper
POST   /api/chat/{analysis_id}      Ask a question, save Q&A
GET    /api/chat/{analysis_id}/history  Get chat history
DELETE /api/papers/{paper_id}       Delete paper + cascading analyses
GET    /api/health                  Health check
GET    /api/analyses/{analysis_id}/download  Download project zip
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session

# ── Ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import SessionLocal, Paper, Analysis, ChatMessage
from backend.schemas import (
    AnalyseRequest,
    AnalysisDetail,
    AnalysisSummary,
    ChatMessageSchema,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    PaperDetail,
    PaperSummary,
    PaperUploadResponse,
)
from config import MAX_PDF_SIZE_MB
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
    build_datasets_prompt,
)
from modules.llm_client import (
    generate_full as ollama_generate,
    is_ollama_running,
    list_models,
)
from modules.gemini_client import generate_full as gemini_generate
from modules.code_generator import parse_code_blocks
from modules.project_builder import build_project_zip
from modules.dataset_extractor import extract_and_verify, results_to_json


# ═════════════════════════════════════════════════════════════════════════════
# App
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Research Paper Analyzer API",
    version="1.0.0",
    description="REST API for uploading, analysing, and querying research papers.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB dependency ────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Helper: run LLM (non-streaming) ─────────────────────────────────────────
def _generate(
    prompt: str,
    model: str,
    provider: str,
    temperature: float,
    gemini_key: str | None = None,
) -> str:
    """Dispatch to the correct LLM provider and return full text."""
    if provider == "gemini":
        if not gemini_key:
            raise HTTPException(400, "Gemini API key is required for the Gemini provider.")
        return gemini_generate(prompt, model, gemini_key, temperature)
    else:
        return ollama_generate(prompt, model, temperature)


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Return API health and provider status."""
    ollama_ok = is_ollama_running()
    return HealthResponse(
        status="ok",
        ollama_running=ollama_ok,
        ollama_models=list_models() if ollama_ok else [],
        database="connected",
    )


# ── Upload ───────────────────────────────────────────────────────────────────

@app.post("/api/upload", response_model=PaperUploadResponse)
def upload_paper(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a PDF, extract text, parse sections, and save to database."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    content = file.file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(
            413, f"File is {size_mb:.1f} MB — max allowed is {MAX_PDF_SIZE_MB} MB."
        )

    # Extract text using existing module
    pdf_stream = io.BytesIO(content)
    full_text, pages = extract_text(pdf_stream)

    # Parse sections
    sections = parse_sections(full_text)
    detected_keys = [k for k in sections if k != "full_text"]

    # Save to DB
    paper = Paper(
        filename=file.filename,
        size_mb=round(size_mb, 2),
        extracted_text=full_text,
        page_count=len(pages),
        sections_json=json.dumps(sections),
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)

    return PaperUploadResponse(
        id=paper.id,
        filename=paper.filename,
        size_mb=paper.size_mb,
        page_count=paper.page_count,
        sections_detected=detected_keys,
    )


# ── Papers CRUD ──────────────────────────────────────────────────────────────

@app.get("/api/papers", response_model=list[PaperSummary])
def list_papers(db: Session = Depends(get_db)):
    """List all uploaded papers (most recent first)."""
    papers = db.query(Paper).order_by(Paper.created_at.desc()).all()
    result = []
    for p in papers:
        result.append(
            PaperSummary(
                id=p.id,
                filename=p.filename,
                size_mb=p.size_mb,
                page_count=p.page_count,
                created_at=p.created_at,
                analysis_count=len(p.analyses),
            )
        )
    return result


@app.get("/api/papers/{paper_id}", response_model=PaperDetail)
def get_paper(paper_id: int, db: Session = Depends(get_db)):
    """Get full details for a specific paper."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(404, "Paper not found.")
    return paper


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: int, db: Session = Depends(get_db)):
    """Delete a paper and all its analyses (cascade)."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(404, "Paper not found.")
    db.delete(paper)
    db.commit()
    return {"detail": "Paper deleted.", "id": paper_id}


# ── Analyse ──────────────────────────────────────────────────────────────────

@app.post("/api/analyse/{paper_id}", response_model=AnalysisDetail)
def analyse_paper(paper_id: int, req: AnalyseRequest, db: Session = Depends(get_db)):
    """Run full LLM analysis on a paper and save all results."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(404, "Paper not found.")

    sections = json.loads(paper.sections_json)
    temp = req.temperature

    # Generate all sections
    explanation = _generate(build_explanation_prompt(sections), req.model, req.provider, temp, req.gemini_key)
    architecture = _generate(build_architecture_prompt(sections), req.model, req.provider, temp, req.gemini_key)
    code_raw = _generate(build_code_prompt(sections), req.model, req.provider, temp, req.gemini_key)
    deployment = _generate(build_deployment_prompt(sections), req.model, req.provider, temp, req.gemini_key)
    flashcards = _generate(build_flashcards_prompt(sections), req.model, req.provider, temp, req.gemini_key)
    equations = _generate(build_equations_prompt(sections), req.model, req.provider, temp, req.gemini_key)
    critique = _generate(build_critique_prompt(sections), req.model, req.provider, temp, req.gemini_key)
    comparison = _generate(build_comparison_prompt(sections), req.model, req.provider, temp, req.gemini_key)

    # Dataset extraction and analysis
    try:
        dataset_results = extract_and_verify(paper.extracted_text)
        datasets_json_str = results_to_json(dataset_results)
    except Exception:
        datasets_json_str = "[]"

    datasets_llm = _generate(build_datasets_prompt(sections), req.model, req.provider, temp, req.gemini_key)

    # Build project zip
    code_files = parse_code_blocks(code_raw)
    zip_bytes = build_project_zip(code_files).getvalue() if code_files else None

    # Save to DB
    analysis = Analysis(
        paper_id=paper.id,
        model_used=req.model,
        provider=req.provider,
        detail_level=req.detail_level,
        explanation=explanation,
        architecture=architecture,
        code_raw=code_raw,
        deployment=deployment,
        flashcards=flashcards,
        equations=equations,
        critique=critique,
        comparison=comparison,
        datasets_json=datasets_json_str,
        datasets_llm=datasets_llm,
        zip_bytes=zip_bytes,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return AnalysisDetail(
        id=analysis.id,
        paper_id=analysis.paper_id,
        model_used=analysis.model_used,
        provider=analysis.provider,
        detail_level=analysis.detail_level,
        explanation=analysis.explanation,
        architecture=analysis.architecture,
        code_raw=analysis.code_raw,
        deployment=analysis.deployment,
        flashcards=analysis.flashcards,
        equations=analysis.equations,
        critique=analysis.critique,
        comparison=analysis.comparison,
        datasets_json=analysis.datasets_json or "",
        datasets_llm=analysis.datasets_llm or "",
        has_zip=analysis.zip_bytes is not None,
        created_at=analysis.created_at,
    )


# ── Analysis retrieval ───────────────────────────────────────────────────────

@app.get("/api/analyses/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    """Get full results for a specific analysis."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(404, "Analysis not found.")
    return AnalysisDetail(
        id=analysis.id,
        paper_id=analysis.paper_id,
        model_used=analysis.model_used,
        provider=analysis.provider,
        detail_level=analysis.detail_level,
        explanation=analysis.explanation,
        architecture=analysis.architecture,
        code_raw=analysis.code_raw,
        deployment=analysis.deployment,
        flashcards=analysis.flashcards,
        equations=analysis.equations,
        critique=analysis.critique,
        comparison=analysis.comparison,
        datasets_json=analysis.datasets_json or "",
        datasets_llm=analysis.datasets_llm or "",
        has_zip=analysis.zip_bytes is not None,
        created_at=analysis.created_at,
    )


@app.get("/api/papers/{paper_id}/analyses", response_model=list[AnalysisSummary])
def list_analyses_for_paper(paper_id: int, db: Session = Depends(get_db)):
    """List all analyses for a specific paper."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(404, "Paper not found.")
    analyses = (
        db.query(Analysis)
        .filter(Analysis.paper_id == paper_id)
        .order_by(Analysis.created_at.desc())
        .all()
    )
    return analyses


@app.get("/api/analyses/{analysis_id}/download")
def download_zip(analysis_id: int, db: Session = Depends(get_db)):
    """Download the generated project zip for an analysis."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(404, "Analysis not found.")
    if not analysis.zip_bytes:
        raise HTTPException(404, "No project zip available for this analysis.")
    return Response(
        content=analysis.zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=research_implementation.zip"},
    )


# ── Chat ─────────────────────────────────────────────────────────────────────

@app.post("/api/chat/{analysis_id}", response_model=ChatResponse)
def chat_with_paper(analysis_id: int, req: ChatRequest, db: Session = Depends(get_db)):
    """Ask a question about a paper. Saves both the question and answer."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(404, "Analysis not found.")

    paper = db.query(Paper).filter(Paper.id == analysis.paper_id).first()
    sections = json.loads(paper.sections_json)

    # Save user message
    user_msg = ChatMessage(
        analysis_id=analysis_id,
        role="user",
        content=req.question,
    )
    db.add(user_msg)
    db.commit()

    # Generate answer
    prompt = build_chat_prompt(sections, req.question)
    answer = _generate(prompt, req.model, req.provider, req.temperature, req.gemini_key)

    # Save assistant message
    assistant_msg = ChatMessage(
        analysis_id=analysis_id,
        role="assistant",
        content=answer,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(answer=answer, message_id=assistant_msg.id)


@app.get("/api/chat/{analysis_id}/history", response_model=list[ChatMessageSchema])
def get_chat_history(analysis_id: int, db: Session = Depends(get_db)):
    """Get all chat messages for an analysis."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(404, "Analysis not found.")
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.analysis_id == analysis_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return messages
