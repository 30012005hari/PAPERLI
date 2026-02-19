"""
Pydantic schemas for request / response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
# Paper
# ═════════════════════════════════════════════════════════════════════════════

class PaperSummary(BaseModel):
    """Lightweight paper listing for the history sidebar."""
    id: int
    filename: str
    size_mb: float
    page_count: int
    created_at: datetime
    analysis_count: int = 0

    class Config:
        from_attributes = True


class PaperDetail(BaseModel):
    """Full paper info including extracted text and sections."""
    id: int
    filename: str
    size_mb: float
    extracted_text: str
    page_count: int
    sections_json: str
    created_at: datetime

    class Config:
        from_attributes = True


class PaperUploadResponse(BaseModel):
    """Response after successful PDF upload."""
    id: int
    filename: str
    size_mb: float
    page_count: int
    sections_detected: list[str]


# ═════════════════════════════════════════════════════════════════════════════
# Analysis
# ═════════════════════════════════════════════════════════════════════════════

class AnalyseRequest(BaseModel):
    """Parameters for running an analysis."""
    model: str = "gemini-2.0-flash"
    provider: str = "gemini"  # "ollama" or "gemini"
    detail_level: str = "Balanced"
    gemini_key: Optional[str] = None

    # Derived temperature from detail level
    @property
    def temperature(self) -> float:
        return {
            "Concise": 0.1,
            "Balanced": 0.4,
            "Detailed": 0.7,
            "Creative": 0.95,
        }.get(self.detail_level, 0.4)


class AnalysisSummary(BaseModel):
    """Lightweight analysis listing."""
    id: int
    paper_id: int
    model_used: str
    provider: str
    detail_level: str
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisDetail(BaseModel):
    """Full analysis results."""
    id: int
    paper_id: int
    model_used: str
    provider: str
    detail_level: str
    explanation: str
    architecture: str
    code_raw: str
    deployment: str
    flashcards: str
    equations: str
    critique: str
    comparison: str
    has_zip: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


# ═════════════════════════════════════════════════════════════════════════════
# Chat
# ═════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """A user question about a paper."""
    question: str
    model: str = "gemini-2.0-flash"
    provider: str = "gemini"
    detail_level: str = "Balanced"
    gemini_key: Optional[str] = None

    @property
    def temperature(self) -> float:
        return {
            "Concise": 0.1,
            "Balanced": 0.4,
            "Detailed": 0.7,
            "Creative": 0.95,
        }.get(self.detail_level, 0.4)


class ChatMessageSchema(BaseModel):
    """A single chat message."""
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    """Response to a chat question."""
    answer: str
    message_id: int


# ═════════════════════════════════════════════════════════════════════════════
# Health
# ═════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    ollama_running: bool = False
    ollama_models: list[str] = []
    database: str = "connected"
