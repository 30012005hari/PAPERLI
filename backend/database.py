"""
Database models and engine setup.

Uses SQLAlchemy ORM with SQLite. The database file is stored at
``data/research_analyzer.db`` (auto-created on first run).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ── Database path ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{_DATA_DIR / 'research_analyzer.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ═════════════════════════════════════════════════════════════════════════════
# Models
# ═════════════════════════════════════════════════════════════════════════════

class Paper(Base):
    """An uploaded research paper."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(512), nullable=False)
    size_mb = Column(Float, nullable=False)
    extracted_text = Column(Text, nullable=False, default="")
    page_count = Column(Integer, nullable=False, default=0)
    sections_json = Column(Text, nullable=False, default="{}")  # JSON string
    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    analyses = relationship(
        "Analysis", back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Paper id={self.id} filename={self.filename!r}>"


class Analysis(Base):
    """A single analysis run for a paper."""

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)

    # LLM config
    model_used = Column(String(256), nullable=False)
    provider = Column(String(64), nullable=False)  # "ollama" or "gemini"
    detail_level = Column(String(32), nullable=False, default="Balanced")

    # Generated content
    explanation = Column(Text, default="")
    architecture = Column(Text, default="")
    code_raw = Column(Text, default="")
    deployment = Column(Text, default="")
    flashcards = Column(Text, default="")
    equations = Column(Text, default="")
    critique = Column(Text, default="")
    comparison = Column(Text, default="")
    zip_bytes = Column(LargeBinary, nullable=True)

    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    paper = relationship("Paper", back_populates="analyses")
    chat_messages = relationship(
        "ChatMessage", back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Analysis id={self.id} paper_id={self.paper_id} model={self.model_used!r}>"


class ChatMessage(Base):
    """A single chat message (user or assistant) tied to an analysis."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(
        Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False, default="")
    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    analysis = relationship("Analysis", back_populates="chat_messages")

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id} role={self.role!r}>"


# ── Create tables ────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


# Auto-init on import
init_db()
