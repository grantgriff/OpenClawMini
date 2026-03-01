"""Memory schema for OpenClawMini - Pydantic models for all memory categories."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class MemoryCategory(str, Enum):
    FACTUAL = "factual"
    STYLISTIC = "stylistic"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"


class DataSource(str, Enum):
    GMAIL = "gmail"
    LINKEDIN = "linkedin"
    LINKEDIN_POSTS = "linkedin_posts"
    WEB_SEARCH = "web_search"
    FILE_UPLOAD = "file_upload"
    CHATGPT_EXPORT = "chatgpt_export"
    CLAUDE_EXPORT = "claude_export"
    GENERIC_LOG = "generic_log"
    ANALYSIS = "analysis"
    MANUAL = "manual"


class FactCategory(str, Enum):
    WORK = "work"
    EDUCATION = "education"
    SKILLS = "skills"
    LOCATION = "location"
    PERSONAL = "personal"
    INTERESTS = "interests"
    ACHIEVEMENTS = "achievements"
    OTHER = "other"


class WritingCategory(str, Enum):
    PROFESSIONAL_EMAIL = "professional_email"
    CASUAL_EMAIL = "casual_email"
    LINKEDIN_MESSAGE = "linkedin_message"
    SLACK_MESSAGE = "slack_message"
    DOCUMENT = "document"
    OTHER = "other"


class PreferenceCategory(str, Enum):
    COMMUNICATION = "communication"
    TECHNOLOGY = "technology"
    WORK_STYLE = "work_style"
    FOOD = "food"
    LIFESTYLE = "lifestyle"
    VALUES = "values"
    OTHER = "other"


# ─────────────────────────────────────────────────────────────
# Memory item models
# ─────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


class Fact(BaseModel):
    """Factual information about the user — drives SFT training."""
    id: str = Field(default_factory=lambda: _new_id("fact"))
    category: FactCategory = FactCategory.OTHER
    content: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: str = DataSource.MANUAL
    extracted_at: datetime = Field(default_factory=_now)

    model_config = {"use_enum_values": True}


class WritingSample(BaseModel):
    """Actual user writing — drives GRPO stylistic training."""
    id: str = Field(default_factory=lambda: _new_id("sample"))
    category: WritingCategory = WritingCategory.OTHER
    text: str
    context: str = ""
    source: str = DataSource.MANUAL
    extracted_at: datetime = Field(default_factory=_now)

    model_config = {"use_enum_values": True}


class Post(BaseModel):
    """Social media posts — drives GRPO stylistic training."""
    id: str = Field(default_factory=lambda: _new_id("post"))
    platform: str = "linkedin"
    text: str
    engagement: dict = Field(default_factory=dict)
    source: str = DataSource.LINKEDIN_POSTS
    extracted_at: datetime = Field(default_factory=_now)

    model_config = {"use_enum_values": True}


class Relationship(BaseModel):
    """People the user interacts with — context for responses."""
    id: str = Field(default_factory=lambda: _new_id("rel"))
    name: str
    relationship: str  # e.g. "coworker", "friend", "manager"
    interaction_frequency: str = "occasional"  # daily, weekly, occasional
    source: str = DataSource.GMAIL
    extracted_at: datetime = Field(default_factory=_now)

    model_config = {"use_enum_values": True}


class Preference(BaseModel):
    """User preferences and opinions — drives GRPO preference alignment."""
    id: str = Field(default_factory=lambda: _new_id("pref"))
    category: PreferenceCategory = PreferenceCategory.OTHER
    content: str
    evidence: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source: str = DataSource.ANALYSIS
    extracted_at: datetime = Field(default_factory=_now)

    model_config = {"use_enum_values": True}


class StyleAnalysis(BaseModel):
    """Computed style metrics from writing samples — informs RULER rubric."""
    vocabulary_level: str = "professional_casual"
    avg_sentence_length: float = 0.0
    formality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    emoji_usage: str = "rare"  # rare, occasional, frequent
    common_phrases: list[str] = Field(default_factory=list)
    tone: str = "friendly_professional"
    avg_email_length: int = 0
    punctuation_style: str = "standard"  # standard, minimal, expressive


class InteractionLog(BaseModel):
    """Metadata for an uploaded AI interaction log file."""
    id: str = Field(default_factory=lambda: _new_id("log"))
    source_file: str
    format: str = "generic"  # chatgpt, claude, generic
    uploaded_at: datetime = Field(default_factory=_now)
    conversations_extracted: int = 0
    training_pairs_generated: int = 0
    facts_extracted: int = 0
    samples_extracted: int = 0


class UserMeta(BaseModel):
    """Top-level user metadata."""
    name: str = ""
    email: str = ""
    created_at: datetime = Field(default_factory=_now)
    last_updated: datetime = Field(default_factory=_now)


# ─────────────────────────────────────────────────────────────
# Root memory model
# ─────────────────────────────────────────────────────────────

class Memory(BaseModel):
    """
    Root memory model — the full structured profile of the user.

    Categories and their training use:
      facts           → SFT (factual Q&A pairs)
      writing_samples → GRPO (style prompts)
      posts           → GRPO (style prompts)
      relationships   → context for both stages
      preferences     → GRPO (preference scenarios)
      style_analysis  → RULER rubric generation
      interaction_logs → metadata for uploaded files
    """
    user: UserMeta = Field(default_factory=UserMeta)
    facts: list[Fact] = Field(default_factory=list)
    writing_samples: list[WritingSample] = Field(default_factory=list)
    posts: list[Post] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)
    style_analysis: StyleAnalysis = Field(default_factory=StyleAnalysis)
    interaction_logs: list[InteractionLog] = Field(default_factory=list)

    def stats(self) -> dict:
        """Return a summary count of each memory category."""
        return {
            "facts": len(self.facts),
            "writing_samples": len(self.writing_samples),
            "posts": len(self.posts),
            "relationships": len(self.relationships),
            "preferences": len(self.preferences),
            "interaction_logs": len(self.interaction_logs),
            "total_items": (
                len(self.facts)
                + len(self.writing_samples)
                + len(self.posts)
                + len(self.relationships)
                + len(self.preferences)
            ),
        }

    def is_empty(self) -> bool:
        return self.stats()["total_items"] == 0


# ─────────────────────────────────────────────────────────────
# Classifier result
# ─────────────────────────────────────────────────────────────

class MemoryClassification(BaseModel):
    """Result of classifying a piece of text into a memory category."""
    category: MemoryCategory
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    sub_category: str = ""  # e.g. FactCategory or PreferenceCategory value
    method: str = "rule_based"  # "rule_based" or "llm"
