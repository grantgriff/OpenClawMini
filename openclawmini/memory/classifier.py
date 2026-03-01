"""
Memory classifier — routes extracted text to the correct memory category.

Two modes:
  1. Rule-based (default) — keyword/heuristic, no API key needed, fast
  2. LLM-based — uses configured LLM for high-accuracy classification
     (called by Research Agent in Task 3)
"""

from __future__ import annotations

import re
from typing import Optional

from openclawmini.memory.schema import (
    FactCategory,
    MemoryCategory,
    MemoryClassification,
    PreferenceCategory,
    WritingCategory,
)

# ─────────────────────────────────────────────────────────────
# Rule-based classifier signal sets
# ─────────────────────────────────────────────────────────────

# Signals that suggest FACTUAL content (things about the user)
_FACTUAL_SIGNALS = [
    r"\bworks? (at|for|with)\b",
    r"\b(founder|co-founder|ceo|cto|engineer|developer|designer|manager|director)\b",
    r"\b(studied|graduated|attended|enrolled|degree|major|university|college|stanford|mit|harvard)\b",
    r"\b(lives? in|based in|located in|from)\b",
    r"\b(years? (of )?experience|background in)\b",
    r"\b(speak[s]?|fluent in|native speaker)\b",
    r"\b(born|raised|grew up)\b",
    r"\b(series [abc]|startup|pre-seed|seed|funded)\b",
    r"\b(skills?|proficient|expertise|specializes? in)\b",
]

# Signals that suggest STYLISTIC content (writing samples)
_STYLISTIC_SIGNALS = [
    r"\b(hey|hi|hello|dear|good morning|good afternoon)\b",  # email greetings
    r"\b(thanks?|thank you|cheers|best|regards|sincerely)\b",  # email closings
    r"^(from|to|subject|date):\s",  # email headers
    r"\b(let me know|feel free|don't hesitate|reach out)\b",
    r"\b(excited to share|happy to announce|thrilled)\b",  # LinkedIn post openers
    r"\b(follow up|following up|checking in|circling back)\b",
    r"\b(quick question|just wanted to|wanted to check)\b",
    r"\n.*\n",  # multi-paragraph text — likely a sample
]

# Signals that suggest PREFERENCE content
_PREFERENCE_SIGNALS = [
    r"\b(prefer[s]?|would rather|rather|instead of)\b",
    r"\b(love[s]?|hate[s]?|dislike[s]?|enjoy[s]?|can'?t stand|big fan)\b",
    r"\b(always|never|usually|typically|tend to|generally)\b",
    r"\b(favorite|favourite|best|worst|top choice|go-to)\b",
    r"\b(important to me|value[s]?|believe[s]? in|prioritize[s]?)\b",
    r"\b(think[s]? that|opinion|feel[s]? that|in my view)\b",
]

# Signals that suggest RELATIONSHIP content
_RELATIONSHIP_SIGNALS = [
    r"\b(colleague|coworker|co-worker|teammate|team member)\b",
    r"\b(manager|boss|report[s]?|direct report|lead)\b",
    r"\b(friend|buddy|roommate|partner|spouse|family)\b",
    r"\b(mentor|mentee|advisor|advisee)\b",
    r"\b(client|customer|vendor|contractor|consultant)\b",
    r"\b(met with|talked to|spoke with|emailed)\b.{0,30}\b(today|yesterday|last week)\b",
    r"\b(interact[s]? with|work[s]? with|collaborate[s]? with|reports? to)\b",
    r"\b(my (boss|manager|colleague|coworker|friend|partner|mentor))\b",
    r"\b(he|she|they) (is|are|was|were) (a|my|the)\b",  # "she is my..." patterns
]


def _score_signals(text: str, patterns: list[str]) -> int:
    """Count how many signal patterns match in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower, re.MULTILINE | re.IGNORECASE))


def _infer_writing_category(text: str) -> str:
    text_lower = text.lower()
    if re.search(r"^(from|to|subject):", text_lower, re.MULTILINE):
        if any(w in text_lower for w in ["sincerely", "regards", "dear", "team", "colleagues"]):
            return WritingCategory.PROFESSIONAL_EMAIL
        return WritingCategory.CASUAL_EMAIL
    if any(w in text_lower for w in ["excited to share", "thrilled", "linkedin", "post"]):
        return WritingCategory.LINKEDIN_MESSAGE
    if len(text.split()) < 30:
        return WritingCategory.SLACK_MESSAGE
    return WritingCategory.OTHER


def _infer_fact_category(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["works", "job", "role", "position", "company", "startup", "engineer", "founder"]):
        return FactCategory.WORK
    if any(w in text_lower for w in ["studied", "degree", "university", "college", "major", "stanford", "school"]):
        return FactCategory.EDUCATION
    if any(w in text_lower for w in ["skill", "proficient", "expert", "language", "python", "javascript"]):
        return FactCategory.SKILLS
    if any(w in text_lower for w in ["lives", "based", "located", "city", "country", "state"]):
        return FactCategory.LOCATION
    if any(w in text_lower for w in ["born", "raised", "family", "personal"]):
        return FactCategory.PERSONAL
    return FactCategory.OTHER


def _infer_preference_category(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["python", "javascript", "typescript", "framework", "tool", "language", "stack"]):
        return PreferenceCategory.TECHNOLOGY
    if any(w in text_lower for w in ["communicate", "email", "slack", "meeting", "async", "sync"]):
        return PreferenceCategory.COMMUNICATION
    if any(w in text_lower for w in ["work", "office", "remote", "schedule", "process"]):
        return PreferenceCategory.WORK_STYLE
    if any(w in text_lower for w in ["food", "eat", "restaurant", "coffee", "drink"]):
        return PreferenceCategory.FOOD
    return PreferenceCategory.OTHER


# ─────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────

class MemoryClassifier:
    """
    Classifies extracted text into memory categories.

    classify(text, source) → MemoryClassification

    Falls back to rule_based if no LLM client is provided.
    """

    def __init__(self, llm_client=None) -> None:
        """
        Args:
            llm_client: Optional LLM client. When provided, classify() uses
                        the LLM path for higher accuracy. When None (default),
                        uses rule-based classification.
        """
        self._llm = llm_client

    def classify(self, text: str, source: str = "unknown") -> MemoryClassification:
        """Classify text into a memory category."""
        if self._llm is not None:
            return self._classify_llm(text, source)
        return self._classify_rule_based(text, source)

    def classify_batch(self, items: list[tuple[str, str]]) -> list[MemoryClassification]:
        """Classify a list of (text, source) tuples."""
        return [self.classify(text, source) for text, source in items]

    # ── Rule-based ─────────────────────────────────────────────

    def _classify_rule_based(self, text: str, source: str) -> MemoryClassification:
        """Keyword/heuristic classifier — no API key required."""
        text = text.strip()
        if not text:
            return MemoryClassification(
                category=MemoryCategory.FACTUAL,
                content=text,
                confidence=0.0,
                reasoning="Empty text",
                method="rule_based",
            )

        scores = {
            MemoryCategory.FACTUAL: _score_signals(text, _FACTUAL_SIGNALS),
            MemoryCategory.STYLISTIC: _score_signals(text, _STYLISTIC_SIGNALS),
            MemoryCategory.PREFERENCE: _score_signals(text, _PREFERENCE_SIGNALS),
            MemoryCategory.RELATIONSHIP: _score_signals(text, _RELATIONSHIP_SIGNALS),
        }

        # Source-based priors
        source_lower = source.lower()
        if "gmail" in source_lower or "email" in source_lower:
            scores[MemoryCategory.STYLISTIC] += 2
        elif "linkedin_posts" in source_lower or "post" in source_lower:
            scores[MemoryCategory.STYLISTIC] += 2
        elif "linkedin" in source_lower or "profile" in source_lower:
            scores[MemoryCategory.FACTUAL] += 2
        elif "web_search" in source_lower or "web" in source_lower:
            scores[MemoryCategory.FACTUAL] += 1
        elif "analysis" in source_lower:
            scores[MemoryCategory.PREFERENCE] += 1

        # Text length heuristic: long texts are more likely writing samples
        word_count = len(text.split())
        if word_count > 50:
            scores[MemoryCategory.STYLISTIC] += 1
        elif word_count < 15:
            scores[MemoryCategory.FACTUAL] += 1

        total = sum(scores.values()) or 1
        best_category = max(scores, key=lambda c: scores[c])
        confidence = round(scores[best_category] / total, 2)

        # Ensure minimum confidence floor
        confidence = max(confidence, 0.3)

        # Infer sub-category
        sub_category = ""
        if best_category == MemoryCategory.FACTUAL:
            sub_category = _infer_fact_category(text)
        elif best_category == MemoryCategory.STYLISTIC:
            sub_category = _infer_writing_category(text)
        elif best_category == MemoryCategory.PREFERENCE:
            sub_category = _infer_preference_category(text)

        reasoning = (
            f"Signal scores: factual={scores[MemoryCategory.FACTUAL]}, "
            f"stylistic={scores[MemoryCategory.STYLISTIC]}, "
            f"preference={scores[MemoryCategory.PREFERENCE]}, "
            f"relationship={scores[MemoryCategory.RELATIONSHIP]}. "
            f"Source: {source}, word_count: {word_count}."
        )

        return MemoryClassification(
            category=best_category,
            content=text,
            confidence=confidence,
            reasoning=reasoning,
            sub_category=sub_category,
            method="rule_based",
        )

    # ── LLM-based ──────────────────────────────────────────────

    def _classify_llm(self, text: str, source: str) -> MemoryClassification:
        """
        LLM-powered classifier — used by Research Agent (Task 3).

        Expects self._llm to have a .complete(prompt: str) -> str method.
        Falls back to rule-based if LLM call fails.
        """
        prompt = f"""You are classifying a piece of text extracted from a user's data source.

Source: {source}
Text: {text[:1000]}

Classify this text into exactly one category:
- FACTUAL: Objective facts about the user (job, education, location, skills, background)
- STYLISTIC: Writing samples showing HOW the user communicates (emails, posts, messages)
- PREFERENCE: User opinions, preferences, values, likes/dislikes
- RELATIONSHIP: Information about specific people the user interacts with

Respond with JSON only:
{{"category": "FACTUAL|STYLISTIC|PREFERENCE|RELATIONSHIP", "confidence": 0.0-1.0, "reasoning": "brief explanation", "sub_category": "specific sub-type"}}"""

        try:
            import json
            raw = self._llm.complete(prompt)
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            data = json.loads(raw)

            category_map = {
                "FACTUAL": MemoryCategory.FACTUAL,
                "STYLISTIC": MemoryCategory.STYLISTIC,
                "PREFERENCE": MemoryCategory.PREFERENCE,
                "RELATIONSHIP": MemoryCategory.RELATIONSHIP,
            }
            category = category_map.get(data.get("category", "").upper(), MemoryCategory.FACTUAL)

            return MemoryClassification(
                category=category,
                content=text,
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                sub_category=data.get("sub_category", ""),
                method="llm",
            )
        except Exception:
            # Fallback to rule-based
            result = self._classify_rule_based(text, source)
            result.method = "rule_based_fallback"
            return result
