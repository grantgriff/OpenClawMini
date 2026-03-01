"""
Gemini-powered fact extractor for OpenClawMini.

Given raw text (email, web page, conversation log), calls Gemini to extract
multiple structured facts about the user and return them as Fact objects.

Unlike the rule-based classifier (one category per text), this extracts ALL
facts present in a piece of text in a single LLM call.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openclawmini.memory.schema import DataSource, Fact, FactCategory, WritingCategory, WritingSample


_CATEGORY_MAP = {
    "work": FactCategory.WORK,
    "education": FactCategory.EDUCATION,
    "skills": FactCategory.SKILLS,
    "location": FactCategory.LOCATION,
    "personal": FactCategory.PERSONAL,
    "interests": FactCategory.INTERESTS,
    "achievements": FactCategory.ACHIEVEMENTS,
    "other": FactCategory.OTHER,
}


def _parse_facts_json(raw: str, source: str) -> list[Fact]:
    """Parse a JSON array of fact dicts from a Gemini response."""
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    facts = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content or len(content) < 10:
            continue
        cat_str = str(item.get("category", "other")).lower()
        category = _CATEGORY_MAP.get(cat_str, FactCategory.OTHER)
        confidence = float(item.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        facts.append(Fact(
            content=content,
            category=category,
            confidence=confidence,
            source=source,
        ))
    return facts


class GeminiExtractor:
    """
    Uses Google Gemini to extract multiple structured facts from text.

    Implements the `.complete(prompt)` interface so it can also be passed
    directly to MemoryClassifier for LLM-based classification.
    """

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._genai_model = None

    def _get_model(self):
        if self._genai_model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai_model = genai.GenerativeModel(self.model)
        return self._genai_model

    def _safety_settings(self):
        """Return permissive safety settings for personal data extraction.

        Personal emails and web pages often contain names, addresses, or
        professional info that Gemini's default filters may block. We disable
        all harm categories so every fact-extraction call goes through.
        """
        try:
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            return {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        except Exception:
            return None

    def complete(self, prompt: str) -> str:
        """Single-turn completion. Used by MemoryClassifier for classification."""
        m = self._get_model()
        safety = self._safety_settings()
        kwargs = {"safety_settings": safety} if safety else {}
        response = m.generate_content(prompt, **kwargs)
        try:
            return response.text
        except Exception:
            # Response was safety-blocked (finish_reason=2) or otherwise empty.
            # Return empty string so callers degrade gracefully instead of crashing.
            return ""

    # ── Email fact extraction ──────────────────────────────────

    def extract_facts_from_email(
        self,
        email_body: str,
        email_subject: str = "",
        user_name: str = "",
    ) -> list[Fact]:
        """
        Extract all facts about the user from a single email they wrote.

        Returns a (possibly empty) list of Fact objects.
        """
        if not self.api_key:
            return []

        user_hint = f"This email was written by {user_name}. " if user_name else ""
        prompt = f"""You are analyzing an email written by a person to extract structured facts about THEM (the author).

{user_hint}Email Subject: {email_subject or "(none)"}
Email Body:
{email_body[:2000]}

Extract ALL factual information about the email's AUTHOR (not recipients) — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, personal background.
Include any fact you can reasonably infer from context, even if only partially stated. More data is better.

Return a JSON array (empty [] if truly nothing found) of fact objects:
[
  {{"content": "Specific fact about the person", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences:"""

        try:
            raw = self.complete(prompt)
            return _parse_facts_json(raw, DataSource.GMAIL)
        except Exception:
            return []

    # ── Web / LinkedIn fact extraction ─────────────────────────

    def extract_facts_from_web(
        self,
        page_text: str,
        url: str = "",
        user_name: str = "",
    ) -> list[Fact]:
        """Extract facts about the user from a web page (LinkedIn, news, etc.)."""
        if not self.api_key or not page_text.strip():
            return []

        source_hint = f"Source URL: {url}\n" if url else ""
        name_hint = f"We are looking for facts about a person named {user_name}. " if user_name else ""
        source = DataSource.LINKEDIN if "linkedin" in url.lower() else DataSource.WEB_SEARCH

        prompt = f"""You are analyzing a web page to extract factual information about a specific person.

{name_hint}{source_hint}
Page content:
{page_text[:3000]}

Extract ALL factual information about {user_name or "this person"} — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, background.
Include partial or implied facts. Skip only clearly irrelevant generic content.

Return a JSON array (empty [] if truly nothing found):
[
  {{"content": "Specific fact", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences:"""

        try:
            raw = self.complete(prompt)
            return _parse_facts_json(raw, source)
        except Exception:
            return []

    # ── Conversation / log file fact extraction ────────────────

    def extract_facts_from_conversation(
        self,
        user_messages: list[str],
        user_name: str = "",
    ) -> list[Fact]:
        """Extract facts about the user from their messages in a conversation log."""
        if not self.api_key or not user_messages:
            return []

        combined = "\n---\n".join(user_messages[:20])
        name_hint = f"These messages were written by {user_name}. " if user_name else ""

        prompt = f"""You are analyzing messages written by a person to extract facts about THEM.

{name_hint}Messages:
{combined[:3000]}

Extract ALL factual information about the MESSAGE AUTHOR — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, personal background.
Include any fact that can be reasonably inferred from context. More data is better.

Return a JSON array (empty [] if truly nothing found):
[
  {{"content": "Specific fact about the person", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences:"""

        try:
            raw = self.complete(prompt)
            return _parse_facts_json(raw, DataSource.FILE_UPLOAD)
        except Exception:
            return []

    @classmethod
    def from_env(cls, model: str = "gemini-2.0-flash") -> Optional["GeminiExtractor"]:
        """Build a GeminiExtractor from environment variables. Returns None if no API key."""
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(model=model, api_key=api_key)
