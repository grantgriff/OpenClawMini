"""
Mistral-powered fact extractor for OpenClawMini.

Mirrors the GeminiExtractor interface so it can be used as a drop-in fallback
when GOOGLE_API_KEY is not available.  Uses mistral-small-latest by default —
good quality structured JSON output at reasonable cost.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openclawmini.memory.schema import DataSource, Fact, FactCategory, WritingSample, WritingCategory


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


class MistralExtractor:
    """
    Uses Mistral API to extract structured facts from text.

    Drop-in fallback for GeminiExtractor when GOOGLE_API_KEY is unavailable.
    Implements the same interface: complete(), extract_facts_from_web(),
    extract_facts_from_email(), extract_facts_from_conversation().
    """

    DEFAULT_MODEL = "mistral-small-latest"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "").strip()
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        response = client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def extract_facts_from_web(
        self,
        page_text: str,
        url: str = "",
        user_name: str = "",
    ) -> list[Fact]:
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

    def extract_facts_from_email(
        self,
        email_body: str,
        email_subject: str = "",
        user_name: str = "",
    ) -> list[Fact]:
        if not self.api_key:
            return []

        user_hint = f"This email was written by {user_name}. " if user_name else ""
        prompt = f"""You are analyzing an email written by a person to extract structured facts about THEM (the author).

{user_hint}Email Subject: {email_subject or "(none)"}
Email Body:
{email_body[:2000]}

Extract ALL factual information about the email's AUTHOR (not recipients) — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, personal background.
Include any fact you can reasonably infer from context, even if only partially stated.

Return a JSON array (empty [] if truly nothing found):
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

    def extract_facts_from_conversation(
        self,
        user_messages: list[str],
        user_name: str = "",
    ) -> list[Fact]:
        if not self.api_key or not user_messages:
            return []

        combined = "\n---\n".join(user_messages[:20])
        name_hint = f"These messages were written by {user_name}. " if user_name else ""

        prompt = f"""You are analyzing messages written by a person to extract facts about THEM.

{name_hint}Messages:
{combined[:3000]}

Extract ALL factual information about the MESSAGE AUTHOR — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, personal background.

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
    def from_env(cls, model: str = DEFAULT_MODEL) -> Optional["MistralExtractor"]:
        """Build from environment variables. Returns None if MISTRAL_API_KEY is not set."""
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(api_key=api_key, model=model)
