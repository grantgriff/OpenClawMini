"""
Gemini-powered fact extractor for OpenClawMini.

Given raw text (email, web page, conversation log), calls Gemini to extract
multiple structured facts about the user and return them as Fact objects.

Unlike the rule-based classifier (one category per text), this extracts ALL
facts present in a piece of text in a single LLM call.

Uses the new google-genai SDK (google.genai) — NOT the deprecated
google.generativeai package.
"""

from __future__ import annotations

import json
import os
import re
import sys
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

    Uses the new google-genai SDK (google.genai).
    """

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _safety_settings(self):
        """Return BLOCK_NONE safety settings for all harm categories."""
        try:
            from google.genai import types
            return [
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"),
            ]
        except Exception:
            return None

    def complete(self, prompt: str) -> str:
        """Single-turn completion. Used by MemoryClassifier for classification."""
        try:
            from google import genai
            from google.genai import types
            client = self._get_client()
            safety = self._safety_settings()
            config = types.GenerateContentConfig(safety_settings=safety) if safety else None
            kwargs = {"config": config} if config else {}
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                **kwargs,
            )
            return response.text or ""
        except Exception as e:
            print(f"[GeminiExtractor] {type(e).__name__}: {e}", file=sys.stderr)
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
{email_body[:6000]}

Extract ALL factual information about the email's AUTHOR (not recipients) — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, personal background.
Include any fact you can reasonably infer from context, even if only partially stated. More data is better.
Aim for at least 3-5 facts per email if any are present.

IMPORTANT: Write each fact as a generic statement without including the person's name.
Use "Works at [company]" not "[Name] works at [company]". Use "Has a degree in..." not "[Name] has a degree in...".

Return a JSON array (empty [] if truly nothing found) of fact objects:
[
  {{"content": "Specific fact about the person (no name)", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences:"""

        try:
            raw = self.complete(prompt)
            return _parse_facts_json(raw, DataSource.GMAIL)
        except Exception as e:
            print(f"[extract_facts_from_email] {type(e).__name__}: {e}", file=sys.stderr)
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
{page_text[:6000]}

Extract ALL factual information about {user_name or "this person"} — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, background.
Include partial or implied facts. Skip only clearly irrelevant generic content.
Aim for at least 5 facts if any are present.

IMPORTANT: Write each fact as a generic statement without including the person's name.
Use "Works at [company]" not "[Name] works at [company]".

Return a JSON array (empty [] if truly nothing found):
[
  {{"content": "Specific fact (no name)", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences:"""

        try:
            raw = self.complete(prompt)
            return _parse_facts_json(raw, source)
        except Exception as e:
            print(f"[extract_facts_from_web] {type(e).__name__}: {e}", file=sys.stderr)
            return []

    # ── Document / resume fact extraction ─────────────────────

    def extract_facts_from_document(
        self,
        text: str,
        filename: str = "",
        user_name: str = "",
    ) -> list[Fact]:
        """
        Extract 50+ structured facts from a document like a resume or bio.

        Uses a more aggressive extraction prompt than email/web since resumes
        are explicitly written to contain professional facts.
        """
        if not self.api_key or not text.strip():
            return []

        name_hint = f"The document belongs to {user_name}. " if user_name else ""
        file_hint = f"Document: {filename}\n" if filename else ""

        prompt = f"""You are a professional career analyst extracting structured facts from a personal document.

{name_hint}{file_hint}
Document content:
{text[:12000]}

This is a personal document (resume, bio, LinkedIn export, etc.) written about or by the person.
Extract EVERY SINGLE factual detail — be exhaustive and thorough. Aim for 50+ facts.

Extract facts across ALL categories:
- work: every job title, company, role, responsibility, team size, budget managed
- education: every degree, school, graduation year, GPA, honors, major, minor
- skills: every technology, tool, language, framework, methodology, certification
- location: every city, country, region mentioned as home or work location
- achievements: every award, promotion, metric (% growth, $ revenue, users, etc.), publication
- personal: name, contact info (no SSN/CC), career goals, personal interests
- interests: hobbies, volunteer work, side projects, communities

IMPORTANT: Write each fact as a generic statement without including the person's name.
Use "Works as a Software Engineer at Acme" not "[Name] works as a Software Engineer at Acme".

Return a JSON array with ALL extracted facts:
[
  {{"content": "Specific fact (no name)", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences. Be exhaustive — more facts = better model training:"""

        all_facts = []
        # If document is long, process in chunks to maximize extraction
        chunk_size = 10000
        chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 40000), chunk_size)]

        for i, chunk in enumerate(chunks):
            chunk_prompt = prompt.replace(text[:12000], chunk) if i > 0 else prompt
            try:
                raw = self.complete(chunk_prompt)
                chunk_facts = _parse_facts_json(raw, DataSource.FILE_UPLOAD)
                all_facts.extend(chunk_facts)
            except Exception as e:
                print(f"[extract_facts_from_document chunk {i}] {type(e).__name__}: {e}", file=sys.stderr)

        # Deduplicate by content
        seen = set()
        unique_facts = []
        for f in all_facts:
            key = f.content.lower()[:80]
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)
        return unique_facts

    # ── Batch email fact extraction ────────────────────────────

    def extract_facts_from_email_batch(
        self,
        emails: list[dict],
        user_name: str = "",
    ) -> list[Fact]:
        """
        Extract facts from a batch of emails in a single Gemini call.

        Each dict should have 'body' and 'subject' keys (already PII-scrubbed).
        Returns a combined deduplicated list of Fact objects.
        """
        if not self.api_key or not emails:
            return []

        user_hint = f"These emails were all written by {user_name}. " if user_name else ""

        # Format batch — cap each email body to keep total prompt < 15k chars
        per_email_limit = max(200, 12000 // len(emails))
        blocks = []
        for i, em in enumerate(emails[:100], 1):
            subject = em.get("subject", "(no subject)")[:100]
            body = em.get("body", "")[:per_email_limit]
            blocks.append(f"=== Email {i} ===\nSubject: {subject}\n{body}")

        combined = "\n\n".join(blocks)

        prompt = f"""You are analyzing {len(emails)} emails written by the same person to extract structured facts about THEM (the author).

{user_hint}
{combined[:15000]}

Extract ALL unique factual information about the email AUTHOR across ALL emails.
Focus on: job/role, company, location, education, skills, interests, achievements, personal background.
Deduplicate — include each unique fact only once.
Aim for at least 10-20 unique facts across the batch.

IMPORTANT: Write each fact as a generic statement without including the person's name.
Use "Works at [company]" not "[Name] works at [company]".

Return a JSON array (empty [] if truly nothing found):
[
  {{"content": "Specific fact about the person (no name)", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences:"""

        try:
            raw = self.complete(prompt)
            return _parse_facts_json(raw, DataSource.GMAIL)
        except Exception as e:
            print(f"[extract_facts_from_email_batch] {type(e).__name__}: {e}", file=sys.stderr)
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
{combined[:6000]}

Extract ALL factual information about the MESSAGE AUTHOR — be generous and inclusive.
Focus on: job/role, company, location, education, skills, interests, achievements, personal background.
Include any fact that can be reasonably inferred from context. More data is better.

IMPORTANT: Write each fact as a generic statement without including the person's name.
Use "Works at [company]" not "[Name] works at [company]".

Return a JSON array (empty [] if truly nothing found):
[
  {{"content": "Specific fact about the person (no name)", "category": "work|education|skills|location|personal|interests|achievements|other", "confidence": 0.0-1.0}},
  ...
]

JSON only, no markdown fences:"""

        try:
            raw = self.complete(prompt)
            return _parse_facts_json(raw, DataSource.FILE_UPLOAD)
        except Exception as e:
            print(f"[extract_facts_from_conversation] {type(e).__name__}: {e}", file=sys.stderr)
            return []

    @classmethod
    def from_env(cls, model: str = "gemini-2.5-flash") -> Optional["GeminiExtractor"]:
        """Build a GeminiExtractor from environment variables. Returns None if no API key."""
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(model=model, api_key=api_key)
