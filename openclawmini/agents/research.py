"""
Research Agent — gathers and classifies user data from configured sources.

Task 3: Gmail support
Task 4: LinkedIn, web search, log file upload (to be added)

All data access is READ-ONLY.
Gmail scope: gmail.readonly (read only — no send/delete/modify)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from openclawmini.memory.schema import (
    DataSource,
    Fact,
    FactCategory,
    Memory,
    Preference,
    Relationship,
    WritingCategory,
    WritingSample,
)
from openclawmini.memory.store import MemoryStore
from openclawmini.memory.classifier import MemoryClassifier, MemoryCategory


@dataclass
class ResearchResult:
    """Summary of what the research agent collected."""
    facts_added: int = 0
    writing_samples_added: int = 0
    posts_added: int = 0
    relationships_added: int = 0
    preferences_added: int = 0
    emails_processed: int = 0
    emails_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total_added(self) -> int:
        return (
            self.facts_added
            + self.writing_samples_added
            + self.posts_added
            + self.relationships_added
            + self.preferences_added
        )

    def summary(self) -> str:
        lines = [
            f"  Facts added:           {self.facts_added}",
            f"  Writing samples added: {self.writing_samples_added}",
            f"  Relationships added:   {self.relationships_added}",
            f"  Preferences added:     {self.preferences_added}",
            f"  Emails processed:      {self.emails_processed}",
        ]
        if self.errors:
            lines.append(f"  Errors:                {len(self.errors)}")
        return "\n".join(lines)


class ResearchAgent:
    """
    Gathers user data from configured sources and classifies it into memory.

    Usage:
        agent = ResearchAgent(store, memory)
        result = agent.run(progress_callback=my_fn)
    """

    def __init__(
        self,
        store: MemoryStore,
        memory: Memory,
        llm_client=None,
        gmail_max_results: int = 200,
    ) -> None:
        self.store = store
        self.memory = memory
        self.classifier = MemoryClassifier(llm_client=llm_client)
        self.gmail_max_results = gmail_max_results

    def run(
        self,
        sources: Optional[dict] = None,
        progress_callback=None,
    ) -> ResearchResult:
        """
        Run research across all configured sources.

        Args:
            sources: Dict of {source_name: bool}. Defaults to env-based detection.
            progress_callback: Optional callable(stage: str, current: int, total: int)

        Returns:
            ResearchResult with counts of items added.
        """
        result = ResearchResult()

        if sources is None:
            sources = _detect_available_sources()

        # ── Gmail ──────────────────────────────────────────────
        if sources.get("gmail"):
            self._research_gmail(result, progress_callback)

        # ── LinkedIn, web, logs — Task 4 ──────────────────────
        # Stub hooks so Task 4 can slot in cleanly
        if sources.get("linkedin"):
            result.errors.append("LinkedIn research not yet implemented (Task 4)")

        if sources.get("web_search"):
            result.errors.append("Web search not yet implemented (Task 4)")

        if sources.get("file_upload"):
            result.errors.append("File upload not yet implemented (Task 4)")

        return result

    # ── Gmail ──────────────────────────────────────────────────

    def _research_gmail(
        self,
        result: ResearchResult,
        progress_callback=None,
    ) -> None:
        """Fetch sent emails, classify each one, and add to memory."""
        from openclawmini.integrations.gmail import gmail_client_from_env

        client = gmail_client_from_env()
        if client is None:
            result.errors.append(
                "Gmail credentials not configured. "
                "Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET in .env"
            )
            return

        # Authenticate (reuses token if valid, otherwise opens browser)
        try:
            client.authenticate()
        except Exception as e:
            result.errors.append(f"Gmail authentication failed: {e}")
            return

        def _gmail_progress(fetched: int, total: int) -> None:
            if progress_callback:
                progress_callback("gmail", fetched, total)

        fetch_result = client.fetch_sent_emails(
            max_results=self.gmail_max_results,
            progress_callback=_gmail_progress,
        )

        if fetch_result.error:
            result.errors.append(f"Gmail fetch error: {fetch_result.error}")

        result.emails_skipped += fetch_result.skipped_short + fetch_result.skipped_no_body

        for email in fetch_result.emails:
            result.emails_processed += 1
            self._classify_and_store_email(email, result)

        # Recompute style analysis after adding new samples
        if fetch_result.emails:
            analysis = self.store.compute_style_analysis(self.memory)
            self.store.update_style_analysis(self.memory, analysis)

    def _classify_and_store_email(self, email, result: ResearchResult) -> None:
        """Classify a single parsed email and add it to the appropriate memory category."""
        classification = self.classifier.classify(email.body, source=DataSource.GMAIL)

        if classification.category == MemoryCategory.FACTUAL:
            fact = Fact(
                content=email.body[:500],  # Truncate long factual content
                category=classification.sub_category or FactCategory.OTHER,
                confidence=classification.confidence,
                source=DataSource.GMAIL,
            )
            before = len(self.memory.facts)
            self.store.add_fact(self.memory, fact)
            if len(self.memory.facts) > before:
                result.facts_added += 1

        elif classification.category == MemoryCategory.STYLISTIC:
            # Determine writing category from email context
            writing_cat = _email_writing_category(email)
            sample = WritingSample(
                text=email.body,
                category=writing_cat,
                context=f"Email: {email.subject[:80]}",
                source=DataSource.GMAIL,
            )
            before = len(self.memory.writing_samples)
            self.store.add_writing_sample(self.memory, sample)
            if len(self.memory.writing_samples) > before:
                result.writing_samples_added += 1

        elif classification.category == MemoryCategory.PREFERENCE:
            pref = Preference(
                content=email.body[:300],
                category=classification.sub_category or "other",
                evidence=f"Extracted from email: {email.subject[:60]}",
                confidence=classification.confidence,
                source=DataSource.GMAIL,
            )
            before = len(self.memory.preferences)
            self.store.add_preference(self.memory, pref)
            if len(self.memory.preferences) > before:
                result.preferences_added += 1

        elif classification.category == MemoryCategory.RELATIONSHIP:
            # Try to extract a name from the email headers
            name = _extract_recipient_name(email.recipient)
            if name:
                rel = Relationship(
                    name=name,
                    relationship=_infer_relationship_type(email),
                    interaction_frequency=_infer_frequency(email),
                    source=DataSource.GMAIL,
                )
                before = len(self.memory.relationships)
                self.store.add_relationship(self.memory, rel)
                if len(self.memory.relationships) > before:
                    result.relationships_added += 1


# ── Email helpers ─────────────────────────────────────────────

def _email_writing_category(email) -> str:
    """Guess writing category from email context."""
    subject_lower = email.subject.lower()
    body_lower = email.body.lower()

    # Signals for professional vs casual
    casual_signals = sum([
        "hey" in body_lower[:50],
        "yo" in body_lower[:20],
        "lol" in body_lower,
        "haha" in body_lower,
        email.body.split()[0].lower() in ("hey", "yo", "hi") if email.body.split() else False,
    ])

    if casual_signals >= 2:
        return WritingCategory.CASUAL_EMAIL
    return WritingCategory.PROFESSIONAL_EMAIL


def _extract_recipient_name(recipient: str) -> Optional[str]:
    """
    Extract a display name from an email To: header.
    e.g. "John Smith <john@example.com>" → "John Smith"
         "john@example.com" → None (no display name)
    """
    import re
    match = re.match(r'^"?([^"<@]+?)"?\s*<', recipient)
    if match:
        name = match.group(1).strip()
        if len(name) > 1 and not name.startswith("@"):
            return name
    return None


def _infer_relationship_type(email) -> str:
    """Guess relationship type from email tone and content."""
    body_lower = email.body.lower()
    casual = sum(w in body_lower for w in ["hey", "yo", "dude", "bro", "hang", "weekend", "party"])
    if casual >= 2:
        return "friend"
    formal = sum(w in body_lower for w in ["meeting", "team", "project", "deadline", "client", "report"])
    if formal >= 1:
        return "coworker"
    return "contact"


def _infer_frequency(email) -> str:
    """Placeholder — real frequency analysis happens across the full email set."""
    return "occasional"


def _detect_available_sources() -> dict:
    """Detect which sources are configured based on environment variables."""
    return {
        "gmail": bool(
            os.getenv("GMAIL_CLIENT_ID", "").strip()
            and os.getenv("GMAIL_CLIENT_SECRET", "").strip()
        ),
        "linkedin": bool(
            os.getenv("LINKEDIN_CLIENT_ID", "").strip()
            and os.getenv("LINKEDIN_CLIENT_SECRET", "").strip()
        ),
        "web_search": False,   # Task 4
        "file_upload": False,  # Task 4
    }
