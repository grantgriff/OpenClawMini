"""
Research Agent — gathers and classifies user data from configured sources.

Sources implemented:
  - Gmail (sent emails) — fact extraction via Gemini + writing samples
  - Web search (DuckDuckGo) — public mentions, LinkedIn public profile
  - File upload (ChatGPT/Claude exports) — conversation logs → facts + samples

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
        agent = ResearchAgent(store, memory, gemini_extractor=extractor)
        result = agent.run(progress_callback=my_fn)
    """

    def __init__(
        self,
        store: MemoryStore,
        memory: Memory,
        llm_client=None,
        gmail_max_results: int = 500,
    ) -> None:
        self.store = store
        self.memory = memory
        # llm_client may be a GeminiExtractor (supports .complete() AND
        # .extract_facts_from_email() etc.) or any object with .complete().
        self.classifier = MemoryClassifier(llm_client=llm_client)
        self._extractor = llm_client  # may be None
        self.gmail_max_results = gmail_max_results

    # ── Public API ─────────────────────────────────────────────

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

        if sources.get("gmail"):
            self._research_gmail(result, progress_callback)

        if sources.get("web_search"):
            self._research_web(result, progress_callback)

        if sources.get("linkedin"):
            self._research_linkedin_public(result, progress_callback)

        if sources.get("file_upload"):
            # File paths passed via sources dict or handled directly in CLI
            file_paths = sources.get("file_paths", [])
            for path in file_paths:
                self._research_file(path, result)

        return result

    def process_uploaded_files(self, file_paths: list[str]) -> ResearchResult:
        """Process a list of uploaded log files and add to memory. Called directly from CLI."""
        result = ResearchResult()
        for path in file_paths:
            self._research_file(path, result)
        return result

    # ── Gmail ──────────────────────────────────────────────────

    def _research_gmail(
        self,
        result: ResearchResult,
        progress_callback=None,
    ) -> None:
        """Fetch sent emails, extract facts with Gemini (or rule-based), always store writing samples."""
        from openclawmini.integrations.gmail import gmail_client_from_env

        client = gmail_client_from_env()
        if client is None:
            result.errors.append(
                "Gmail credentials not configured. "
                "Set GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET in .env"
            )
            return

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
            self._process_email(email, result)

        if fetch_result.emails:
            analysis = self.store.compute_style_analysis(self.memory)
            self.store.update_style_analysis(self.memory, analysis)

    def _process_email(self, email, result: ResearchResult) -> None:
        """
        Process a single parsed email:
          1. Always store it as a writing sample (captures style).
          2. Use Gemini to extract multiple facts (if available), else rule-based.
          3. Always do rule-based relationship + preference extraction.
        """
        # ── 1. Writing sample (always) ─────────────────────────
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

        # ── 2. Fact extraction ─────────────────────────────────
        if self._extractor and hasattr(self._extractor, "extract_facts_from_email"):
            # Gemini multi-fact extraction
            facts = self._extractor.extract_facts_from_email(
                email_body=email.body,
                email_subject=email.subject,
                user_name=self.memory.user.name,
            )
            for fact in facts:
                before = len(self.memory.facts)
                self.store.add_fact(self.memory, fact)
                if len(self.memory.facts) > before:
                    result.facts_added += 1
        else:
            # Rule-based fallback: only store if classified as FACTUAL
            classification = self.classifier.classify(email.body, source=DataSource.GMAIL)
            if classification.category == MemoryCategory.FACTUAL:
                fact = Fact(
                    content=email.body[:500],
                    category=classification.sub_category or FactCategory.OTHER,
                    confidence=classification.confidence,
                    source=DataSource.GMAIL,
                )
                before = len(self.memory.facts)
                self.store.add_fact(self.memory, fact)
                if len(self.memory.facts) > before:
                    result.facts_added += 1

        # ── 3. Relationship extraction (rule-based, always) ────
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

        # ── 4. Preference extraction (rule-based, always) ──────
        pref_cls = self.classifier._classify_rule_based(email.body, source=DataSource.GMAIL)
        if pref_cls.category == MemoryCategory.PREFERENCE:
            pref = Preference(
                content=email.body[:300],
                category=pref_cls.sub_category or "other",
                evidence=f"Extracted from email: {email.subject[:60]}",
                confidence=pref_cls.confidence,
                source=DataSource.GMAIL,
            )
            before = len(self.memory.preferences)
            self.store.add_preference(self.memory, pref)
            if len(self.memory.preferences) > before:
                result.preferences_added += 1

    # ── Web search ─────────────────────────────────────────────

    def _research_web(
        self,
        result: ResearchResult,
        progress_callback=None,
    ) -> None:
        """Search DuckDuckGo for public mentions and scrape for facts."""
        from openclawmini.integrations.web_search import (
            search_duckduckgo,
            scrape_page,
            build_search_queries,
        )

        extractor = self._get_extractor(result, "Web search")
        if extractor is None:
            return

        name = self.memory.user.name
        email = self.memory.user.email
        linkedin_url = os.getenv("LINKEDIN_PROFILE_URL", "").strip()

        if not name:
            result.errors.append(
                "No user name configured for web search. Run openclawmini init first."
            )
            return

        queries = build_search_queries(name, email, linkedin_url)

        seen_urls: set[str] = set()
        total_queries = min(len(queries), 3)

        for i, query in enumerate(queries[:3]):
            if progress_callback:
                progress_callback("web_search", i + 1, total_queries)

            search_result = search_duckduckgo(query, max_results=3)
            if search_result.error:
                result.errors.append(f"Web search error: {search_result.error}")
                break

            for r in search_result.results:
                if not r.url or r.url in seen_urls:
                    continue
                seen_urls.add(r.url)

                # Scrape page for richer content; fall back to snippet
                page_text = scrape_page(r.url) or r.snippet
                if not page_text:
                    continue

                facts = extractor.extract_facts_from_web(
                    page_text=page_text,
                    url=r.url,
                    user_name=name,
                )
                for fact in facts:
                    before = len(self.memory.facts)
                    self.store.add_fact(self.memory, fact)
                    if len(self.memory.facts) > before:
                        result.facts_added += 1

    # ── LinkedIn public profile ────────────────────────────────

    def _research_linkedin_public(
        self,
        result: ResearchResult,
        progress_callback=None,
    ) -> None:
        """Scrape public LinkedIn profile if LINKEDIN_PROFILE_URL is set."""
        from openclawmini.integrations.web_search import scrape_linkedin_public

        linkedin_url = os.getenv("LINKEDIN_PROFILE_URL", "").strip()
        if not linkedin_url:
            result.errors.append(
                "LinkedIn profile URL not set. "
                "Add LINKEDIN_PROFILE_URL=https://linkedin.com/in/yourhandle to .env"
            )
            return

        extractor = self._get_extractor(result, "LinkedIn scraping")
        if extractor is None:
            return

        if progress_callback:
            progress_callback("linkedin", 1, 1)

        page_text, err = scrape_linkedin_public(linkedin_url)
        if err:
            result.errors.append(f"LinkedIn scrape: {err}")
            return

        facts = extractor.extract_facts_from_web(
            page_text=page_text,
            url=linkedin_url,
            user_name=self.memory.user.name,
        )
        for fact in facts:
            before = len(self.memory.facts)
            self.store.add_fact(self.memory, fact)
            if len(self.memory.facts) > before:
                result.facts_added += 1

    # ── File upload ────────────────────────────────────────────

    def _research_file(self, file_path: str, result: ResearchResult) -> None:
        """Parse an uploaded conversation log and extract facts + writing samples."""
        from openclawmini.memory.log_parser import parse_log_file
        from openclawmini.memory.schema import InteractionLog

        try:
            conversations = parse_log_file(file_path)
        except (ValueError, FileNotFoundError) as e:
            result.errors.append(f"Could not parse file {file_path}: {e}")
            return

        log = InteractionLog(
            source_file=file_path,
            format=conversations[0].source_format if conversations else "generic",
        )

        for conv in conversations:
            user_msgs = conv.user_messages()

            # Writing samples from user messages
            for msg in user_msgs:
                if len(msg.split()) >= 15:
                    sample = WritingSample(
                        text=msg,
                        category=WritingCategory.OTHER,
                        context=f"Conversation: {conv.title[:60]}",
                        source=DataSource.FILE_UPLOAD,
                    )
                    before = len(self.memory.writing_samples)
                    self.store.add_writing_sample(self.memory, sample)
                    if len(self.memory.writing_samples) > before:
                        result.writing_samples_added += 1
                        log.samples_extracted += 1

            # Fact extraction with Gemini (if available)
            extractor = self._extractor
            if extractor and hasattr(extractor, "extract_facts_from_conversation"):
                facts = extractor.extract_facts_from_conversation(
                    user_messages=user_msgs,
                    user_name=self.memory.user.name,
                )
                for fact in facts:
                    before = len(self.memory.facts)
                    self.store.add_fact(self.memory, fact)
                    if len(self.memory.facts) > before:
                        result.facts_added += 1
                        log.facts_extracted += 1
            else:
                # Rule-based fallback: classify each user message
                for msg in user_msgs:
                    cls = self.classifier.classify(msg, source=DataSource.FILE_UPLOAD)
                    if cls.category == MemoryCategory.FACTUAL:
                        fact = Fact(
                            content=msg[:500],
                            category=cls.sub_category or FactCategory.OTHER,
                            confidence=cls.confidence,
                            source=DataSource.FILE_UPLOAD,
                        )
                        before = len(self.memory.facts)
                        self.store.add_fact(self.memory, fact)
                        if len(self.memory.facts) > before:
                            result.facts_added += 1
                            log.facts_extracted += 1

            log.conversations_extracted += 1
            log.training_pairs_generated += len(conv.as_pairs())

        self.store.add_interaction_log(self.memory, log)

    # ── Helpers ────────────────────────────────────────────────

    def _get_extractor(self, result: ResearchResult, context: str):
        """
        Return a GeminiExtractor, building one from env if llm_client wasn't set.
        Appends an error to result and returns None if no API key available.
        """
        if self._extractor and hasattr(self._extractor, "extract_facts_from_web"):
            return self._extractor

        # Try to build one on the fly
        from openclawmini.integrations.gemini_extractor import GeminiExtractor
        extractor = GeminiExtractor.from_env()
        if extractor is None:
            result.errors.append(
                f"{context} requires GOOGLE_API_KEY. "
                "Add it to .env or run openclawmini init."
            )
            return None
        self._extractor = extractor
        return extractor


# ── Email helpers ─────────────────────────────────────────────

def _email_writing_category(email) -> str:
    """Guess writing category from email context."""
    body_lower = email.body.lower()
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
         "john@example.com" → None
    """
    import re
    match = re.match(r'^"?([^"<@]+?)"?\s*<', recipient)
    if match:
        name = match.group(1).strip()
        if len(name) > 1 and not name.startswith("@"):
            return name
    return None


def _infer_relationship_type(email) -> str:
    """Guess relationship type from email tone."""
    body_lower = email.body.lower()
    casual = sum(w in body_lower for w in ["hey", "yo", "dude", "bro", "hang", "weekend", "party"])
    if casual >= 2:
        return "friend"
    formal = sum(w in body_lower for w in ["meeting", "team", "project", "deadline", "client", "report"])
    if formal >= 1:
        return "coworker"
    return "contact"


def _infer_frequency(email) -> str:
    """Placeholder — frequency analysis happens across the full email set."""
    return "occasional"


def _detect_available_sources() -> dict:
    """Detect which sources are configured based on environment variables."""
    return {
        "gmail": bool(
            os.getenv("GMAIL_CLIENT_ID", "").strip()
            and os.getenv("GMAIL_CLIENT_SECRET", "").strip()
        ),
        "linkedin": bool(os.getenv("LINKEDIN_PROFILE_URL", "").strip()),
        "web_search": bool(
            os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("LINKEDIN_PROFILE_URL", "").strip()
        ),
        "file_upload": False,  # Triggered directly from CLI
    }
