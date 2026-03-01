"""
Research Agent — gathers and classifies user data from configured sources.

Sources implemented:
  - Gmail (sent emails) — fact extraction via Gemini + writing samples
  - Web search (Brave API primary, DuckDuckGo fallback) — public mentions
  - LinkedIn public profile — with fallback messaging
  - GitHub profile + repos — via public GitHub API (no auth needed)
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

    def targeted_research(
        self,
        user_name: str,
        category: str,
        memory,
        max_results: int = 7,
    ) -> ResearchResult:
        """
        Run targeted web research to gather more facts about a specific category
        the model is weak on (e.g. "work", "education", "skills").

        Uses Brave/DuckDuckGo with 5 category-specific queries + parallel page
        scraping. For "skills" and "achievements", also checks GitHub profile.

        Args:
            user_name: The user's name for search queries.
            category: Fact category to target (work, education, skills, etc.)
            memory: Memory object to add new facts to.
            max_results: Max web search results per query.

        Returns:
            ResearchResult with counts of items added.
        """
        from openclawmini.integrations.web_search import (
            search,
            scrape_pages_parallel,
            fetch_github_profile,
            github_profile_to_text,
            detect_github_username,
        )

        result = ResearchResult()
        extractor = self._get_extractor(result, f"Targeted research ({category})")
        if extractor is None:
            return result

        queries = _build_targeted_queries(user_name, category)
        seen_urls: set[str] = set()
        all_search_results = []

        # Run up to 5 queries, collect all unique URLs
        for query in queries[:5]:
            sr = search(query, max_results=max_results)
            for r in sr.results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_search_results.append(r)

        # Parallel scraping of all collected URLs
        urls_to_scrape = [r.url for r in all_search_results if r.url]
        scraped = scrape_pages_parallel(urls_to_scrape, max_workers=6)

        for r in all_search_results:
            page_text = scraped.get(r.url, "") or r.snippet
            if not page_text:
                continue
            facts = _extract_targeted_facts(
                extractor=extractor,
                page_text=page_text,
                url=r.url,
                user_name=user_name,
                category=category,
            )
            for fact in facts:
                before = len(memory.facts)
                self.store.add_fact(memory, fact)
                if len(memory.facts) > before:
                    result.facts_added += 1

        # GitHub profile is especially useful for skills + achievements
        if category in ("skills", "achievements", "work"):
            email = getattr(self.memory.user, "email", "") if self.memory else ""
            github_username = (
                os.getenv("GITHUB_USERNAME", "").strip()
                or detect_github_username(user_name, email)
            )
            if github_username:
                gh_data = fetch_github_profile(github_username)
                if gh_data:
                    gh_text = github_profile_to_text(gh_data)
                    facts = _extract_targeted_facts(
                        extractor=extractor,
                        page_text=gh_text,
                        url=f"https://github.com/{github_username}",
                        user_name=user_name,
                        category=category,
                    )
                    for fact in facts:
                        before = len(memory.facts)
                        self.store.add_fact(memory, fact)
                        if len(memory.facts) > before:
                            result.facts_added += 1

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
          0. Redact PII from body before any storage or extraction.
          1. Always store it as a writing sample (captures style).
          2. Use Gemini to extract multiple facts (if available), else rule-based.
          3. Always do rule-based relationship + preference extraction.
        """
        # ── 0. PII scrubbing (always, before anything else) ───
        from openclawmini.utils.pii_scrubber import redact_pii
        safe_body = redact_pii(email.body)

        # ── 1. Writing sample (always) ─────────────────────────
        writing_cat = _email_writing_category(email)
        sample = WritingSample(
            text=safe_body,
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
                email_body=safe_body,
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
            classification = self.classifier.classify(safe_body, source=DataSource.GMAIL)
            if classification.category == MemoryCategory.FACTUAL:
                fact = Fact(
                    content=safe_body[:500],
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
        pref_cls = self.classifier._classify_rule_based(safe_body, source=DataSource.GMAIL)
        if pref_cls.category == MemoryCategory.PREFERENCE:
            pref = Preference(
                content=safe_body[:300],
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
        """
        Multi-source web research:
          1. Brave Search (primary) or DuckDuckGo fallback — up to 5 queries
          2. GitHub profile scrape (if detectable from name/email)
          3. Parallel page scraping for all found URLs
        """
        from openclawmini.integrations.web_search import (
            search,
            scrape_pages_parallel,
            build_search_queries,
            fetch_github_profile,
            github_profile_to_text,
            detect_github_username,
        )

        extractor = self._get_extractor(result, "Web search")
        if extractor is None:
            return

        name = self.memory.user.name
        email = self.memory.user.email
        linkedin_url = os.getenv("LINKEDIN_PROFILE_URL", "").strip()
        github_username = os.getenv("GITHUB_USERNAME", "").strip()

        if not name:
            result.errors.append(
                "No user name configured for web search. Run openclawmini init first."
            )
            return

        # Generate 20 personalized queries with Gemini; fall back to static list
        gemini_queries = _generate_web_search_queries(extractor, name, email, self.memory, count=20)
        if gemini_queries:
            queries = gemini_queries
        else:
            # Gemini query generation failed; use static fallback
            queries = build_search_queries(name, email, linkedin_url)
            result.errors.append(
                f"Gemini query generation failed — using {len(queries)} static queries. "
                "Check GOOGLE_API_KEY if you expected 20 personalized queries."
            )
        seen_urls: set[str] = set()
        total_queries = min(len(queries), 20)

        # ── Collect all URLs from search queries ──────────────
        all_results = []
        for i, query in enumerate(queries[:20]):
            if progress_callback:
                progress_callback("web_search", i + 1, total_queries)
            sr = search(query, max_results=10)
            for r in sr.results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

        # ── Parallel page scraping ────────────────────────────
        urls_to_scrape = [r.url for r in all_results if r.url]
        if progress_callback and urls_to_scrape:
            progress_callback("web_scrape", 0, len(urls_to_scrape))
        scraped = scrape_pages_parallel(urls_to_scrape, max_workers=6)
        if progress_callback and urls_to_scrape:
            progress_callback("web_scrape", len(urls_to_scrape), len(urls_to_scrape))

        for r in all_results:
            page_text = scraped.get(r.url, "") or r.snippet
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

        # ── GitHub profile (extra signal for devs) ────────────
        if not github_username:
            github_username = detect_github_username(name, email)

        if github_username:
            gh_data = fetch_github_profile(github_username)
            if gh_data:
                gh_text = github_profile_to_text(gh_data)
                facts = extractor.extract_facts_from_web(
                    page_text=gh_text,
                    url=f"https://github.com/{github_username}",
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
        Appends an error and returns None if GOOGLE_API_KEY is not available.
        """
        if self._extractor and hasattr(self._extractor, "extract_facts_from_web"):
            return self._extractor

        from openclawmini.integrations.gemini_extractor import GeminiExtractor
        extractor = GeminiExtractor.from_env()
        if extractor is not None:
            self._extractor = extractor
            return extractor

        result.errors.append(
            f"{context} requires GOOGLE_API_KEY. "
            "Add it to .env or run openclawmini init."
        )
        return None


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
    # Web search runs if we have any extraction backend (Gemini or Mistral).
    # DuckDuckGo is free so search itself never needs a key; Brave is preferred
    # if BRAVE_API_KEY is set.
    has_extractor = bool(
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("MISTRAL_API_KEY", "").strip()
    )
    return {
        "gmail": bool(
            os.getenv("GMAIL_CLIENT_ID", "").strip()
            and os.getenv("GMAIL_CLIENT_SECRET", "").strip()
        ),
        "linkedin": bool(os.getenv("LINKEDIN_PROFILE_URL", "").strip()),
        "web_search": has_extractor or bool(os.getenv("LINKEDIN_PROFILE_URL", "").strip()),
        "file_upload": False,  # Triggered directly from CLI
    }


# ── Targeted research helpers ──────────────────────────────────

_CATEGORY_QUERIES = {
    "work": [
        "{name} job title company",
        "{name} career employment",
        "{name} professional role",
    ],
    "education": [
        "{name} university degree education",
        "{name} college graduation",
        "{name} studied academic background",
    ],
    "skills": [
        "{name} skills expertise technology",
        "{name} proficient tools programming",
        "{name} professional skills",
    ],
    "location": [
        "{name} location city lives",
        "{name} based where",
        "{name} hometown current location",
    ],
    "personal": [
        "{name} personal background story",
        "{name} about bio",
        "{name} interests hobbies",
    ],
    "interests": [
        "{name} interests hobbies activities",
        "{name} passion projects",
        "{name} extracurricular",
    ],
    "achievements": [
        "{name} achievements awards recognition",
        "{name} accomplishments built created",
        "{name} notable projects",
    ],
    "other": [
        "{name} profile background",
        "{name} about",
    ],
}


def _build_targeted_queries(user_name: str, category: str) -> list[str]:
    """Build search queries for a specific fact category."""
    templates = _CATEGORY_QUERIES.get(category.lower(), _CATEGORY_QUERIES["other"])
    return [t.format(name=user_name) for t in templates]


def _extract_targeted_facts(
    extractor,
    page_text: str,
    url: str,
    user_name: str,
    category: str,
) -> list:
    """Extract facts focused on a specific category."""
    from openclawmini.memory.schema import DataSource, Fact, FactCategory

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
    target_cat = _CATEGORY_MAP.get(category.lower(), FactCategory.OTHER)
    source = DataSource.LINKEDIN if "linkedin" in url.lower() else DataSource.WEB_SEARCH

    prompt = f"""Extract facts about {user_name} from this page, focusing specifically on their {category}.
Source: {url}
Content: {page_text[:2500]}

Focus exclusively on {category}-related facts: {', '.join(_CATEGORY_QUERIES.get(category, ['background'])[:1])}.
Be generous — include any fact you can reasonably infer.

Return JSON array:
[{{"content": "fact text", "category": "{category}", "confidence": 0.0-1.0}}]
JSON only:"""

    try:
        import json
        import re
        raw = extractor.complete(prompt)
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        facts = []
        for item in data:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content or len(content) < 10:
                continue
            confidence = float(item.get("confidence", 0.7))
            facts.append(Fact(
                content=content,
                category=target_cat,
                confidence=max(0.0, min(1.0, confidence)),
                source=source,
            ))
        return facts
    except Exception:
        return []


def _generate_web_search_queries(
    extractor,
    user_name: str,
    user_email: str = "",
    memory=None,
    count: int = 20,
) -> list[str]:
    """
    Use Gemini Flash to generate personalized web search queries for the user.

    Returns up to `count` queries. Falls back to an empty list on any error
    so the caller can fall back to static queries from build_search_queries().
    """
    import json
    import re

    if extractor is None:
        return []

    # Give Gemini context: name, email domain, and a sample of known facts
    known_facts = ""
    if memory and getattr(memory, "facts", None):
        sample = [f.content for f in memory.facts[:10]]
        if sample:
            known_facts = "\nKnown facts so far:\n" + "\n".join(f"- {f}" for f in sample)

    email_hint = ""
    if user_email and "@" in user_email:
        domain = user_email.split("@")[-1]
        if domain not in {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"}:
            email_hint = f"\nEmail domain: {domain}"

    prompt = (
        f'Generate {count} diverse web search queries to find public information '
        f'about a person named "{user_name}".{email_hint}'
        f'{known_facts}\n\n'
        f"Goal: Discover verifiable facts about their career, education, skills, "
        f"achievements, background, interests, and projects.\n"
        f"Use varied query styles: LinkedIn profiles, interviews, GitHub repos, "
        f"news mentions, company bios, conference speaker bios, publications, etc.\n"
        f"Target a different angle with each query — don't repeat similar queries.\n\n"
        f'Return ONLY a JSON array of {count} search query strings, no explanation:\n'
        f'["query 1", "query 2", ...]'
    )

    try:
        raw = extractor.complete(prompt)
        if raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                if isinstance(queries, list) and queries:
                    return [str(q).strip() for q in queries if str(q).strip()][:count]
    except Exception:
        pass

    return []
