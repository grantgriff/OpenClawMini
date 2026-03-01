"""Memory store — JSON file persistence for the structured memory profile."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from openclawmini.memory.schema import (
    Fact,
    InteractionLog,
    Memory,
    Post,
    Preference,
    Relationship,
    StyleAnalysis,
    WritingSample,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    """
    File-based JSON store for the user's memory profile.

    Usage:
        store = MemoryStore("./data/memory.json")
        mem = store.load()
        store.add_fact(mem, Fact(content="Works at a pre-Series A AI startup"))
        store.save(mem)
    """

    def __init__(self, path: str = "./data/memory.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load / Save ────────────────────────────────────────────

    def load(self) -> Memory:
        """Load memory from disk. Returns an empty Memory if file doesn't exist."""
        if not self.path.exists():
            return Memory()
        with open(self.path) as f:
            raw = json.load(f)
        return Memory.model_validate(raw)

    def save(self, memory: Memory) -> None:
        """Persist memory to disk as pretty-printed JSON."""
        memory.user.last_updated = _now()
        with open(self.path, "w") as f:
            json.dump(
                memory.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

    def exists(self) -> bool:
        return self.path.exists()

    # ── Append helpers ─────────────────────────────────────────

    def add_fact(self, memory: Memory, fact: Fact) -> None:
        """Add a fact, deduplicating by content similarity (exact match)."""
        existing_contents = {f.content.lower().strip() for f in memory.facts}
        if fact.content.lower().strip() not in existing_contents:
            memory.facts.append(fact)

    def add_writing_sample(self, memory: Memory, sample: WritingSample) -> None:
        existing_texts = {s.text.lower().strip() for s in memory.writing_samples}
        if sample.text.lower().strip() not in existing_texts:
            memory.writing_samples.append(sample)

    def add_post(self, memory: Memory, post: Post) -> None:
        existing_texts = {p.text.lower().strip() for p in memory.posts}
        if post.text.lower().strip() not in existing_texts:
            memory.posts.append(post)

    def add_relationship(self, memory: Memory, rel: Relationship) -> None:
        existing_names = {r.name.lower().strip() for r in memory.relationships}
        if rel.name.lower().strip() not in existing_names:
            memory.relationships.append(rel)

    def add_preference(self, memory: Memory, pref: Preference) -> None:
        existing = {p.content.lower().strip() for p in memory.preferences}
        if pref.content.lower().strip() not in existing:
            memory.preferences.append(pref)

    def update_style_analysis(self, memory: Memory, analysis: StyleAnalysis) -> None:
        memory.style_analysis = analysis

    def add_interaction_log(self, memory: Memory, log: InteractionLog) -> None:
        memory.interaction_logs.append(log)

    # ── Merge ──────────────────────────────────────────────────

    def merge(self, memory: Memory, other: Memory) -> int:
        """
        Merge items from `other` into `memory`, deduplicating.
        Returns the total number of new items added.
        """
        added = 0
        before = memory.stats()

        for fact in other.facts:
            self.add_fact(memory, fact)
        for sample in other.writing_samples:
            self.add_writing_sample(memory, sample)
        for post in other.posts:
            self.add_post(memory, post)
        for rel in other.relationships:
            self.add_relationship(memory, rel)
        for pref in other.preferences:
            self.add_preference(memory, pref)
        for log in other.interaction_logs:
            memory.interaction_logs.append(log)

        after = memory.stats()
        added = after["total_items"] - before["total_items"]
        return added

    # ── Style analysis computation ─────────────────────────────

    def compute_style_analysis(self, memory: Memory) -> StyleAnalysis:
        """
        Compute style metrics from writing samples and posts.
        Lightweight heuristics — no LLM required.
        """
        all_texts = [s.text for s in memory.writing_samples] + [p.text for p in memory.posts]
        if not all_texts:
            return StyleAnalysis()

        # Average sentence length (words per sentence)
        total_words = 0
        total_sentences = 0
        for text in all_texts:
            sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
            total_sentences += len(sentences)
            total_words += len(text.split())

        avg_sentence_length = round(total_words / max(total_sentences, 1), 1)

        # Formality score: higher = more formal
        casual_markers = ["hey", "yo", "lol", "haha", "awesome", "cool", "gonna", "wanna", "yeah", "nope"]
        formal_markers = ["sincerely", "regards", "dear", "pursuant", "herewith", "furthermore", "accordingly"]
        casual_count = sum(
            sum(1 for word in text.lower().split() if word in casual_markers)
            for text in all_texts
        )
        formal_count = sum(
            sum(1 for word in text.lower().split() if word in formal_markers)
            for text in all_texts
        )
        total_markers = casual_count + formal_count
        formality_score = round(formal_count / max(total_markers, 1), 2) if total_markers else 0.5

        # Emoji usage
        emoji_count = sum(
            sum(1 for ch in text if ord(ch) > 127 and not ch.isalpha())
            for text in all_texts
        )
        avg_emoji = emoji_count / len(all_texts)
        emoji_usage = "frequent" if avg_emoji > 2 else ("occasional" if avg_emoji > 0.3 else "rare")

        # Common short phrases (2-gram frequency, top 10)
        from collections import Counter
        all_words = " ".join(all_texts).lower().split()
        bigrams = [f"{all_words[i]} {all_words[i+1]}" for i in range(len(all_words) - 1)]
        common_phrases = [phrase for phrase, _ in Counter(bigrams).most_common(10)]

        # Average email length (words)
        email_samples = [s for s in memory.writing_samples if "email" in s.category.lower()]
        avg_email_length = int(
            sum(len(s.text.split()) for s in email_samples) / max(len(email_samples), 1)
        ) if email_samples else 0

        # Tone guess
        tone = "friendly_professional"
        if formality_score > 0.7:
            tone = "formal"
        elif formality_score < 0.3:
            tone = "casual"

        return StyleAnalysis(
            avg_sentence_length=avg_sentence_length,
            formality_score=formality_score,
            emoji_usage=emoji_usage,
            common_phrases=common_phrases,
            avg_email_length=avg_email_length,
            tone=tone,
        )
