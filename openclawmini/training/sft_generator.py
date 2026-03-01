"""
SFT Data Generator — creates supervised fine-tuning Q&A pairs from memory.facts.

Output format: JSONL with chat messages, compatible with Mistral fine-tuning API.
Each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return f"sft_{uuid.uuid4().hex[:8]}"


@dataclass
class SFTSample:
    id: str = field(default_factory=_new_id)
    messages: list[dict] = field(default_factory=list)
    source_fact_id: str = ""
    source_fact_content: str = ""
    quality_score: float = 7.0
    category: str = "factual"
    generated_at: datetime = field(default_factory=_now)

    def to_jsonl_line(self) -> str:
        """Return the Mistral fine-tuning JSONL format."""
        return json.dumps({"messages": self.messages})

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": self.messages,
            "source_fact_id": self.source_fact_id,
            "source_fact_content": self.source_fact_content,
            "quality_score": self.quality_score,
            "category": self.category,
            "generated_at": self.generated_at.isoformat(),
        }


_CATEGORY_QUESTIONS: dict[str, list[str]] = {
    "work": [
        "What do you do for work?",
        "Where do you work?",
        "What is your job?",
        "Can you tell me about your current role?",
    ],
    "education": [
        "Where did you go to school?",
        "What did you study?",
        "What is your educational background?",
        "What degrees do you have?",
    ],
    "skills": [
        "What skills do you have?",
        "What are you good at?",
        "What technical skills do you have?",
        "What are your strongest skills?",
    ],
    "location": [
        "Where are you located?",
        "Where do you live?",
        "What city are you based in?",
        "Where are you from?",
    ],
    "personal": [
        "Tell me something about yourself.",
        "What are your interests outside of work?",
        "How would you describe yourself?",
    ],
    "interests": [
        "What are your hobbies?",
        "What do you enjoy doing in your free time?",
        "What are your interests?",
    ],
    "achievements": [
        "What are you most proud of?",
        "What have you accomplished?",
        "Tell me about a significant achievement.",
    ],
    "other": [
        "Tell me about yourself.",
        "What should I know about you?",
        "Share something interesting about yourself.",
    ],
}


class SFTDataGenerator:
    """
    Generates supervised fine-tuning Q&A data pairs from memory.facts.

    Uses an LLM (Gemini Pro) to create diverse question-answer pairs
    that teach the base model factual knowledge about the user.
    Falls back to template-based generation if LLM is unavailable.

    Args:
        llm_client: LLM client with .complete(prompt) -> str interface.
        user_name: The user's name, used for persona framing.
        quality_threshold: Minimum quality score (0-10) to keep a sample.
        samples_per_fact: Target Q&A pairs per fact when using LLM.
    """

    BATCH_SIZE = 8
    DEFAULT_SAMPLES_PER_FACT = 2

    def __init__(
        self,
        llm_client=None,
        user_name: str = "the user",
        quality_threshold: float = 6.0,
        samples_per_fact: int = DEFAULT_SAMPLES_PER_FACT,
    ) -> None:
        self._llm = llm_client
        self.user_name = user_name
        self.quality_threshold = quality_threshold
        self.samples_per_fact = samples_per_fact

    def generate(
        self,
        memory,
        target_count: int = 200,
        progress_callback=None,
    ) -> list[SFTSample]:
        """
        Generate SFT samples from memory.facts.

        Args:
            memory: Memory object with .facts list.
            target_count: Maximum samples to return.
            progress_callback: optional callable(current, total)

        Returns:
            Quality-filtered, deduplicated list of SFTSample objects.
        """
        facts = memory.facts
        if not facts:
            return []

        sorted_facts = sorted(facts, key=lambda f: f.confidence, reverse=True)
        samples: list[SFTSample] = []
        total = len(sorted_facts)

        for i in range(0, total, self.BATCH_SIZE):
            if len(samples) >= target_count:
                break
            batch = sorted_facts[i : i + self.BATCH_SIZE]
            if self._llm:
                batch_samples = self._batch_generate_llm(batch)
            else:
                batch_samples = self._batch_generate_template(batch)

            for s in batch_samples:
                if s.quality_score >= self.quality_threshold:
                    samples.append(s)

            if progress_callback:
                progress_callback(min(i + self.BATCH_SIZE, total), total)

        return _deduplicate_sft(samples, target_count)

    def _batch_generate_llm(self, facts) -> list[SFTSample]:
        """Generate Q&A pairs for a batch of facts using LLM."""
        facts_text = "\n".join(
            f"{idx + 1}. [{f.category}] {f.content}"
            for idx, f in enumerate(facts)
        )
        prompt = (
            f"You are generating training data for a personalized AI model that represents {self.user_name}.\n\n"
            f"For each fact below, generate {self.samples_per_fact} diverse question-answer pairs.\n"
            f"The AI answers AS {self.user_name} in first person, naturally and conversationally.\n"
            f"Vary the question phrasing — use different question types (what, where, when, who, how, tell me about).\n"
            f"Answers should sound like how a real person would respond, not a resume.\n\n"
            f"Facts about {self.user_name}:\n{facts_text}\n\n"
            f"Return a JSON array. Each object must have:\n"
            f'- "fact_index": integer (1-based, matching the fact above)\n'
            f'- "question": the user question\n'
            f'- "answer": {self.user_name}\'s first-person answer\n'
            f'- "quality": float 0-10 (10 = perfect training example)\n\n'
            f"JSON array only, no explanation:"
        )

        try:
            response = self._llm.complete(prompt)
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            pairs = json.loads(text)
        except Exception:
            return self._batch_generate_template(facts)

        samples = []
        for pair in pairs:
            try:
                idx = int(pair.get("fact_index", 1)) - 1
                fact = facts[idx] if 0 <= idx < len(facts) else facts[0]
                sample = SFTSample(
                    messages=[
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]},
                    ],
                    source_fact_id=fact.id,
                    source_fact_content=fact.content,
                    quality_score=float(pair.get("quality", 7.0)),
                    category=str(fact.category),
                )
                samples.append(sample)
            except (KeyError, ValueError, TypeError):
                continue
        return samples

    def _batch_generate_template(self, facts) -> list[SFTSample]:
        """Template-based fallback when LLM is unavailable."""
        samples = []
        for fact in facts:
            cat = str(fact.category).split(".")[-1].lower()
            questions = _CATEGORY_QUESTIONS.get(cat, _CATEGORY_QUESTIONS["other"])
            sample = SFTSample(
                messages=[
                    {"role": "user", "content": random.choice(questions)},
                    {"role": "assistant", "content": fact.content},
                ],
                source_fact_id=fact.id,
                source_fact_content=fact.content,
                quality_score=6.0,
                category=str(fact.category),
            )
            samples.append(sample)
        return samples

    def save_jsonl(self, samples: list[SFTSample], output_path: str | Path) -> Path:
        """Save samples to JSONL file (Mistral fine-tuning format)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in samples:
                f.write(s.to_jsonl_line() + "\n")
        return path

    def save_metadata(self, samples: list[SFTSample], output_path: str | Path) -> Path:
        """Save full metadata as JSON for inspection."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in samples], f, indent=2)
        return path


def _deduplicate_sft(samples: list[SFTSample], max_count: int) -> list[SFTSample]:
    """Remove samples with identical questions (exact match)."""
    seen: set[str] = set()
    unique = []
    for s in samples:
        if s.messages:
            key = s.messages[0]["content"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        if len(unique) >= max_count:
            break
    return unique
