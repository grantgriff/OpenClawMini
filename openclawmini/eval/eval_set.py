"""
Persistent evaluation set — generated ONCE from memory, reused across
BASE → SFT → GRPO for fair apples-to-apples comparison.

Two dimensions:
  1. FactualQuestion  — binary scoring, tests if model knows user facts
  2. StylisticPrompt  — 0-1 LLM scoring, tests if model writes like user
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


# ── Data models ────────────────────────────────────────────────


class FactualQuestion(BaseModel):
    """Tests whether the model knows a specific fact about the user."""
    id: str = Field(default_factory=lambda: _new_id("fq"))
    question: str
    expected_keywords: list[str]   # Any keyword present in response → correct
    category: str                  # FactCategory value
    source_fact_id: str
    source_fact_content: str       # Original fact kept for LLM judge reference


class StylisticPrompt(BaseModel):
    """Tests whether the model writes like the user (scored by LLM judge)."""
    id: str = Field(default_factory=lambda: _new_id("sp"))
    prompt: str
    reference_sample: str          # What the user actually wrote
    style_markers: list[str]       # Key stylistic elements to match
    category: str                  # WritingCategory or "preference"
    expected_behavior: str = ""    # For preference-based prompts


class EvalSet(BaseModel):
    """
    Persistent eval set saved to data/evals/eval_set.json.
    Generated once after initial research; same set used for all stages.
    """
    id: str = Field(default_factory=lambda: _new_id("evalset"))
    generated_at: datetime = Field(default_factory=_now)
    factual_questions: list[FactualQuestion] = Field(default_factory=list)
    stylistic_prompts: list[StylisticPrompt] = Field(default_factory=list)
    # Memory snapshot at generation time (for reference)
    memory_facts_count: int = 0
    memory_samples_count: int = 0
    memory_prefs_count: int = 0

    def stats(self) -> dict:
        return {
            "factual_questions": len(self.factual_questions),
            "stylistic_prompts": len(self.stylistic_prompts),
            "total": len(self.factual_questions) + len(self.stylistic_prompts),
        }

    def is_empty(self) -> bool:
        return len(self.factual_questions) == 0 and len(self.stylistic_prompts) == 0


# ── Storage ────────────────────────────────────────────────────


class EvalSetStore:
    """Persist the eval set to/from disk."""

    DEFAULT_PATH = "./data/evals/eval_set.json"

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def save(self, eval_set: EvalSet) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(eval_set.model_dump(mode="json"), f, indent=2, default=str)

    def load(self) -> Optional[EvalSet]:
        if not self.path.exists():
            return None
        with open(self.path) as f:
            return EvalSet.model_validate(json.load(f))

    def delete(self) -> None:
        """Remove the saved eval set so it will be regenerated on next run."""
        self.path.unlink(missing_ok=True)


# ── Generator ─────────────────────────────────────────────────


class EvalSetGenerator:
    """
    Generates factual questions and stylistic prompts from memory.

    Uses Gemini (via GeminiExtractor) for high-quality generation;
    falls back to templates if no LLM is available.

    Guarantees a minimum of MIN_TOTAL_QUESTIONS total questions across
    both dimensions (pads with template questions if memory is sparse).
    """

    MAX_FACTUAL = 50
    MAX_STYLISTIC = 30
    MIN_TOTAL = 50          # hard floor — always generate at least this many

    def __init__(self, gemini_extractor=None) -> None:
        self._llm = gemini_extractor

    def generate(self, memory) -> EvalSet:
        """Generate a complete eval set from a Memory object."""
        # Pick highest-confidence facts
        facts = sorted(memory.facts, key=lambda f: f.confidence, reverse=True)
        facts = facts[: self.MAX_FACTUAL]

        samples = memory.writing_samples[: self.MAX_STYLISTIC // 2]
        posts = memory.posts[:5]
        prefs = memory.preferences[: self.MAX_STYLISTIC // 4]

        factual_qs = self._build_factual_questions(facts)
        stylistic_ps = self._build_stylistic_prompts(samples, posts, prefs)

        # ── Enforce minimum total ──────────────────────────────
        factual_qs, stylistic_ps = self._enforce_minimum(
            factual_qs, stylistic_ps, facts, samples
        )

        return EvalSet(
            factual_questions=factual_qs,
            stylistic_prompts=stylistic_ps,
            memory_facts_count=len(memory.facts),
            memory_samples_count=len(memory.writing_samples),
            memory_prefs_count=len(memory.preferences),
        )

    def _enforce_minimum(
        self,
        factual_qs: list[FactualQuestion],
        stylistic_ps: list[StylisticPrompt],
        facts,
        samples,
    ) -> tuple[list[FactualQuestion], list[StylisticPrompt]]:
        """
        Pad the eval set up to MIN_TOTAL questions if the initial generation
        produced fewer (e.g. sparse memory with only a handful of facts).

        Strategy:
          1. Generate extra factual questions from already-used facts with
             different phrasings (template-based, so always available).
          2. If still short and we have samples, add more stylistic prompts.
        """
        total = len(factual_qs) + len(stylistic_ps)
        if total >= self.MIN_TOTAL:
            return factual_qs, stylistic_ps

        deficit = self.MIN_TOTAL - total

        # Extra factual questions via alternative phrasings
        if facts:
            extra_factual = self._extra_factual_questions(facts, deficit)
            factual_qs = factual_qs + extra_factual
            deficit = max(0, self.MIN_TOTAL - len(factual_qs) - len(stylistic_ps))

        # Still short? Pad with stylistic templates
        if deficit > 0 and samples:
            extra_stylistic = self._stylistic_template(
                samples[:deficit], [], []
            )
            # Avoid duplicating already-added prompts
            existing_prompts = {sp.prompt for sp in stylistic_ps}
            extra_stylistic = [
                sp for sp in extra_stylistic if sp.prompt not in existing_prompts
            ][:deficit]
            stylistic_ps = stylistic_ps + extra_stylistic

        return factual_qs, stylistic_ps

    def _extra_factual_questions(self, facts, count: int) -> list[FactualQuestion]:
        """Generate additional factual questions with alternative phrasings."""
        _ALT_PHRASING = [
            ("Can you tell me about your {category}?",   ["about", "background"]),
            ("How would you describe your {category}?",  ["describe", "experience"]),
            ("What can you share about your {category}?",["share", "about"]),
            ("Give me an overview of your {category}.",  ["overview", "experience"]),
        ]
        extra = []
        for i, fact in enumerate(facts * 4):  # cycle through facts
            if len(extra) >= count:
                break
            template, base_kw = _ALT_PHRASING[i % len(_ALT_PHRASING)]
            cat = str(fact.category).split(".")[-1].lower().replace("_", " ")
            q_text = template.format(category=cat)
            content_kw = [w.lower() for w in fact.content.split() if len(w) > 3][:4]
            extra.append(FactualQuestion(
                question=q_text,
                expected_keywords=list(set(base_kw + content_kw)),
                category=str(fact.category),
                source_fact_id=fact.id,
                source_fact_content=fact.content,
            ))
        return extra

    # ── Factual question generation ────────────────────────────

    def _build_factual_questions(self, facts) -> list[FactualQuestion]:
        if self._llm and facts:
            return self._factual_llm(facts)
        return self._factual_template(facts)

    def _factual_llm(self, facts) -> list[FactualQuestion]:
        questions: list[FactualQuestion] = []
        batch_size = 10
        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            fact_lines = "\n".join(
                f'{j+1}. [ID:{f.id}] ({f.category}) {f.content}'
                for j, f in enumerate(batch)
            )
            prompt = f"""Given these facts about a person, generate one natural question per fact that tests if an AI assistant knows that specific fact about the person.

Facts:
{fact_lines}

For each fact create:
- A short, natural question (e.g. "Where do you work?", "What did you study?")
- 3-5 keywords that should appear in a correct answer

Return a JSON array:
[
  {{"fact_id": "...", "question": "...", "expected_keywords": ["word1", "word2"]}},
  ...
]
JSON only, no markdown:"""
            try:
                raw = self._llm.complete(prompt)
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if not match:
                    raise ValueError("no JSON array")
                data = json.loads(match.group())
                fact_map = {f.id: f for f in batch}
                for item in data:
                    fact = fact_map.get(item.get("fact_id", ""))
                    if not fact or not item.get("question"):
                        continue
                    questions.append(FactualQuestion(
                        question=item["question"],
                        expected_keywords=[str(k).lower() for k in item.get("expected_keywords", [])],
                        category=str(fact.category),
                        source_fact_id=fact.id,
                        source_fact_content=fact.content,
                    ))
            except Exception:
                questions.extend(self._factual_template(batch))
        return questions

    def _factual_template(self, facts) -> list[FactualQuestion]:
        _TMPL = {
            "work":         ("What is your current job or role?",       ["work", "job", "company", "role", "position"]),
            "education":    ("Where did you study?",                    ["university", "college", "school", "degree", "studied"]),
            "skills":       ("What are your main skills?",              ["skill", "proficient", "expert", "experience"]),
            "location":     ("Where are you based?",                    ["city", "country", "located", "based", "live"]),
            "personal":     ("Can you tell me about your background?",  ["background", "grew", "family", "born", "raised"]),
            "interests":    ("What are your interests or hobbies?",     ["hobby", "interest", "enjoy", "passion", "like"]),
            "achievements": ("What are your notable achievements?",     ["achievement", "award", "built", "launched"]),
            "other":        ("Tell me about yourself.",                 ["about", "background", "experience"]),
        }
        questions = []
        for fact in facts:
            category = str(fact.category).lower()
            q_text, base_kw = _TMPL.get(category, _TMPL["other"])
            # Add keywords from the fact content itself
            extra = [w.lower() for w in fact.content.split() if len(w) > 3][:5]
            questions.append(FactualQuestion(
                question=q_text,
                expected_keywords=list(set(base_kw + extra)),
                category=category,
                source_fact_id=fact.id,
                source_fact_content=fact.content,
            ))
        return questions

    # ── Stylistic prompt generation ────────────────────────────

    def _build_stylistic_prompts(self, samples, posts, prefs) -> list[StylisticPrompt]:
        if self._llm:
            return self._stylistic_llm(samples, posts, prefs)
        return self._stylistic_template(samples, posts, prefs)

    def _stylistic_llm(self, samples, posts, prefs) -> list[StylisticPrompt]:
        prompts: list[StylisticPrompt] = []

        # Writing samples → prompts
        for sample in samples:
            p = f"""Given this writing sample (first 300 chars):
Context: {sample.context}
Text: {sample.text[:300]}

Generate:
1. A writing prompt that would elicit a similar response from this person
2. 3-5 style markers visible in the sample (e.g. "concise", "uses Hey greeting", "ends with question")

Return JSON: {{"prompt": "...", "style_markers": ["m1", "m2", "m3"]}}
JSON only:"""
            try:
                raw = self._llm.complete(p)
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not match:
                    raise ValueError("no JSON")
                data = json.loads(match.group())
                if data.get("prompt"):
                    prompts.append(StylisticPrompt(
                        prompt=data["prompt"],
                        reference_sample=sample.text[:600],
                        style_markers=data.get("style_markers", []),
                        category=str(sample.category),
                    ))
            except Exception:
                prompts.extend(self._stylistic_template([sample], [], []))

        # Posts → prompts
        for post in posts:
            p = f"""Given this social media post:
{post.text[:300]}

Generate a writing prompt that would elicit a similar post, and 3 style markers.
Return JSON: {{"prompt": "...", "style_markers": ["m1", "m2", "m3"]}}
JSON only:"""
            try:
                raw = self._llm.complete(p)
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not match:
                    raise ValueError("no JSON")
                data = json.loads(match.group())
                if data.get("prompt"):
                    prompts.append(StylisticPrompt(
                        prompt=data["prompt"],
                        reference_sample=post.text[:600],
                        style_markers=data.get("style_markers", []),
                        category="linkedin_post",
                    ))
            except Exception:
                pass

        # Preferences → scenario prompts
        for pref in prefs:
            p = f"""Given this preference about a person:
"{pref.content}"

Create a scenario/question that would reveal whether an AI assistant reflects this preference.
Return JSON: {{"prompt": "...", "style_markers": ["expected behavior 1", "expected behavior 2"]}}
JSON only:"""
            try:
                raw = self._llm.complete(p)
                raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if not match:
                    raise ValueError("no JSON")
                data = json.loads(match.group())
                if data.get("prompt"):
                    prompts.append(StylisticPrompt(
                        prompt=data["prompt"],
                        reference_sample="",
                        style_markers=data.get("style_markers", []),
                        category="preference",
                        expected_behavior=pref.content,
                    ))
            except Exception:
                pass

        return prompts

    def _stylistic_template(self, samples, posts, prefs) -> list[StylisticPrompt]:
        _TMPL = {
            "professional_email": "Write a professional email to a colleague about an upcoming project update.",
            "casual_email":       "Write a casual email to a friend making plans for the weekend.",
            "linkedin_message":   "Write a LinkedIn message to a new professional contact you met at a conference.",
            "slack_message":      "Write a quick Slack message to your team about today's priorities.",
            "other":              "Write a message to someone you know about something you've been working on.",
        }
        prompts = []
        for sample in samples:
            category = str(sample.category).lower()
            text_lower = sample.text.lower()
            markers = []
            if len(sample.text.split()) < 60:
                markers.append("concise")
            if any(w in text_lower for w in ["hey", "hi", "yo"]):
                markers.append("casual greeting")
            if any(w in text_lower for w in ["regards", "sincerely", "best,"]):
                markers.append("formal closing")
            if "?" in sample.text:
                markers.append("includes a question")
            prompts.append(StylisticPrompt(
                prompt=_TMPL.get(category, _TMPL["other"]),
                reference_sample=sample.text[:600],
                style_markers=markers or ["natural", "conversational"],
                category=category,
            ))
        return prompts
