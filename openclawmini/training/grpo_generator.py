"""
GRPO Data Generator — creates style-alignment scenarios from writing samples.

GRPO (Group Relative Policy Optimization) trains the model to match a user's
writing style. Each scenario has a prompt and a reference response (from actual
user writing) plus style metadata for the RULER judge.

Output format: JSONL where each line is a scenario dict.
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
    return f"grpo_{uuid.uuid4().hex[:8]}"


@dataclass
class GRPOScenario:
    id: str = field(default_factory=_new_id)
    prompt: str = ""
    reference_response: str = ""
    style_markers: list[str] = field(default_factory=list)
    scenario_type: str = "general"  # email, message, rewrite, social, casual, general
    source_sample_id: str = ""
    quality_score: float = 7.0
    generated_at: datetime = field(default_factory=_now)

    def to_jsonl_line(self) -> str:
        return json.dumps({
            "prompt": self.prompt,
            "reference_response": self.reference_response,
            "style_markers": self.style_markers,
            "scenario_type": self.scenario_type,
        })

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "reference_response": self.reference_response,
            "style_markers": self.style_markers,
            "scenario_type": self.scenario_type,
            "source_sample_id": self.source_sample_id,
            "quality_score": self.quality_score,
            "generated_at": self.generated_at.isoformat(),
        }


_SCENARIO_TEMPLATES: dict[str, list[str]] = {
    "professional_email": [
        "Write a professional email responding to a colleague about a project update.",
        "Draft a brief professional email to schedule a meeting.",
        "Write an email to a client about a deliverable status.",
        "Compose a follow-up email after a meeting.",
    ],
    "casual_email": [
        "Write a casual email to a friend about catching up.",
        "Send a quick email to a colleague about grabbing lunch.",
        "Write a friendly email checking in with someone you haven't spoken to in a while.",
    ],
    "linkedin_message": [
        "Write a LinkedIn message to connect with someone in your industry.",
        "Compose a LinkedIn message following up on a conversation at a conference.",
        "Write a brief LinkedIn message reaching out about a collaboration opportunity.",
    ],
    "slack_message": [
        "Write a Slack message to your team about a project update.",
        "Compose a quick Slack message asking for help with a blocker.",
        "Write a Slack message sharing good news with your team.",
    ],
    "general": [
        "Write a short message introducing yourself to a new colleague.",
        "Compose a brief response to someone asking about your experience.",
        "Write a message giving an update on something you're working on.",
    ],
}

_REWRITE_TEMPLATES = [
    "Rewrite the following in your own voice:\n\n{text}",
    "How would you say this in your own words?\n\n{text}",
    "Rephrase this message as you would naturally write it:\n\n{text}",
    "Express this idea the way you would naturally say it:\n\n{text}",
]


class GRPODataGenerator:
    """
    Generates GRPO style-alignment scenarios from memory writing samples.

    Creates (prompt, reference_response) pairs where the RULER judge scores
    model output against the user's actual writing.

    Distribution of generated scenarios:
      - 60%: from writing_samples (email, message, etc.)
      - 20%: from posts (social content)
      - 20%: rewrite prompts (style transfer from actual samples)

    Args:
        llm_client: LLM client with .complete(prompt) -> str interface.
        user_name: The user's name for persona framing.
        quality_threshold: Min quality score (0-10) to keep a scenario.
        min_sample_length: Skip writing samples shorter than this (chars).
    """

    BATCH_SIZE = 6
    MIN_SAMPLE_LENGTH = 50

    def __init__(
        self,
        llm_client=None,
        user_name: str = "the user",
        quality_threshold: float = 6.0,
        min_sample_length: int = MIN_SAMPLE_LENGTH,
    ) -> None:
        self._llm = llm_client
        self.user_name = user_name
        self.quality_threshold = quality_threshold
        self.min_sample_length = min_sample_length

    def generate(
        self,
        memory,
        target_count: int = 100,
        progress_callback=None,
    ) -> list[GRPOScenario]:
        """
        Generate GRPO scenarios from writing samples, posts, and preferences.

        Args:
            memory: Memory object.
            target_count: Maximum scenarios to return.
            progress_callback: optional callable(current, total)

        Returns:
            List of GRPOScenario objects.
        """
        usable_samples = [
            s for s in memory.writing_samples
            if len(s.text) >= self.min_sample_length
        ]
        usable_posts = [
            p for p in memory.posts
            if len(p.text) >= self.min_sample_length
        ]

        sample_target = int(target_count * 0.6)
        post_target = int(target_count * 0.2)
        rewrite_target = target_count - sample_target - post_target

        scenarios: list[GRPOScenario] = []
        total_sources = len(usable_samples) + len(usable_posts)
        processed = 0

        # 1. From writing samples (~60%)
        for i in range(0, len(usable_samples), self.BATCH_SIZE):
            if len(scenarios) >= sample_target:
                break
            batch = usable_samples[i : i + self.BATCH_SIZE]
            if self._llm:
                batch_scenarios = self._from_samples_llm(batch)
            else:
                batch_scenarios = self._from_samples_template(batch)
            for sc in batch_scenarios:
                if sc.quality_score >= self.quality_threshold:
                    scenarios.append(sc)
            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total_sources)

        # 2. From posts (~20%)
        for i in range(0, len(usable_posts), self.BATCH_SIZE):
            post_count = sum(1 for s in scenarios if s.scenario_type == "social")
            if post_count >= post_target:
                break
            batch = usable_posts[i : i + self.BATCH_SIZE]
            if self._llm:
                batch_scenarios = self._from_posts_llm(batch)
            else:
                batch_scenarios = self._from_posts_template(batch)
            for sc in batch_scenarios:
                if sc.quality_score >= self.quality_threshold:
                    scenarios.append(sc)
            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total_sources)

        # 3. Rewrite prompts (~20%)
        if usable_samples and len(scenarios) < target_count:
            rewrites = self._rewrite_scenarios(
                usable_samples, rewrite_target
            )
            scenarios.extend(rewrites)

        return scenarios[:target_count]

    def _from_samples_llm(self, samples) -> list[GRPOScenario]:
        """Use LLM to generate a prompt that would naturally elicit each sample."""
        samples_text = "\n\n---\n\n".join(
            f"Sample {idx + 1} (type: {s.category}, context: {s.context or 'n/a'}):\n{s.text[:500]}"
            for idx, s in enumerate(samples)
        )
        prompt = (
            f"You are creating training scenarios for a personalized AI model that writes like {self.user_name}.\n\n"
            f"Below are actual writing samples from {self.user_name}.\n"
            f"For each sample, create a realistic writing prompt that would naturally lead to that response.\n"
            f"The prompt should be the kind of request/message that would result in someone writing that reply.\n\n"
            f"Writing samples:\n{samples_text}\n\n"
            f"Return a JSON array. Each object must have:\n"
            f'- "sample_index": integer (1-based)\n'
            f'- "prompt": the writing task or message that would elicit this response\n'
            f'- "style_markers": list of 2-4 strings describing the writing style '
            f'(e.g. "professional tone", "uses bullet points", "brief sentences")\n'
            f'- "scenario_type": one of: email, message, rewrite, social, casual, general\n'
            f'- "quality": float 0-10\n\n'
            f"JSON array only:"
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
            return self._from_samples_template(samples)

        scenarios = []
        for pair in pairs:
            try:
                idx = int(pair.get("sample_index", 1)) - 1
                sample = samples[idx] if 0 <= idx < len(samples) else samples[0]
                sc = GRPOScenario(
                    prompt=pair["prompt"],
                    reference_response=sample.text,
                    style_markers=pair.get("style_markers", []),
                    scenario_type=pair.get("scenario_type", "general"),
                    source_sample_id=sample.id,
                    quality_score=float(pair.get("quality", 7.0)),
                )
                scenarios.append(sc)
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        return scenarios

    def _from_samples_template(self, samples) -> list[GRPOScenario]:
        """Template-based fallback for writing samples."""
        scenarios = []
        for sample in samples:
            cat = str(sample.category).split(".")[-1].lower()
            templates = _SCENARIO_TEMPLATES.get(cat, _SCENARIO_TEMPLATES["general"])
            sc = GRPOScenario(
                prompt=random.choice(templates),
                reference_response=sample.text,
                style_markers=[],
                scenario_type=cat,
                source_sample_id=sample.id,
                quality_score=6.0,
            )
            scenarios.append(sc)
        return scenarios

    def _from_posts_llm(self, posts) -> list[GRPOScenario]:
        """Generate social media scenarios from posts using LLM."""
        posts_text = "\n\n---\n\n".join(
            f"Post {idx + 1} (platform: {p.platform}):\n{p.text[:400]}"
            for idx, p in enumerate(posts)
        )
        prompt = (
            f"You are creating training scenarios for {self.user_name}'s personalized AI model.\n\n"
            f"Below are social media posts by {self.user_name}.\n"
            f"For each post, create a prompt/request that would naturally produce that post.\n\n"
            f"Posts:\n{posts_text}\n\n"
            f"Return a JSON array. Each object must have:\n"
            f'- "post_index": integer (1-based)\n'
            f'- "prompt": the writing task (e.g. "Write a LinkedIn post about a professional accomplishment")\n'
            f'- "style_markers": list of 2-4 style descriptors\n'
            f'- "quality": float 0-10\n\n'
            f"JSON array only:"
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
            return self._from_posts_template(posts)

        scenarios = []
        for pair in pairs:
            try:
                idx = int(pair.get("post_index", 1)) - 1
                post = posts[idx] if 0 <= idx < len(posts) else posts[0]
                sc = GRPOScenario(
                    prompt=pair["prompt"],
                    reference_response=post.text,
                    style_markers=pair.get("style_markers", []),
                    scenario_type="social",
                    source_sample_id=post.id,
                    quality_score=float(pair.get("quality", 7.0)),
                )
                scenarios.append(sc)
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        return scenarios

    def _from_posts_template(self, posts) -> list[GRPOScenario]:
        """Template-based fallback for posts."""
        templates = [
            "Write a LinkedIn post about a recent professional accomplishment.",
            "Write a social media post sharing a professional insight.",
            "Create a post sharing something you learned recently.",
        ]
        scenarios = []
        for post in posts:
            sc = GRPOScenario(
                prompt=random.choice(templates),
                reference_response=post.text,
                style_markers=[],
                scenario_type="social",
                source_sample_id=post.id,
                quality_score=6.0,
            )
            scenarios.append(sc)
        return scenarios

    def _rewrite_scenarios(self, samples, target_count: int) -> list[GRPOScenario]:
        """Create 'rewrite in your style' scenarios from actual writing samples."""
        scenarios = []
        for sample in samples[:target_count]:
            if len(sample.text) < self.min_sample_length:
                continue
            template = random.choice(_REWRITE_TEMPLATES)
            snippet = sample.text[:200].strip()
            sc = GRPOScenario(
                prompt=template.format(text=snippet),
                reference_response=sample.text,
                style_markers=[],
                scenario_type="rewrite",
                source_sample_id=sample.id,
                quality_score=7.0,
            )
            scenarios.append(sc)
        return scenarios

    def save_jsonl(
        self, scenarios: list[GRPOScenario], output_path: str | Path
    ) -> Path:
        """Save scenarios to JSONL file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for sc in scenarios:
                f.write(sc.to_jsonl_line() + "\n")
        return path

    def save_metadata(
        self, scenarios: list[GRPOScenario], output_path: str | Path
    ) -> Path:
        """Save full metadata as JSON for inspection."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([sc.to_dict() for sc in scenarios], f, indent=2)
        return path
