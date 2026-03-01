"""
RULER — LLM-as-Judge for stylistic alignment scoring.

Scores a model's response against the user's actual writing style
using Gemini Flash as the judge. Returns a reward in [0.0, 1.0].

Used by the GRPO Agent as the online reward function during training.

Trace logging:
  Every score() call is decorated with @weave.op() so W&B Weave captures
  the full input/output trace (prompt sent, score returned, latency).
  Traces appear in W&B under the "Traces" tab of the openclawmini project.
  Requires WANDB_API_KEY — silently skips Weave init if not set.
"""

from __future__ import annotations

import os
import re
from typing import Optional


def _maybe_init_weave(project: str = "openclawmini") -> bool:
    """Initialize W&B Weave for trace logging. Returns True if successful."""
    if not os.getenv("WANDB_API_KEY", ""):
        return False
    try:
        import weave  # type: ignore[import]
        weave.init(project)
        return True
    except Exception:
        return False


# ── Writing style snapshot (compiled from memory at training start) ──────────


class WritingStyleProfile:
    """
    Compact snapshot of the user's writing style derived from memory.
    Passed to RULER.score() to guide the LLM judge.
    """

    def __init__(
        self,
        user_name: str,
        sample_texts: list[str],          # Actual user writing (up to 5)
        style_markers: list[str],         # e.g. ["concise", "casual greeting", "ends with ?"]
        preferences: list[str],           # e.g. ["direct communication", "no fluff"]
        avg_length_words: int = 80,
    ) -> None:
        self.user_name = user_name
        self.sample_texts = sample_texts[:5]
        self.style_markers = style_markers[:10]
        self.preferences = preferences[:5]
        self.avg_length_words = avg_length_words

    @classmethod
    def from_memory(cls, memory, user_name: str = "") -> "WritingStyleProfile":
        """Build a style profile from a Memory object."""
        name = user_name or getattr(memory.user, "name", "the user")

        samples = [s.text for s in memory.writing_samples[:5] if len(s.text) >= 40]
        posts = [p.text for p in memory.posts[:3] if len(p.text) >= 40]
        all_samples = samples + posts

        # Derive rough style markers from samples
        markers: list[str] = []
        all_text = " ".join(all_samples).lower()
        if all_text:
            avg_len = sum(len(t.split()) for t in all_samples) // max(len(all_samples), 1)
            if avg_len < 60:
                markers.append("concise responses")
            elif avg_len > 150:
                markers.append("detailed responses")
            if any(w in all_text for w in ["hey", "yo", "haha", "lol"]):
                markers.append("casual/friendly tone")
            if any(w in all_text for w in ["regards", "sincerely", "best,"]):
                markers.append("professional closing")
            if "?" in " ".join(all_samples):
                markers.append("often ends with a question")
            if "i " in all_text:
                markers.append("first-person voice")
        if not markers:
            markers = ["natural conversational tone", "first-person voice"]

        prefs = [p.content for p in memory.preferences[:5]]

        return cls(
            user_name=name,
            sample_texts=all_samples,
            style_markers=markers,
            preferences=prefs,
            avg_length_words=avg_len if all_samples else 80,
        )

    def to_rubric(self) -> str:
        """Format the style profile as a RULER scoring rubric."""
        lines = [
            f"You are scoring how well an AI response matches {self.user_name}'s writing style.",
            "",
            "## Reference Writing Samples (actual writing by this person):",
        ]
        for i, text in enumerate(self.sample_texts, 1):
            lines.append(f"Sample {i}: {text[:300]}")
        lines.append("")

        lines.append("## Style Markers to Match:")
        for marker in self.style_markers:
            lines.append(f"  - {marker}")
        lines.append("")

        if self.preferences:
            lines.append("## Communication Preferences:")
            for pref in self.preferences:
                lines.append(f"  - {pref}")
            lines.append("")

        lines.append(
            f"Typical response length: ~{self.avg_length_words} words "
            f"({'short' if self.avg_length_words < 60 else 'medium' if self.avg_length_words < 150 else 'long'})"
        )
        return "\n".join(lines)


# ── RULER ────────────────────────────────────────────────────────────────────


class RULER:
    """
    LLM-as-judge that scores stylistic alignment of model outputs.

    score(response, profile) → float in [0.0, 1.0]

    Design:
      - Single-call scoring (1 Gemini call per response)
      - Structured prompt forces the model to reason then output a score
      - Falls back to 0.5 on any error (neutral — does not punish or reward)

    Args:
        api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        model: Gemini model ID. Should be Flash (fast + cheap for reward calls).
        temperature: Low temperature for consistent scoring.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        weave_project: str = "openclawmini",
    ) -> None:
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._model = model
        self._temperature = temperature
        self._client = None
        # Init Weave trace logging, then wrap score() with @weave.op()
        if _maybe_init_weave(weave_project):
            self._wrap_with_weave()

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai  # type: ignore[import]
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(
                self._model,
                generation_config={"temperature": self._temperature, "max_output_tokens": 200},
            )
        return self._client

    def score(self, response: str, profile: WritingStyleProfile) -> float:
        """
        Score a single model response against the user's style profile.

        Decorated with @weave.op() when Weave is available — every call is
        traced in W&B (input prompt, raw judge output, final score, latency).

        Returns a float in [0.0, 1.0]:
          1.0 = perfectly matches user's style
          0.5 = neutral (fallback on error)
          0.0 = completely misaligned
        """
        return self._score_impl(response, profile)

    def _score_impl(self, response: str, profile: WritingStyleProfile) -> float:
        """Core scoring logic, called directly or via Weave-wrapped score()."""
        if not response or not response.strip():
            return 0.0

        prompt = self._build_prompt(response, profile)
        try:
            client = self._get_client()
            result = client.generate_content(prompt)
            return self._parse_score(result.text)
        except Exception:
            return 0.5  # neutral fallback

    def _wrap_with_weave(self) -> None:
        """Replace score() with a @weave.op()-wrapped version for trace logging."""
        try:
            import weave  # type: ignore[import]
            self.score = weave.op()(self._score_impl)
        except Exception:
            pass  # Weave unavailable — use plain score()

    def score_batch(
        self,
        responses: list[str],
        profile: WritingStyleProfile,
    ) -> list[float]:
        """Score a batch of responses. Returns list of [0, 1] scores."""
        return [self.score(r, profile) for r in responses]

    def _build_prompt(self, response: str, profile: WritingStyleProfile) -> str:
        rubric = profile.to_rubric()
        return f"""{rubric}

## Response to Score:
\"\"\"{response[:600]}\"\"\"

## Scoring Instructions:
Rate how well this response matches {profile.user_name}'s writing style on a scale of 0 to 10.

Criteria:
  - Tone and formality match (0-3 pts)
  - Sentence length and rhythm match (0-2 pts)
  - Vocabulary and phrasing feel authentic (0-2 pts)
  - Response length is appropriate (0-2 pts)
  - Would this be mistaken for something {profile.user_name} actually wrote? (0-1 pt)

Think briefly, then output ONLY a number like:
SCORE: 7.5
"""

    def _parse_score(self, text: str) -> float:
        """Extract numeric score from RULER output."""
        match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text or "")
        if match:
            raw = float(match.group(1))
            return max(0.0, min(1.0, raw / 10.0))
        # Fallback: extract any number 0-10 from end of response
        nums = re.findall(r"\b([0-9](?:\.[0-9]+)?|10(?:\.0+)?)\b", text or "")
        if nums:
            return max(0.0, min(1.0, float(nums[-1]) / 10.0))
        return 0.5

    @classmethod
    def from_env(cls) -> "RULER":
        return cls(api_key=os.getenv("GOOGLE_API_KEY"))
