"""
Stylistic evaluator — tests whether the model writes like the user.

Uses a RULER-style LLM judge: Gemini ranks/scores the model's response
against the user's reference writing sample and style markers.

Scoring: 0.0-1.0 per prompt, averaged for stylistic_accuracy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from openclawmini.eval.eval_set import EvalSet, StylisticPrompt


@dataclass
class StylisticResult:
    prompt_id: str
    prompt: str
    response: str
    score: float           # 0-1
    reasoning: str = ""
    reference_sample: str = ""
    style_markers: list[str] = field(default_factory=list)


@dataclass
class StylisticEvalSummary:
    results: list[StylisticResult]
    accuracy: float        # mean score across all prompts
    total_count: int


class StylisticEvaluator:
    """
    RULER-style stylistic evaluator.

    For each stylistic prompt:
      1. Get model response via model_fn
      2. Ask Gemini judge to score 0-1: how well does it match user's style?
      3. If no LLM judge, fall back to heuristic scoring

    Args:
        llm_judge: GeminiExtractor (or any object with .complete(str)->str)
    """

    def __init__(self, llm_judge=None) -> None:
        self._judge = llm_judge

    def run(
        self,
        eval_set: EvalSet,
        model_fn: Callable[[str], str],
        progress_callback=None,
    ) -> StylisticEvalSummary:
        """
        Run stylistic eval.

        Args:
            eval_set: The persistent eval set.
            model_fn: callable(prompt: str) -> str
            progress_callback: optional callable(current, total)

        Returns:
            StylisticEvalSummary with per-prompt scores + mean accuracy.
        """
        prompts = eval_set.stylistic_prompts
        results: list[StylisticResult] = []

        for i, p in enumerate(prompts):
            if progress_callback:
                progress_callback(i + 1, len(prompts))

            try:
                response = model_fn(p.prompt)
            except Exception as e:
                response = f"[ERROR: {e}]"

            if self._judge:
                score, reasoning = self._llm_score(response, p)
            else:
                score, reasoning = _heuristic_score(response, p)

            results.append(StylisticResult(
                prompt_id=p.id,
                prompt=p.prompt,
                response=response,
                score=score,
                reasoning=reasoning,
                reference_sample=p.reference_sample,
                style_markers=p.style_markers,
            ))

        accuracy = sum(r.score for r in results) / max(len(results), 1)

        return StylisticEvalSummary(
            results=results,
            accuracy=round(accuracy, 4),
            total_count=len(results),
        )

    def _llm_score(self, response: str, prompt: StylisticPrompt) -> tuple[float, str]:
        """Use Gemini to score how well the response matches the user's style."""
        ref_section = (
            f"Reference sample (how this person actually writes):\n{prompt.reference_sample[:400]}\n\n"
            if prompt.reference_sample
            else ""
        )
        behavior_section = (
            f"Expected behavior: {prompt.expected_behavior}\n\n"
            if prompt.expected_behavior
            else ""
        )
        markers_str = ", ".join(prompt.style_markers) if prompt.style_markers else "natural writing style"

        judge_prompt = f"""You are evaluating how well an AI response matches a specific person's writing style and preferences.

{ref_section}{behavior_section}Style markers to look for: {markers_str}

AI Response to evaluate:
{response[:600]}

Score this response 0.0-1.0 on how well it matches this person's style:
- 1.0: Nearly identical (tone, length, vocabulary, structure all match)
- 0.7: Similar style with minor differences
- 0.5: Somewhat similar, some markers present
- 0.3: Different style, few markers
- 0.0: Completely different

Return JSON: {{"score": 0.0-1.0, "reasoning": "one sentence"}}
JSON only:"""
        try:
            raw = self._judge.complete(judge_prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                raise ValueError("no JSON")
            data = json.loads(match.group())
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
        except Exception:
            return _heuristic_score(response, prompt)


# ── Heuristic fallback ────────────────────────────────────────


def _heuristic_score(response: str, prompt: StylisticPrompt) -> tuple[float, str]:
    """
    Simple heuristic scoring when no LLM judge is available.
    Checks for style markers in the response text.
    """
    if not response or response.startswith("[ERROR"):
        return 0.0, "Error generating response"

    response_lower = response.lower()
    markers = [m.lower() for m in prompt.style_markers]
    if not markers:
        return 0.5, "No style markers to compare"

    # Count how many style markers are hinted at in the response
    hits = 0
    for marker in markers:
        # Check for keywords from the marker phrase
        marker_words = [w for w in marker.split() if len(w) > 3]
        if any(w in response_lower for w in marker_words):
            hits += 1

    # Length similarity (if we have a reference)
    length_bonus = 0.0
    if prompt.reference_sample:
        ref_words = len(prompt.reference_sample.split())
        resp_words = len(response.split())
        ratio = min(ref_words, resp_words) / max(ref_words, resp_words, 1)
        length_bonus = ratio * 0.2

    base_score = hits / max(len(markers), 1)
    score = min(1.0, base_score * 0.8 + length_bonus)
    reasoning = f"Matched {hits}/{len(markers)} style markers"
    return round(score, 3), reasoning
