"""
Factual evaluator — tests whether the model knows user facts.

Scoring:
  - Primary: keyword matching (any expected keyword in response → correct)
  - Optional secondary: Gemini judge for borderline / low-keyword cases
  - Final: binary correct/incorrect per question, accuracy = correct / total
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from openclawmini.eval.eval_set import EvalSet, FactualQuestion


@dataclass
class FactualResult:
    question_id: str
    question: str
    response: str
    correct: bool
    matched_keywords: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    source_fact: str = ""


@dataclass
class FactualEvalSummary:
    results: list[FactualResult]
    accuracy: float         # 0-1
    correct_count: int
    total_count: int

    def by_category(self) -> dict[str, float]:
        """Per-category accuracy."""
        from collections import defaultdict
        cat_correct: dict[str, list[bool]] = defaultdict(list)
        for r in self.results:
            # We don't track category here but we can add it later
            pass
        return {}


class FactualEvaluator:
    """
    Runs factual evaluation against a model function.

    Args:
        llm_judge: Optional Gemini extractor for LLM-based secondary scoring.
                   When provided, used for questions where keyword matching
                   returns no match (to catch paraphrased correct answers).
        use_llm_judge: Whether to use the LLM judge at all (costs API calls).
    """

    def __init__(
        self,
        llm_judge=None,
        use_llm_judge: bool = True,
    ) -> None:
        self._judge = llm_judge
        self._use_llm_judge = use_llm_judge and llm_judge is not None

    def run(
        self,
        eval_set: EvalSet,
        model_fn: Callable[[str], str],
        progress_callback=None,
    ) -> FactualEvalSummary:
        """
        Run factual eval.

        Args:
            eval_set: The persistent eval set.
            model_fn: callable(prompt: str) -> str
            progress_callback: optional callable(current, total)

        Returns:
            FactualEvalSummary with per-question results + accuracy.
        """
        questions = eval_set.factual_questions
        results: list[FactualResult] = []

        for i, q in enumerate(questions):
            if progress_callback:
                progress_callback(i + 1, len(questions))

            try:
                response = model_fn(q.question)
            except Exception as e:
                response = f"[ERROR: {e}]"

            correct, matched = _keyword_match(response, q.expected_keywords)

            # LLM judge for failed keyword matches (catch paraphrasing)
            if not correct and self._use_llm_judge:
                correct = self._llm_judge(response, q)

            results.append(FactualResult(
                question_id=q.id,
                question=q.question,
                response=response,
                correct=correct,
                matched_keywords=matched,
                expected_keywords=q.expected_keywords,
                source_fact=q.source_fact_content,
            ))

        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / max(len(results), 1)

        return FactualEvalSummary(
            results=results,
            accuracy=round(accuracy, 4),
            correct_count=correct_count,
            total_count=len(results),
        )

    def _llm_judge(self, response: str, question: FactualQuestion) -> bool:
        """
        Ask the LLM judge if the response correctly answers the question,
        even if it doesn't contain the exact expected keywords.
        """
        prompt = f"""Given this fact about a person:
"{question.source_fact_content}"

Question: "{question.question}"
AI Response: "{response[:500]}"

Does the AI response correctly answer the question based on the fact?
Answer with just YES or NO:"""
        try:
            raw = self._judge.complete(prompt).strip().lower()
            return raw.startswith("yes")
        except Exception:
            return False


# ── Helpers ───────────────────────────────────────────────────


def _keyword_match(response: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Check if any expected keyword appears in the response."""
    response_lower = response.lower()
    matched = [kw for kw in keywords if kw.lower() in response_lower]
    return len(matched) > 0, matched
