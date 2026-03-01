"""
Routing logic — decides what training action to take based on eval results.

Also defines EvalResults (the combined output of factual + stylistic eval)
and EvalResultsStore (persists results per stage).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openclawmini.eval.factual import FactualEvalSummary, FactualResult
from openclawmini.eval.stylistic import StylisticEvalSummary, StylisticResult


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── EvalResults ────────────────────────────────────────────────


@dataclass
class EvalResults:
    """
    Combined evaluation results for one training stage.
    Stored in W&B and used to drive the SFT ↔ GRPO routing decision.
    """
    stage: str                    # "base", "sft", "grpo", "grpo_v2", ...
    timestamp: datetime = field(default_factory=_now)

    # Factual dimension
    factual_accuracy: float = 0.0
    factual_correct: int = 0
    factual_total: int = 0
    factual_details: list[FactualResult] = field(default_factory=list)

    # Stylistic dimension
    stylistic_accuracy: float = 0.0
    stylistic_total: int = 0
    stylistic_details: list[StylisticResult] = field(default_factory=list)

    # Combined
    overall_accuracy: float = 0.0

    # Per-category factual accuracy {category: accuracy}
    category_accuracy: dict = field(default_factory=dict)

    # Routing signal
    weak_dimension: str = ""       # "factual" or "stylistic"
    weak_categories: list = field(default_factory=list)  # categories below threshold
    recommended_action: str = ""   # "sft", "grpo", or "complete"

    @classmethod
    def from_summaries(
        cls,
        stage: str,
        factual: FactualEvalSummary,
        stylistic: StylisticEvalSummary,
    ) -> "EvalResults":
        overall = round(0.5 * factual.accuracy + 0.5 * stylistic.accuracy, 4)
        weak = "factual" if factual.accuracy <= stylistic.accuracy else "stylistic"
        cat_acc = factual.by_category()
        # Categories below 60% accuracy are "weak"
        weak_cats = sorted(
            [cat for cat, acc in cat_acc.items() if acc < 0.60],
            key=lambda c: cat_acc[c],
        )
        return cls(
            stage=stage,
            factual_accuracy=factual.accuracy,
            factual_correct=factual.correct_count,
            factual_total=factual.total_count,
            factual_details=factual.results,
            stylistic_accuracy=stylistic.accuracy,
            stylistic_total=stylistic.total_count,
            stylistic_details=stylistic.results,
            overall_accuracy=overall,
            category_accuracy=cat_acc,
            weak_dimension=weak,
            weak_categories=weak_cats,
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"  Factual accuracy:    {self.factual_accuracy:.0%}  ({self.factual_correct}/{self.factual_total})",
            f"  Stylistic accuracy:  {self.stylistic_accuracy:.0%}  ({self.stylistic_total} prompts)",
            f"  Overall accuracy:    {self.overall_accuracy:.0%}",
            f"  Weak dimension:      {self.weak_dimension}",
            f"  Recommended action:  {self.recommended_action}",
        ]
        if self.category_accuracy:
            lines.append("  Per-category factual accuracy:")
            for cat, acc in sorted(self.category_accuracy.items(), key=lambda x: x[1]):
                marker = " ⚠" if acc < 0.60 else ""
                lines.append(f"    {cat:<16} {acc:.0%}{marker}")
        return lines

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "factual_accuracy": self.factual_accuracy,
            "factual_correct": self.factual_correct,
            "factual_total": self.factual_total,
            "stylistic_accuracy": self.stylistic_accuracy,
            "stylistic_total": self.stylistic_total,
            "overall_accuracy": self.overall_accuracy,
            "category_accuracy": self.category_accuracy,
            "weak_categories": self.weak_categories,
            "weak_dimension": self.weak_dimension,
            "recommended_action": self.recommended_action,
            # Store a sample of details (not all, to keep file sizes manageable)
            "factual_sample": [
                {
                    "question": r.question,
                    "correct": r.correct,
                    "matched": r.matched_keywords,
                    "fact": r.source_fact,
                }
                for r in self.factual_details[:10]
            ],
            "stylistic_sample": [
                {
                    "prompt": r.prompt,
                    "score": r.score,
                    "reasoning": r.reasoning,
                }
                for r in self.stylistic_details[:10]
            ],
        }


# ── Routing logic ──────────────────────────────────────────────


@dataclass
class Action:
    type: str           # "sft", "grpo", "complete"
    reason: str
    data_focus: str     # "factual", "stylistic", "balanced"


def decide_next_action(eval_results: EvalResults, config) -> Action:
    """
    Routing decision based on eval results and config thresholds.

    Rules (from PRD §7):
      1. overall >= FINAL_TARGET → complete
      2. factual < SFT_FACTUAL_THRESHOLD → need SFT
      3. factual >= threshold AND overall < target → need GRPO
    """
    sft_threshold = getattr(config.training, "sft_factual_threshold", 0.70)
    final_target = getattr(config.training, "final_target_accuracy", 0.80)

    if eval_results.overall_accuracy >= final_target:
        return Action(
            type="complete",
            reason=(
                f"Target accuracy reached! "
                f"Overall {eval_results.overall_accuracy:.0%} >= {final_target:.0%}"
            ),
            data_focus="balanced",
        )

    if eval_results.factual_accuracy < sft_threshold:
        return Action(
            type="sft",
            reason=(
                f"Factual accuracy ({eval_results.factual_accuracy:.0%}) "
                f"below threshold ({sft_threshold:.0%}). "
                f"More SFT training on user facts needed."
            ),
            data_focus="factual",
        )

    # Factual is good — work on stylistic
    return Action(
        type="grpo",
        reason=(
            f"Factual accuracy ({eval_results.factual_accuracy:.0%}) is above threshold. "
            f"Stylistic accuracy ({eval_results.stylistic_accuracy:.0%}) needs improvement. "
            f"Running GRPO for style/preference alignment."
        ),
        data_focus="stylistic",
    )


# ── Results store ──────────────────────────────────────────────


class EvalResultsStore:
    """Save eval results per stage to data/evals/results/."""

    RESULTS_DIR = "./data/evals/results"

    def __init__(self, results_dir: str = RESULTS_DIR) -> None:
        self.dir = Path(results_dir)

    def save(self, results: EvalResults) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        ts = results.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{results.stage}_{ts}.json"
        path = self.dir / filename
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        return path

    def load_all(self) -> list[dict]:
        """Load all saved results, sorted by timestamp."""
        if not self.dir.exists():
            return []
        files = sorted(self.dir.glob("*.json"))
        results = []
        for f in files:
            try:
                with open(f) as fh:
                    results.append(json.load(fh))
            except Exception:
                pass
        return results

    def load_stage(self, stage: str) -> Optional[dict]:
        """Load most recent result for a given stage."""
        all_results = [r for r in self.load_all() if r.get("stage") == stage]
        return all_results[-1] if all_results else None
