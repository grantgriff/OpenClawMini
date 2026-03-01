"""
Weights & Biases logging for OpenClawMini eval pipeline.

Logs per-stage eval results to a W&B run so you can track improvement
across BASE → SFT → GRPO on a single dashboard.

Usage:
    logger = WBLogger.from_env()        # reads WANDB_API_KEY etc
    logger.log_eval(results)            # call after each eval stage
    logger.log_eval_set(eval_set)       # call after generating eval set
    logger.log_training_data(result)    # call after data cleansing

W&B project: WANDB_PROJECT env var (default: "openclawmini")
W&B entity:  WANDB_ENTITY env var (optional)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openclawmini.eval.router import EvalResults
    from openclawmini.eval.eval_set import EvalSet
    from openclawmini.agents.data_cleansing import DataCleansingResult


class WBLogger:
    """
    Thin wrapper around the wandb Python SDK for OpenClawMini metrics.

    Each call to log_eval() creates a new W&B run (one per stage) so
    the dashboard shows a clean progression: base → sft → grpo.

    If WANDB_API_KEY is not set, all methods silently no-op so the
    pipeline continues without W&B.
    """

    def __init__(
        self,
        project: str = "openclawmini",
        entity: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.project = project
        self.entity = entity
        self.enabled = enabled
        self._wandb = None

        if self.enabled:
            try:
                import wandb  # type: ignore[import]
                self._wandb = wandb
            except ImportError:
                self.enabled = False

    # ── Core logging ───────────────────────────────────────────

    def log_eval(self, results: "EvalResults") -> None:
        """
        Log factual, stylistic, and overall accuracy for one eval stage.

        Creates a new W&B run named after the stage (e.g. "eval_base",
        "eval_sft") under the openclawmini project.
        """
        if not self.enabled or self._wandb is None:
            return

        try:
            run = self._wandb.init(
                project=self.project,
                entity=self.entity,
                name=f"eval_{results.stage}",
                tags=["eval", results.stage],
                config={
                    "stage": results.stage,
                    "factual_total": results.factual_total,
                    "stylistic_total": results.stylistic_total,
                },
                reinit=True,
            )
            self._wandb.log({
                "factual_accuracy": results.factual_accuracy,
                "stylistic_accuracy": results.stylistic_accuracy,
                "overall_accuracy": results.overall_accuracy,
                "factual_correct": results.factual_correct,
                "factual_total": results.factual_total,
                "stylistic_total": results.stylistic_total,
                "weak_dimension": results.weak_dimension,
                "recommended_action": results.recommended_action,
                "stage": results.stage,
            })

            # Per-question factual breakdown (as a W&B Table)
            if results.factual_details:
                try:
                    table = self._wandb.Table(
                        columns=["question", "correct", "matched_keywords", "source_fact"]
                    )
                    for r in results.factual_details:
                        table.add_data(
                            r.get("question", ""),
                            r.get("correct", False),
                            ", ".join(r.get("matched_keywords", [])),
                            r.get("source_fact", ""),
                        )
                    self._wandb.log({"factual_details": table})
                except Exception:
                    pass  # Table logging is best-effort

            run.finish()
        except Exception:
            pass  # W&B errors must never crash the pipeline

    def log_eval_set(self, eval_set: "EvalSet") -> None:
        """Log eval set statistics (run once after generation)."""
        if not self.enabled or self._wandb is None:
            return

        try:
            run = self._wandb.init(
                project=self.project,
                entity=self.entity,
                name="eval_set_generated",
                tags=["eval_set"],
                reinit=True,
            )
            stats = eval_set.stats()
            self._wandb.log({
                "eval_set/factual_questions": stats["factual_questions"],
                "eval_set/stylistic_prompts": stats["stylistic_prompts"],
                "eval_set/total": stats["total"],
                "eval_set/memory_facts_count": eval_set.memory_facts_count,
                "eval_set/memory_samples_count": eval_set.memory_samples_count,
            })
            run.finish()
        except Exception:
            pass

    def log_training_data(self, result: "DataCleansingResult") -> None:
        """Log data cleansing results (SFT + GRPO sample counts)."""
        if not self.enabled or self._wandb is None:
            return

        try:
            run = self._wandb.init(
                project=self.project,
                entity=self.entity,
                name="data_cleansing",
                tags=["data", "cleansing"],
                reinit=True,
            )
            self._wandb.log({
                "data/sft_samples": len(result.sft_samples),
                "data/grpo_scenarios": len(result.grpo_scenarios),
                "data/facts_used": result.facts_used,
                "data/writing_samples_used": result.samples_used,
                "data/posts_used": result.posts_used,
                "data/used_datasimulator": result.used_datasimulator,
            })
            run.finish()
        except Exception:
            pass

    # ── Factory ────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "WBLogger":
        """Build from environment variables."""
        api_key = os.getenv("WANDB_API_KEY", "").strip()
        if not api_key:
            return cls(enabled=False)

        try:
            import wandb  # type: ignore[import]
            wandb.login(key=api_key, relogin=False)
        except Exception:
            return cls(enabled=False)

        return cls(
            project=os.getenv("WANDB_PROJECT", "openclawmini"),
            entity=os.getenv("WANDB_ENTITY") or None,
            enabled=True,
        )
