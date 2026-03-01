"""
Evals Agent — generates the persistent eval set and runs evaluations.

Key design decisions (from PRD §8.3):
  - Eval set generated ONCE after initial research, saved to disk
  - Same eval set used across BASE → SFT → GRPO for fair comparison
  - Two dimensions: factual (binary) + stylistic (LLM-scored 0-1)
  - Results drive routing: weak factual → SFT, weak stylistic → GRPO
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from openclawmini.eval.eval_set import EvalSet, EvalSetGenerator, EvalSetStore
from openclawmini.eval.factual import FactualEvaluator
from openclawmini.eval.router import Action, EvalResults, EvalResultsStore, decide_next_action
from openclawmini.eval.stylistic import StylisticEvaluator
from openclawmini.integrations.wb_logger import WBLogger


class EvalsAgent:
    """
    Orchestrates eval set generation and evaluation runs.

    Usage:
        agent = EvalsAgent(gemini_extractor=extractor)

        # Generate eval set once (after research):
        eval_set = agent.generate_eval_set(memory)

        # Run eval against any model function:
        results = agent.run_eval(model_fn=mistral_fn, stage="base")
        action = agent.get_routing_action(results, config)
    """

    def __init__(
        self,
        gemini_extractor=None,
        eval_store_path: str = "./data/evals/eval_set.json",
        results_dir: str = "./data/evals/results",
        wb_logger: Optional[WBLogger] = None,
        data_cleansing_agent=None,
    ) -> None:
        self._extractor = gemini_extractor
        self._eval_store = EvalSetStore(eval_store_path)
        self._results_store = EvalResultsStore(results_dir)
        self._wb = wb_logger or WBLogger.from_env()
        # Optional DataCleansingAgent for DataSimulator-based question augmentation
        self._data_agent = data_cleansing_agent

    # ── Eval set generation ────────────────────────────────────

    def generate_eval_set(self, memory, force: bool = False) -> EvalSet:
        """
        Generate the persistent eval set from memory.

        If an eval set already exists on disk and force=False, loads and
        returns it without regenerating (preserves fair comparison).

        Args:
            memory: Memory object from MemoryStore.
            force: If True, regenerate even if an existing set is found.

        Returns:
            EvalSet with factual questions + stylistic prompts.
        """
        if not force and self._eval_store.exists():
            existing = self._eval_store.load()
            if existing and not existing.is_empty():
                return existing

        generator = EvalSetGenerator(gemini_extractor=self._extractor)
        eval_set = generator.generate(memory)

        # Augment factual questions with DataSimulator-generated prompts
        if self._data_agent is not None:
            try:
                extra_prompts = self._data_agent.generate_question_prompts(
                    memory, count=30
                )
                from openclawmini.eval.eval_set import FactualQuestion
                existing_qs = {q.question for q in eval_set.factual_questions}
                for prompt_text in extra_prompts:
                    if prompt_text and prompt_text not in existing_qs:
                        eval_set.factual_questions.append(FactualQuestion(
                            question=prompt_text,
                            expected_keywords=[],
                            category="other",
                            source_fact_id="datasimulator",
                            source_fact_content="",
                        ))
                        existing_qs.add(prompt_text)
            except Exception:
                pass  # Augmentation is best-effort

        self._eval_store.save(eval_set)
        self._wb.log_eval_set(eval_set)
        return eval_set

    def load_eval_set(self) -> Optional[EvalSet]:
        """Load the eval set from disk. Returns None if not generated yet."""
        return self._eval_store.load()

    def eval_set_exists(self) -> bool:
        return self._eval_store.exists()

    # ── Eval running ───────────────────────────────────────────

    def run_eval(
        self,
        model_fn: Callable[[str], str],
        stage: str,
        progress_callback=None,
    ) -> EvalResults:
        """
        Run the full eval (factual + stylistic) against a model function.

        Args:
            model_fn: callable(prompt: str) -> str  (any model, base or fine-tuned)
            stage: one of "base", "sft", "grpo", "grpo_v2", ...
            progress_callback: optional callable(dimension: str, current: int, total: int)

        Returns:
            EvalResults with both dimensions + routing recommendation.
        """
        eval_set = self.load_eval_set()
        if eval_set is None or eval_set.is_empty():
            raise RuntimeError(
                "No eval set found. Call generate_eval_set(memory) first."
            )

        # ── Factual eval ──────────────────────────────────────
        factual_evaluator = FactualEvaluator(
            llm_judge=self._extractor,
            use_llm_judge=self._extractor is not None,
        )

        def factual_progress(current, total):
            if progress_callback:
                progress_callback("factual", current, total)

        factual_summary = factual_evaluator.run(
            eval_set=eval_set,
            model_fn=model_fn,
            progress_callback=factual_progress,
        )

        # ── Stylistic eval ────────────────────────────────────
        stylistic_evaluator = StylisticEvaluator(llm_judge=self._extractor)

        def stylistic_progress(current, total):
            if progress_callback:
                progress_callback("stylistic", current, total)

        stylistic_summary = stylistic_evaluator.run(
            eval_set=eval_set,
            model_fn=model_fn,
            progress_callback=stylistic_progress,
        )

        # ── Combine results ───────────────────────────────────
        results = EvalResults.from_summaries(
            stage=stage,
            factual=factual_summary,
            stylistic=stylistic_summary,
        )

        # ── Save results + log to W&B ─────────────────────────
        self._results_store.save(results)
        self._wb.log_eval(results)

        return results

    def run_base_eval(
        self,
        progress_callback=None,
    ) -> EvalResults:
        """
        Convenience method: run eval against the base Ministral 8B model
        via the Mistral API (no fine-tuning required).

        Requires MISTRAL_API_KEY in environment.
        """
        from openclawmini.utils.llm_client import MistralClient

        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        model = os.getenv("BASE_MODEL", "ministral-8b-2412")
        client = MistralClient(api_key=api_key, model=model)
        return self.run_eval(
            model_fn=client.complete,
            stage="base",
            progress_callback=progress_callback,
        )

    # ── Routing ────────────────────────────────────────────────

    def get_routing_action(self, results: EvalResults, config) -> Action:
        """Determine the next training action based on eval results."""
        action = decide_next_action(results, config)
        results.recommended_action = action.type
        # Re-save with the routing recommendation filled in
        self._results_store.save(results)
        return action

    # ── History ────────────────────────────────────────────────

    def load_all_results(self) -> list[dict]:
        """Load all saved eval results sorted by timestamp."""
        return self._results_store.load_all()

    def load_stage_result(self, stage: str) -> Optional[dict]:
        """Load the most recent saved result for a given stage."""
        return self._results_store.load_stage(stage)

    @classmethod
    def from_env(cls) -> "EvalsAgent":
        """Build EvalsAgent with Gemini extractor from environment variables."""
        try:
            from openclawmini.integrations.gemini_extractor import GeminiExtractor
            extractor = GeminiExtractor.from_env()
        except ImportError:
            extractor = None
        return cls(gemini_extractor=extractor)
