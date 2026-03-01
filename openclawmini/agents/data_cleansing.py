"""
Data Cleansing Agent — orchestrates SFT and GRPO data generation.

Transforms raw memory into clean, high-quality training datasets:
  - SFT data:  Q&A pairs from facts  (factual knowledge training)
  - GRPO data: style scenarios from writing samples (style alignment)

Key design decisions:
  - SFT target:  200 samples (config.training.sft_sample_count)
  - GRPO target: 100 scenarios (config.training.grpo_scenario_count)
  - Quality threshold filters out low-quality generated pairs
  - Uses Gemini Pro for high-quality generation; template fallback if unavailable
  - Output: timestamped JSONL files in data/training/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openclawmini.training.sft_generator import SFTDataGenerator, SFTSample
from openclawmini.training.grpo_generator import GRPODataGenerator, GRPOScenario


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class DataCleansingResult:
    sft_samples: list[SFTSample] = field(default_factory=list)
    grpo_scenarios: list[GRPOScenario] = field(default_factory=list)
    sft_path: Optional[Path] = None
    grpo_path: Optional[Path] = None
    sft_metadata_path: Optional[Path] = None
    grpo_metadata_path: Optional[Path] = None
    facts_used: int = 0
    samples_used: int = 0
    posts_used: int = 0
    stage: str = "cleansing"

    def summary_lines(self) -> list[str]:
        lines = [
            f"SFT samples generated:    {len(self.sft_samples)}",
            f"GRPO scenarios generated: {len(self.grpo_scenarios)}",
            f"Facts processed:          {self.facts_used}",
            f"Writing samples used:     {self.samples_used}",
            f"Posts used:               {self.posts_used}",
        ]
        if self.sft_path:
            lines.append(f"SFT JSONL:                {self.sft_path}")
        if self.grpo_path:
            lines.append(f"GRPO JSONL:               {self.grpo_path}")
        return lines


class DataCleansingAgent:
    """
    Orchestrates generation of SFT and GRPO training data from memory.

    SFT data  — teaches the model WHAT the user knows (factual Q&A pairs).
    GRPO data — teaches the model HOW the user writes (style scenarios).

    Args:
        llm_client: Gemini Pro (or similar) LLM with .complete(prompt) -> str.
        sft_target: Target number of SFT training examples.
        grpo_target: Target number of GRPO scenarios.
        quality_threshold: Min quality score (0-10) to include a sample.
        output_dir: Directory to write training JSONL files.
        user_name: User's name for persona framing in prompts.
    """

    DEFAULT_OUTPUT_DIR = "./data/training"

    def __init__(
        self,
        llm_client=None,
        sft_target: int = 200,
        grpo_target: int = 100,
        quality_threshold: float = 6.0,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        user_name: str = "the user",
    ) -> None:
        self._llm = llm_client
        self.sft_target = sft_target
        self.grpo_target = grpo_target
        self.quality_threshold = quality_threshold
        self.output_dir = Path(output_dir)
        self.user_name = user_name

    def generate_sft_data(
        self,
        memory,
        progress_callback=None,
    ) -> tuple[list[SFTSample], Path]:
        """
        Generate SFT Q&A pairs from memory.facts.

        Args:
            memory: Memory object with .facts list.
            progress_callback: optional callable(current, total)

        Returns:
            (samples, jsonl_output_path)
        """
        generator = SFTDataGenerator(
            llm_client=self._llm,
            user_name=self.user_name,
            quality_threshold=self.quality_threshold,
        )
        samples = generator.generate(
            memory=memory,
            target_count=self.sft_target,
            progress_callback=progress_callback,
        )
        tag = _now_tag()
        jsonl_path = self.output_dir / f"sft_{tag}.jsonl"
        meta_path = self.output_dir / f"sft_{tag}_meta.json"
        generator.save_jsonl(samples, jsonl_path)
        generator.save_metadata(samples, meta_path)
        return samples, jsonl_path

    def generate_grpo_data(
        self,
        memory,
        progress_callback=None,
    ) -> tuple[list[GRPOScenario], Path]:
        """
        Generate GRPO style scenarios from memory.writing_samples and memory.posts.

        Args:
            memory: Memory object with .writing_samples and .posts lists.
            progress_callback: optional callable(current, total)

        Returns:
            (scenarios, jsonl_output_path)
        """
        generator = GRPODataGenerator(
            llm_client=self._llm,
            user_name=self.user_name,
            quality_threshold=self.quality_threshold,
        )
        scenarios = generator.generate(
            memory=memory,
            target_count=self.grpo_target,
            progress_callback=progress_callback,
        )
        tag = _now_tag()
        jsonl_path = self.output_dir / f"grpo_{tag}.jsonl"
        meta_path = self.output_dir / f"grpo_{tag}_meta.json"
        generator.save_jsonl(scenarios, jsonl_path)
        generator.save_metadata(scenarios, meta_path)
        return scenarios, jsonl_path

    def run(
        self,
        memory,
        sft_progress_callback=None,
        grpo_progress_callback=None,
    ) -> DataCleansingResult:
        """
        Run full data cleansing: generate both SFT and GRPO datasets.

        Args:
            memory: Memory object.
            sft_progress_callback: optional callable(current, total)
            grpo_progress_callback: optional callable(current, total)

        Returns:
            DataCleansingResult with paths, counts, and sample lists.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sft_samples, sft_path = self.generate_sft_data(
            memory, progress_callback=sft_progress_callback
        )
        grpo_scenarios, grpo_path = self.generate_grpo_data(
            memory, progress_callback=grpo_progress_callback
        )

        sft_meta = Path(str(sft_path).replace(".jsonl", "_meta.json"))
        grpo_meta = Path(str(grpo_path).replace(".jsonl", "_meta.json"))

        return DataCleansingResult(
            sft_samples=sft_samples,
            grpo_scenarios=grpo_scenarios,
            sft_path=sft_path if sft_path.exists() else None,
            grpo_path=grpo_path if grpo_path.exists() else None,
            sft_metadata_path=sft_meta if sft_meta.exists() else None,
            grpo_metadata_path=grpo_meta if grpo_meta.exists() else None,
            facts_used=len(memory.facts),
            samples_used=len(memory.writing_samples),
            posts_used=len(memory.posts),
        )

    @classmethod
    def from_env(cls, config=None) -> "DataCleansingAgent":
        """Build from environment variables + optional config."""
        try:
            from openclawmini.integrations.gemini_extractor import GeminiExtractor
            llm_client = GeminiExtractor.from_env()
        except Exception:
            llm_client = None

        user_name = os.getenv("USER_NAME", "the user")
        sft_target = 200
        grpo_target = 100
        quality_threshold = 6.0

        if config is not None:
            sft_target = getattr(config.training, "sft_sample_count", 200)
            grpo_target = getattr(config.training, "grpo_scenario_count", 100)
            # config.training.quality_threshold is 0-10; generator uses same scale
            quality_threshold = getattr(config.training, "quality_threshold", 7.0) * 0.7
            user_name = getattr(config.user, "name", "") or user_name

        return cls(
            llm_client=llm_client,
            sft_target=sft_target,
            grpo_target=grpo_target,
            quality_threshold=quality_threshold,
            user_name=user_name,
        )
