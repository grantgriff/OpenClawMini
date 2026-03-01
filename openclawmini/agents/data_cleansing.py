"""
Data Cleansing Agent — orchestrates SFT and GRPO data generation.

SFT data  → uses DataSimulator SDK (grantgriff/datasimulator):
  Writes memory.facts + preferences to a profile .txt, then calls
  DataSimulator(source=profile.txt, data_type="sft") to generate
  high-quality Q&A pairs in Mistral fine-tuning JSONL format.

GRPO data → custom GRPODataGenerator:
  DataSimulator generates prompts only (no reference_response), so we
  keep the custom approach that derives prompts FROM actual writing
  samples, preserving the (prompt, reference) pairing RULER needs.

Output: timestamped JSONL files in data/training/
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openclawmini.training.grpo_generator import GRPODataGenerator, GRPOScenario
from openclawmini.training.sft_generator import SFTDataGenerator, SFTSample


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
    used_datasimulator: bool = False
    stage: str = "cleansing"

    def summary_lines(self) -> list[str]:
        lines = [
            f"SFT samples generated:    {len(self.sft_samples)}"
            + (" (DataSimulator)" if self.used_datasimulator else " (template)"),
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
    Orchestrates SFT and GRPO training data generation from memory.

    SFT  — uses DataSimulator SDK with memory.facts as source document.
           Falls back to Gemini-direct or template generation if unavailable.
    GRPO — uses custom GRPODataGenerator (needs actual writing as reference).

    Args:
        llm_client: Gemini extractor for fallback/GRPO generation.
        sft_target: Target number of SFT training examples.
        grpo_target: Target number of GRPO scenarios.
        quality_threshold: Min quality score (0-10) to keep a sample.
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

    # ── SFT generation (DataSimulator primary) ─────────────────

    def generate_sft_data(
        self,
        memory,
        progress_callback=None,
    ) -> tuple[list[SFTSample], Path, bool]:
        """
        Generate SFT Q&A pairs from memory.facts.

        Tries DataSimulator SDK first; falls back to Gemini-direct or template.

        Returns:
            (samples, jsonl_path, used_datasimulator)
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tag = _now_tag()

        try:
            samples, path = self._generate_sft_datasimulator(memory, tag)
            return samples, path, True
        except Exception as ds_err:
            # Log the reason for fallback (visible in verbose runs)
            _warn(f"DataSimulator SFT unavailable ({ds_err}), using fallback generator")

        # Fallback: Gemini-direct / template
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
        path = self.output_dir / f"sft_{tag}.jsonl"
        meta = self.output_dir / f"sft_{tag}_meta.json"
        generator.save_jsonl(samples, path)
        generator.save_metadata(samples, meta)
        return samples, path, False

    def _generate_sft_datasimulator(self, memory, tag: str) -> tuple[list[SFTSample], Path]:
        """Run DataSimulator SDK for SFT data generation."""
        from datasimulator import DataSimulator  # type: ignore[import]

        if not memory.facts:
            raise ValueError("No facts in memory — nothing to generate SFT data from")

        # Write source files to disk (DataSimulator requires file-based inputs)
        # Pass both facts profile AND writing samples so DataSimulator has full context
        profile_path = self._write_memory_profile(memory)
        sources = [str(profile_path)]
        if memory.writing_samples or memory.posts:
            samples_path = self._write_writing_samples(memory)
            sources.append(str(samples_path))

        output_path = self.output_dir / f"sft_{tag}.jsonl"

        sdk = DataSimulator(
            source=sources,
            data_type="sft",
            models={
                "generator": "gemini-2.0-flash",
                "verifier": "gemini-2.0-flash",
            },
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            quality_threshold=self.quality_threshold,
            max_cost=15.0,
            batch_size=10,
            parallel_batches=2,
            interactive=False,
            checkpoint_dir=str(self.output_dir / "checkpoints"),
        )

        domain_context = self._sft_domain_context(memory)
        dataset = sdk.generate(
            num_samples=self.sft_target,
            domain_context=domain_context,
            show_progress=False,
        )
        dataset.save(str(output_path))

        # Parse JSONL; inject our persona system message (replaces DataSimulator's generic one)
        persona_system = (
            f"You are {self.user_name}. Answer all questions about yourself honestly "
            f"and naturally in first person. Draw only from what you know about yourself."
        )
        samples = _parse_sft_jsonl(output_path, persona_system=persona_system)
        return samples, output_path

    def _write_memory_profile(self, memory) -> Path:
        """Write all facts + preferences as a structured plain-text profile file."""
        lines = [f"# Personal Profile: {self.user_name}", ""]

        # Group facts by category
        by_cat: dict[str, list[str]] = {}
        for fact in memory.facts:
            cat = str(fact.category).split(".")[-1].replace("_", " ").title()
            by_cat.setdefault(cat, []).append(fact.content)

        for cat in sorted(by_cat):
            lines.append(f"## {cat}")
            for content in by_cat[cat]:
                lines.append(f"- {content}")
            lines.append("")

        if memory.preferences:
            lines.append("## Preferences & Work Style")
            for pref in memory.preferences:
                lines.append(f"- {pref.content}")
            lines.append("")

        if memory.relationships:
            lines.append("## Key Relationships")
            for rel in memory.relationships[:20]:
                lines.append(f"- {rel.name}: {rel.relationship} ({rel.interaction_frequency})")
            lines.append("")

        profile_path = self.output_dir / "memory_profile.txt"
        profile_path.write_text("\n".join(lines), encoding="utf-8")
        return profile_path

    def _write_writing_samples(self, memory) -> Path:
        """Write writing samples and posts as a plain-text file for DataSimulator."""
        lines = [
            f"# Writing Samples: {self.user_name}",
            "",
            f"The following are actual messages and posts written by {self.user_name}.",
            "Use these to understand their natural voice, tone, and communication style.",
            "",
        ]
        for i, sample in enumerate(memory.writing_samples[:60]):
            cat = str(sample.category).split(".")[-1].replace("_", " ").title()
            lines.append(f"## Writing Sample {i + 1} ({cat})")
            if sample.context:
                lines.append(f"Context: {sample.context}")
            lines.append(sample.text)
            lines.append("")

        for i, post in enumerate(memory.posts[:20]):
            lines.append(f"## LinkedIn Post {i + 1}")
            lines.append(post.text)
            lines.append("")

        samples_path = self.output_dir / "memory_writing_samples.txt"
        samples_path.write_text("\n".join(lines), encoding="utf-8")
        return samples_path

    def _sft_domain_context(self, memory) -> str:
        fact_count = len(memory.facts)
        return (
            f"Generate diverse question-answer training pairs where an AI assistant IS {self.user_name}, "
            f"responding in first person.\n\n"
            f"The source document contains {fact_count} verified facts about {self.user_name}'s "
            f"background, work, education, skills, interests, and personal life.\n\n"
            f"Requirements:\n"
            f"- Answer AS {self.user_name} in first person (use 'I', 'my', 'me')\n"
            f"- Draw ONLY from facts in the source document — no fabrication\n"
            f"- Vary question types: direct ('Where do you work?'), conversational "
            f"('Tell me about yourself'), situational, reflective ('What are you proud of?')\n"
            f"- Answers should sound natural and personal, NOT like a resume bullet point\n"
            f"- Include a mix of short (1-2 sentence) and longer (3-4 sentence) answers\n"
            f"- Cover all fact categories: work, education, skills, location, interests, achievements"
        )

    # ── GRPO generation (custom — needs actual writing as reference) ─

    def generate_grpo_data(
        self,
        memory,
        progress_callback=None,
    ) -> tuple[list[GRPOScenario], Path]:
        """
        Generate GRPO style scenarios from memory.writing_samples and memory.posts.

        Uses custom generator (not DataSimulator) because GRPO requires
        (prompt, reference_response) pairs where reference_response is ACTUAL
        user writing — DataSimulator only generates prompts with no reference.
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

    # ── Full run ───────────────────────────────────────────────

    def run(
        self,
        memory,
        sft_progress_callback=None,
        grpo_progress_callback=None,
    ) -> DataCleansingResult:
        """
        Run full data cleansing: generate both SFT and GRPO datasets.

        Returns:
            DataCleansingResult with paths, counts, and sample lists.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sft_samples, sft_path, used_ds = self.generate_sft_data(
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
            used_datasimulator=used_ds,
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
            quality_threshold = getattr(config.training, "quality_threshold", 7.0) * 0.7
            user_name = getattr(config.user, "name", "") or user_name

        return cls(
            llm_client=llm_client,
            sft_target=sft_target,
            grpo_target=grpo_target,
            quality_threshold=quality_threshold,
            user_name=user_name,
        )


# ── Helpers ────────────────────────────────────────────────────

def _parse_sft_jsonl(
    path: Path,
    persona_system: Optional[str] = None,
) -> list[SFTSample]:
    """
    Read a DataSimulator-saved JSONL file into SFTSample objects.

    If persona_system is provided, DataSimulator's generic system message is
    replaced with the persona-framing system message (e.g. "You are Grant.
    Answer questions about yourself in first person.").
    """
    samples = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    user_msg = next((m for m in messages if m.get("role") == "user"), None)
                    asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
                    if user_msg and asst_msg:
                        # Build final message list
                        msg_list = []
                        if persona_system:
                            msg_list.append({"role": "system", "content": persona_system})
                        msg_list.extend([user_msg, asst_msg])
                        samples.append(SFTSample(
                            messages=msg_list,
                            quality_score=7.5,
                            category="factual",
                            source_fact_id="datasimulator",
                        ))
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return samples


def _warn(msg: str) -> None:
    """Print a dim warning to stderr-friendly output."""
    import sys
    print(f"[warn] {msg}", file=sys.stderr)
