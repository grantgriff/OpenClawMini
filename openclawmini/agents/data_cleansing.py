"""
Data Cleansing Agent — orchestrates SFT and GRPO data generation.

SFT data  → DataSimulator SDK (grantgriff/datasimulator):
  Writes memory profile + writing samples to disk, calls DataSimulator
  with data_type="sft" to generate high-quality Q&A pairs.

GRPO data → DataSimulator SDK 100%:
  Feeds ALL memory data (facts + writing_samples + posts + preferences)
  to DataSimulator, generates `grpo_target` prompts, discards answers.
  Output: grpo_prompts_*.jsonl — just {"prompt": "..."} per line.
  ART/RULER handles online scoring at training time; no reference_response needed.

generate_question_prompts() is also public so EvalsAgent can augment
the eval set with DataSimulator-generated factual questions.

Models (wired from Config):
  datasimulator_generator_model: Gemini Pro  (high-quality generation)
  datasimulator_verifier_model:  Gemini Flash (fast quality scoring)

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

    # Default DataSimulator model assignments (match Config defaults):
    #   generator → Gemini Pro  (data_generation model)
    #   verifier  → Gemini Flash (eval_judge / research_extraction model)
    DEFAULT_DS_GENERATOR = "gemini-2.5-pro"
    DEFAULT_DS_VERIFIER  = "gemini-2.5-flash"

    def __init__(
        self,
        llm_client=None,
        sft_target: int = 200,
        grpo_target: int = 100,
        quality_threshold: float = 6.0,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        user_name: str = "the user",
        datasimulator_generator_model: str = DEFAULT_DS_GENERATOR,
        datasimulator_verifier_model: str = DEFAULT_DS_VERIFIER,
    ) -> None:
        self._llm = llm_client
        self.sft_target = sft_target
        self.grpo_target = grpo_target
        self.quality_threshold = quality_threshold
        self.output_dir = Path(output_dir)
        self.user_name = user_name
        self.datasimulator_generator_model = datasimulator_generator_model
        self.datasimulator_verifier_model = datasimulator_verifier_model

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
                "generator": self.datasimulator_generator_model,
                "verifier": self.datasimulator_verifier_model,
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

    # ── DataSimulator prompt extraction ────────────────────────

    def generate_question_prompts(
        self,
        memory,
        count: int = 50,
    ) -> list[str]:
        """
        Use DataSimulator SFT generation to extract ONLY the question side.

        Runs DataSimulator with data_type="sft" targeting `count` samples,
        then discards all assistant answers and returns just the user
        questions as plain strings. These can be used as:
          - GRPO writing prompts (paired with random writing samples as reference)
          - Additional eval factual questions (via EvalSetGenerator augmentation)

        Returns:
            List of question strings (empty list if DataSimulator unavailable).
        """
        import tempfile
        try:
            from datasimulator import DataSimulator  # type: ignore[import]
        except ImportError:
            return []

        if not memory.facts:
            return []

        profile_path = self._write_memory_profile(memory)
        output_dir = self.output_dir / "prompts_cache"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"prompts_{_now_tag()}.jsonl"

        try:
            sdk = DataSimulator(
                source=str(profile_path),
                data_type="sft",
                models={
                    "generator": self.datasimulator_generator_model,
                    "verifier": self.datasimulator_verifier_model,
                },
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                quality_threshold=self.quality_threshold,
                max_cost=3.0,      # small budget — prompts only
                batch_size=10,
                parallel_batches=2,
                interactive=False,
                domain_context=(
                    f"Generate diverse question prompts that someone might ask {self.user_name} "
                    f"about their life, work, background, skills, and personality. "
                    f"Questions should be natural and conversational. "
                    f"Cover all fact categories in the source document."
                ),
            )
            dataset = sdk.generate(num_samples=count, show_progress=False)
            dataset.save(str(output_path))
        except Exception:
            return []

        # Extract user questions only
        questions: list[str] = []
        try:
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        messages = data.get("messages", [])
                        user_msg = next(
                            (m for m in messages if m.get("role") == "user"), None
                        )
                        if user_msg and user_msg.get("content"):
                            questions.append(user_msg["content"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        except FileNotFoundError:
            pass

        return questions

    # ── GRPO generation (DataSimulator 100%) ──────────────────────

    def generate_grpo_data(
        self,
        memory,
        progress_callback=None,
    ) -> tuple[list[GRPOScenario], Path]:
        """
        Generate GRPO training prompts using DataSimulator 100%.

        Feeds ALL memory data (facts + writing samples + posts + preferences)
        to DataSimulator in SFT mode. Discards generated answers and saves
        only the questions/prompts as grpo_prompts_*.jsonl.

        Output format per line: {"prompt": "..."} — no reference_response.
        ART handles online GRPO scoring via RULER at training time.

        Falls back to GRPODataGenerator if DataSimulator unavailable.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tag = _now_tag()
        jsonl_path = self.output_dir / f"grpo_prompts_{tag}.jsonl"

        try:
            prompts = self._generate_grpo_datasimulator(memory)
            if not prompts:
                raise ValueError("DataSimulator returned no prompts")
        except Exception as ds_err:
            _warn(f"DataSimulator GRPO unavailable ({ds_err}), using fallback generator")
            prompts = self._generate_grpo_fallback(memory, progress_callback)

        # Build lightweight GRPOScenario list (prompt only — no reference needed for ART)
        scenarios = [
            GRPOScenario(
                prompt=p,
                reference_response="",       # ART RULER scores online; no pre-set reference
                style_markers=["match user's natural writing style"],
                scenario_type="datasimulator",
                quality_score=7.5,
            )
            for p in prompts
        ]

        # Save as prompts-only JSONL for ART
        with open(jsonl_path, "w") as f:
            for sc in scenarios:
                f.write(json.dumps({"prompt": sc.prompt}) + "\n")

        return scenarios, jsonl_path

    def _generate_grpo_datasimulator(self, memory) -> list[str]:
        """Run DataSimulator across ALL memory sources and return user questions only."""
        from datasimulator import DataSimulator  # type: ignore[import]

        if not memory.facts and not memory.writing_samples and not memory.posts:
            raise ValueError("Memory is empty")

        # Build source list: facts profile + writing samples + preferences
        profile_path = self._write_memory_profile(memory)
        sources = [str(profile_path)]
        if memory.writing_samples or memory.posts:
            samples_path = self._write_writing_samples(memory)
            sources.append(str(samples_path))
        if memory.preferences:
            prefs_path = self._write_preferences(memory)
            sources.append(str(prefs_path))

        output_dir = self.output_dir / "grpo_cache"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"grpo_raw_{_now_tag()}.jsonl"

        sdk = DataSimulator(
            source=sources,
            data_type="sft",           # SFT mode → rich Q&A, we keep only questions
            models={
                "generator": self.datasimulator_generator_model,
                "verifier": self.datasimulator_verifier_model,
            },
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            quality_threshold=self.quality_threshold,
            max_cost=8.0,
            batch_size=10,
            parallel_batches=2,
            interactive=False,
            checkpoint_dir=str(self.output_dir / "checkpoints"),
        )

        domain_context = (
            f"Generate diverse writing prompts and questions that test whether an AI "
            f"model responds in {self.user_name}'s voice, style, and persona.\n\n"
            f"Cover all dimensions:\n"
            f"- Factual persona questions ('Where do you work?', 'Tell me about yourself')\n"
            f"- Writing tasks ('Draft a quick email to a colleague about X')\n"
            f"- Stylistic prompts ('How would you respond to a recruiter message?')\n"
            f"- Preference probes ('What's your take on remote work?')\n"
            f"- Situational ('You just shipped a big feature — what do you say to your team?')\n\n"
            f"Vary difficulty and register. Draw on all categories in the source material."
        )

        dataset = sdk.generate(
            num_samples=self.grpo_target,
            domain_context=domain_context,
            show_progress=False,
        )
        dataset.save(str(output_path))

        # Extract only the user-turn questions
        prompts: list[str] = []
        try:
            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        messages = data.get("messages", [])
                        user_msg = next(
                            (m for m in messages if m.get("role") == "user"), None
                        )
                        if user_msg and user_msg.get("content"):
                            prompts.append(user_msg["content"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        except FileNotFoundError:
            pass

        return prompts[:self.grpo_target]

    def _generate_grpo_fallback(self, memory, progress_callback=None) -> list[str]:
        """GRPODataGenerator fallback — returns prompt strings only."""
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
        return [sc.prompt for sc in scenarios if sc.prompt]

    def _write_preferences(self, memory) -> Path:
        """Write user preferences as a plain-text file for DataSimulator."""
        lines = [
            f"# Preferences & Opinions: {self.user_name}",
            "",
            f"The following reflect {self.user_name}'s documented preferences and opinions.",
            "",
        ]
        for i, pref in enumerate(memory.preferences[:40]):
            cat = str(pref.category).split(".")[-1].replace("_", " ").title() if hasattr(pref, "category") else "Preference"
            lines.append(f"## {cat} Preference {i + 1}")
            lines.append(pref.content)
            lines.append("")

        prefs_path = self.output_dir / "memory_preferences.txt"
        prefs_path.write_text("\n".join(lines), encoding="utf-8")
        return prefs_path

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

        # Pull DataSimulator model overrides from config if available
        ds_generator = cls.DEFAULT_DS_GENERATOR
        ds_verifier  = cls.DEFAULT_DS_VERIFIER
        if config is not None:
            # data_generation model → generator (Pro for quality)
            gen_model = getattr(config.data_generation, "model", None)
            if gen_model:
                ds_generator = gen_model
            # eval_judge model → verifier (Flash for speed/cost)
            ver_model = getattr(config.eval_judge, "model", None)
            if ver_model:
                ds_verifier = ver_model

        return cls(
            llm_client=llm_client,
            sft_target=sft_target,
            grpo_target=grpo_target,
            quality_threshold=quality_threshold,
            user_name=user_name,
            datasimulator_generator_model=ds_generator,
            datasimulator_verifier_model=ds_verifier,
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
