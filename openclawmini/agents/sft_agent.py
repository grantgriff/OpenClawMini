"""
SFT Agent — supervised fine-tuning via W&B ART (openpipe-art).

Loads sft_*.jsonl from data/training/, converts each Q&A pair into an
ART Trajectory with reward=1.0 (behavioral cloning), then calls
model.train_sft() on the ServerlessBackend (CoreWeave GPU).

W&B logging happens automatically through ART's backend integration.

Auth:
  WANDB_API_KEY  — used for both W&B logging AND ServerlessBackend GPU access
  OPENPIPE_API_KEY — optional, enables OpenPipe trace logging

Output:
  ART model artifact registered in W&B project "openclawmini"
  Saved checkpoint path returned for EvalsAgent + GRPO hand-off
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Result ───────────────────────────────────────────────────────────────────


@dataclass
class SFTResult:
    model_name: str            # ART model name (for GRPO hand-off)
    project: str               # W&B project name
    base_model: str            # HuggingFace model ID used
    samples_trained: int
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoint_path: Optional[str] = None  # Local checkpoint (LocalBackend only)
    wandb_run_url: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "project": self.project,
            "base_model": self.base_model,
            "samples_trained": self.samples_trained,
            "trained_at": self.trained_at.isoformat(),
            "checkpoint_path": self.checkpoint_path,
        }


# ── SFT Agent ────────────────────────────────────────────────────────────────


class SFTAgent:
    """
    Supervised fine-tuning via W&B ART ServerlessBackend.

    Converts SFT JSONL (messages format) → ART Trajectories with reward=1.0
    → model.train_sft() on CoreWeave GPU via W&B serverless.

    Args:
        base_model:    ART model ID (e.g. "ministral-8b-2512"). Must match ART's
                       supported models list — check `art list-models` if training fails.
        model_name:    ART model name (unique identifier within project)
        project:       W&B project name
        learning_rate: SFT learning rate
        epochs:        Number of training epochs passed to TrainSFTConfig
        use_serverless: True → CoreWeave via W&B; False → local GPU
        wandb_api_key: Falls back to WANDB_API_KEY env var
    """

    ART_BASE_MODEL = "ministral-8b-2512"
    DEFAULT_PROJECT = "openclawmini"

    def __init__(
        self,
        base_model: str = ART_BASE_MODEL,
        model_name: str = "openclawmini-sft",
        project: str = DEFAULT_PROJECT,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        use_serverless: bool = True,
        wandb_api_key: Optional[str] = None,
    ) -> None:
        self.base_model = base_model
        self.model_name = model_name
        self.project = project
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_serverless = use_serverless
        self._wandb_api_key = wandb_api_key or os.getenv("WANDB_API_KEY", "")

    # ── Public API ──────────────────────────────────────────────

    def train(self, sft_jsonl_path: Path) -> SFTResult:
        """
        Run SFT training. Blocks until training is complete.

        Args:
            sft_jsonl_path: Path to sft_*.jsonl file with {"messages": [...]} per line.

        Returns:
            SFTResult with model_name for use by GRPOAgent.
        """
        return asyncio.run(self._train_async(sft_jsonl_path))

    # ── Internal async logic ────────────────────────────────────

    async def _train_async(self, sft_jsonl_path: Path) -> SFTResult:
        try:
            import art  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "ART not installed. Run: pip install openpipe-art"
            )

        samples = self._load_jsonl(sft_jsonl_path)
        if not samples:
            raise ValueError(f"No valid samples found in {sft_jsonl_path}")

        # Build ART model
        model = art.TrainableModel(
            name=self.model_name,
            project=self.project,
            base_model=self.base_model,
        )

        # Select backend
        if self.use_serverless:
            if not self._wandb_api_key:
                raise RuntimeError(
                    "WANDB_API_KEY is required for ART ServerlessBackend. "
                    "Set it in .env or pass use_serverless=False for local GPU."
                )
            backend = art.ServerlessBackend(api_key=self._wandb_api_key)
        else:
            backend = art.LocalBackend()

        await model.register(backend)

        # Convert SFT samples → Trajectories (reward=1.0 for supervised data)
        trajectories = self._build_trajectories(samples)

        # SFT config — pass epochs and learning_rate from TrainingConfig
        try:
            sft_config = art.TrainSFTConfig(
                learning_rate=self.learning_rate,
                num_epochs=self.epochs,
            )
        except (AttributeError, TypeError):
            sft_config = None  # Older ART version — use defaults

        print(f"  Starting SFT: {len(trajectories)} samples, lr={self.learning_rate}, epochs={self.epochs}")
        await model.train_sft(trajectories, config=sft_config)
        print(f"  SFT complete. Model: {self.project}/{self.model_name}")

        result = SFTResult(
            model_name=self.model_name,
            project=self.project,
            base_model=self.base_model,
            samples_trained=len(trajectories),
        )

        # Save result metadata
        output_dir = Path("./data/models/sft")
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / f"sft_result_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        meta_path.write_text(json.dumps(result.to_dict(), indent=2))

        return result

    def _build_trajectories(self, samples: list[dict]):
        """Convert SFT JSONL samples into ART Trajectory objects."""
        import art  # type: ignore[import]

        trajectories = []
        for item in samples:
            messages = item.get("messages", [])
            if not messages:
                continue
            # ART Trajectory: messages_and_choices is the full conversation
            # For SFT from pre-written data, pass all messages directly
            traj = art.Trajectory(
                messages_and_choices=messages,
                reward=1.0,    # Supervised — always reward the labeled response
            )
            trajectories.append(traj)
        return trajectories

    def _load_jsonl(self, path: Path) -> list[dict]:
        """Load JSONL file into list of sample dicts."""
        samples = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "messages" in data and data["messages"]:
                            samples.append(data)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        return samples

    # ── Factory ────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config, user_name: str = "") -> "SFTAgent":
        """Build SFTAgent from Config object."""
        # Use config's base_model directly as the ART model ID.
        # ART's supported model list uses short names like "ministral-8b-2512".
        # Run `art list-models` to verify the exact ID if training fails.
        art_model = getattr(config.base_model, "model", cls.ART_BASE_MODEL)

        name = (
            f"openclawmini-sft-{user_name.lower().replace(' ', '-')}"
            if user_name else "openclawmini-sft"
        )

        return cls(
            base_model=art_model,
            model_name=name,
            project=os.getenv("WANDB_PROJECT", cls.DEFAULT_PROJECT),
            learning_rate=getattr(config.training, "sft_learning_rate", 2e-5),
            epochs=getattr(config.training, "sft_epochs", 3),
        )

    @classmethod
    def from_env(cls, config=None) -> "SFTAgent":
        """Build from environment + optional config."""
        if config is not None:
            user_name = getattr(config.user, "name", "")
            return cls.from_config(config, user_name=user_name)
        return cls()
