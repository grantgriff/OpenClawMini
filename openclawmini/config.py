"""Config loading for OpenClawMini - YAML model config + .env secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-opus-4-5-20251101", "claude-sonnet-4-20250514"],
    "google": ["gemini-2.5-pro", "gemini-2.5-flash"],
    "openai": ["gpt-5.2", "gpt-5.2-chat-latest", "gpt-5-mini"],
    "mistral": ["mistral-small-2506", "mistral-large-latest"],
    "openpipe": ["OpenPipe/Qwen3-14B-Instruct"],
}

ORCHESTRATOR_OPTIONS: list[dict] = [
    {"label": "Claude Opus 4.5 (claude-opus-4-5-20251101)", "provider": "anthropic", "model": "claude-opus-4-5-20251101", "default": True},
    {"label": "GPT-5.2 (gpt-5.2)", "provider": "openai", "model": "gpt-5.2"},
    {"label": "Gemini 2.5 Pro (gemini-2.5-pro)", "provider": "google", "model": "gemini-2.5-pro"},
    {"label": "Mistral Large (mistral-large-latest)", "provider": "mistral", "model": "mistral-large-latest"},
]

RESEARCH_OPTIONS: list[dict] = [
    {"label": "Gemini 2.5 Flash (gemini-2.5-flash)", "provider": "google", "model": "gemini-2.5-flash", "default": True},
    {"label": "GPT-5 Mini (gpt-5-mini)", "provider": "openai", "model": "gpt-5-mini"},
    {"label": "Gemini 2.5 Pro (gemini-2.5-pro)", "provider": "google", "model": "gemini-2.5-pro"},
]

DATA_GEN_OPTIONS: list[dict] = [
    {"label": "Gemini 2.5 Pro (gemini-2.5-pro)", "provider": "google", "model": "gemini-2.5-pro", "default": True},
    {"label": "GPT-5.2 (gpt-5.2)", "provider": "openai", "model": "gpt-5.2"},
    {"label": "Claude Opus 4.5 (claude-opus-4-5-20251101)", "provider": "anthropic", "model": "claude-opus-4-5-20251101"},
]


@dataclass
class ModelConfig:
    provider: str
    model: str


@dataclass
class TrainingConfig:
    sft_sample_count: int = 30
    grpo_scenario_count: int = 15
    quality_threshold: float = 7.0
    sft_factual_threshold: float = 0.70
    final_target_accuracy: float = 0.80
    sft_epochs: int = 1
    sft_learning_rate: float = 2e-5
    grpo_steps: int = 30
    grpo_trajectories_per_scenario: int = 2
    orchestrator_budget: float = 25.0   # Max USD for autonomous orchestration


@dataclass
class UserConfig:
    name: str = ""
    email: str = ""
    data_sources: dict = field(default_factory=lambda: {
        "gmail": False,
        "linkedin": False,
        "web_search": True,
        "file_upload": False,
    })


@dataclass
class Config:
    orchestrator: ModelConfig = field(default_factory=lambda: ModelConfig("anthropic", "claude-opus-4-5-20251101"))
    research_extraction: ModelConfig = field(default_factory=lambda: ModelConfig("google", "gemini-2.5-flash"))
    data_generation: ModelConfig = field(default_factory=lambda: ModelConfig("google", "gemini-2.5-pro"))
    grpo_judge: ModelConfig = field(default_factory=lambda: ModelConfig("google", "gemini-2.5-flash"))
    eval_judge: ModelConfig = field(default_factory=lambda: ModelConfig("google", "gemini-2.5-flash"))
    base_model: ModelConfig = field(default_factory=lambda: ModelConfig("openpipe", "OpenPipe/Qwen3-14B-Instruct"))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    user: UserConfig = field(default_factory=UserConfig)
    memory_file_path: str = "./data/memory.json"
    memory_storage_type: str = "file"


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def load_env(env_path: str = ".env") -> None:
    """Load .env file into environment."""
    load_dotenv(env_path, override=False)


def _parse_model_config(raw: dict) -> ModelConfig:
    return ModelConfig(provider=raw["provider"], model=raw["model"])


def load_config(config_path: str = "config.yaml") -> Config:
    """Load config.yaml into a Config dataclass. Falls back to defaults if file missing."""
    path = Path(config_path)
    if not path.exists():
        return Config()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    models_raw = raw.get("models", {})
    training_raw = raw.get("training", {})
    storage_raw = raw.get("storage", {})
    user_raw = raw.get("user", {})

    training = TrainingConfig(
        sft_sample_count=training_raw.get("sft_sample_count", 30),
        grpo_scenario_count=training_raw.get("grpo_scenario_count", 15),
        quality_threshold=training_raw.get("quality_threshold", 7.0),
        sft_factual_threshold=training_raw.get("sft_factual_threshold", 0.70),
        final_target_accuracy=training_raw.get("final_target_accuracy", 0.80),
        sft_epochs=training_raw.get("sft_epochs", 1),
        sft_learning_rate=training_raw.get("sft_learning_rate", 2e-5),
        grpo_steps=training_raw.get("grpo_steps", 30),
        grpo_trajectories_per_scenario=training_raw.get("grpo_trajectories_per_scenario", 2),
        orchestrator_budget=training_raw.get("orchestrator_budget", 25.0),
    )

    user = UserConfig(
        name=user_raw.get("name", ""),
        email=user_raw.get("email", ""),
        data_sources=user_raw.get("data_sources", {
            "gmail": False, "linkedin": False, "web_search": True, "file_upload": False,
        }),
    )

    def _mc(key: str, default_provider: str, default_model: str) -> ModelConfig:
        r = models_raw.get(key, {})
        return ModelConfig(
            provider=r.get("provider", default_provider),
            model=r.get("model", default_model),
        )

    return Config(
        orchestrator=_mc("orchestrator", "anthropic", "claude-opus-4-5-20251101"),
        research_extraction=_mc("research_extraction", "google", "gemini-2.5-flash"),
        data_generation=_mc("data_generation", "google", "gemini-2.5-pro"),
        grpo_judge=_mc("grpo_judge", "google", "gemini-2.5-flash"),
        eval_judge=_mc("eval_judge", "google", "gemini-2.5-flash"),
        base_model=_mc("base_model", "openpipe", "OpenPipe/Qwen3-14B-Instruct"),
        training=training,
        user=user,
        memory_file_path=storage_raw.get("memory_file_path", "./data/memory.json"),
        memory_storage_type=storage_raw.get("memory_storage_type", "file"),
    )


def save_config(config: Config, config_path: str = "config.yaml") -> None:
    """Serialize Config back to YAML."""
    data = {
        "models": {
            "orchestrator": {"provider": config.orchestrator.provider, "model": config.orchestrator.model},
            "research_extraction": {"provider": config.research_extraction.provider, "model": config.research_extraction.model},
            "data_generation": {"provider": config.data_generation.provider, "model": config.data_generation.model},
            "grpo_judge": {"provider": config.grpo_judge.provider, "model": config.grpo_judge.model},
            "eval_judge": {"provider": config.eval_judge.provider, "model": config.eval_judge.model},
            "base_model": {"provider": config.base_model.provider, "model": config.base_model.model},
        },
        "training": {
            "sft_sample_count": config.training.sft_sample_count,
            "grpo_scenario_count": config.training.grpo_scenario_count,
            "quality_threshold": config.training.quality_threshold,
            "sft_factual_threshold": config.training.sft_factual_threshold,
            "final_target_accuracy": config.training.final_target_accuracy,
            "sft_epochs": config.training.sft_epochs,
            "sft_learning_rate": config.training.sft_learning_rate,
            "grpo_steps": config.training.grpo_steps,
            "grpo_trajectories_per_scenario": config.training.grpo_trajectories_per_scenario,
            "orchestrator_budget": config.training.orchestrator_budget,
        },
        "storage": {
            "memory_file_path": config.memory_file_path,
            "memory_storage_type": config.memory_storage_type,
        },
        "user": {
            "name": config.user.name,
            "email": config.user.email,
            "data_sources": config.user.data_sources,
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_env(env_vars: dict, env_path: str = ".env") -> None:
    """Write a dict of env vars to a .env file, merging with existing values."""
    existing: dict[str, str] = {}
    path = Path(env_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    existing[k.strip()] = v.strip()

    existing.update({k: v for k, v in env_vars.items() if v})

    with open(env_path, "w") as f:
        f.write("# OpenClawMini environment configuration\n")
        f.write("# Generated by openclawmini init\n\n")
        for k, v in existing.items():
            f.write(f"{k}={v}\n")


def get_env_key(key: str) -> Optional[str]:
    """Get an env var, returning None if not set or empty."""
    val = os.getenv(key, "").strip()
    return val if val else None
