"""
HuggingFace Model Exporter for OpenClawMini fine-tuned models.

ART stores LoRA adapter checkpoints in .art/{project}/{model_name}/step_{N}/
when using LocalBackend. This exporter reads those checkpoints and uploads
them to HuggingFace Hub.

Important limitations:
  - Works only with LocalBackend checkpoints stored in .art/
  - ServerlessBackend (CoreWeave) models live on CoreWeave managed infra and
    cannot be directly exported — they must be queried via ART's openai_client().
  - ART does NOT have an export_weights() method.

Model naming on HF: {hf_username}/openclawmini-{user_slug}-{stage}-{MMDDYYYY}-{n}
  e.g.  grgriff/openclawmini-alice-sft-03012026-1
        grgriff/openclawmini-alice-grpo-03012026-1

Auth:
  HF_TOKEN    — HuggingFace write token
  HF_USERNAME — HuggingFace username (default: grgriff)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class ModelExporter:
    """
    Export ART LocalBackend fine-tuned LoRA checkpoints to HuggingFace Hub.

    Usage:
        exporter = ModelExporter.from_env()
        repo_id = exporter.export(
            project="openclawmini",
            model_name="openclawmini-sft",
            user_name="Alice",
            stage="sft",
        )
        # → "grgriff/openclawmini-alice-sft-03012026-1"
    """

    DEFAULT_HF_USERNAME = "grgriff"
    DEFAULT_ART_DIR = ".art"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        hf_username: Optional[str] = None,
        art_base_dir: str = DEFAULT_ART_DIR,
    ) -> None:
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.hf_username = hf_username or os.getenv("HF_USERNAME", self.DEFAULT_HF_USERNAME)
        self.art_base_dir = Path(art_base_dir)

    # ── Public API ──────────────────────────────────────────────

    def export(
        self,
        project: str,
        model_name: str,
        user_name: str,
        stage: str,
    ) -> Optional[str]:
        """
        Export the latest checkpoint from .art/{project}/{model_name}/ to HF Hub.

        Args:
            project:    ART project name (e.g. "openclawmini")
            model_name: ART model name (e.g. "openclawmini-sft")
            user_name:  User's real name for the HF repo slug
            stage:      Training stage: "sft" or "grpo"

        Returns:
            HuggingFace repo ID if successful, None if no checkpoint found.
        """
        if not self.hf_token:
            print("  [HF Export] HF_TOKEN not set — skipping upload.")
            return None

        checkpoint_path = self._find_latest_checkpoint(project, model_name)
        if checkpoint_path is None:
            print(
                f"  [HF Export] No local checkpoint found at "
                f"{self.art_base_dir / project / model_name}/\n"
                f"  (ServerlessBackend models live on CoreWeave — local export not supported)"
            )
            return None

        repo_id = self._build_repo_id(user_name, stage)
        print(f"  [HF Export] Uploading {checkpoint_path} → {repo_id}")
        self._push_to_hub(checkpoint_path, repo_id)
        print(f"  [HF Export] Uploaded: https://huggingface.co/{repo_id}")
        return repo_id

    # ── Internal ────────────────────────────────────────────────

    def _find_latest_checkpoint(self, project: str, model_name: str) -> Optional[Path]:
        """Find the highest-numbered step checkpoint in .art/{project}/{model_name}/."""
        base = self.art_base_dir / project / model_name
        if not base.exists():
            return None

        # ART stores step checkpoints as step_N directories
        step_dirs = []
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step_num = int(d.name.split("_")[1])
                    step_dirs.append((step_num, d))
                except (IndexError, ValueError):
                    pass

        if not step_dirs:
            # Fallback: return the base directory if it has model files
            if any(base.glob("*.safetensors")) or any(base.glob("adapter_config.json")):
                return base
            return None

        # Return the highest-step checkpoint
        _, latest = max(step_dirs, key=lambda x: x[0])
        return latest

    def _build_repo_id(self, user_name: str, stage: str) -> str:
        """
        Build HF repo ID: {username}/openclawmini-{user_slug}-{stage}-{MMDDYYYY}-{n}
        Appends -1, -2, etc. to avoid collisions on the same date.
        """
        user_slug = user_name.lower().replace(" ", "-") if user_name else "user"
        date_str = datetime.now().strftime("%m%d%Y")
        base_name = f"openclawmini-{user_slug}-{stage}-{date_str}"
        base_repo = f"{self.hf_username}/{base_name}"
        return self._find_available_repo(base_repo)

    def _find_available_repo(self, base_repo: str) -> str:
        """Append -1, -2, etc. until we find an unused HF repo name."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            for n in range(1, 20):
                candidate = f"{base_repo}-{n}"
                try:
                    api.repo_info(repo_id=candidate)
                    # Repo exists — try next number
                except Exception:
                    return candidate  # Repo doesn't exist — use it
        except ImportError:
            pass
        return f"{base_repo}-1"

    def _push_to_hub(self, checkpoint_path: Path, repo_id: str) -> None:
        """Upload the checkpoint directory to HuggingFace Hub."""
        from huggingface_hub import HfApi
        api = HfApi(token=self.hf_token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            commit_message=(
                f"Upload OpenClawMini fine-tuned model "
                f"({checkpoint_path.name})"
            ),
        )

    @classmethod
    def from_env(cls) -> "ModelExporter":
        """Build from HF_TOKEN and HF_USERNAME environment variables."""
        return cls(
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_username=os.getenv("HF_USERNAME", cls.DEFAULT_HF_USERNAME),
        )
