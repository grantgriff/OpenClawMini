"""
HuggingFace Model Exporter for OpenClawMini fine-tuned models.

Export strategy (in priority order):
  1. W&B artifact download (ServerlessBackend): ART logs the LoRA adapter as a
     W&B artifact after each training run. Download it via the W&B API, then
     upload to HuggingFace.
  2. LocalBackend checkpoint: ART saves step checkpoints in
     .art/{project}/{model_name}/step_{N}/. Read directly and upload to HF.

Model naming on HF:
  {hf_username}/openclawmini-{user_slug}-{stage}-{MMDDYYYY}-{n}
  e.g.  grgriff/openclawmini-grant-sft-03012026-1
        grgriff/openclawmini-grant-grpo-03012026-1

Auth:
  HF_TOKEN       — HuggingFace write token
  HF_USERNAME    — HuggingFace username (default: grgriff)
  WANDB_API_KEY  — W&B API key (for downloading ServerlessBackend artifacts)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


class ModelExporter:
    """
    Export ART fine-tuned LoRA adapters to HuggingFace Hub.

    For ServerlessBackend (CoreWeave), downloads the LoRA adapter from W&B
    artifacts then uploads to HF. For LocalBackend, reads from .art/ directly.

    Usage:
        exporter = ModelExporter.from_env()
        repo_id = exporter.export(
            project="openclawmini",
            model_name="openclawmini-sft",
            user_name="Grant",
            stage="sft",
        )
        # → "grgriff/openclawmini-grant-sft-03012026-1"
    """

    DEFAULT_HF_USERNAME = "grgriff"
    DEFAULT_ART_DIR = ".art"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        hf_username: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        art_base_dir: str = DEFAULT_ART_DIR,
    ) -> None:
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.hf_username = hf_username or os.getenv("HF_USERNAME", self.DEFAULT_HF_USERNAME)
        self.wandb_api_key = wandb_api_key or os.getenv("WANDB_API_KEY", "")
        self.art_base_dir = Path(art_base_dir)

    # ── Public API ──────────────────────────────────────────────

    def export(
        self,
        project: str,
        model_name: str,
        user_name: str,
        stage: str,
        wandb_entity: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export the latest fine-tuned LoRA adapter to HuggingFace Hub.

        Tries W&B artifact download first (ServerlessBackend), then falls back
        to local .art/ checkpoint (LocalBackend).

        Args:
            project:       ART/W&B project name (e.g. "openclawmini")
            model_name:    ART model name (e.g. "openclawmini-sft")
            user_name:     User's real name for the HF repo slug
            stage:         Training stage: "sft" or "grpo"
            wandb_entity:  W&B entity/username (defaults to WANDB_ENTITY env var)

        Returns:
            HuggingFace repo ID if successful, None otherwise.
        """
        if not self.hf_token:
            print("  [HF Export] HF_TOKEN not set — skipping upload.")
            return None

        repo_id = self._build_repo_id(user_name, stage)

        # Strategy 1: W&B artifact (ServerlessBackend)
        if self.wandb_api_key:
            adapter_path = self._download_wandb_artifact(
                project, model_name, wandb_entity
            )
            if adapter_path:
                try:
                    print(f"  [HF Export] W&B artifact → {repo_id}")
                    self._push_to_hub(adapter_path, repo_id, stage, model_name)
                    print(f"  [HF Export] Done: https://huggingface.co/{repo_id}")
                    return repo_id
                finally:
                    # Clean up temp dir if we created one
                    if str(adapter_path).startswith(tempfile.gettempdir()):
                        shutil.rmtree(adapter_path, ignore_errors=True)

        # Strategy 2: Local .art/ checkpoint (LocalBackend)
        checkpoint_path = self._find_local_checkpoint(project, model_name)
        if checkpoint_path:
            print(f"  [HF Export] Local checkpoint → {repo_id}")
            self._push_to_hub(checkpoint_path, repo_id, stage, model_name)
            print(f"  [HF Export] Done: https://huggingface.co/{repo_id}")
            return repo_id

        print(
            f"  [HF Export] No checkpoint found.\n"
            f"    W&B artifact: set WANDB_API_KEY to download ServerlessBackend adapter\n"
            f"    Local: no .art/{project}/{model_name}/ directory found"
        )
        return None

    # ── W&B artifact download ───────────────────────────────────

    def _download_wandb_artifact(
        self, project: str, model_name: str, entity: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download the latest LoRA adapter artifact from W&B.

        ART logs adapter checkpoints as W&B artifacts during training.
        The artifact path is: {entity}/{project}/{model_name}:latest
        """
        try:
            import wandb  # type: ignore[import]
        except ImportError:
            return None

        entity = entity or os.getenv("WANDB_ENTITY", "")
        if not entity:
            # Try to get from W&B API
            try:
                wandb.login(key=self.wandb_api_key, relogin=False)
                entity = wandb.api.default_entity or ""
            except Exception:
                return None

        artifact_path = f"{entity}/{project}/{model_name}:latest"

        try:
            wandb.login(key=self.wandb_api_key, relogin=False)
            api = wandb.Api()
            artifact = api.artifact(artifact_path)
            download_dir = Path(tempfile.mkdtemp(prefix="openclawmini_export_"))
            artifact.download(root=str(download_dir))
            # Verify it looks like a LoRA adapter
            if list(download_dir.glob("*.safetensors")) or list(
                download_dir.glob("adapter_config.json")
            ):
                return download_dir
            # Maybe it's nested
            for subdir in download_dir.iterdir():
                if subdir.is_dir() and (
                    list(subdir.glob("*.safetensors"))
                    or list(subdir.glob("adapter_config.json"))
                ):
                    return subdir
        except Exception as e:
            print(f"  [HF Export] W&B artifact not found ({artifact_path}): {e}")

        return None

    # ── Local checkpoint ────────────────────────────────────────

    def _find_local_checkpoint(self, project: str, model_name: str) -> Optional[Path]:
        """Find the highest-numbered step checkpoint in .art/{project}/{model_name}/."""
        base = self.art_base_dir / project / model_name
        if not base.exists():
            return None

        step_dirs = []
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                try:
                    step_num = int(d.name.split("_")[1])
                    step_dirs.append((step_num, d))
                except (IndexError, ValueError):
                    pass

        if step_dirs:
            _, latest = max(step_dirs, key=lambda x: x[0])
            return latest

        # Fallback: base dir itself has adapter files
        if any(base.glob("*.safetensors")) or any(base.glob("adapter_config.json")):
            return base

        return None

    # ── HuggingFace upload ──────────────────────────────────────

    def _build_repo_id(self, user_name: str, stage: str) -> str:
        """Build HF repo ID with date + auto-incremented run number."""
        user_slug = user_name.lower().replace(" ", "-") if user_name else "user"
        date_str = datetime.now().strftime("%m%d%Y")
        base_repo = f"{self.hf_username}/openclawmini-{user_slug}-{stage}-{date_str}"
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
                    # Repo exists — try next
                except Exception:
                    return candidate  # Repo doesn't exist — safe to use
        except ImportError:
            pass
        return f"{base_repo}-1"

    def _push_to_hub(
        self,
        adapter_path: Path,
        repo_id: str,
        stage: str,
        model_name: str,
    ) -> None:
        """Upload the adapter directory to HuggingFace Hub."""
        from huggingface_hub import HfApi
        api = HfApi(token=self.hf_token)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_id,
            commit_message=(
                f"OpenClawMini fine-tuned LoRA adapter — "
                f"stage={stage}, model={model_name}, "
                f"date={datetime.now().strftime('%Y-%m-%d')}"
            ),
        )

    @classmethod
    def from_env(cls) -> "ModelExporter":
        """Build from environment variables."""
        return cls(
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_username=os.getenv("HF_USERNAME", cls.DEFAULT_HF_USERNAME),
            wandb_api_key=os.getenv("WANDB_API_KEY", ""),
        )
