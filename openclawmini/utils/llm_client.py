"""
Unified LLM clients for inference.

MistralClient   — wraps the Mistral API (legacy, kept for compatibility).
OpenPipeClient  — wraps OpenPipe's OpenAI-compatible API for calling
                  OpenPipe/Qwen3-14B-Instruct (or any fine-tuned checkpoint).
"""

from __future__ import annotations

import os
from typing import Optional


class MistralClient:
    """
    Thin wrapper around the Mistral API for model inference.

    Used by EvalsAgent to run the base model eval before any fine-tuning,
    and later to eval SFT/GRPO checkpoints via Mistral API.

    Args:
        api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
        model: Model ID (defaults to "mistral-small-2506")
        system_prompt: Optional system prompt prepended to every call.
        max_tokens: Max output tokens per call.
        temperature: Sampling temperature (0 = deterministic).
    """

    DEFAULT_MODEL = "mistral-small-2506"
    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant. "
        "Answer questions concisely and accurately."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "").strip()
        self.model = model or os.getenv("BASE_MODEL", self.DEFAULT_MODEL)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str) -> str:
        """
        Single-turn completion.

        Args:
            prompt: User message.

        Returns:
            Model response text.

        Raises:
            RuntimeError: if MISTRAL_API_KEY is not set.
        """
        if not self.api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY not set. "
                "Add it to .env or run openclawmini init."
            )
        client = self._get_client()
        response = client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def complete_as_user(self, prompt: str, persona_context: str = "") -> str:
        """
        Completion with a persona system prompt (used for stylistic eval).

        Args:
            prompt: The stylistic writing prompt.
            persona_context: Brief description of the person's style.
        """
        system = (
            f"You are responding as a specific person. {persona_context}"
            if persona_context
            else "Respond naturally in your own voice."
        )
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY not set.")
        client = self._get_client()
        response = client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "MistralClient":
        """Build from environment variables."""
        return cls(
            api_key=os.getenv("MISTRAL_API_KEY", "").strip(),
            model=model or os.getenv("BASE_MODEL", cls.DEFAULT_MODEL),
        )


class OpenPipeClient:
    """
    Thin wrapper around HuggingFace's OpenAI-compatible serverless inference API.

    Used by EvalsAgent and OrchestratorAgent to run the base model eval
    against OpenPipe/Qwen3-14B-Instruct (a HuggingFace model) before fine-tuning.

    No separate OpenPipe API key needed — uses HF_TOKEN from environment.

    Args:
        api_key: HuggingFace token (defaults to HF_TOKEN env var)
        model: HuggingFace model ID (defaults to "OpenPipe/Qwen3-14B-Instruct")
        system_prompt: Optional system prompt prepended to every call.
        max_tokens: Max output tokens per call.
        temperature: Sampling temperature (0 = deterministic).
    """

    DEFAULT_MODEL = "OpenPipe/Qwen3-14B-Instruct"
    BASE_URL = "https://api-inference.huggingface.co/v1"
    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant. "
        "Answer questions concisely and accurately."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key or os.getenv("HF_TOKEN", "").strip()
        self.model = model or os.getenv("BASE_MODEL", self.DEFAULT_MODEL)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        return self._client

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "HF_TOKEN not set. Add it to .env or run openclawmini init."
            )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def complete_as_user(self, prompt: str, persona_context: str = "") -> str:
        system = (
            f"You are responding as a specific person. {persona_context}"
            if persona_context
            else "Respond naturally in your own voice."
        )
        if not self.api_key:
            raise RuntimeError("HF_TOKEN not set.")
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "OpenPipeClient":
        """Build from environment variables."""
        return cls(
            api_key=os.getenv("HF_TOKEN", "").strip(),
            model=model or os.getenv("BASE_MODEL", cls.DEFAULT_MODEL),
        )
