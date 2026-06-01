"""Prompt management — versioning, loading, and formatting."""

import os
from pathlib import Path
from string import Template
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class PromptManager:
    """Loads and manages versioned prompts from YAML files.

    Usage:
        pm = PromptManager(version="v1")
        prompt = pm.render("generation_prompt", question="...", context="...")
    """

    def __init__(self, version: str = "v1", prompts_dir: Path = PROMPTS_DIR):
        self.version = version
        self.prompts_dir = prompts_dir
        self._config = self._load()

    def _load(self) -> dict[str, Any]:
        """Load the prompt YAML for the configured version."""
        yaml_path = self.prompts_dir / "rag_prompts" / f"{self.version}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Prompt version '{self.version}' not found at {yaml_path}"
            )
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key: str) -> str:
        """Get a raw prompt template string by key."""
        if key not in self._config:
            raise KeyError(f"Prompt key '{key}' not found in v{self.version}")
        return self._config[key]

    def render(self, key: str, **kwargs) -> str:
        """Render a prompt template with variable substitution.

        Args:
            key: The prompt template key.
            **kwargs: Variables to substitute.

        Returns:
            Rendered prompt string.
        """
        template_str = self.get(key)
        return template_str.format(**kwargs)

    def system_prompt(self) -> str:
        return self.get("system_prompt")

    def generation_prompt(self, *, question: str, context: str) -> str:
        """Render the generation prompt with question and context.

        Citations are enforced by the prompt template itself.
        """
        return self.render(
            "generation_prompt",
            question=question,
            context=context,
        )

    @property
    def citation_declaration(self) -> str:
        """The exact string used to decline unanswerable questions."""
        return self.get("citation_declaration")

    @property
    def version_info(self) -> dict[str, str]:
        return {
            "name": self._config.get("name", "unknown"),
            "version": self._config.get("version", "unknown"),
            "description": self._config.get("description", ""),
        }
