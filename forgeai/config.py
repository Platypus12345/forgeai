"""Runtime configuration for ForgeAI.

Everything the factory needs to know about *where the model lives* and *how much
it is allowed to spend* is resolved here, once, at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
RUNS_DIR = REPO_ROOT / "runs"
DEMOS_DIR = REPO_ROOT / "demos"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Real env vars always win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(REPO_ROOT / ".env")


@dataclass
class ModelConfig:
    """Which model every stage of the factory runs on."""

    provider: str = "openai"          # openai | ollama | heuristic
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.1-8b-instant"
    api_key: str = ""
    temperature: float = 0.2
    max_output_tokens: int = 2600
    request_timeout: int = 120
    max_retries: int = 4
    # Provider token-per-minute ceiling, enforced client-side. 0 disables.
    tokens_per_minute: int = 8000

    @classmethod
    def from_env(cls) -> "ModelConfig":
        provider = os.environ.get("FORGEAI_PROVIDER", "").strip().lower()
        api_key = os.environ.get("FORGEAI_API_KEY", "").strip()
        base_url = os.environ.get("FORGEAI_BASE_URL", "").strip()
        model = os.environ.get("FORGEAI_MODEL", "").strip()

        if not provider:
            # Infer: a key means a hosted endpoint, a local Ollama means ollama,
            # nothing at all means the deterministic fallback engine.
            provider = "openai" if api_key else "heuristic"

        defaults = {
            "openai": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
            "ollama": ("http://localhost:11434", "qwen2.5:7b-instruct"),
            "heuristic": ("", "heuristic-v1"),
        }
        d_url, d_model = defaults.get(provider, defaults["openai"])
        return cls(
            provider=provider,
            base_url=base_url or d_url,
            model=model or d_model,
            api_key=api_key,
            temperature=float(os.environ.get("FORGEAI_TEMPERATURE", "0.2")),
            max_output_tokens=int(os.environ.get("FORGEAI_MAX_OUTPUT_TOKENS", "2600")),
            request_timeout=int(os.environ.get("FORGEAI_TIMEOUT", "120")),
            max_retries=int(os.environ.get("FORGEAI_MAX_RETRIES", "4")),
            tokens_per_minute=int(
                os.environ.get(
                    "FORGEAI_TPM", "0" if provider in {"ollama", "heuristic"} else "8000"
                )
            ),
        )

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class BudgetConfig:
    """Stretch goal: a token/latency budget per run, with early stopping."""

    max_tokens: int = 400_000
    max_seconds: int = 7200
    # An iteration that improves the score by less than this is not worth
    # another one; the loop stops early.
    min_gain_per_iteration: float = 0.02

    @classmethod
    def from_env(cls) -> "BudgetConfig":
        return cls(
            max_tokens=int(os.environ.get("FORGEAI_MAX_TOKENS", "400000")),
            max_seconds=int(os.environ.get("FORGEAI_MAX_SECONDS", "7200")),
            min_gain_per_iteration=float(
                os.environ.get("FORGEAI_MIN_GAIN", "0.02")
            ),
        )


@dataclass
class FactoryConfig:
    model: ModelConfig = field(default_factory=ModelConfig.from_env)
    budget: BudgetConfig = field(default_factory=BudgetConfig.from_env)
    max_iterations: int = 3          # hard cap from the brief
    min_eval_cases: int = 20         # hard floor from the brief
    max_clarifying_questions: int = 5
    # Score at which the loop stops happy - overridable so a reviewer can
    # demand a stricter bar than the default without editing code.
    pass_threshold: float = field(
        default_factory=lambda: float(os.environ.get("FORGEAI_PASS_THRESHOLD", "0.85"))
    )
    prompt_tournament: bool = True   # stretch goal
    seed: int = 20260101
