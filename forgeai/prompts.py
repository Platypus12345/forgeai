"""Loader for the committed stage prompts in `prompts/`.

Prompts live on disk, not in string literals, because the brief requires the
generation prompts to be committed as evidence and because it makes them
diffable and reviewable on their own. Substitution uses `{{name}}` rather than
str.format so that the JSON examples inside the prompts - which are full of
braces - need no escaping.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import PROMPTS_DIR

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, **values: object) -> str:
    """Load prompt `name` and substitute every {{placeholder}}."""
    text = _read(name)
    missing: list[str] = []

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            missing.append(key)
            return match.group(0)
        return str(values[key])

    out = _PLACEHOLDER.sub(sub, text)
    if missing:
        raise KeyError(f"prompt '{name}' missing values for: {sorted(set(missing))}")
    return out


def prompt_path(name: str) -> str:
    """Repo-relative path, recorded in artifacts so a run is reproducible."""
    return f"prompts/{name}.md"
