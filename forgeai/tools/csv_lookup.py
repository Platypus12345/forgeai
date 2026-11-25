"""Row lookup over the CSV documents in the agent's own corpus.

The tool is bound to the run's document folder by `set_corpus()` so a produced
agent can only ever read the knowledge base it was built for.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

_CORPUS: Path | None = None


def set_corpus(folder: str | Path | None) -> None:
    global _CORPUS
    _CORPUS = Path(folder) if folder else None


def _csv_files() -> list[Path]:
    if _CORPUS is None or not _CORPUS.exists():
        return []
    return sorted(
        p for p in _CORPUS.rglob("*") if p.suffix.lower() in {".csv", ".tsv"}
    )


def _load(path: Path) -> tuple[list[str], list[list[str]]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows = [r for r in csv.reader(io.StringIO(raw), delimiter=delim) if any(r)]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def run(file: str = "", query: str = "", limit: int = 5, **_ignored) -> str:
    files = _csv_files()
    if not files:
        return "ERROR: no CSV document is available in this agent's knowledge base"

    target = None
    if file:
        wanted = file.strip().lower()
        for p in files:
            if p.name.lower() == wanted or wanted in p.name.lower():
                target = p
                break
        if target is None:
            names = ", ".join(p.name for p in files)
            return f"ERROR: no CSV named '{file}'. Available: {names}"
    else:
        target = files[0]

    header, rows = _load(target)
    if not header:
        return f"ERROR: {target.name} is empty"

    try:
        limit = max(1, min(int(limit), 25))
    except (TypeError, ValueError):
        limit = 5

    query = (query or "").strip()
    if not query:
        matched = rows[:limit]
    elif "=" in query and query.split("=", 1)[0].strip().lower() in {
        h.strip().lower() for h in header
    }:
        col, _, value = query.partition("=")
        idx = [h.strip().lower() for h in header].index(col.strip().lower())
        value = value.strip().lower()
        matched = [r for r in rows if idx < len(r) and value in r[idx].strip().lower()]
    else:
        needle = query.lower()
        matched = [r for r in rows if any(needle in cell.lower() for cell in r)]

    if not matched:
        return f"No rows in {target.name} match '{query}'."

    out = [f"{target.name} - {len(matched)} row(s) matched, showing {min(len(matched), limit)}:"]
    out.append(" | ".join(header))
    for row in matched[:limit]:
        out.append(" | ".join(row))
    return "\n".join(out)
