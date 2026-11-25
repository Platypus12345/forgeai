"""The tool registry the factory selects from, plus the selection guard.

The brief is explicit that attaching a tool the brief does not need is a
*factory defect* ("an HR agent with a weather tool"). So tool selection is
deliberately two-stage:

1. the model proposes a set of tools with a rationale, and
2. `vet_selection` can veto any of them on deterministic evidence.

The veto is what makes this robust on an unseen brief. A small model asked
"which of these four tools do you need?" has a strong bias toward answering
"all of them" - it reads the list as a menu to be satisfied rather than a
constraint to be justified. The guard checks the world instead of the model:
a CSV tool needs an actual CSV in the corpus, a weather tool needs an actual
weather/voyage signal in the brief. Both directions are logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from . import calculator, csv_lookup, current_date, weather


@dataclass
class Tool:
    name: str
    description: str
    when_to_use: str
    args_schema: dict[str, str]
    run: Callable[..., str]
    requires_network: bool = False
    # Deterministic evidence that this tool is actually warranted.
    brief_signals: list[str] = field(default_factory=list)
    needs_csv_corpus: bool = False


REGISTRY: dict[str, Tool] = {
    "csv_lookup": Tool(
        name="csv_lookup",
        description=(
            "Query a CSV/tabular document in the knowledge base: filter rows by a "
            "column value or substring and return matching rows."
        ),
        when_to_use=(
            "The user asks for a specific record, price, part number, stock level "
            "or row from a table."
        ),
        args_schema={
            "file": "CSV file name as it appears in the corpus",
            "query": "substring or 'column=value' filter",
            "limit": "max rows to return (default 5)",
        },
        run=csv_lookup.run,
        needs_csv_corpus=True,
        brief_signals=[
            "catalogue", "catalog", "csv", "table", "price", "pricing", "stock",
            "inventory", "part", "spare", "sku", "lookup", "quantity", "supplier",
            "order", "procure",
        ],
    ),
    "calculator": Tool(
        name="calculator",
        description="Evaluate an arithmetic expression exactly (+ - * / % ** parentheses).",
        when_to_use=(
            "The answer requires arithmetic: totals, percentages, pro-rata amounts, "
            "unit conversions, accrual over months."
        ),
        args_schema={"expression": "an arithmetic expression, e.g. '2.5 * 12 - 4'"},
        run=calculator.run,
        brief_signals=[
            "calculat", "comput", "total", "sum", "percent", "pro-rata", "prorata",
            "accru", "budget", "cost", "price", "discount", "tax", "quantity",
            "entitle", "balance", "convert",
        ],
    ),
    "current_date": Tool(
        name="current_date",
        description="Return today's date and time (UTC and local ISO-8601).",
        when_to_use=(
            "The answer depends on today: notice periods, deadlines, 'how many days "
            "until', leave-year boundaries, delivery ETAs."
        ),
        args_schema={"timezone": "optional IANA name, default UTC"},
        run=current_date.run,
        brief_signals=[
            "today", "current", "deadline", "notice period", "expir", "due",
            "remaining", "until", "schedule", "eta", "delivery", "renew",
            "leave year", "anniversar",
        ],
    ),
    "weather": Tool(
        name="weather",
        description=(
            "Current weather and short forecast for a latitude/longitude via the "
            "keyless open-meteo.com public API."
        ),
        when_to_use=(
            "The answer depends on real-world weather or sea conditions at a place."
        ),
        args_schema={
            "latitude": "decimal degrees",
            "longitude": "decimal degrees",
            "place": "optional label for the location",
        },
        run=weather.run,
        requires_network=True,
        brief_signals=[
            "weather", "forecast", "sea state", "wind", "storm", "temperature",
            "rain", "swell", "voyage condition", "port condition", "sailing",
            "meteo", "climate",
        ],
    ),
}


def tool_catalogue() -> str:
    """The registry rendered for a selection prompt."""
    lines = []
    for t in REGISTRY.values():
        lines.append(
            f"- {t.name}: {t.description}\n  Use when: {t.when_to_use}\n"
            f"  Arguments: {t.args_schema}"
        )
    return "\n".join(lines)


def vet_selection(
    proposed: list[str],
    *,
    brief_text: str,
    corpus_kinds: set[str],
    rationale: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Filter a proposed tool list down to what the evidence supports.

    Returns ``(accepted, rationale, decisions)`` where `decisions` is the
    human-readable audit trail written into the agent definition.
    """
    rationale = dict(rationale or {})
    text = (brief_text or "").lower()
    accepted: list[str] = []
    decisions: list[str] = []

    for name in dict.fromkeys(proposed):
        tool = REGISTRY.get(name)
        if tool is None:
            decisions.append(f"rejected '{name}': not in the registry")
            continue

        if tool.needs_csv_corpus and "csv" not in corpus_kinds:
            decisions.append(
                f"rejected '{name}': no tabular document in the corpus to query"
            )
            continue

        hits = [s for s in tool.brief_signals if s in text]
        if tool.brief_signals and not hits:
            decisions.append(
                f"rejected '{name}': nothing in the brief or requirements calls for it "
                f"(no signal among {tool.brief_signals[:4]}...)"
            )
            continue

        accepted.append(name)
        why = rationale.get(name) or tool.when_to_use
        evidence = f" Brief signal(s): {', '.join(hits[:3])}." if hits else ""
        rationale[name] = f"{why}{evidence}"
        decisions.append(f"accepted '{name}': {', '.join(hits[:3]) or 'warranted'}")

    for name in list(rationale):
        if name not in accepted:
            rationale.pop(name, None)
    return accepted, rationale, decisions


def propose_from_signals(brief_text: str, corpus_kinds: set[str]) -> list[str]:
    """Deterministic tool proposal - the fallback when the model is unusable."""
    text = (brief_text or "").lower()
    proposed = []
    for name, tool in REGISTRY.items():
        if tool.needs_csv_corpus and "csv" not in corpus_kinds:
            continue
        if any(s in text for s in tool.brief_signals):
            proposed.append(name)
    return proposed


_CALL = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*$", re.DOTALL)


def execute(name: str, args: dict[str, Any]) -> str:
    tool = REGISTRY.get(name)
    if tool is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        return tool.run(**args)
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:  # a tool must never take the agent down
        return f"ERROR: {name} failed: {exc}"


__all__ = [
    "REGISTRY",
    "Tool",
    "tool_catalogue",
    "vet_selection",
    "propose_from_signals",
    "execute",
]
