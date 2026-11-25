"""Stage 6 - the critic-optimizer.

Failures arrive already classified by the judge from hard evidence (was the
gold document retrieved? did it refuse? are its numbers in the retrieved
text?). This module turns a cluster of failures into *one targeted edit* and
nothing more. The brief forbids full regeneration, and there is a practical
reason beyond compliance: regenerating the whole prompt makes the score move
for reasons nobody can attribute, so you learn nothing from the iteration.

Two kinds of fix:

- **retrieval_miss** is a *parameter* bug, not a wording bug. Telling a prompt
  to "retrieve better" does nothing. So these are fixed by walking a ladder of
  concrete retrieval changes (widen top_k, raise overlap, spread across
  documents, shrink chunks, enable query expansion), one rung per iteration,
  never repeating a rung.
- everything else is fixed by rewriting exactly one named prompt section,
  chosen from the evidence, with the other six passed through untouched.

Every fix records what changed, why, which case ids it was aimed at, and the
score either side of it. If the score does not improve, the factory rolls back
to the best definition and says so in the log.
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMClient, first_str
from ..prompts import render
from ..schemas import AgentDefinition, CaseResult, Requirements

CATEGORY_MEANING = {
    "retrieval_miss": (
        "the passage that contains the answer was never put in front of the agent"
    ),
    "prompt_defect": (
        "the agent had the right passage and stayed grounded, but its instructions "
        "led it to the wrong shape of answer or the wrong scope decision"
    ),
    "hallucination": (
        "the agent asserted things that are not in the retrieved text"
    ),
    "wrong_refusal": (
        "the agent declined a question its documents could answer"
    ),
}

# One rung per iteration, in order of how cheap and how likely to help.
RETRIEVAL_LADDER: list[tuple[str, str]] = [
    ("widen_top_k", "Increase top_k so more candidate passages reach the agent."),
    ("raise_overlap", "Increase chunk overlap so facts are not split across a boundary."),
    ("spread_documents", "Cap hits per document so one document cannot crowd out the others."),
    ("shrink_chunks", "Reduce chunk size so BM25 scores a passage on its own content."),
    ("query_expansion", "Enable pseudo-relevance feedback for vocabulary mismatch."),
    ("lower_floor", "Lower the minimum score so weakly-matching passages still reach the agent."),
]

# Which prompt sections own which failure mode, in order of preference. If the
# first choice has already been tried and rolled back, the next one is used
# rather than rewriting the same section again and expecting a different result.
SECTIONS_FOR: dict[str, list[str]] = {
    "wrong_refusal": ["refusal_policy", "edge_cases", "scope"],
    "hallucination": ["grounding_rules", "output_format", "role"],
    "prompt_defect": ["scope", "edge_cases", "output_format", "role"],
}


def fix_key(category: str, kind: str, target: str) -> str:
    """Identity of a fix, so the loop never tries the same one twice."""
    return f"{category}|{kind}|{target}"


class FixPlan:
    def __init__(
        self,
        category: str,
        kind: str,          # "retrieval" | "prompt"
        target: str,        # ladder rung or section name
        case_ids: list[str],
        reason: str,
    ):
        self.category = category
        self.kind = kind
        self.target = target
        self.case_ids = case_ids
        self.reason = reason


def _applied_rungs(definition: AgentDefinition) -> set[str]:
    done = set()
    for entry in definition.changelog:
        if entry.get("event") == "retrieval_fix":
            done.add(entry.get("rung", ""))
    return done


def _section_preferences(category: str, failures: list[CaseResult]) -> list[str]:
    """Order the candidate sections, using what the failures actually show."""
    order = list(SECTIONS_FOR.get(category, ["scope"]))
    if category != "prompt_defect":
        return order

    contract_broken = sum(1 for f in failures if not f.contract_ok)
    edge_cases = sum(1 for f in failures if f.kind == "edge")
    answered_when_should_refuse = sum(
        1 for f in failures if f.kind == "out_of_scope" and not f.refused
    )
    half = max(1, len(failures) // 2)
    if contract_broken >= half:
        first = "output_format"
    elif answered_when_should_refuse >= half:
        first = "scope"
    elif edge_cases >= half:
        first = "edge_cases"
    else:
        first = "scope"
    return [first] + [s for s in order if s != first]


def plan_fixes(
    results: list[CaseResult],
    definition: AgentDefinition,
    *,
    max_fixes: int = 2,
    exclude: set[str] | None = None,
) -> list[FixPlan]:
    """Group failures by category and plan at most `max_fixes` targeted edits.

    `exclude` holds the `fix_key()` of every fix already tried in this run,
    including ones that were rolled back. Repeating a fix that did not work is
    pure budget burn, so an exhausted category falls through to its next
    candidate section (or next retrieval rung) and, failing that, is reported
    as having no fix left.
    """
    exclude = exclude or set()
    buckets: dict[str, list[CaseResult]] = {}
    for r in results:
        if not r.passed and r.failure_category:
            buckets.setdefault(r.failure_category, []).append(r)
    if not buckets:
        return []

    ordered = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    plans: list[FixPlan] = []

    for category, failures in ordered:
        if len(plans) >= max_fixes:
            break
        ids = [f.case_id for f in failures]

        if category == "retrieval_miss":
            done = _applied_rungs(definition)
            rung = next(
                (
                    r
                    for r, _ in RETRIEVAL_LADDER
                    if r not in done and fix_key(category, "retrieval", r) not in exclude
                ),
                None,
            )
            if rung is None:
                plans.append(
                    FixPlan(
                        category, "none", "", ids,
                        "Every retrieval adjustment on the ladder has been tried; "
                        "the remaining misses are a corpus/question mismatch, not a "
                        "parameter problem.",
                    )
                )
                continue
            plans.append(
                FixPlan(category, "retrieval", rung, ids, dict(RETRIEVAL_LADDER)[rung])
            )
            continue

        section = next(
            (
                s
                for s in _section_preferences(category, failures)
                if fix_key(category, "prompt", s) not in exclude
            ),
            None,
        )
        if section is None:
            plans.append(
                FixPlan(
                    category, "none", "", ids,
                    f"Every prompt section that governs {category} has already been "
                    "rewritten in this run without improving the score; the cause is "
                    "not in the prompt.",
                )
            )
            continue
        plans.append(
            FixPlan(
                category, "prompt", section, ids,
                f"{len(failures)} case(s) classified {category}; "
                f"the `{section}` section governs this behaviour.",
            )
        )
    return plans


def _apply_retrieval(definition: AgentDefinition, rung: str) -> tuple[dict, bool]:
    r = definition.retrieval
    before = r.to_dict()
    if rung == "widen_top_k":
        r.top_k = min(r.top_k + 2, 8)
    elif rung == "raise_overlap":
        r.chunk_overlap = min(r.chunk_overlap + 100, r.chunk_size // 2)
    elif rung == "spread_documents":
        r.per_doc_cap = 2 if r.per_doc_cap == 0 else max(1, r.per_doc_cap - 1)
    elif rung == "shrink_chunks":
        r.chunk_size = max(350, r.chunk_size - 250)
        r.chunk_overlap = min(r.chunk_overlap, r.chunk_size // 2)
    elif rung == "query_expansion":
        r.query_expansion = True
    elif rung == "lower_floor":
        r.min_score = round(max(0.0, r.min_score - 0.03), 3)
    after = r.to_dict()
    changed = {k: (before[k], after[k]) for k in after if before[k] != after[k]}
    # chunk_size / chunk_overlap changes mean the corpus must be re-chunked.
    needs_rechunk = any(k in changed for k in ("chunk_size", "chunk_overlap"))
    return {"before": before, "after": after, "changed": changed}, needs_rechunk


def _patch_section(
    definition: AgentDefinition,
    plan: FixPlan,
    failures: list[CaseResult],
    reqs: Requirements,
    client: LLMClient,
) -> tuple[str, str, str]:
    """Returns (new_text, what_changed, diagnosis)."""
    section = plan.target
    current = definition.system_prompt.get(section, "")
    failures_block = "\n\n".join(
        f"- case {f.case_id} ({f.kind}): {f.question}\n"
        f"  agent answered: {(f.answer or '(empty)')[:260]}\n"
        f"  refused={f.refused}  grounded={f.grounded}  fact_recall={f.correct}  "
        f"gold_doc_retrieved={f.gold_chunk_retrieved}"
        for f in failures[:6]
    )

    diag, _ = client.structured(
        render(
            "critic_diagnose",
            category=plan.category,
            category_meaning=CATEGORY_MEANING.get(plan.category, ""),
            failures_block=failures_block,
            section=section,
            current_section=current or "(empty)",
        ),
        stage="critic.diagnose",
        validate=lambda v: v if isinstance(v, dict) and first_str(v.get("diagnosis")) else
        (_ for _ in ()).throw(ValueError("no diagnosis")),
        fallback=lambda: {
            "diagnosis": (
                f"{len(failures)} case(s) failed as {plan.category}; the `{section}` "
                "section does not currently constrain this behaviour explicitly."
            )
        },
        temperature=0.0,
    )
    diagnosis = first_str(diag.get("diagnosis"))

    def validate(value: Any) -> dict:
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        text = first_str(value.get("new_text"))
        if len(text) < 40:
            raise ValueError("replacement section too short to be a real fix")
        return value

    patch, source = client.structured(
        render(
            "critic_patch",
            category=plan.category,
            section=section,
            scope=reqs.scope,
            audience=reqs.audience,
            must_refuse="; ".join(reqs.must_refuse_topics) or "(none)",
            current_section=current or "(empty)",
            failures_block=failures_block,
            diagnosis=diagnosis,
        ),
        stage="critic.patch",
        validate=validate,
        fallback=lambda: {
            "new_text": _deterministic_patch(section, current, plan, reqs),
            "what_changed": "Deterministic reinforcement appended (model unusable).",
        },
        max_tokens=800,
    )
    return (
        first_str(patch.get("new_text")),
        first_str(patch.get("what_changed")) + f" [patch source: {source}]",
        diagnosis,
    )


def _deterministic_patch(
    section: str, current: str, plan: FixPlan, reqs: Requirements
) -> str:
    """The no-model fix: append a specific, non-generic clause to the section."""
    additions = {
        "refusal_policy": (
            "Do not refuse a question merely because it is phrased unusually or "
            "because only part of it is covered. If the excerpts contain the "
            "answer, answer it. Refuse only when the excerpts genuinely lack the "
            "information or the topic is on the out-of-scope list."
        ),
        "grounding_rules": (
            "Never state a number, date, amount or name that does not appear "
            "verbatim in the excerpts you were given. If you cannot point to the "
            "excerpt containing a figure, leave the figure out and say the "
            "documents do not give it."
        ),
        "scope": (
            "Treat these as out of scope and decline them: "
            + ("; ".join(reqs.must_refuse_topics[:6]) or "anything the excerpts do not cover")
            + ". A question that merely sounds related is still out of scope if the "
            "excerpts do not address it."
        ),
        "output_format": (
            "The reply must be a single JSON object with exactly the keys "
            '"answer", "citations" and "refused". No text before or after it. '
            "Every label in `citations` must be copied from an excerpt you were given."
        ),
        "edge_cases": (
            "If a question is vague, answer the most likely reading and state the "
            "assumption in one clause. If the answer needs two documents, use both "
            "and cite both."
        ),
    }
    extra = additions.get(section, "Be specific and stay within the excerpts.")
    return (current + "\n" + extra).strip()


def apply_fix(
    definition: AgentDefinition,
    plan: FixPlan,
    results: list[CaseResult],
    reqs: Requirements,
    client: LLMClient,
    iteration: int,
) -> tuple[AgentDefinition, dict, bool]:
    """Apply one plan to a *copy* of the definition. Returns (new_def, record, rechunk)."""
    new_def = definition.copy()
    failures = [r for r in results if r.case_id in set(plan.case_ids)]

    if plan.kind == "none":
        record = {
            "event": "no_fix_available",
            "iteration": iteration,
            "category": plan.category,
            "targeted_cases": plan.case_ids,
            "why": plan.reason,
        }
        new_def.changelog.append(record)
        return new_def, record, False

    if plan.kind == "retrieval":
        detail, rechunk = _apply_retrieval(new_def, plan.target)
        record = {
            "event": "retrieval_fix",
            "iteration": iteration,
            "rung": plan.target,
            "category": plan.category,
            "why": plan.reason,
            "targeted_cases": plan.case_ids,
            "parameters_changed": detail["changed"],
        }
        new_def.changelog.append(record)
        new_def.version += 1
        new_def.lineage = f"iteration-{iteration} retrieval fix: {plan.target}"
        return new_def, record, rechunk

    new_text, what_changed, diagnosis = _patch_section(
        new_def, plan, failures, reqs, client
    )
    section = plan.target
    old_text = new_def.system_prompt.get(section, "")
    new_def.system_prompt[section] = new_text
    record = {
        "event": "prompt_fix",
        "iteration": iteration,
        "section": section,
        "category": plan.category,
        "why": plan.reason,
        "diagnosis": diagnosis,
        "what_changed": what_changed,
        "targeted_cases": plan.case_ids,
        "section_before": old_text,
        "section_after": new_text,
    }
    new_def.changelog.append(record)
    new_def.version += 1
    new_def.lineage = f"iteration-{iteration} prompt fix: {section}"
    return new_def, record, False
