"""Stage 7 - honest reporting.

"A factory that claims 100% is suspicious; a factory that explains its 78% is
hired." So the report is built to make the number harder to believe, not
easier: the remaining failures are printed with the actual question and the
actual answer, the judge's own audit score sits above the scorecard it
produced, and a standing limitations section lists the reasons this number
could be wrong - including the circularity of generating the tests and the
agent from the same small model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import CaseResult, Scorecard

CATEGORY_EXPLANATION = {
    "retrieval_miss": "the passage holding the answer never reached the agent",
    "prompt_defect": "the agent had the evidence but its instructions led it astray",
    "hallucination": "the agent asserted something the retrieved text does not say",
    "wrong_refusal": "the agent declined a question its documents could answer",
}


def _load_cases(run_dir: Path, iteration: int) -> list[CaseResult]:
    path = run_dir / "iterations" / f"iteration-{iteration}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [CaseResult(**c) for c in data.get("cases", [])]


def _best_iteration(result: Any) -> int:
    if not result.scorecards:
        return 0
    best = max(result.scorecards, key=lambda s: s.score)
    return best.iteration


def _scorecard_table(scorecards: list[Scorecard]) -> list[str]:
    rows = [
        "| Iteration | Score | Passed | Answerable | Out-of-scope | Edge | Failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in scorecards:
        def kind(name: str) -> str:
            b = s.by_kind.get(name)
            return f"{b['passed']}/{b['total']}" if b else "-"

        failures = (
            ", ".join(f"{k}:{v}" for k, v in sorted(s.failures_by_category.items()))
            or "none"
        )
        rows.append(
            f"| {s.iteration} | **{s.score:.3f}** | {s.passed}/{s.total} | "
            f"{kind('answerable')} | {kind('out_of_scope')} | {kind('edge')} | {failures} |"
        )
    return rows


def write_report(result: Any, meta: dict, run_dir: Path) -> Path:
    reqs = result.requirements
    definition = result.definition
    eval_set = result.eval_set
    best_iter = _best_iteration(result)
    best_card = next(
        (s for s in result.scorecards if s.iteration == best_iter), None
    )
    cases = _load_cases(run_dir, best_iter)
    failures = [c for c in cases if not c.passed]

    L: list[str] = []
    name = definition.name if definition else (reqs.agent_name if reqs else "(not built)")
    L += [f"# ForgeAI run report - {name}", ""]

    if result.error:
        L += [
            "> **This run did not complete.** The error and whatever artifacts were",
            "> produced before it are below.",
            "",
            "```",
            result.error.strip()[:2000],
            "```",
            "",
        ]

    # -- headline ---------------------------------------------------------
    score_line = f"{best_card.score:.3f} ({best_card.passed}/{best_card.total} cases passed)" if best_card else "no evaluation completed"
    L += [
        "## Headline",
        "",
        f"- **Final score:** {score_line}",
        f"- **Winning iteration:** {best_iter}",
        f"- **Model (every stage):** `{meta.get('model')}`",
        f"- **Wall clock:** {meta.get('wall_seconds')}s",
        f"- **Stop reason:** {meta.get('stop_reason')}",
        "",
    ]
    if meta.get("provider") == "heuristic":
        L += [
            "> :warning: **No language model was configured for this run.** Every",
            "> stage ran on its deterministic fallback, and the produced agent",
            "> answers extractively. The pipeline is exercised end to end, but these",
            "> numbers say nothing about how the factory performs with a real model.",
            "",
        ]

    # -- brief ------------------------------------------------------------
    if reqs:
        L += ["## The brief", "", f"> {reqs.brief}", ""]

    # -- scorecard --------------------------------------------------------
    L += ["## Per-iteration scorecard", ""]
    L += _scorecard_table(result.scorecards) if result.scorecards else ["_No evaluation ran._"]
    L += [""]

    # -- the agent --------------------------------------------------------
    if definition:
        r = definition.retrieval
        L += [
            "## The agent that was produced",
            "",
            f"- **Name:** {definition.name}",
            f"- **Fingerprint:** `{definition.fingerprint()}` (version {definition.version})",
            f"- **Lineage:** {definition.lineage}",
            f"- **Retrieval:** chunk {r.chunk_size}/{r.chunk_overlap} overlap, "
            f"top_k {r.top_k}, min_score {r.min_score}, "
            f"per-doc cap {r.per_doc_cap or 'none'}, "
            f"query expansion {'on' if r.query_expansion else 'off'}",
            f"- **Output contract:** {definition.output_contract.format}, fields "
            f"{definition.output_contract.required_fields}, "
            f"max {definition.output_contract.max_answer_words} words",
            "",
            "### Tools",
            "",
        ]
        if definition.tools:
            for tool in definition.tools:
                L.append(f"- `{tool}` - {definition.tool_rationale.get(tool, '')}")
        else:
            L.append(
                "- **None.** Nothing in this brief requires a tool; the agent answers "
                "purely from retrieved documents."
            )
        decisions = next(
            (c.get("tool_decisions") for c in definition.changelog if c.get("tool_decisions")),
            [],
        )
        if decisions:
            L += ["", "Selection audit trail:", ""]
            L += [f"- {d}" for d in decisions]
        L += [""]

    # -- groundability ----------------------------------------------------
    if reqs and reqs.groundability:
        L += ["## Groundability guard", ""]
        unsupported = [f for f in reqs.groundability if f.status == "unsupported"]
        partial = [f for f in reqs.groundability if f.status == "partial"]
        L += [
            f"- supported: {sum(1 for f in reqs.groundability if f.status == 'supported')}",
            f"- partial: {len(partial)}",
            f"- **unsupported: {len(unsupported)}**",
            "",
        ]
        if unsupported:
            L += [
                "The brief asked for these, the documents do not support them, and the",
                "factory moved them into the refusal policy rather than generating an",
                "agent that would invent them:",
                "",
            ]
            L += [f"- **{f.topic}** - {f.action}" for f in unsupported]
            L += [""]
        if partial:
            L += ["Thin coverage (answered only as far as the documents go):", ""]
            L += [f"- {f.topic}" for f in partial]
            L += [""]

    # -- eval set ---------------------------------------------------------
    if eval_set:
        counts = eval_set.counts()
        L += [
            "## The eval set",
            "",
            f"{len(eval_set.cases)} cases generated from the documents themselves: "
            + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            + ".",
            "",
            "Generation prompts (committed): "
            + ", ".join(f"`{p}`" for p in eval_set.generation_prompt_files),
            "",
        ]
        origins: dict[str, int] = {}
        for c in eval_set.cases:
            origins[c.origin] = origins.get(c.origin, 0) + 1
        L += [
            "Case provenance: "
            + ", ".join(f"{v} from {k}" for k, v in sorted(origins.items()))
            + ".",
            "",
        ]
        for note in eval_set.notes:
            L += [f"- :warning: {note}"]
        if eval_set.notes:
            L += [""]

    # -- tournament -------------------------------------------------------
    if result.tournament:
        L += [
            "## Prompt tournament",
            "",
            "Three candidate system prompts were drafted in different styles and",
            "scored on a stratified subset of the generated eval set. The eval set",
            "picked the winner, not the author.",
            "",
            "| Style | Score | Passed | Winner |",
            "|---|---|---|---|",
        ]
        for entry in result.tournament:
            L.append(
                f"| {entry['style']} | {entry['score']:.3f} | "
                f"{entry['passed']}/{entry['subset_size']} | "
                f"{'**yes**' if entry.get('winner') else 'no'} |"
            )
        L += [""]

    # -- iterations -------------------------------------------------------
    L += ["## What the optimizer changed", ""]
    if not result.iterations:
        L += ["_No iteration ran: the baseline already met the threshold, or the "
              "loop stopped before the first fix._", ""]
    for it in result.iterations:
        verdict = "accepted" if it.accepted else "**rolled back**"
        L += [
            f"### Iteration {it.iteration} - {it.score_before:.3f} -> "
            f"{it.score_after:.3f} ({it.score_after - it.score_before:+.3f}, {verdict})",
            "",
        ]
        for change in it.changes:
            if change.get("event") == "retrieval_fix":
                params = ", ".join(
                    f"`{k}` {v[0]} -> {v[1]}"
                    for k, v in (change.get("parameters_changed") or {}).items()
                )
                L += [
                    f"- **Retrieval fix** ({change['category']}): {change['rung']}",
                    f"  - changed: {params or 'nothing'}",
                    f"  - why: {change.get('why', '')}",
                    f"  - aimed at: {', '.join(change.get('targeted_cases', [])[:8])}",
                ]
            elif change.get("event") == "prompt_fix":
                L += [
                    f"- **Prompt fix** ({change['category']}): rewrote the "
                    f"`{change['section']}` section only",
                    f"  - diagnosis: {change.get('diagnosis', '')}",
                    f"  - what changed: {change.get('what_changed', '')}",
                    f"  - aimed at: {', '.join(change.get('targeted_cases', [])[:8])}",
                ]
            else:
                L += [f"- **No fix available** ({change.get('category')}): "
                      f"{change.get('why', '')}"]
        if it.note:
            L += [f"- _{it.note}_"]
        L += [""]

    # -- the honest part --------------------------------------------------
    L += ["## What still fails, and why", ""]
    if not cases:
        L += ["_No case-level results were produced._", ""]
    elif not failures:
        L += [
            "Every generated case passes. **Treat this with suspicion rather than",
            "satisfaction**: the same small model wrote the questions and the agent",
            "that answers them, so a clean sweep is at least as likely to mean the",
            "eval set is too easy as it is to mean the agent is good. The unseen-brief",
            "run at review is the real test.",
            "",
        ]
    else:
        by_cat: dict[str, list[CaseResult]] = {}
        for f in failures:
            by_cat.setdefault(f.failure_category or "unclassified", []).append(f)
        L += [
            f"{len(failures)} of {len(cases)} cases still fail at iteration {best_iter}.",
            "",
        ]
        for category, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            L += [
                f"### {category} ({len(items)}) - "
                f"{CATEGORY_EXPLANATION.get(category, '')}",
                "",
            ]
            for f in items[:6]:
                L += [
                    f"- **{f.case_id}** ({f.kind}): {f.question}",
                    f"  - agent said: {(f.answer or '(empty)')[:240]}",
                    f"  - refused={f.refused}, grounded={f.grounded}, "
                    f"fact recall={f.correct}, gold doc retrieved="
                    f"{f.gold_chunk_retrieved}, contract ok={f.contract_ok}",
                ]
                if f.judge_notes:
                    L += [f"  - {f.judge_notes}"]
            if len(items) > 6:
                L += [f"- _...and {len(items) - 6} more in `iterations/`._"]
            L += [""]

    # -- judge audit ------------------------------------------------------
    L += ["## Can you trust the judge?", ""]
    audit = result.judge_audit or {}
    if audit.get("ran"):
        L += [
            f"The judge was tested against planted answers on "
            f"{audit['cases_sampled']} case(s):",
            "",
            f"- **Recall on known-good answers:** {audit['recall_on_known_good']:.2f} "
            "(it should mark every fact present)",
            f"- **Specificity on known-bad answers:** "
            f"{audit['specificity_on_known_bad']:.2f} (it should mark every fact absent)",
            f"- **Verbosity resistance:** {audit['verbosity_resistance']:.2f} "
            "(the same wrong answer, padded five times longer)",
            f"- **Verdict:** {audit['verdict']}",
            "",
        ]
        if audit.get("verbosity_bias_detected"):
            L += [
                "> :warning: The judge scored the padded wrong answer better than the",
                "> short wrong answer. That is verbosity bias, measured rather than",
                "> assumed, and it means the scores above are slightly generous to",
                "> long answers.",
                "",
            ]
    else:
        L += [f"_Judge audit did not run: {audit.get('reason', 'unknown')}._", ""]

    L += [
        "### How judging actually works here",
        "",
        "Scoring is deterministic-first. Refusal correctness, citation validity,",
        "whether the gold document was retrieved, and the contract check are pure",
        "code with no model involved. Expected facts are matched literally first;",
        "the model is asked only about facts the matcher did not find, i.e. only",
        "to catch paraphrase, and then twice with the fact order reversed - a fact",
        "counts as present only if both passes agree.",
        "",
    ]
    if best_card and best_card.judge_disagreements:
        L += [
            f"On this run, {best_card.judge_disagreements} case(s) contained at least",
            "one fact the judge graded differently depending on its position in the",
            "list. Those facts were counted as absent.",
            "",
        ]

    # -- budget -----------------------------------------------------------
    budget = meta.get("budget", {})
    L += [
        "## Budget",
        "",
        f"- model calls: {budget.get('calls')}",
        f"- tokens (estimated): {budget.get('tokens'):,} of "
        f"{budget.get('token_budget'):,}" if budget.get("tokens") is not None else "",
        f"- wall clock: {budget.get('seconds')}s of {budget.get('latency_budget_s')}s",
        f"- JSON repair turns needed: {budget.get('json_repairs')}",
        f"- deterministic fallbacks taken: {budget.get('deterministic_fallbacks')}",
        "",
    ]
    per_stage = budget.get("per_stage") or {}
    if per_stage:
        L += ["| Stage | Calls | Tokens | Seconds | Repairs | Fallbacks |", "|---|---|---|---|---|---|"]
        for stage, s in sorted(per_stage.items(), key=lambda kv: -kv[1]["tokens"]):
            L.append(
                f"| {stage} | {s['calls']} | {s['tokens']:,} | {s['seconds']} | "
                f"{s['repairs']} | {s['fallbacks']} |"
            )
        L += [""]
    if budget.get("deterministic_fallbacks"):
        L += [
            f"> {budget['deterministic_fallbacks']} stage call(s) fell back to",
            "> deterministic logic because the model's output could not be parsed or",
            "> validated. Those parts of the pipeline did not benefit from the model.",
            "",
        ]

    # -- standing limitations ---------------------------------------------
    L += [
        "## Limitations of this report",
        "",
        "These apply to every run, including the good-looking ones:",
        "",
        "1. **The tests and the agent share an author.** The same model wrote the",
        "   eval questions and the agent's prompt, so they share blind spots. A",
        "   question the model would not think to ask is a weakness neither the",
        "   agent nor the score will show.",
        "2. **Expected facts are the model's opinion of what matters.** The grading",
        "   key was extracted from the documents by the same small model. A wrong",
        "   expected fact marks a correct answer as a failure, and the reverse.",
        "3. **Groundedness is lexical.** It measures whether the answer's numbers and",
        "   terms appear in the retrieved text. A fluent sentence that recombines",
        "   real terms into a false claim can score as grounded.",
        "4. **The eval set is small.** Twenty-odd cases put a wide confidence",
        "   interval on any score; a few points of difference between iterations is",
        "   noise, not progress.",
        "5. **Refusal is scored as a binary.** An agent that refuses politely and an",
        "   agent that refuses rudely score identically, and an over-cautious agent",
        "   can bank easy points on the out-of-scope third of the set.",
        "",
    ]

    text = "\n".join(line for line in L if line is not None)
    path = run_dir / "REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path
