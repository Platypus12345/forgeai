"""Stage 3 - the agent generator.

Produces a complete declarative agent definition: a sectioned system prompt,
chunking and retrieval parameters derived from the actual corpus, a vetted
tool selection, and the output contract.

The prompt tournament (a stretch goal) lives here too: `candidate_styles()`
returns three genuinely different drafting instructions, the factory builds one
definition per style, and the generated eval set - not the author's taste -
picks the winner.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import FactoryConfig
from ..docs import corpus_summary
from ..llm import LLMClient, as_str_list, first_str
from ..prompts import render
from ..schemas import (
    PROMPT_SECTIONS,
    AgentDefinition,
    Chunk,
    Document,
    OutputContract,
    RetrievalConfig,
    Requirements,
)
from ..tools import REGISTRY, propose_from_signals, tool_catalogue, vet_selection

CANDIDATE_STYLES: list[tuple[str, str]] = [
    (
        "terse-rules",
        "Write in short imperative rules. Numbered or bulleted lines, no prose "
        "paragraphs. Optimise for an instruction-following model that skims: the "
        "most important constraint first in every section.",
    ),
    (
        "explanatory",
        "Write in short explanatory paragraphs that give the assistant the reason "
        "behind each rule, not only the rule. A model that understands why it must "
        "cite is better at citing in situations the rules did not anticipate.",
    ),
    (
        "procedural",
        "Write as an explicit procedure the assistant follows for every question: "
        "a numbered decision path from 'read the excerpts' through 'decide whether "
        "they answer the question' to 'answer or refuse'. Make the refusal branch "
        "as concrete as the answer branch.",
    ),
]


def candidate_styles(cfg: FactoryConfig) -> list[tuple[str, str]]:
    return CANDIDATE_STYLES if cfg.prompt_tournament else CANDIDATE_STYLES[:1]


# ---------------------------------------------------------------------------
# Retrieval parameters, derived from the corpus rather than hard-coded
# ---------------------------------------------------------------------------


def choose_retrieval(docs: list[Document], chunks: list[Chunk]) -> RetrievalConfig:
    kinds = {d.kind for d in docs}
    n_docs = len(docs)
    avg_doc_chars = (sum(d.n_chars for d in docs) / n_docs) if n_docs else 0

    cfg = RetrievalConfig()

    # Tabular corpora want smaller chunks: a row is a self-contained fact and
    # packing many rows together dilutes the BM25 signal for any single one.
    if "csv" in kinds and len(kinds) == 1:
        cfg.chunk_size, cfg.chunk_overlap = 600, 0
    elif "csv" in kinds:
        cfg.chunk_size, cfg.chunk_overlap = 800, 100
    elif avg_doc_chars > 40_000:
        cfg.chunk_size, cfg.chunk_overlap = 1100, 180
    else:
        cfg.chunk_size, cfg.chunk_overlap = 900, 150

    # More documents means more chance the answer spans two of them.
    if n_docs >= 3:
        cfg.top_k, cfg.per_doc_cap = 5, 2
    elif n_docs == 2:
        cfg.top_k, cfg.per_doc_cap = 4, 3
    else:
        cfg.top_k, cfg.per_doc_cap = 4, 0

    if len(chunks) <= 6:
        cfg.top_k = max(2, min(cfg.top_k, len(chunks)))
        cfg.per_doc_cap = 0
    return cfg


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def select_tools(
    reqs: Requirements, docs: list[Document], client: LLMClient
) -> tuple[list[str], dict[str, str], list[str]]:
    corpus_kinds = {d.kind for d in docs}
    # The vet reads the brief *and* the derived requirements: "compute the order
    # total" usually shows up in a must-answer topic, not in the one-line brief.
    vet_text = " ".join(
        [reqs.brief, reqs.scope, " ".join(reqs.must_answer_topics),
         " ".join(reqs.requested_capabilities)]
    )

    def fallback() -> dict:
        return {"tools": propose_from_signals(vet_text, corpus_kinds), "rationale": {}}

    def validate(value: Any) -> dict:
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        value["tools"] = as_str_list(value.get("tools"))
        return value

    prompt = render(
        "tool_selection",
        scope=reqs.scope,
        must_answer="; ".join(reqs.must_answer_topics) or "(none)",
        capabilities="; ".join(reqs.requested_capabilities) or "(none stated)",
        corpus_summary=reqs.corpus_summary,
        catalogue=tool_catalogue(),
    )
    data, source = client.structured(
        prompt, stage="generator.tools", validate=validate, fallback=fallback
    )

    proposed = [t for t in as_str_list(data.get("tools")) if t in REGISTRY]
    rationale_raw = data.get("rationale")
    rationale = (
        {k: first_str(v) for k, v in rationale_raw.items()}
        if isinstance(rationale_raw, dict)
        else {}
    )
    accepted, rationale, decisions = vet_selection(
        proposed, brief_text=vet_text, corpus_kinds=corpus_kinds, rationale=rationale
    )
    decisions.insert(0, f"model proposed: {proposed or '[]'} (source: {source})")
    return accepted, rationale, decisions


# ---------------------------------------------------------------------------
# The system prompt
# ---------------------------------------------------------------------------


def _fallback_sections(
    reqs: Requirements, tools: list[str], contract: OutputContract
) -> dict[str, str]:
    """The deterministic prompt. Used when the model cannot produce usable JSON.

    This is written to be genuinely serviceable, not a placeholder: it is the
    floor on the quality of anything this factory ships.
    """
    must_answer = "; ".join(reqs.must_answer_topics) or "the contents of the documents"
    must_refuse = "; ".join(reqs.must_refuse_topics) or "anything outside the documents"
    ungrounded = "; ".join(reqs.ungrounded_topics)
    tool_line = (
        f" You may call these tools when they are needed: {', '.join(tools)}."
        if tools
        else ""
    )
    return {
        "role": (
            f"You are {reqs.agent_name}, assisting {reqs.audience}. You answer "
            f"strictly from excerpts supplied to you from a fixed set of documents."
            f"{tool_line}"
        ),
        "scope": (
            f"{reqs.scope}\nIn scope: {must_answer}.\nOut of scope: {must_refuse}."
        ),
        "grounding_rules": (
            "Answer only from the excerpts provided with the question. Never use "
            "knowledge from outside them, and never fill a gap with what is "
            "typical or likely. Every factual claim must be traceable to an "
            "excerpt, and you must cite the excerpt label for it. If the excerpts "
            "are thin, answer only the part they support and say which part they "
            "do not. If two excerpts conflict, say so and cite both rather than "
            "silently choosing one."
        ),
        "refusal_policy": (
            f"Refuse when the excerpts do not contain the answer, and when the "
            f"question is about: {must_refuse}."
            + (f" Also refuse these, which the documents do not cover: {ungrounded}." if ungrounded else "")
            + f" To refuse, set \"refused\": true and make the answer: "
            f"\"{contract.refusal_message}\" Add one sentence on what you can help "
            "with instead. Refusing is correct behaviour, not failure. Never guess "
            "to avoid refusing. But do not refuse a question the excerpts do "
            "answer - an unnecessary refusal is just as wrong as a fabrication."
        ),
        "tone": f"{reqs.tone} Write for {reqs.audience}. Plain words, no filler.",
        "output_format": (
            "Reply with a single JSON object and nothing else: "
            '{"answer": "...", "citations": ["document.pdf#2"], "refused": false}. '
            f"`citations` lists only labels of excerpts you were actually given. "
            f"Keep `answer` under {contract.max_answer_words} words. Set `refused` "
            "to true only when you are declining."
        ),
        "edge_cases": (
            "If the question is ambiguous, answer the most likely reading and state "
            "the assumption in one clause; ask for clarification only if the "
            "readings have genuinely different answers. If the answer spans two "
            "documents, combine them and cite both. If part of the question is in "
            "scope, answer that part and decline the rest in the same reply rather "
            "than refusing wholesale."
        ),
    }


def build_definition(
    reqs: Requirements,
    docs: list[Document],
    chunks: list[Chunk],
    client: LLMClient,
    cfg: FactoryConfig,
    *,
    style: tuple[str, str] | None = None,
    tools: list[str] | None = None,
    tool_rationale: dict[str, str] | None = None,
    tool_decisions: list[str] | None = None,
) -> AgentDefinition:
    style_name, style_directive = style or ("terse-rules", CANDIDATE_STYLES[0][1])
    contract = OutputContract()

    if tools is None:
        tools, tool_rationale, tool_decisions = select_tools(reqs, docs, client)
    tool_rationale = dict(tool_rationale or {})
    tool_decisions = list(tool_decisions or [])

    tools_block = (
        "\n".join(f"- {t}: {REGISTRY[t].description}" for t in tools)
        if tools
        else "(no tools - this agent answers purely from retrieved documents)"
    )

    def validate(value: Any) -> dict:
        if not isinstance(value, dict):
            raise ValueError("expected an object of prompt sections")
        out = {
            k: first_str(v)
            for k, v in value.items()
            if isinstance(k, str) and first_str(v)
        }
        present = [k for k in PROMPT_SECTIONS if k in out]
        if len(present) < 5:
            raise ValueError(f"only {len(present)} of 7 sections usable: {present}")
        return out

    prompt = render(
        "agent_generator",
        style_directive=f"## Drafting style\n{style_directive}",
        agent_name=reqs.agent_name,
        scope=reqs.scope,
        audience=reqs.audience,
        tone=reqs.tone,
        must_answer="; ".join(reqs.must_answer_topics) or "(none)",
        must_refuse="; ".join(reqs.must_refuse_topics) or "(none)",
        ungrounded="; ".join(reqs.ungrounded_topics) or "(none)",
        corpus_summary=corpus_summary(docs, chunks),
        tools_block=tools_block,
        max_words=contract.max_answer_words,
    )
    sections, source = client.structured(
        prompt,
        stage="generator.prompt",
        validate=validate,
        fallback=lambda: _fallback_sections(reqs, tools, contract),
        max_tokens=2000,
    )

    # Backfill any section the model skipped from the deterministic template, so
    # the shipped prompt is always complete.
    baseline = _fallback_sections(reqs, tools, contract)
    merged: dict[str, str] = {}
    backfilled: list[str] = []
    for key in PROMPT_SECTIONS:
        if sections.get(key):
            merged[key] = sections[key]
        else:
            merged[key] = baseline[key]
            backfilled.append(key)
    for key, body in sections.items():
        if key not in merged:
            merged[key] = body

    # The output contract is machine-enforced, so the instruction for it is
    # never left to the model's discretion.
    merged["output_format"] = baseline["output_format"]

    agent_id = re.sub(r"[^a-z0-9]+", "-", reqs.agent_name.lower()).strip("-") or "agent"
    definition = AgentDefinition(
        agent_id=f"{agent_id}-{style_name}",
        name=reqs.agent_name,
        description=f"{reqs.scope[:200]}",
        system_prompt=merged,
        retrieval=choose_retrieval(docs, chunks),
        tools=tools,
        output_contract=contract,
        tool_rationale=tool_rationale,
        source_documents=[d.name for d in docs],
        lineage=f"initial (style: {style_name}, prompt source: {source})",
        changelog=[
            {
                "event": "generated",
                "style": style_name,
                "prompt_source": source,
                "sections_backfilled": backfilled,
                "tool_decisions": tool_decisions,
            }
        ],
    )
    return definition
