"""Stage 1 - the intake interviewer.

Reads the brief, inspects the documents, asks at most five clarifying
questions, and writes the structured requirements document.

Two modes. `--interactive` puts the questions to a human at the CLI, which is
also where the optional human checkpoint on the finished requirements lives.
Unattended (the default, and what happens at the review) the factory answers
its own questions from the brief and the document evidence, and every answer is
recorded with `answered_by: "auto (unattended)"` so nobody mistakes a factory
assumption for something the requester said.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import FactoryConfig
from ..docs import corpus_summary, preview_corpus
from ..llm import LLMClient, as_str_list, first_str
from ..prompts import render
from ..retrieval import Retriever, tokenize
from ..schemas import Chunk, Clarification, Document, Requirements

# Used by the deterministic fallback to find candidate topics in the corpus.
_HEADING = re.compile(
    r"^(?:#{1,4}\s+(?P<md>.+)"
    r"|(?P<num>\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Z][^.]{3,60})"
    r"|(?P<caps>[A-Z][A-Z \-/&]{5,60}))\s*$"
)

_GENERIC_HEADINGS = {
    "scope", "purpose", "introduction", "general", "overview", "definitions",
    "contents", "appendix", "background", "summary", "policy", "application",
    "purpose and scope", "other", "other benefits", "disputes and liability",
}

_FALLBACK_QUESTIONS = [
    {
        "question": "Who exactly will use this assistant, and is there anyone who must NOT see parts of this material?",
        "why": "Sets the audience and the confidentiality boundary.",
        "default_if_unanswered": "The audience named in the brief, with no privileged material.",
    },
    {
        "question": "Which closely-related topics should the assistant refuse rather than attempt?",
        "why": "Defines the refusal policy and the out-of-scope eval cases.",
        "default_if_unanswered": "Refuse anything outside the supplied documents, and all individual advice.",
    },
    {
        "question": "When the documents do not contain an answer, should the assistant say so plainly or point somewhere else?",
        "why": "Determines the refusal wording.",
        "default_if_unanswered": "Say plainly that the documents do not cover it.",
    },
    {
        "question": "How long should a typical answer be, and how formal?",
        "why": "Sets tone and the answer length cap in the output contract.",
        "default_if_unanswered": "Short, plain, professional; under 160 words.",
    },
    {
        "question": "Does any required answer depend on today's date, arithmetic, or a lookup in a table?",
        "why": "Decides which tools, if any, get attached.",
        "default_if_unanswered": "Infer tool needs from the brief and the document types.",
    },
]


def _candidate_topics(docs: list[Document], limit: int = 8) -> list[str]:
    """Pull plausible topic names out of document structure, no model needed."""
    found: list[str] = []
    for doc in docs:
        for line in doc.text.splitlines():
            line = line.strip()
            if not (6 <= len(line) <= 70):
                continue
            m = _HEADING.match(line)
            if not m:
                continue
            text = (m.group("md") or m.group("num") or m.group("caps") or "").strip()
            text = re.sub(r"^\d{1,2}(\.\d{1,2})?\.?\s*", "", text).strip(" :-")
            if len(text) < 4 or text.lower().startswith("page "):
                continue
            # Structural headings are not topics a user asks about.
            if text.lower() in _GENERIC_HEADINGS:
                continue
            found.append(text if text.isupper() is False else text.title())
    seen: set[str] = set()
    out: list[str] = []
    for f in found:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            out.append(f)
    if len(out) < 4:  # structureless corpus: fall back to salient terms
        counts: dict[str, int] = {}
        for doc in docs:
            for tok in tokenize(doc.text):
                if len(tok) > 4:
                    counts[tok] = counts.get(tok, 0) + 1
        out += [
            t.replace("_", " ").title()
            for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:6]
        ]
    return out[:limit]


def _ask_questions(
    brief: str, docs: list[Document], chunks: list[Chunk], client: LLMClient, cfg: FactoryConfig
) -> list[dict]:
    def fallback() -> list[dict]:
        return _FALLBACK_QUESTIONS[: cfg.max_clarifying_questions]

    def validate(value: Any) -> list[dict]:
        if not isinstance(value, list) or not value:
            raise ValueError("expected a non-empty array of questions")
        out = []
        for item in value[: cfg.max_clarifying_questions]:
            if isinstance(item, str):
                item = {"question": item}
            if not isinstance(item, dict):
                continue
            q = first_str(item.get("question"))
            if not q:
                continue
            out.append(
                {
                    "question": q,
                    "why": first_str(item.get("why")),
                    "default_if_unanswered": first_str(
                        item.get("default_if_unanswered"), "Not specified."
                    ),
                }
            )
        if not out:
            raise ValueError("no usable questions")
        return out

    prompt = render(
        "intake_questions",
        brief=brief,
        max_questions=cfg.max_clarifying_questions,
        corpus_summary=corpus_summary(docs, chunks),
        corpus_preview=preview_corpus(docs, 900),
    )
    value, _ = client.structured(
        prompt, stage="intake.questions", validate=validate, fallback=fallback
    )
    return value[: cfg.max_clarifying_questions]


def _collect_answers(
    questions: list[dict],
    brief: str,
    docs: list[Document],
    chunks: list[Chunk],
    client: LLMClient,
    *,
    interactive: bool,
) -> list[Clarification]:
    if interactive:
        print("\n" + "=" * 70)
        print("ForgeAI intake - please answer these (blank accepts the default)")
        print("=" * 70)
        out: list[Clarification] = []
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {q['question']}")
            if q.get("why"):
                print(f"      (why: {q['why']})")
            print(f"      (default: {q['default_if_unanswered']})")
            try:
                reply = input("      > ").strip()
            except EOFError:
                reply = ""
            if reply:
                out.append(Clarification(q["question"], reply, "requester"))
            else:
                out.append(
                    Clarification(
                        q["question"],
                        q["default_if_unanswered"],
                        "default (requester skipped)",
                    )
                )
        return out

    # Unattended: the factory answers its own questions from the evidence.
    questions_block = "\n".join(
        f"{i}. {q['question']}\n   (default if unknown: {q['default_if_unanswered']})"
        for i, q in enumerate(questions, 1)
    )

    def fallback() -> list[dict]:
        return [
            {"question": q["question"], "answer": q["default_if_unanswered"]}
            for q in questions
        ]

    def validate(value: Any) -> list[dict]:
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError("expected an array of answers")
        return value

    prompt = render(
        "intake_auto_answer",
        brief=brief,
        corpus_summary=corpus_summary(docs, chunks),
        corpus_preview=preview_corpus(docs, 900),
        questions_block=questions_block,
    )
    answers, _ = client.structured(
        prompt, stage="intake.auto_answer", validate=validate, fallback=fallback
    )

    out = []
    for i, q in enumerate(questions):
        answer = q["default_if_unanswered"]
        if i < len(answers) and isinstance(answers[i], dict):
            answer = first_str(answers[i].get("answer"), answer) or answer
        out.append(Clarification(q["question"], answer, "auto (unattended)"))
    return out


def _requirements_fallback(
    brief: str, docs: list[Document], chunks: list[Chunk], clarifications: list[Clarification]
) -> dict:
    topics = _candidate_topics(docs)
    skip = {"assistant", "agent", "answer", "question", "about", "would", "their",
            "should", "people", "thing", "which", "needs", "staff"}
    name_seed = next(
        (t for t in tokenize(brief) if len(t) > 4 and t not in skip),
        docs[0].doc_id.replace("_", " ") if docs else "document",
    )
    return {
        "agent_name": f"{name_seed.title()} Assistant",
        "scope": (
            f"Answers questions that the supplied documents support, on: "
            f"{', '.join(topics[:5]) or 'the contents of the corpus'}. "
            "Anything outside these documents is out of scope."
        ),
        "audience": "The audience named in the brief.",
        "tone": "Short, plain, professional.",
        "must_answer_topics": topics[:6],
        "must_refuse_topics": [
            "Any topic the supplied documents do not cover",
            "Individual legal, medical or financial advice",
            "Decisions about a specific named person's case",
            "Information about other organisations or systems",
        ],
        "requested_capabilities": [],
    }


def run_intake(
    brief: str,
    docs: list[Document],
    chunks: list[Chunk],
    retriever: Retriever,
    client: LLMClient,
    cfg: FactoryConfig,
    *,
    interactive: bool = False,
    warnings: list[str] | None = None,
) -> Requirements:
    questions = _ask_questions(brief, docs, chunks, client, cfg)
    clarifications = _collect_answers(
        questions, brief, docs, chunks, client, interactive=interactive
    )

    qa_block = "\n".join(
        f"Q{i}: {c.question}\nA{i} ({c.answered_by}): {c.answer}"
        for i, c in enumerate(clarifications, 1)
    )

    def validate(value: Any) -> dict:
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        if not as_str_list(value.get("must_answer_topics")):
            raise ValueError("must_answer_topics is empty")
        return value

    prompt = render(
        "intake_requirements",
        brief=brief,
        corpus_summary=corpus_summary(docs, chunks),
        corpus_preview=preview_corpus(docs, 1200),
        qa_block=qa_block or "(none)",
    )
    data, source = client.structured(
        prompt,
        stage="intake.requirements",
        validate=validate,
        fallback=lambda: _requirements_fallback(brief, docs, chunks, clarifications),
    )

    reqs = Requirements(
        brief=brief.strip(),
        agent_name=first_str(data.get("agent_name"), "Document Assistant") or "Document Assistant",
        scope=first_str(data.get("scope"), brief.strip()),
        audience=first_str(data.get("audience"), "Not specified"),
        tone=first_str(data.get("tone"), "Short, plain, professional"),
        must_answer_topics=as_str_list(data.get("must_answer_topics"), 8),
        must_refuse_topics=as_str_list(data.get("must_refuse_topics"), 8),
        requested_capabilities=as_str_list(data.get("requested_capabilities"), 6),
        clarifications=clarifications,
        corpus_summary=corpus_summary(docs, chunks),
        warnings=list(warnings or []),
        generated_by=f"{client.cfg.describe()} (requirements source: {source})",
    )
    if not reqs.must_refuse_topics:
        reqs.must_refuse_topics = [
            "Any topic the supplied documents do not cover",
            "Individual legal, medical or financial advice",
        ]
    return reqs


def human_checkpoint(reqs: Requirements) -> Requirements:
    """Stretch goal: let the requester approve or edit the requirements."""
    print("\n" + "=" * 70)
    print("HUMAN CHECKPOINT - review the requirements before generation")
    print("=" * 70)
    print(reqs.to_markdown())
    print("\n" + "-" * 70)
    print("[a]pprove   [s]cope   [t]opics (must-answer)   [r]efuse topics   [q]uit")
    while True:
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            choice = "a"
        if choice in {"a", ""}:
            reqs.approved_by_human = True
            print("Approved.")
            return reqs
        if choice == "q":
            raise SystemExit("Aborted at the human checkpoint.")
        if choice == "s":
            new = input("New scope: ").strip()
            if new:
                reqs.scope = new
        elif choice == "t":
            new = input("Must-answer topics (semicolon separated): ").strip()
            if new:
                reqs.must_answer_topics = [t.strip() for t in new.split(";") if t.strip()]
        elif choice == "r":
            new = input("Must-refuse topics (semicolon separated): ").strip()
            if new:
                reqs.must_refuse_topics = [t.strip() for t in new.split(";") if t.strip()]
        else:
            print("Unrecognised option.")
            continue
        print("\nUpdated. [a]pprove to continue, or edit again.")
