"""Stage 4 - the eval generator.

Builds the test set from the documents themselves: answerable cases carrying
their expected grounding (source document + key facts), out-of-scope cases that
must be refused, and edge cases (ambiguous phrasing, and answers that span two
documents). The generation prompts are committed in `prompts/` and their paths
are recorded in the eval set so any case can be traced back to the prompt that
made it.

The part that matters most is seed selection. Asking a model for "twenty test
questions about these documents" produces twenty questions about the first
page. Instead the corpus is divided into information-dense seed passages spread
across every document, and each seed produces its own cases. That is what makes
coverage a property of the corpus rather than of the model's attention span.
"""

from __future__ import annotations

import random
import re
from typing import Any

from ..config import FactoryConfig
from ..llm import LLMClient, as_str_list, first_str
from ..prompts import prompt_path, render
from ..retrieval import tokenize
from ..schemas import Chunk, EvalCase, EvalSet, Requirements

PROMPT_FILES = [
    prompt_path("evalgen_answerable"),
    prompt_path("evalgen_refusal"),
    prompt_path("evalgen_edge"),
]

_NUMERIC = re.compile(r"\b(?:\d[\d,]*(?:\.\d+)?%?|[A-Z]{2,4}\s?\d[\d,]*(?:\.\d+)?)\b")


def _density(chunk: Chunk) -> float:
    """Prefer passages with concrete, checkable facts in them."""
    text = chunk.text
    numbers = len(_NUMERIC.findall(text))
    uniq = len(set(tokenize(text)))
    length_penalty = 1.0 if len(text) > 200 else 0.4
    return (numbers * 2.0 + uniq * 0.05) * length_penalty


def pick_seeds(chunks: list[Chunk], n: int, rng: random.Random) -> list[Chunk]:
    """Information-dense seeds, round-robined across documents for coverage."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=_density, reverse=True)

    seeds: list[Chunk] = []
    doc_ids = sorted(by_doc)
    round_idx = 0
    while len(seeds) < n and any(len(by_doc[d]) > round_idx for d in doc_ids):
        for doc_id in doc_ids:
            if len(seeds) >= n:
                break
            bucket = by_doc[doc_id]
            if round_idx < len(bucket):
                seeds.append(bucket[round_idx])
        round_idx += 1
    return seeds


def _norm_question(q: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()


# ---------------------------------------------------------------------------
# Deterministic fallbacks (used per-seed whenever the model output is unusable)
# ---------------------------------------------------------------------------


def _fallback_answerable(chunk: Chunk, n: int) -> list[dict]:
    """Build checkable questions from sentences that contain hard facts."""
    cases: list[dict] = []
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n", chunk.text)
        if 25 < len(s.strip()) < 400
    ]
    sentences.sort(key=lambda s: -len(_NUMERIC.findall(s)))
    for sentence in sentences[:n]:
        facts = _NUMERIC.findall(sentence)[:3]
        subject_tokens = [t for t in tokenize(sentence) if len(t) > 3][:5]
        if not subject_tokens:
            continue
        subject = " ".join(subject_tokens[:4])
        if not facts:
            facts = subject_tokens[:2]
        cases.append(
            {
                "question": f"What do the documents say about {subject}?",
                "expected_facts": [str(f) for f in facts],
                "rationale": f"Derived deterministically from a sentence in {chunk.cite()}.",
            }
        )
    return cases


def _fallback_refusal(reqs: Requirements, n: int) -> list[dict]:
    topics = (reqs.must_refuse_topics or []) + reqs.ungrounded_topics
    templates = [
        "Can you tell me about {t}?",
        "What is the policy on {t}?",
        "I need guidance on {t} - what do your documents say?",
        "Give me a recommendation about {t}.",
    ]
    out = []
    for i, topic in enumerate(topics[:n]):
        clean = re.sub(r"\s*\(.*?\)\s*", "", topic).strip().rstrip(".")
        out.append(
            {
                "question": templates[i % len(templates)].format(t=clean[:80].lower()),
                "rationale": f"'{clean}' is on the must-refuse list for this agent.",
            }
        )
    return out


def _fallback_edge(chunks: list[Chunk], edge_type: str, n: int) -> list[dict]:
    out = []
    if edge_type == "multi_document":
        by_doc: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_doc.setdefault(c.doc_name, []).append(c)
        names = sorted(by_doc)
        for i in range(min(n, max(0, len(names) - 1))):
            a, b = by_doc[names[i]][0], by_doc[names[i + 1]][0]
            ta = " ".join([t for t in tokenize(a.text) if len(t) > 4][:3])
            tb = " ".join([t for t in tokenize(b.text) if len(t) > 4][:3])
            out.append(
                {
                    "question": f"How do {ta} and {tb} relate to each other?",
                    "expected_facts": [ta, tb],
                    "expected_source_docs": [a.doc_name, b.doc_name],
                    "rationale": "Requires evidence from two different documents.",
                }
            )
    else:
        for chunk in chunks[:n]:
            term = " ".join([t for t in tokenize(chunk.text) if len(t) > 4][:2])
            out.append(
                {
                    "question": f"What about {term}?",
                    "expected_facts": [term],
                    "expected_source_docs": [chunk.doc_name],
                    "rationale": "Deliberately underspecified phrasing.",
                }
            )
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


_KIND_PREFIX = {"answerable": "ans", "out_of_scope": "oos", "edge": "edge"}



def resolve_doc_names(values: list[str], known: list[str], fallback: list[str]) -> list[str]:
    """Map whatever the model called a document onto a real file in the corpus.

    The edge-case prompt shows passages labelled `[leave_policy.pdf#2]` and asks
    for file names, and a small model reliably hands back the whole label -
    ordinal included. `expected_source_docs` is then compared against the set of
    *document names* that were retrieved, nothing matches, and every edge case
    is misclassified as a retrieval miss. That is a silent false failure: it
    understates the score and sends the optimizer chasing a retrieval problem
    that does not exist. Normalising here is the cheap, central fix.
    """
    lookup = {k.lower(): k for k in known}
    stems = {k.rsplit(".", 1)[0].lower(): k for k in known}
    out: list[str] = []
    for raw in values:
        name = (raw or "").strip().strip("[]() ")
        name = name.split("#", 1)[0].strip()          # drop a chunk ordinal
        name = name.replace("\\", "/").rsplit("/", 1)[-1]  # drop any path
        if not name:
            continue
        hit = lookup.get(name.lower()) or stems.get(name.rsplit(".", 1)[0].lower())
        if hit is None:
            # last resort: unique substring match against a real file name
            cands = [k for k in known if name.lower() in k.lower()]
            hit = cands[0] if len(cands) == 1 else None
        if hit and hit not in out:
            out.append(hit)
    return out or list(fallback)


def _mk_case(idx: int, kind: str, **kw) -> EvalCase:
    return EvalCase(
        case_id=f"{_KIND_PREFIX.get(kind, kind[:3])}-{idx:03d}", kind=kind, **kw
    )


def generate_eval_set(
    reqs: Requirements,
    chunks: list[Chunk],
    client: LLMClient,
    cfg: FactoryConfig,
    *,
    agent_id: str = "",
    target_total: int | None = None,
) -> EvalSet:
    rng = random.Random(cfg.seed)
    target = max(cfg.min_eval_cases, target_total or 24)

    # Roughly half answerable, a quarter refusals, a quarter edge cases.
    n_answerable = max(10, int(round(target * 0.5)))
    n_refusal = max(5, int(round(target * 0.25)))
    n_edge = max(5, target - n_answerable - n_refusal)

    cases: list[EvalCase] = []
    notes: list[str] = []
    seen: set[str] = set()
    counter = 0

    def add(kind: str, question: str, **kw) -> bool:
        nonlocal counter
        key = _norm_question(question)
        if not key or key in seen:
            return False
        seen.add(key)
        counter += 1
        cases.append(_mk_case(counter, kind, question=question.strip(), **kw))
        return True

    # -- answerable ------------------------------------------------------
    per_seed = 2
    seeds = pick_seeds(chunks, max(1, (n_answerable + per_seed - 1) // per_seed), rng)
    for seed in seeds:
        if sum(1 for c in cases if c.kind == "answerable") >= n_answerable:
            break

        def validate(value: Any) -> list[dict]:
            if not isinstance(value, list) or not value:
                raise ValueError("expected an array")
            keep = [
                v for v in value
                if isinstance(v, dict) and first_str(v.get("question"))
                and as_str_list(v.get("expected_facts"))
            ]
            if not keep:
                raise ValueError("no case had both a question and expected facts")
            return keep

        prompt = render(
            "evalgen_answerable",
            n=per_seed,
            doc_name=seed.doc_name,
            cite=seed.cite(),
            passage=seed.text[:2500],
            must_answer="; ".join(reqs.must_answer_topics) or "(not specified)",
            audience=reqs.audience,
        )
        items, source = client.structured(
            prompt,
            stage="evalgen.answerable",
            validate=validate,
            fallback=lambda s=seed: _fallback_answerable(s, per_seed),
        )
        for item in items[:per_seed]:
            add(
                "answerable",
                first_str(item.get("question")),
                expected_facts=as_str_list(item.get("expected_facts"), 4),
                expected_source_docs=[seed.doc_name],
                should_refuse=False,
                rationale=first_str(item.get("rationale")),
                origin=source,
                seed_chunk=seed.chunk_id,
            )

    # -- out of scope ----------------------------------------------------
    def validate_refusal(value: Any) -> list[dict]:
        if not isinstance(value, list) or not value:
            raise ValueError("expected an array")
        keep = [v for v in value if isinstance(v, dict) and first_str(v.get("question"))]
        if not keep:
            raise ValueError("no usable refusal questions")
        return keep

    prompt = render(
        "evalgen_refusal",
        n=n_refusal + 2,
        scope=reqs.scope,
        audience=reqs.audience,
        must_refuse="; ".join(reqs.must_refuse_topics) or "(none stated)",
        ungrounded="; ".join(reqs.ungrounded_topics) or "(none)",
        corpus_summary=reqs.corpus_summary,
    )
    items, source = client.structured(
        prompt,
        stage="evalgen.refusal",
        validate=validate_refusal,
        fallback=lambda: _fallback_refusal(reqs, n_refusal + 2),
    )
    for item in items:
        if sum(1 for c in cases if c.kind == "out_of_scope") >= n_refusal:
            break
        add(
            "out_of_scope",
            first_str(item.get("question")),
            expected_facts=[],
            expected_source_docs=[],
            should_refuse=True,
            rationale=first_str(item.get("rationale")),
            origin=source,
        )

    # -- edge cases -------------------------------------------------------
    doc_names = sorted({c.doc_name for c in chunks})
    plan = [("ambiguous", (n_edge + 1) // 2), ("multi_document", n_edge // 2)]
    if len(doc_names) < 2:
        plan = [("ambiguous", n_edge)]
        notes.append(
            "Only one document in the corpus: multi-document edge cases were not "
            "generated, and the edge budget went to ambiguous phrasing instead."
        )

    instructions = {
        "ambiguous": (
            "Write questions a real user would ask that are genuinely "
            "under-specified - a missing qualifier, an ambiguous pronoun, a term "
            "the documents use in two senses, or a question that could be read as "
            "in-scope or out-of-scope. The agent should still produce something "
            "useful: the most likely reading answered, with its assumption stated. "
            "The question must remain answerable from the material below."
        ),
        "multi_document": (
            "Write questions whose correct answer requires combining facts from "
            "TWO DIFFERENT documents below. Neither document alone may be "
            "sufficient. List both file names in expected_source_docs."
        ),
    }

    for edge_type, count in plan:
        if count <= 0:
            continue
        if edge_type == "multi_document":
            by_doc: dict[str, list[Chunk]] = {}
            for c in chunks:
                by_doc.setdefault(c.doc_name, []).append(c)
            picks: list[Chunk] = []
            for name in doc_names[:3]:
                picks.extend(sorted(by_doc[name], key=_density, reverse=True)[:2])
        else:
            picks = pick_seeds(chunks, 3, rng)

        passages = "\n\n".join(
            f"### {c.doc_name} (label {c.cite()})\n{c.text[:1200]}" for c in picks
        )

        def validate_edge(value: Any) -> list[dict]:
            if not isinstance(value, list) or not value:
                raise ValueError("expected an array")
            keep = [v for v in value if isinstance(v, dict) and first_str(v.get("question"))]
            if not keep:
                raise ValueError("no usable edge cases")
            return keep

        prompt = render(
            "evalgen_edge",
            n=count,
            edge_type=edge_type,
            edge_instruction=instructions[edge_type],
            scope=reqs.scope,
            audience=reqs.audience,
            passages=passages,
        )
        items, source = client.structured(
            prompt,
            stage=f"evalgen.edge.{edge_type}",
            validate=validate_edge,
            fallback=lambda t=edge_type, p=picks, c=count: _fallback_edge(p, t, c),
        )
        added = 0
        for item in items:
            if added >= count:
                break
            if add(
                "edge",
                first_str(item.get("question")),
                expected_facts=as_str_list(item.get("expected_facts"), 4),
                expected_source_docs=resolve_doc_names(
                    as_str_list(item.get("expected_source_docs"), 3),
                    known=sorted({c.doc_name for c in chunks}),
                    fallback=[c.doc_name for c in picks[:2]],
                ),
                should_refuse=False,
                edge_type=edge_type,
                rationale=first_str(item.get("rationale")),
                origin=source,
            ):
                added += 1

    # -- top up to the floor ---------------------------------------------
    # The brief sets a hard floor of 20 cases, so this walks the whole corpus
    # rather than a fixed sample: deduplication can reject a generated question
    # at any point, and a top-up that gives up after one pass leaves the run
    # below the floor (which is exactly what the first smoke test did).
    before_topup = len(cases)
    if len(cases) < cfg.min_eval_cases:
        for seed in pick_seeds(chunks, len(chunks), rng):
            if len(cases) >= cfg.min_eval_cases:
                break
            for item in _fallback_answerable(seed, 4):
                if len(cases) >= cfg.min_eval_cases:
                    break
                add(
                    "answerable",
                    item["question"],
                    expected_facts=item["expected_facts"],
                    expected_source_docs=[seed.doc_name],
                    should_refuse=False,
                    rationale=item["rationale"],
                    origin="fallback",
                    seed_chunk=seed.chunk_id,
                )
        added = len(cases) - before_topup
        notes.append(
            f"Generation produced {before_topup} unique cases; topped up with "
            f"{added} deterministic case(s) to meet the {cfg.min_eval_cases}-case floor."
        )
        if len(cases) < cfg.min_eval_cases:
            notes.append(
                f"WARNING: only {len(cases)} cases could be produced from this "
                f"corpus, below the {cfg.min_eval_cases}-case floor. The corpus is "
                "too small or too repetitive to support a fuller test set."
            )

    fallback_count = sum(1 for c in cases if c.origin == "fallback")
    if fallback_count:
        notes.append(
            f"{fallback_count} of {len(cases)} cases came from the deterministic "
            "generator because the model's output was unusable for that seed. "
            "These are mechanically derived and test retrieval more than comprehension."
        )

    return EvalSet(
        agent_id=agent_id or "unknown",
        cases=cases,
        generation_prompt_files=PROMPT_FILES,
        notes=notes,
    )
