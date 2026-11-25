"""Every artifact the factory produces, as a dataclass that round-trips to JSON.

The one design decision worth calling out: `AgentDefinition.system_prompt` is
not a string. It is an ordered dict of *named sections*. The critic-optimizer
is required by the brief to apply "a targeted fix, not a full regeneration",
and you cannot target anything inside an opaque blob of text. Sections make a
fix addressable ("rewrite `refusal_policy`, leave the other six alone") and
make the diff in the iteration log readable by a human.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Order matters: this is the order sections are rendered into the final prompt.
PROMPT_SECTIONS = [
    "role",
    "scope",
    "grounding_rules",
    "refusal_policy",
    "tone",
    "output_format",
    "edge_cases",
]

FAILURE_CATEGORIES = [
    "retrieval_miss",
    "prompt_defect",
    "hallucination",
    "wrong_refusal",
]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False)


class JsonMixin:
    def to_dict(self) -> dict:
        return asdict(self)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@dataclass
class Document:
    doc_id: str
    name: str
    path: str
    kind: str           # pdf | csv | md | txt
    text: str
    n_chars: int = 0
    note: str = ""      # e.g. "scanned PDF, no extractable text"

    def __post_init__(self) -> None:
        self.n_chars = len(self.text)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    ordinal: int
    text: str

    def cite(self) -> str:
        return f"{self.doc_name}#{self.ordinal}"


# ---------------------------------------------------------------------------
# Stage 1-2: requirements + groundability
# ---------------------------------------------------------------------------


@dataclass
class Clarification:
    question: str
    answer: str
    answered_by: str  # "requester" | "auto (unattended)"


@dataclass
class GroundabilityFinding:
    topic: str
    status: str        # supported | partial | unsupported
    best_evidence: str
    score: float
    action: str        # what the factory did about it


@dataclass
class Requirements(JsonMixin):
    brief: str
    agent_name: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[str] = field(default_factory=list)
    must_refuse_topics: list[str] = field(default_factory=list)
    clarifications: list[Clarification] = field(default_factory=list)
    groundability: list[GroundabilityFinding] = field(default_factory=list)
    requested_capabilities: list[str] = field(default_factory=list)
    corpus_summary: str = ""
    warnings: list[str] = field(default_factory=list)
    approved_by_human: bool = False
    generated_by: str = ""

    @property
    def ungrounded_topics(self) -> list[str]:
        return [f.topic for f in self.groundability if f.status == "unsupported"]

    @classmethod
    def from_dict(cls, d: dict) -> "Requirements":
        d = dict(d)
        d["clarifications"] = [Clarification(**c) for c in d.get("clarifications", [])]
        d["groundability"] = [
            GroundabilityFinding(**g) for g in d.get("groundability", [])
        ]
        return cls(**d)

    def to_markdown(self) -> str:
        lines = [
            f"# Requirements - {self.agent_name}",
            "",
            "## Brief (verbatim)",
            f"> {self.brief.strip()}",
            "",
            "## Scope",
            self.scope,
            "",
            f"**Audience:** {self.audience}",
            "",
            f"**Tone:** {self.tone}",
            "",
            "## Must answer",
        ]
        lines += [f"- {t}" for t in self.must_answer_topics] or ["- (none captured)"]
        lines += ["", "## Must refuse"]
        lines += [f"- {t}" for t in self.must_refuse_topics] or ["- (none captured)"]

        if self.clarifications:
            lines += ["", "## Clarifying questions asked (max 5)"]
            for i, c in enumerate(self.clarifications, 1):
                lines += [
                    f"{i}. **{c.question}**",
                    f"   - _{c.answered_by}:_ {c.answer}",
                ]

        lines += ["", "## Groundability guard", ""]
        if self.groundability:
            lines += [
                "| Topic | Verdict | Evidence score | Best evidence | Action |",
                "|---|---|---|---|---|",
            ]
            for g in self.groundability:
                ev = g.best_evidence.replace("|", "/")[:70]
                lines.append(
                    f"| {g.topic} | **{g.status}** | {g.score:.2f} | {ev} | {g.action} |"
                )
        else:
            lines.append("_No topics to check._")

        if self.warnings:
            lines += ["", "## Warnings", ""]
            lines += [f"- :warning: {w}" for w in self.warnings]

        lines += [
            "",
            "## Corpus",
            self.corpus_summary,
            "",
            f"_Requirements produced by {self.generated_by}._",
            f"_Human approval: {'yes' if self.approved_by_human else 'not requested'}._",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 3: the agent definition
# ---------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    min_score: float = 0.05
    query_expansion: bool = False
    per_doc_cap: int = 0   # 0 = no cap; >0 forces spread across documents

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OutputContract:
    """What a well-formed answer from the produced agent must look like."""

    format: str = "json"
    required_fields: list[str] = field(
        default_factory=lambda: ["answer", "citations", "refused"]
    )
    citation_style: str = "document name + chunk ordinal, e.g. handbook.pdf#3"
    max_answer_words: int = 160
    refusal_message: str = (
        "I can't answer that from the documents I've been given."
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentDefinition(JsonMixin):
    agent_id: str
    name: str
    description: str
    system_prompt: dict[str, str]
    retrieval: RetrievalConfig
    tools: list[str]
    output_contract: OutputContract
    tool_rationale: dict[str, str] = field(default_factory=dict)
    version: int = 1
    lineage: str = "initial"            # or "iteration-2 patch: refusal_policy"
    source_documents: list[str] = field(default_factory=list)
    changelog: list[dict] = field(default_factory=list)

    def render_prompt(self) -> str:
        parts = []
        for key in PROMPT_SECTIONS:
            body = (self.system_prompt.get(key) or "").strip()
            if body:
                parts.append(f"## {key.replace('_', ' ').upper()}\n{body}")
        # Any non-standard sections the generator invented, appended at the end.
        for key, body in self.system_prompt.items():
            if key not in PROMPT_SECTIONS and body.strip():
                parts.append(f"## {key.replace('_', ' ').upper()}\n{body.strip()}")
        return "\n\n".join(parts)

    def fingerprint(self) -> str:
        blob = self.render_prompt() + _dumps(self.retrieval.to_dict()) + ",".join(
            sorted(self.tools)
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rendered_system_prompt"] = self.render_prompt()
        d["fingerprint"] = self.fingerprint()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AgentDefinition":
        d = dict(d)
        d.pop("rendered_system_prompt", None)
        d.pop("fingerprint", None)
        d["retrieval"] = RetrievalConfig(**d["retrieval"])
        d["output_contract"] = OutputContract(**d["output_contract"])
        return cls(**d)

    def copy(self) -> "AgentDefinition":
        return AgentDefinition.from_dict(json.loads(_dumps(self.to_dict())))


# ---------------------------------------------------------------------------
# Stage 4-5: evals and judging
# ---------------------------------------------------------------------------


@dataclass
class EvalCase:
    case_id: str
    kind: str                 # answerable | out_of_scope | edge
    question: str
    expected_facts: list[str] = field(default_factory=list)
    expected_source_docs: list[str] = field(default_factory=list)
    should_refuse: bool = False
    edge_type: str = ""       # ambiguous | multi_document | ""
    rationale: str = ""
    origin: str = "model"     # model | repair | fallback
    seed_chunk: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        return cls(**d)


@dataclass
class EvalSet(JsonMixin):
    agent_id: str
    cases: list[EvalCase] = field(default_factory=list)
    generation_prompt_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cases:
            out[c.kind] = out.get(c.kind, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "counts": self.counts(),
            "total": len(self.cases),
            "generation_prompt_files": self.generation_prompt_files,
            "notes": self.notes,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalSet":
        return cls(
            agent_id=d.get("agent_id", ""),
            cases=[EvalCase.from_dict(c) for c in d.get("cases", [])],
            generation_prompt_files=d.get("generation_prompt_files", []),
            notes=d.get("notes", []),
        )


@dataclass
class CaseResult:
    case_id: str
    kind: str
    question: str
    answer: str
    refused: bool
    citations: list[str]
    retrieved_chunks: list[str]
    gold_chunk_retrieved: bool
    grounded: float            # 0..1 deterministic groundedness
    correct: float             # 0..1 fact recall (deterministic + judge)
    refusal_correct: bool
    contract_ok: bool
    score: float
    passed: bool
    failure_category: str = ""
    judge_notes: str = ""
    judge_agreed_with_checks: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scorecard:
    iteration: int
    definition_fingerprint: str
    total: int
    passed: int
    score: float
    by_kind: dict[str, dict[str, Any]]
    failures_by_category: dict[str, int]
    judge_disagreements: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IterationLog:
    iteration: int
    score_before: float
    score_after: float
    failures_addressed: list[str]
    changes: list[dict]
    reasoning: str
    accepted: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
