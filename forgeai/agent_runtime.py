"""The produced agent, running.

This loads any agent definition the factory emits and answers questions with
it. It is the engine behind both the chat runner (core requirement 8) and the
evaluation harness - the agent the judge scores is exactly the agent the
reviewer talks to, which is the only way the scorecard means anything.

The output contract is enforced here, in code, rather than hoped for:
- citations that name an excerpt the agent was not given are stripped and the
  contract is marked violated,
- over-long answers are flagged,
- a reply that is not JSON at all is salvaged into the contract shape.
A contract violation is recorded on the reply, not hidden, so it can be scored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .docs import chunk_documents, load_documents
from .llm import LLMClient, extract_json, first_str
from .retrieval import Hit, Retriever, format_context, tokenize
from .schemas import AgentDefinition, Chunk
from .tools import REGISTRY, execute
from .tools import csv_lookup

MAX_TOOL_ROUNDS = 2


@dataclass
class AgentReply:
    answer: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    retrieved: list[str] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    contract_ok: bool = True
    contract_notes: list[str] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "refused": self.refused,
            "retrieved": self.retrieved,
            "tool_calls": self.tool_calls,
            "contract_ok": self.contract_ok,
            "contract_notes": self.contract_notes,
        }


_REFUSAL_HINTS = (
    "i can't answer", "i cannot answer", "i can't help", "cannot help with",
    "don't have", "do not have", "not covered", "no information", "outside",
    "out of scope", "unable to", "not in the documents", "don't cover",
    "does not cover", "isn't covered", "not something i can",
)


class RunnableAgent:
    """Loads a definition + corpus and answers questions under its contract."""

    def __init__(
        self,
        definition: AgentDefinition,
        chunks: list[Chunk],
        client: LLMClient,
        *,
        corpus_dir: str | Path | None = None,
    ):
        self.definition = definition
        self.chunks = chunks
        self.client = client
        self.retriever = Retriever(chunks, definition.retrieval)
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        csv_lookup.set_corpus(self.corpus_dir)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_files(
        cls,
        definition_path: str | Path,
        corpus_dir: str | Path,
        client: LLMClient,
    ) -> "RunnableAgent":
        definition = AgentDefinition.load(definition_path)
        docs, _ = load_documents(corpus_dir)
        chunks = chunk_documents(
            docs,
            definition.retrieval.chunk_size,
            definition.retrieval.chunk_overlap,
        )
        return cls(definition, chunks, client, corpus_dir=corpus_dir)

    # -- prompting -------------------------------------------------------
    def _tool_instructions(self) -> str:
        if not self.definition.tools:
            return ""
        lines = ["", "## AVAILABLE TOOLS", "You may call a tool before answering:"]
        for name in self.definition.tools:
            tool = REGISTRY.get(name)
            if tool:
                lines.append(f"- {name}: {tool.description} Arguments: {tool.args_schema}")
        lines.append(
            'To call one, reply with exactly: {"tool_call": {"name": "<tool>", '
            '"arguments": {...}}} and nothing else. You will be given the result '
            "and may then answer. Call a tool only when the excerpts alone cannot "
            "answer the question."
        )
        return "\n".join(lines)

    def _build_user_message(self, question: str, hits: list[Hit], tool_log: list[dict]) -> str:
        parts = [
            "## RETRIEVED EXCERPTS",
            format_context(hits),
        ]
        if tool_log:
            parts += ["", "## TOOL RESULTS"]
            for call in tool_log:
                parts.append(
                    f"{call['name']}({json.dumps(call['arguments'])}) ->\n{call['result']}"
                )
        parts += [
            "",
            "## QUESTION",
            question,
            "",
            "Reply with the JSON object required by your output format.",
        ]
        return "\n".join(parts)

    # -- the no-model path ------------------------------------------------
    def _extractive_answer(self, question: str, hits: list[Hit]) -> AgentReply:
        """Deterministic extractive fallback used when no model is configured.

        It is a real (if blunt) agent: it refuses when retrieval is weak and
        otherwise returns the best-matching sentences with a correct citation.
        """
        contract = self.definition.output_contract
        if not hits or hits[0].score < max(self.definition.retrieval.min_score, 0.12):
            return AgentReply(
                answer=contract.refusal_message,
                refused=True,
                retrieved=[h.chunk_id for h in hits],
                retrieval_scores={h.chunk_id: round(h.score, 3) for h in hits},
            )
        q_tokens = set(tokenize(question))
        scored: list[tuple[float, str, str]] = []
        for hit in hits[:3]:
            for sentence in re.split(r"(?<=[.!?])\s+|\n", hit.chunk.text):
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                overlap = len(q_tokens & set(tokenize(sentence)))
                if overlap:
                    scored.append((overlap / (1 + len(sentence) / 400), sentence, hit.chunk.cite()))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            best = hits[0]
            scored = [(0.0, " ".join(best.chunk.text.split())[:400], best.chunk.cite())]
        picked, cites, words = [], [], 0
        for _, sentence, cite in scored[:4]:
            n = len(sentence.split())
            if words + n > contract.max_answer_words:
                break
            picked.append(sentence)
            words += n
            if cite not in cites:
                cites.append(cite)
        return AgentReply(
            answer=" ".join(picked),
            citations=cites,
            refused=False,
            retrieved=[h.chunk_id for h in hits],
            retrieval_scores={h.chunk_id: round(h.score, 3) for h in hits},
        )

    # -- contract enforcement ---------------------------------------------
    def _coerce(self, raw: str, hits: list[Hit]) -> AgentReply:
        contract = self.definition.output_contract
        notes: list[str] = []
        ok = True
        data: Any
        try:
            data = extract_json(raw)
            if not isinstance(data, dict):
                raise ValueError("not an object")
        except Exception:
            ok = False
            notes.append("reply was not valid JSON; salvaged as plain text")
            data = {"answer": raw.strip(), "citations": [], "refused": None}

        answer = first_str(data.get("answer"))
        if not answer:
            answer = first_str(data.get("response") or data.get("text") or raw)
            if answer:
                notes.append("'answer' field missing; recovered from another field")
                ok = False

        refused = data.get("refused")
        if not isinstance(refused, bool):
            low = answer.lower()
            refused = any(h in low for h in _REFUSAL_HINTS)
            notes.append("'refused' missing or not boolean; inferred from wording")
            ok = False

        raw_cites = data.get("citations")
        cites: list[str] = []
        if isinstance(raw_cites, str):
            raw_cites = [raw_cites]
        if isinstance(raw_cites, list):
            for c in raw_cites:
                label = first_str(c if not isinstance(c, dict) else
                                  (c.get("source") or c.get("citation") or c.get("id")))
                if label:
                    cites.append(label)

        valid_labels = {h.chunk.cite() for h in hits}
        valid_docs = {h.chunk.doc_name for h in hits}
        kept, dropped = [], []
        for c in cites:
            if c in valid_labels or c in valid_docs:
                kept.append(c)
            else:
                # Tolerate "handbook.pdf #2" / "[handbook.pdf#2]" style noise.
                cleaned = c.strip("[]() ").replace(" #", "#")
                if cleaned in valid_labels or cleaned in valid_docs:
                    kept.append(cleaned)
                else:
                    dropped.append(c)
        if dropped:
            ok = False
            notes.append(
                f"fabricated citation(s) removed: {dropped} (not among retrieved excerpts)"
            )

        if not refused and not kept:
            ok = False
            notes.append("answered without citing any retrieved excerpt")

        words = len(answer.split())
        if words > contract.max_answer_words:
            ok = False
            notes.append(f"answer is {words} words, over the {contract.max_answer_words} cap")

        return AgentReply(
            answer=answer,
            citations=list(dict.fromkeys(kept)),
            refused=bool(refused),
            retrieved=[h.chunk_id for h in hits],
            retrieval_scores={h.chunk_id: round(h.score, 3) for h in hits},
            contract_ok=ok,
            contract_notes=notes,
            raw=raw,
        )

    # -- main entry point --------------------------------------------------
    def answer(self, question: str) -> AgentReply:
        hits = self.retriever.retrieve(question)

        if not self.client.available:
            return self._extractive_answer(question, hits)

        system = self.definition.render_prompt() + self._tool_instructions()
        tool_log: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS + 1):
            user = self._build_user_message(question, hits, tool_log)
            raw = self.client.chat(user, system=system, stage="agent.answer")
            if not raw:
                return self._extractive_answer(question, hits)

            # Did it ask for a tool?
            call = None
            try:
                parsed = extract_json(raw)
                if isinstance(parsed, dict) and isinstance(parsed.get("tool_call"), dict):
                    call = parsed["tool_call"]
            except Exception:
                call = None

            if call and self.definition.tools:
                name = first_str(call.get("name"))
                args = call.get("arguments") or call.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                if name not in self.definition.tools:
                    result = (
                        f"ERROR: '{name}' is not attached to this agent. "
                        f"Available: {self.definition.tools}"
                    )
                else:
                    result = execute(name, args)
                tool_log.append({"name": name, "arguments": args, "result": result})
                continue

            reply = self._coerce(raw, hits)
            reply.tool_calls = tool_log
            return reply

        # Ran out of tool rounds without a final answer.
        reply = self._extractive_answer(question, hits)
        reply.tool_calls = tool_log
        reply.contract_ok = False
        reply.contract_notes.append("agent kept calling tools; fell back to extraction")
        return reply
