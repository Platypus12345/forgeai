"""Tests for the parts of ForgeAI that must not quietly break.

These deliberately concentrate on the *deterministic* machinery - JSON
salvage, chunking, retrieval, the tool veto, contract enforcement, failure
classification - because that is the layer the factory leans on when the small
model misbehaves, and it is the layer whose bugs would silently inflate a
score rather than crash a run.

Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forgeai.agent_runtime import AgentReply, RunnableAgent
from forgeai.config import BudgetConfig, FactoryConfig, ModelConfig
from forgeai.docs import chunk_documents, load_documents
from forgeai.llm import Ledger, LLMClient, as_str_list, extract_json
from forgeai.retrieval import Retriever
from forgeai.schemas import (
    AgentDefinition,
    Chunk,
    Document,
    EvalCase,
    OutputContract,
    RetrievalConfig,
)
from forgeai.stages import critic
from forgeai.stages.judge import (
    fact_matched_deterministically,
    groundedness,
    score_case,
)
from forgeai.tools import calculator, propose_from_signals, vet_selection
from forgeai.tools import csv_lookup

REPO = Path(__file__).resolve().parent.parent
HR_DOCS = REPO / "demos" / "hr_policy" / "docs"
SPARES_DOCS = REPO / "demos" / "ship_spares" / "docs"


def _client() -> LLMClient:
    """A client with no model: every structured call takes its fallback."""
    return LLMClient(ModelConfig(provider="heuristic"), Ledger(BudgetConfig()))


# ---------------------------------------------------------------------------
# JSON salvage
# ---------------------------------------------------------------------------


class TestJsonSalvage:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_with_prose_and_trailing_comma(self):
        raw = 'Sure!\n```json\n{"a": [1, 2,], "b": "x"}\n```\nHope that helps.'
        assert extract_json(raw) == {"a": [1, 2], "b": "x"}

    def test_brace_inside_string_does_not_confuse_the_scanner(self):
        assert extract_json('noise {"k": "}"} tail') == {"k": "}"}

    def test_array_at_top_level(self):
        assert extract_json("here: [1, 2, 3]") == [1, 2, 3]

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            extract_json("there is no json here at all")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            extract_json("")

    def test_as_str_list_normalises_shapes(self):
        assert as_str_list([{"topic": "Leave"}, "Pay", {"name": "Pay"}]) == ["Leave", "Pay"]


# ---------------------------------------------------------------------------
# Documents and chunking
# ---------------------------------------------------------------------------


class TestDocuments:
    def test_loads_pdfs(self):
        docs, warnings = load_documents(HR_DOCS)
        assert len(docs) == 3
        assert all(d.kind == "pdf" for d in docs)
        assert all(d.n_chars > 500 for d in docs)

    def test_csv_chunks_repeat_the_header(self):
        """Every CSV chunk must carry the header or its columns are meaningless."""
        docs, _ = load_documents(SPARES_DOCS)
        csv_docs = [d for d in docs if d.kind == "csv"]
        assert csv_docs, "the spares corpus should contain a CSV"
        chunks = chunk_documents(csv_docs, 600, 0)
        assert len(chunks) > 1, "the catalogue should split into several chunks"
        for chunk in chunks:
            assert chunk.text.splitlines()[0].startswith("part_no")

    def test_unsupported_files_are_warned_not_fatal(self, tmp_path):
        (tmp_path / "notes.txt").write_text("Annual leave is 25 days.", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        docs, warnings = load_documents(tmp_path)
        assert len(docs) == 1
        assert any("image.png" in w for w in warnings)

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nope")

    def test_overlap_is_clamped_below_chunk_size(self):
        docs, _ = load_documents(HR_DOCS)
        chunks = chunk_documents(docs, 400, 9999)
        assert chunks and all(len(c.text) < 3000 for c in chunks)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestRetrieval:
    def _retriever(self, top_k: int = 4) -> Retriever:
        docs, _ = load_documents(HR_DOCS)
        chunks = chunk_documents(docs, 900, 150)
        return Retriever(chunks, RetrievalConfig(top_k=top_k, min_score=0.0))

    def test_finds_the_right_document(self):
        hits = self._retriever().retrieve("How many days of annual leave do I get?")
        assert hits
        assert hits[0].chunk.doc_name == "leave_policy.pdf"
        assert "25 days" in hits[0].chunk.text

    def test_respects_top_k(self):
        assert len(self._retriever(top_k=2).retrieve("pension contribution")) <= 2

    def test_per_doc_cap_spreads_across_documents(self):
        docs, _ = load_documents(HR_DOCS)
        chunks = chunk_documents(docs, 900, 150)
        r = Retriever(chunks, RetrievalConfig(top_k=6, min_score=0.0, per_doc_cap=1))
        hits = r.retrieve("notice period leave entitlement pay")
        names = [h.chunk.doc_name for h in hits]
        assert len(names) == len(set(names)), "per_doc_cap=1 must not repeat a document"

    def test_evidence_bundle_returns_several_readable_passages(self):
        """The groundability guard must judge on more than one truncated snippet.

        Judging a topic on a single 160-character snippet wrongly marked
        "Annual Leave Entitlement" unsupported on the HR corpus, because the
        best lexical match was the policy's title page.
        """
        r = self._retriever()
        score, bundle = r.evidence_bundle("annual leave entitlement", k=3, chars=700)
        assert score > 0
        assert bundle.count("[") >= 2, "expected several cited passages"
        assert len(bundle) > 400, "passages must be long enough to judge"

    def test_evidence_bundle_does_not_silently_cut_a_table_row(self):
        """A live run marked part AX-3310 ungroundable because the catalogue row
        containing it sat past a 700-character cut in the evidence shown to the
        guard. The model judged correctly on evidence the pipeline had mangled."""
        from forgeai.docs import load_documents as _load

        docs, _ = _load(SPARES_DOCS)
        chunks = chunk_documents(docs, 900, 150)
        r = Retriever(chunks, RetrievalConfig(top_k=4, min_score=0.0))
        _, bundle = r.evidence_bundle(
            "How many units of AX-3310 are currently in stock?", k=3
        )
        assert "AX-3310" in bundle, "the part must be visible in its own evidence"
        assert "stock_qty" in bundle, "the column that answers the topic must be visible"

    def test_truncated_evidence_announces_itself(self):
        """Absence of evidence must not read as evidence of absence."""
        r = self._retriever()
        _, bundle = r.evidence_bundle("annual leave entitlement", k=1, chars=120)
        assert "truncated" in bundle

    def test_evidence_bundle_empty_for_absent_topic(self):
        r = self._retriever()
        score, bundle = r.evidence_bundle("quantum chromodynamics lattice gauge")
        assert score == 0.0 and bundle == ""

    def test_evidence_score_discriminates(self):
        r = self._retriever()
        good, _ = r.evidence_score("annual leave entitlement")
        bad, _ = r.evidence_score("shipboard rotation sea service revalidation")
        assert good > bad


# ---------------------------------------------------------------------------
# Tool registry and the selection veto
# ---------------------------------------------------------------------------


class TestTools:
    HR_BRIEF = "answers HR policy questions for shore staff about leave and conduct"
    SPARES_BRIEF = "look up catalogue prices, compute order totals and delivery dates"

    def test_hr_agent_never_gets_a_weather_tool(self):
        accepted, _, decisions = vet_selection(
            ["weather", "csv_lookup", "calculator", "current_date"],
            brief_text=self.HR_BRIEF,
            corpus_kinds={"pdf"},
        )
        assert "weather" not in accepted
        assert any("weather" in d and "rejected" in d for d in decisions)

    def test_csv_tool_requires_a_csv_in_the_corpus(self):
        accepted, _, _ = vet_selection(
            ["csv_lookup"], brief_text=self.SPARES_BRIEF, corpus_kinds={"pdf"}
        )
        assert accepted == []

    def test_csv_tool_accepted_when_corpus_has_one(self):
        accepted, _, _ = vet_selection(
            ["csv_lookup"], brief_text=self.SPARES_BRIEF, corpus_kinds={"csv", "pdf"}
        )
        assert accepted == ["csv_lookup"]

    def test_unknown_tool_is_rejected(self):
        accepted, _, _ = vet_selection(
            ["definitely_not_a_tool"], brief_text=self.SPARES_BRIEF, corpus_kinds={"csv"}
        )
        assert accepted == []

    def test_deterministic_proposal_differs_by_brief(self):
        hr = set(propose_from_signals(self.HR_BRIEF, {"pdf"}))
        spares = set(propose_from_signals(self.SPARES_BRIEF, {"csv", "pdf"}))
        assert "weather" not in hr and "weather" not in spares
        assert "csv_lookup" in spares and "csv_lookup" not in hr

    def test_calculator_evaluates(self):
        assert calculator.run("2.5 * 12 - 4").endswith("= 26")
        assert calculator.run("1245.00 * 2").endswith("= 2490")

    def test_calculator_refuses_code(self):
        assert calculator.run("__import__('os').system('echo hi')").startswith("ERROR")
        assert calculator.run("open('x')").startswith("ERROR")

    def test_calculator_handles_division_by_zero(self):
        assert "ERROR" in calculator.run("1/0")

    def test_csv_lookup_finds_a_row(self):
        csv_lookup.set_corpus(SPARES_DOCS)
        out = csv_lookup.run(file="spares_catalogue.csv", query="MP-2210")
        assert "18750.00" in out and "Kobe Turbo Systems" in out

    def test_csv_lookup_reports_no_match_cleanly(self):
        csv_lookup.set_corpus(SPARES_DOCS)
        assert "No rows" in csv_lookup.run(file="spares_catalogue.csv", query="ZZ-0000")


# ---------------------------------------------------------------------------
# Judge internals
# ---------------------------------------------------------------------------


class TestJudgeChecks:
    def test_numeric_fact_requires_the_number(self):
        assert fact_matched_deterministically("25 days", "You get 25 days of leave.")
        assert not fact_matched_deterministically("25 days", "You get 20 days of leave.")

    def test_numeric_fact_tolerates_thousands_separators(self):
        assert fact_matched_deterministically("18750.00", "The price is 18,750.00 USD.")

    def test_textual_fact_allows_partial_overlap(self):
        assert fact_matched_deterministically(
            "medical certificate", "A medical certificate is required."
        )
        assert not fact_matched_deterministically(
            "medical certificate", "Call your line manager before 09:30."
        )

    def test_groundedness_punishes_invented_numbers(self):
        context = "Full-time staff accrue 25 days of paid annual leave per year."
        honest = groundedness("You accrue 25 days of annual leave.", context)
        invented = groundedness("You accrue 47 days of annual leave.", context)
        assert honest > invented
        assert invented < 0.6

    def test_groundedness_of_unrelated_answer_is_low(self):
        context = "Full-time staff accrue 25 days of paid annual leave."
        assert groundedness("The cafeteria closes at two o'clock.", context) < 0.5


class TestEvalDocNameResolution:
    """A real run misclassified all 6 edge cases as retrieval misses because the
    model returned chunk labels where document names were expected."""

    KNOWN = ["leave_policy.pdf", "pay_and_benefits.pdf", "workplace_conduct.pdf"]

    def test_chunk_labels_are_reduced_to_file_names(self):
        from forgeai.stages.evalgen import resolve_doc_names

        out = resolve_doc_names(
            ["leave_policy.pdf#2", "pay_and_benefits.pdf#3"], self.KNOWN, ["x.pdf"]
        )
        assert out == ["leave_policy.pdf", "pay_and_benefits.pdf"]

    def test_paths_brackets_and_stems_all_resolve(self):
        from forgeai.stages.evalgen import resolve_doc_names

        assert resolve_doc_names(["[docs/leave_policy.pdf#1]"], self.KNOWN, []) == [
            "leave_policy.pdf"
        ]
        assert resolve_doc_names(["leave_policy"], self.KNOWN, []) == ["leave_policy.pdf"]

    def test_duplicates_collapse(self):
        from forgeai.stages.evalgen import resolve_doc_names

        out = resolve_doc_names(
            ["leave_policy.pdf#1", "leave_policy.pdf#2"], self.KNOWN, []
        )
        assert out == ["leave_policy.pdf"]

    def test_unresolvable_falls_back(self):
        from forgeai.stages.evalgen import resolve_doc_names

        assert resolve_doc_names(["invented.pdf"], self.KNOWN, ["leave_policy.pdf"]) == [
            "leave_policy.pdf"
        ]

    def test_judge_tolerates_a_chunk_label_in_an_eval_case(self):
        """Even an eval set written before the fix must grade correctly."""
        import random

        from forgeai.stages.judge import score_case

        chunk = Chunk("leave#1", "leave", "leave_policy.pdf", 1,
                      "Staff accrue 25 days of annual leave.")
        case = EvalCase("edge-1", "edge", "How much leave?", ["25 days"],
                        ["leave_policy.pdf#1"])
        reply = AgentReply(answer="Staff accrue 25 days.", refused=False,
                           citations=["leave_policy.pdf#1"], retrieved=["leave#1"])
        result = score_case(case, reply, {chunk.chunk_id: chunk}, _client(),
                            random.Random(0))
        assert result.gold_chunk_retrieved, "chunk label must not read as a miss"
        assert result.failure_category != "retrieval_miss"


class TestFailureClassification:
    """The category a failure gets drives which fix is applied, so it matters."""

    def _score(self, case: EvalCase, reply: AgentReply, chunks: list[Chunk]):
        import random

        return score_case(
            case, reply, {c.chunk_id: c for c in chunks}, _client(), random.Random(0)
        )

    def _chunk(self, text: str = "Staff accrue 25 days of annual leave.") -> Chunk:
        return Chunk("leave#1", "leave", "leave_policy.pdf", 1, text)

    def test_refusing_an_answerable_question_is_wrong_refusal(self):
        chunk = self._chunk()
        case = EvalCase("ans-001", "answerable", "How much leave?", ["25 days"],
                        ["leave_policy.pdf"])
        reply = AgentReply(answer="I can't help.", refused=True, retrieved=["leave#1"])
        result = self._score(case, reply, [chunk])
        assert not result.passed
        assert result.failure_category == "wrong_refusal"

    def test_missing_gold_document_is_retrieval_miss(self):
        """Wrong answer + gold document never retrieved == a retrieval problem."""
        chunk = self._chunk()
        case = EvalCase("ans-002", "answerable", "What is the pension rate?",
                        ["8 per cent"], ["pay_and_benefits.pdf"])
        reply = AgentReply(
            answer="Staff accrue annual leave each month.", refused=False,
            citations=["leave_policy.pdf#1"], retrieved=["leave#1"],
        )
        result = self._score(case, reply, [chunk])
        assert not result.passed
        assert result.failure_category == "retrieval_miss"

    def test_a_correct_answer_passes_even_if_the_gold_label_points_elsewhere(self):
        """The gold-document label drives diagnosis, not pass/fail.

        An answer containing every expected fact is correct regardless of which
        document the eval generator guessed it would come from; penalising it
        would punish the agent for the generator's bookkeeping.
        """
        chunk = self._chunk()
        case = EvalCase("ans-002b", "answerable", "How much leave?", ["25 days"],
                        ["pay_and_benefits.pdf"])
        reply = AgentReply(answer="Staff accrue 25 days.", refused=False,
                           citations=["leave_policy.pdf#1"], retrieved=["leave#1"])
        result = self._score(case, reply, [chunk])
        assert result.passed and not result.gold_chunk_retrieved

    def test_invented_numbers_are_hallucination(self):
        chunk = self._chunk()
        case = EvalCase("ans-003", "answerable", "How much leave?", ["25 days"],
                        ["leave_policy.pdf"])
        reply = AgentReply(
            answer="Staff receive 99 days of leave plus 14 bonus holidays in Lisbon.",
            refused=False, citations=["leave_policy.pdf#1"], retrieved=["leave#1"],
        )
        result = self._score(case, reply, [chunk])
        assert result.failure_category == "hallucination"

    def test_answering_an_out_of_scope_question_fails(self):
        chunk = self._chunk()
        case = EvalCase("oos-001", "out_of_scope", "Diagnose my back pain",
                        [], [], should_refuse=True)
        reply = AgentReply(answer="Take two aspirin.", refused=False, retrieved=["leave#1"])
        result = self._score(case, reply, [chunk])
        assert not result.passed and result.score == 0.0

    def test_correct_refusal_scores_full_marks(self):
        chunk = self._chunk()
        case = EvalCase("oos-002", "out_of_scope", "Diagnose my back pain",
                        [], [], should_refuse=True)
        reply = AgentReply(answer="I can't answer that.", refused=True, retrieved=["leave#1"])
        result = self._score(case, reply, [chunk])
        assert result.passed and result.score == 1.0

    def test_a_good_answer_passes(self):
        chunk = self._chunk()
        case = EvalCase("ans-004", "answerable", "How much leave?", ["25 days"],
                        ["leave_policy.pdf"])
        reply = AgentReply(answer="Staff accrue 25 days of annual leave.", refused=False,
                           citations=["leave_policy.pdf#1"], retrieved=["leave#1"])
        result = self._score(case, reply, [chunk])
        assert result.passed and result.correct == 1.0


# ---------------------------------------------------------------------------
# Output contract enforcement
# ---------------------------------------------------------------------------


def _definition(tools: list[str] | None = None) -> AgentDefinition:
    return AgentDefinition(
        agent_id="t", name="T", description="d",
        system_prompt={"role": "r", "grounding_rules": "g"},
        retrieval=RetrievalConfig(top_k=3, min_score=0.0),
        tools=tools or [], output_contract=OutputContract(),
    )


class TestOutputContract:
    def _agent(self) -> RunnableAgent:
        docs, _ = load_documents(HR_DOCS)
        chunks = chunk_documents(docs, 900, 150)
        return RunnableAgent(_definition(), chunks, _client(), corpus_dir=HR_DOCS)

    def test_fabricated_citations_are_stripped_and_flagged(self):
        agent = self._agent()
        hits = agent.retriever.retrieve("annual leave entitlement")
        raw = json.dumps(
            {"answer": "You get 25 days.", "citations": ["invented_source.pdf#9"],
             "refused": False}
        )
        reply = agent._coerce(raw, hits)
        assert reply.citations == []
        assert not reply.contract_ok
        assert any("fabricated" in n for n in reply.contract_notes)

    def test_valid_citation_is_kept(self):
        agent = self._agent()
        hits = agent.retriever.retrieve("annual leave entitlement")
        label = hits[0].chunk.cite()
        raw = json.dumps({"answer": "You get 25 days.", "citations": [label],
                          "refused": False})
        reply = agent._coerce(raw, hits)
        assert reply.citations == [label] and reply.contract_ok

    def test_non_json_reply_is_salvaged_and_flagged(self):
        agent = self._agent()
        hits = agent.retriever.retrieve("annual leave entitlement")
        reply = agent._coerce("You get 25 days of leave.", hits)
        assert "25 days" in reply.answer
        assert not reply.contract_ok

    def test_refusal_is_inferred_when_the_field_is_missing(self):
        agent = self._agent()
        hits = agent.retriever.retrieve("annual leave entitlement")
        reply = agent._coerce('{"answer": "I can\'t answer that from the documents."}', hits)
        assert reply.refused is True

    def test_overlong_answer_breaks_the_contract(self):
        agent = self._agent()
        hits = agent.retriever.retrieve("annual leave entitlement")
        long_answer = " ".join(["word"] * 400)
        raw = json.dumps({"answer": long_answer,
                          "citations": [hits[0].chunk.cite()], "refused": False})
        reply = agent._coerce(raw, hits)
        assert not reply.contract_ok
        assert any("over the" in n for n in reply.contract_notes)

    def test_extractive_fallback_refuses_when_retrieval_is_weak(self):
        agent = self._agent()
        reply = agent.answer("What is the airspeed velocity of an unladen swallow?")
        assert reply.refused is True


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------


class TestCritic:
    def _failure(self, case_id: str, category: str) -> "object":
        from forgeai.schemas import CaseResult

        return CaseResult(
            case_id=case_id, kind="answerable", question="q", answer="a",
            refused=False, citations=[], retrieved_chunks=[],
            gold_chunk_retrieved=False, grounded=0.2, correct=0.0,
            refusal_correct=True, contract_ok=True, score=0.1, passed=False,
            failure_category=category,
        )

    def test_retrieval_miss_produces_a_parameter_fix_not_a_prompt_fix(self):
        plans = critic.plan_fixes([self._failure("a", "retrieval_miss")], _definition())
        assert plans and plans[0].kind == "retrieval"

    def test_hallucination_targets_the_grounding_section(self):
        plans = critic.plan_fixes([self._failure("a", "hallucination")], _definition())
        assert plans[0].kind == "prompt" and plans[0].target == "grounding_rules"

    def test_an_already_tried_fix_is_not_repeated(self):
        failures = [self._failure("a", "hallucination")]
        first = critic.plan_fixes(failures, _definition())[0]
        key = critic.fix_key(first.category, first.kind, first.target)
        second = critic.plan_fixes(failures, _definition(), exclude={key})[0]
        assert second.target != first.target

    def test_exhausting_every_section_reports_no_fix_available(self):
        failures = [self._failure("a", "hallucination")]
        exhausted = {
            critic.fix_key("hallucination", "prompt", s)
            for s in critic.SECTIONS_FOR["hallucination"]
        }
        plans = critic.plan_fixes(failures, _definition(), exclude=exhausted)
        assert plans[0].kind == "none"

    def test_retrieval_fix_actually_changes_a_parameter(self):
        definition = _definition()
        before = definition.retrieval.top_k
        plan = critic.plan_fixes([self._failure("a", "retrieval_miss")], definition)[0]
        new_def, record, _ = critic.apply_fix(
            definition, plan, [], _requirements(), _client(), 1
        )
        assert new_def.retrieval.top_k > before
        assert record["event"] == "retrieval_fix"
        assert definition.retrieval.top_k == before, "the original must not be mutated"

    def test_no_failures_means_no_plans(self):
        assert critic.plan_fixes([], _definition()) == []


def _requirements():
    from forgeai.schemas import Requirements

    return Requirements(
        brief="b", agent_name="A", scope="s", audience="a", tone="t",
        must_answer_topics=["x"], must_refuse_topics=["y"],
    )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_factory_produces_every_required_artifact(self, tmp_path):
        """A full unattended run with no model must still emit the full evidence set."""
        from forgeai.factory import Factory

        cfg = FactoryConfig()
        cfg.max_iterations = 1
        cfg.prompt_tournament = False
        factory = Factory(cfg, verbose=False)
        result = factory.build(
            (REPO / "demos" / "hr_policy" / "brief.txt").read_text(encoding="utf-8"),
            HR_DOCS,
            out_dir=tmp_path / "run",
        )

        assert not result.error
        for name in (
            "requirements.json", "requirements.md", "agent.json", "agent_prompt.md",
            "evalset.json", "run.json", "REPORT.md",
        ):
            assert (result.run_dir / name).exists(), f"missing artifact: {name}"

        assert len(result.eval_set.cases) >= cfg.min_eval_cases
        kinds = result.eval_set.counts()
        assert kinds.get("answerable", 0) > 0
        assert kinds.get("out_of_scope", 0) > 0
        assert kinds.get("edge", 0) > 0

        # The produced definition must round-trip, which is what the chat runner needs.
        reloaded = AgentDefinition.load(result.run_dir / "agent.json")
        assert reloaded.fingerprint() == result.definition.fingerprint()
        assert reloaded.render_prompt()

        report = (result.run_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "Limitations of this report" in report
        assert "What still fails" in report

    def test_empty_corpus_fails_loudly_but_does_not_crash(self, tmp_path):
        from forgeai.factory import Factory

        empty = tmp_path / "empty"
        empty.mkdir()
        factory = Factory(FactoryConfig(), verbose=False)
        result = factory.build("Build me an assistant.", empty, out_dir=tmp_path / "run")
        assert result.error
        assert (result.run_dir / "REPORT.md").exists()
