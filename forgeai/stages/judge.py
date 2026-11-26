"""Stage 5 - the judge.

An LLM-as-judge running on the same small model that wrote the agent is not
trustworthy on its own, so it is not asked to be. The scoring here is
**deterministic-first**:

- *Did it refuse?* Read off the structured `refused` field. No model.
- *Did it cite real excerpts?* Set membership against what was retrieved. No model.
- *Was the gold document retrieved?* Set membership. No model.
- *Is the answer grounded?* Overlap of the answer's numbers and content terms
  with the retrieved text. Numbers are weighted hardest, because an invented
  number is the failure mode that matters in HR and procurement documents. No model.
- *Does the answer contain the expected facts?* Exact/normalised matching
  first; the model is consulted **only** for facts the deterministic matcher
  did not find, i.e. only to catch paraphrase.

The three named biases are handled concretely:

- **Position bias** - the expected facts are shuffled, and the judge is run
  twice with the order reversed. A fact counts as present only if both passes
  agree. Disagreements are counted and reported, not averaged away.
- **Verbosity bias** - the judge grades each fact independently and is told
  explicitly that length is not quality; and because a fact the deterministic
  matcher already found never reaches the judge, padding an answer cannot buy
  credit for facts that are actually present.
- **Self-preference** - the judge never sees the agent's prompt, its identity,
  or that the answer came from the system being optimised. It sees a question,
  a list of facts, and an answer.

Finally `audit_judge()` scores the judge itself against planted known-good and
known-bad answers, and that accuracy is printed in the report. A judge that
cannot tell a correct answer from an unrelated one invalidates the scorecard
above it, and the reader deserves to know that before believing the number.
"""

from __future__ import annotations

import random
import re
from typing import Any

from ..agent_runtime import AgentReply, RunnableAgent
from ..llm import LLMClient, first_str
from ..prompts import render
from ..retrieval import STOPWORDS, tokenize
from ..schemas import CaseResult, Chunk, EvalCase, EvalSet, Scorecard

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A case passes only if it clears all of these.
PASS_FACT_RECALL = 0.6
PASS_GROUNDED = 0.5


def _norm_num(text: str) -> str:
    return text.replace(",", "").rstrip("0").rstrip(".") if "." in text else text.replace(",", "")


def _numbers(text: str) -> set[str]:
    return {_norm_num(m) for m in _NUM.findall(text or "")}


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text or "") if len(t) > 3 and t not in STOPWORDS}


def fact_matched_deterministically(fact: str, answer: str) -> bool:
    """High-precision check: does the answer literally contain this fact?"""
    fact = (fact or "").strip()
    if not fact:
        return False
    a_low = (answer or "").lower()
    if fact.lower() in a_low:
        return True

    fact_nums = _numbers(fact)
    if fact_nums:
        # A numeric fact is only present if every number in it is present.
        return fact_nums.issubset(_numbers(answer))

    f_tokens = _content_tokens(fact)
    if not f_tokens:
        return False
    a_tokens = _content_tokens(answer)
    return len(f_tokens & a_tokens) / len(f_tokens) >= 0.67


def groundedness(answer: str, context: str) -> float:
    """How much of the answer is traceable to the retrieved text?

    Numbers dominate the score when the answer contains any, because an
    invented figure is the failure that matters in a policy or price document.

    When the answer contains *no* numbers the score rests entirely on term
    support. An earlier version still awarded the numeric component in that
    case, which handed every number-free answer a floor of 0.6 - above the
    pass threshold - so a fluent, wholly fabricated prose answer scored as
    grounded. A test caught it; the weighting below is the fix.
    """
    if not (answer or "").strip():
        return 0.0
    ctx_nums, ctx_tokens = _numbers(context), _content_tokens(context)

    a_nums = _numbers(answer)
    a_tokens = _content_tokens(answer)
    tok_support = len(a_tokens & ctx_tokens) / len(a_tokens) if a_tokens else 0.0

    if not a_nums:
        return round(tok_support, 3)

    num_support = len(a_nums & ctx_nums) / len(a_nums)
    return round(0.6 * num_support + 0.4 * tok_support, 3)


# ---------------------------------------------------------------------------
# The LLM half
# ---------------------------------------------------------------------------


def _judge_pass(question: str, facts: list[str], answer: str, client: LLMClient) -> dict[str, bool]:
    def validate(value: Any) -> dict:
        if not isinstance(value, dict) or not isinstance(value.get("facts"), list):
            raise ValueError("expected {'facts': [...]}")
        return value

    facts_block = "\n".join(f"- {f}" for f in facts)
    prompt = render("judge", question=question, facts_block=facts_block, answer=answer)
    data, _ = client.structured(
        prompt,
        stage="judge",
        validate=validate,
        fallback=lambda: {"facts": [{"fact": f, "present": False} for f in facts]},
        temperature=0.0,
    )
    out: dict[str, bool] = {}
    for entry in data.get("facts", []):
        if isinstance(entry, dict):
            name = first_str(entry.get("fact"))
            if name:
                out[name.strip().lower()] = bool(entry.get("present"))
    return out


def judge_facts(
    question: str,
    facts: list[str],
    answer: str,
    client: LLMClient,
    rng: random.Random,
) -> tuple[dict[str, bool], int]:
    """Two order-swapped passes; a fact is present only if both agree."""
    if not facts or not client.available:
        return {}, 0
    order = list(facts)
    rng.shuffle(order)
    first = _judge_pass(question, order, answer, client)
    second = _judge_pass(question, list(reversed(order)), answer, client)

    verdicts: dict[str, bool] = {}
    disagreements = 0
    for fact in facts:
        key = fact.strip().lower()
        a, b = first.get(key, False), second.get(key, False)
        if a != b:
            disagreements += 1
            verdicts[fact] = False  # unstable judgement is not evidence of presence
        else:
            verdicts[fact] = a
    return verdicts, disagreements


# ---------------------------------------------------------------------------
# Scoring one case
# ---------------------------------------------------------------------------


def score_case(
    case: EvalCase,
    reply: AgentReply,
    chunk_index: dict[str, Chunk],
    client: LLMClient,
    rng: random.Random,
) -> CaseResult:
    context = "\n".join(
        chunk_index[cid].text for cid in reply.retrieved if cid in chunk_index
    )
    retrieved_docs = {
        chunk_index[cid].doc_name for cid in reply.retrieved if cid in chunk_index
    }
    # Compare on bare document names: an eval case may carry a full chunk label
    # ("leave_policy.pdf#2") rather than a file name, and treating that as a
    # miss would invent a retrieval failure that never happened.
    expected_docs = {
        (d or "").split("#", 1)[0].strip().strip("[]() ")
        for d in case.expected_source_docs
    } - {""}
    gold_retrieved = (
        True if not expected_docs else bool(expected_docs & retrieved_docs)
    )

    refusal_correct = reply.refused == case.should_refuse
    grounded = 1.0 if reply.refused else groundedness(reply.answer, context)

    # --- fact recall -----------------------------------------------------
    det_hits = {f: fact_matched_deterministically(f, reply.answer) for f in case.expected_facts}
    unresolved = [f for f, hit in det_hits.items() if not hit]
    judge_verdicts, disagreements = ({}, 0)
    if unresolved and not reply.refused:
        judge_verdicts, disagreements = judge_facts(
            case.question, unresolved, reply.answer, client, rng
        )

    present = {f: det_hits[f] or judge_verdicts.get(f, False) for f in case.expected_facts}
    fact_recall = (
        sum(1 for v in present.values() if v) / len(present) if present else 1.0
    )
    judge_added = [f for f in unresolved if judge_verdicts.get(f)]

    # --- aggregate -------------------------------------------------------
    if case.should_refuse:
        score = 1.0 if reply.refused else 0.0
        passed = reply.refused
    elif reply.refused:
        score, passed = 0.0, False
    else:
        score = round(
            0.6 * fact_recall + 0.3 * grounded + 0.1 * (1.0 if reply.contract_ok else 0.0),
            3,
        )
        passed = (
            fact_recall >= PASS_FACT_RECALL
            and grounded >= PASS_GROUNDED
            and reply.contract_ok
        )

    # --- failure classification, from hard evidence ----------------------
    category = ""
    if not passed:
        if case.should_refuse and not reply.refused:
            # It answered something it should have declined. Whether that is a
            # scope defect or an invention depends on whether it had support.
            category = "prompt_defect" if grounded >= 0.6 else "hallucination"
        elif not case.should_refuse and reply.refused:
            category = "wrong_refusal"
        elif not gold_retrieved:
            category = "retrieval_miss"
        elif grounded < PASS_GROUNDED:
            category = "hallucination"
        else:
            category = "prompt_defect"

    notes = []
    if judge_added:
        notes.append(f"judge credited paraphrase for: {judge_added}")
    if disagreements:
        notes.append(f"{disagreements} fact(s) unstable across order-swapped passes")
    if reply.contract_notes:
        notes.append("contract: " + "; ".join(reply.contract_notes))

    return CaseResult(
        case_id=case.case_id,
        kind=case.kind,
        question=case.question,
        answer=reply.answer,
        refused=reply.refused,
        citations=reply.citations,
        retrieved_chunks=reply.retrieved,
        gold_chunk_retrieved=gold_retrieved,
        grounded=grounded,
        correct=round(fact_recall, 3),
        refusal_correct=refusal_correct,
        contract_ok=reply.contract_ok,
        score=score,
        passed=passed,
        failure_category=category,
        judge_notes=" | ".join(notes),
        judge_agreed_with_checks=disagreements == 0,
    )


def run_evaluation(
    agent: RunnableAgent,
    eval_set: EvalSet,
    client: LLMClient,
    iteration: int,
    *,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[Scorecard, list[CaseResult]]:
    rng = random.Random(seed)
    chunk_index = {c.chunk_id: c for c in agent.chunks}
    results: list[CaseResult] = []

    for i, case in enumerate(eval_set.cases, 1):
        reply = agent.answer(case.question)
        result = score_case(case, reply, chunk_index, client, rng)
        results.append(result)
        if verbose:
            mark = "PASS" if result.passed else f"FAIL/{result.failure_category}"
            print(f"    [{i:2d}/{len(eval_set.cases)}] {result.case_id} {mark}")

    by_kind: dict[str, dict[str, Any]] = {}
    for r in results:
        bucket = by_kind.setdefault(r.kind, {"total": 0, "passed": 0, "score": 0.0})
        bucket["total"] += 1
        bucket["passed"] += int(r.passed)
        bucket["score"] += r.score
    for bucket in by_kind.values():
        bucket["score"] = round(bucket["score"] / bucket["total"], 3)

    failures: dict[str, int] = {}
    for r in results:
        if r.failure_category:
            failures[r.failure_category] = failures.get(r.failure_category, 0) + 1

    scorecard = Scorecard(
        iteration=iteration,
        definition_fingerprint=agent.definition.fingerprint(),
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        score=round(sum(r.score for r in results) / len(results), 3) if results else 0.0,
        by_kind=by_kind,
        failures_by_category=failures,
        judge_disagreements=sum(1 for r in results if not r.judge_agreed_with_checks),
    )
    return scorecard, results


# ---------------------------------------------------------------------------
# Auditing the judge itself
# ---------------------------------------------------------------------------


def audit_judge(
    eval_set: EvalSet, client: LLMClient, *, seed: int = 0, sample: int = 3
) -> dict[str, Any]:
    """Score the judge against planted answers whose grade we already know.

    For each sampled answerable case we synthesise:
      - a *good* answer that states every expected fact (judge must say present),
      - a *bad* answer about something else entirely (judge must say absent),
      - a *padded* answer: the bad answer at five times the length, which is
        the verbosity-bias trap.
    """
    rng = random.Random(seed)
    cases = [c for c in eval_set.cases if c.kind == "answerable" and c.expected_facts]
    if not cases:
        return {"ran": False, "reason": "no answerable cases with expected facts"}
    if not client.available:
        return {
            "ran": False,
            "reason": "no model configured; deterministic matching was used alone",
        }

    sampled = cases[: max(1, min(sample, len(cases)))]
    checks = {"good": [0, 0], "bad": [0, 0], "padded": [0, 0]}  # [correct, total]
    padding = (
        " This is a thorough and carefully considered response that takes into "
        "account the full context of your question and all relevant factors."
    )

    for case in sampled:
        facts = case.expected_facts
        good = "According to the documents: " + "; ".join(facts) + "."
        bad = (
            "The office cafeteria serves lunch between noon and two, and the car "
            "park barrier needs a staff pass."
        )
        padded = bad + padding * 5

        for label, answer, expect_present in (
            ("good", good, True),
            ("bad", bad, False),
            ("padded", padded, False),
        ):
            verdicts, _ = judge_facts(case.question, facts, answer, client, rng)
            for fact in facts:
                checks[label][1] += 1
                if verdicts.get(fact, False) == expect_present:
                    checks[label][0] += 1

    def rate(pair: list[int]) -> float:
        return round(pair[0] / pair[1], 3) if pair[1] else 0.0

    recall = rate(checks["good"])
    specificity = rate(checks["bad"])
    verbosity_resistance = rate(checks["padded"])
    return {
        "ran": True,
        "cases_sampled": len(sampled),
        "recall_on_known_good": recall,
        "specificity_on_known_bad": specificity,
        "verbosity_resistance": verbosity_resistance,
        "verbosity_bias_detected": verbosity_resistance < specificity - 0.1,
        "verdict": (
            "trustworthy"
            if recall >= 0.8 and specificity >= 0.8
            else "weak - treat the scorecard as indicative, not exact"
        ),
    }
