"""Stage 2 - the groundability guard.

The brief: "If the brief asks for things the documents cannot support, the
factory flags it at the requirements stage - it must not generate an agent that
will hallucinate the gap away."

So this runs *before* generation and it has teeth. A topic judged unsupported
is not merely annotated: it is moved out of `must_answer_topics` and into
`must_refuse_topics`, which means the generator writes it into the refusal
policy and the eval generator writes refusal test cases for it. The gap becomes
something the agent is tested on declining, instead of something it quietly
fills in.

Evidence first, model second. The retrieval score over the real corpus is the
primary signal; the model is only consulted in the ambiguous middle band, where
"the words appear but is the substance there?" is a judgement call. A model
that is unavailable or incoherent therefore degrades the guard to a
conservative keyword-evidence check rather than disabling it.
"""

from __future__ import annotations

from typing import Any

from ..llm import LLMClient, first_str
from ..prompts import render
from ..retrieval import Retriever
from ..schemas import GroundabilityFinding, Requirements

# Below LOW the corpus clearly has nothing; above HIGH it clearly has something.
# Between them we ask the model, giving it the top passages in full.
#
# LOW is deliberately very low. A false "unsupported" is the expensive error in
# the other direction: it pushes a topic the documents DO cover into the refusal
# policy, and the agent then declines its own core subject. So the deterministic
# path only claims "unsupported" when retrieval found essentially nothing, and
# everything else goes to the model with real evidence attached.
LOW_EVIDENCE = 0.06
HIGH_EVIDENCE = 0.55


def _probe(topic: str, evidence: str, client: LLMClient) -> tuple[str, str]:
    def fallback() -> dict:
        return {
            "status": "partial",
            "reason": "Model unavailable; graded partial on retrieval evidence alone.",
        }

    def validate(value: Any) -> dict:
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        status = first_str(value.get("status")).lower()
        if status not in {"supported", "partial", "unsupported"}:
            raise ValueError(f"bad status {status!r}")
        value["status"] = status
        return value

    prompt = render("groundability_probe", topic=topic, evidence=evidence or "(nothing found)")
    data, _ = client.structured(
        prompt, stage="groundability", validate=validate, fallback=fallback, temperature=0.0
    )
    return data["status"], first_str(data.get("reason"), "")


def check_groundability(
    reqs: Requirements, retriever: Retriever, client: LLMClient
) -> Requirements:
    findings: list[GroundabilityFinding] = []
    still_answerable: list[str] = []
    demoted: list[str] = []

    for topic in reqs.must_answer_topics:
        score, evidence = retriever.evidence_bundle(topic, k=3, chars=0)

        if score < LOW_EVIDENCE:
            status = "unsupported"
            reason = "No passage in the corpus matches this topic at all."
        elif score >= HIGH_EVIDENCE:
            status = "supported"
            reason = "Strong lexical and passage-level match in the corpus."
        else:
            status, reason = _probe(topic, evidence, client)

        if status == "unsupported":
            action = (
                "Removed from must-answer and added to must-refuse; the agent is "
                "instructed to decline it and is eval-tested on refusing it."
            )
            demoted.append(topic)
        elif status == "partial":
            action = (
                "Kept in scope but flagged: the agent is instructed to answer only "
                "the part the documents cover and to say what is missing."
            )
            still_answerable.append(topic)
        else:
            action = "Kept in scope."
            still_answerable.append(topic)

        findings.append(
            GroundabilityFinding(
                topic=topic,
                status=status,
                best_evidence=(
                    " ".join(evidence.split())[:200] if evidence
                    else "(no matching passage)"
                ),
                score=round(score, 3),
                action=f"{action} {reason}".strip(),
            )
        )

    reqs.groundability = findings
    reqs.must_answer_topics = still_answerable

    for topic in demoted:
        entry = f"{topic} (requested in the brief, but the documents do not support it)"
        if entry not in reqs.must_refuse_topics:
            reqs.must_refuse_topics.append(entry)

    if demoted:
        reqs.warnings.append(
            "GROUNDABILITY: the brief asks for "
            + str(len(demoted))
            + " topic(s) the documents cannot support - "
            + "; ".join(demoted)
            + ". These were moved to must-refuse rather than generated over."
        )
    partials = [f.topic for f in findings if f.status == "partial"]
    if partials:
        reqs.warnings.append(
            "GROUNDABILITY: thin coverage for - " + "; ".join(partials)
            + ". The agent will answer these only as far as the documents go."
        )
    if not reqs.must_answer_topics:
        reqs.warnings.append(
            "GROUNDABILITY: no requested topic is supported by this corpus. The "
            "produced agent will refuse almost everything; the brief and the "
            "document folder do not match."
        )
    return reqs
