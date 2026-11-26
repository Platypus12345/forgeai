"""The factory: brief + documents in, tested agent + evidence out.

Pipeline
--------
  load documents
  -> intake interview (<=5 questions)            [stage 1]
  -> groundability guard                         [stage 2]
  -> tool selection + candidate prompts          [stage 3]
  -> eval set generated from the documents       [stage 4]
  -> prompt tournament, judged by that eval set  [stretch]
  -> baseline evaluation                         [stage 5]
  -> up to 3 critic-optimizer iterations         [stage 6]
  -> scorecards, iteration log, honest report    [stage 7]

Three rules hold the loop together:

* **Keep the best, not the last.** An iteration that lowers the score is rolled
  back and recorded as rolled back. The shipped agent is the best-scoring one
  the factory ever held, which is not always the final one.
* **Stop when iterating stops paying.** Budget exhaustion, the pass threshold,
  or a gain smaller than `min_gain_per_iteration` all end the loop early, and
  the reason is written into the report.
* **Never crash on a bad brief.** An empty folder, a scanned PDF, a brief the
  documents cannot support - these produce a *reported* bad outcome, not a
  traceback. At the review this runs unattended on an unseen folder.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_runtime import RunnableAgent
from .config import RUNS_DIR, FactoryConfig
from .docs import chunk_documents, corpus_summary, load_documents
from .llm import BudgetExceeded, Ledger, LLMClient
from .retrieval import Retriever
from .schemas import (
    AgentDefinition,
    CaseResult,
    Document,
    EvalSet,
    IterationLog,
    Requirements,
    Scorecard,
)
from .stages import critic, evalgen, generator, groundability, intake
from .stages.judge import audit_judge, run_evaluation
from .report import write_report


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "run").lower()).strip("-")[:40] or "run"


@dataclass
class RunResult:
    run_dir: Path
    requirements: Requirements | None = None
    definition: AgentDefinition | None = None
    eval_set: EvalSet | None = None
    scorecards: list[Scorecard] = field(default_factory=list)
    iterations: list[IterationLog] = field(default_factory=list)
    tournament: list[dict] = field(default_factory=list)
    judge_audit: dict = field(default_factory=dict)
    stop_reason: str = ""
    error: str = ""

    @property
    def final_score(self) -> float:
        return max((s.score for s in self.scorecards), default=0.0)


class Factory:
    def __init__(self, cfg: FactoryConfig | None = None, *, verbose: bool = True):
        self.cfg = cfg or FactoryConfig()
        self.ledger = Ledger(self.cfg.budget)
        self.client = LLMClient(self.cfg.model, self.ledger, verbose=verbose)
        self.verbose = verbose

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------
    def _build_agent(
        self, definition: AgentDefinition, docs: list[Document], corpus_dir: Path
    ) -> RunnableAgent:
        chunks = chunk_documents(
            docs, definition.retrieval.chunk_size, definition.retrieval.chunk_overlap
        )
        return RunnableAgent(definition, chunks, self.client, corpus_dir=corpus_dir)

    # ------------------------------------------------------------------
    def build(
        self,
        brief: str,
        corpus_dir: str | Path,
        *,
        run_name: str = "",
        interactive: bool = False,
        human_checkpoint: bool = False,
        out_dir: str | Path | None = None,
    ) -> RunResult:
        started = time.time()
        corpus_dir = Path(corpus_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = Path(out_dir) if out_dir else RUNS_DIR / f"{_slug(run_name or corpus_dir.name)}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = RunResult(run_dir=run_dir)

        try:
            return self._build_inner(
                brief, corpus_dir, run_dir, result, started,
                interactive=interactive, human_checkpoint=human_checkpoint,
            )
        except BudgetExceeded as exc:
            result.stop_reason = f"budget exhausted: {exc}"
            result.error = str(exc)
        except KeyboardInterrupt:
            result.stop_reason = "interrupted by user"
            result.error = "KeyboardInterrupt"
        except Exception as exc:  # unattended runs must still hand back evidence
            result.stop_reason = f"factory error: {exc}"
            result.error = traceback.format_exc()
            self.log(f"\n!! factory error: {exc}")

        self._persist(result, brief, corpus_dir, started)
        return result

    # ------------------------------------------------------------------
    def _build_inner(
        self,
        brief: str,
        corpus_dir: Path,
        run_dir: Path,
        result: RunResult,
        started: float,
        *,
        interactive: bool,
        human_checkpoint: bool,
    ) -> RunResult:
        cfg = self.cfg
        self.log("=" * 72)
        self.log(f"ForgeAI  |  model: {cfg.model.describe()}  |  run dir: {run_dir}")
        self.log("=" * 72)

        # -- documents ---------------------------------------------------
        self.log("\n[0/7] Reading the document folder")
        docs, warnings = load_documents(corpus_dir)
        if not docs:
            raise RuntimeError(
                f"no readable documents in {corpus_dir}: " + "; ".join(warnings)
            )
        base_chunks = chunk_documents(docs, 900, 150)
        self.log("      " + corpus_summary(docs, base_chunks).replace("\n", "\n      "))
        for w in warnings:
            self.log(f"      ! {w}")

        base_retriever = Retriever(base_chunks, generator.choose_retrieval(docs, base_chunks))

        # -- stage 1: intake ---------------------------------------------
        self.log("\n[1/7] Intake interview")
        reqs = intake.run_intake(
            brief, docs, base_chunks, base_retriever, self.client, cfg,
            interactive=interactive, warnings=warnings,
        )
        self.log(f"      agent: {reqs.agent_name}")
        self.log(f"      must answer: {len(reqs.must_answer_topics)} topics")
        self.log(f"      must refuse: {len(reqs.must_refuse_topics)} topics")

        # -- stage 2: groundability --------------------------------------
        self.log("\n[2/7] Groundability guard")
        reqs = groundability.check_groundability(reqs, base_retriever, self.client)
        for f in reqs.groundability:
            self.log(f"      {f.status:<12} {f.topic[:58]}  (evidence {f.score:.2f})")
        for w in reqs.warnings:
            if w.startswith("GROUNDABILITY"):
                self.log(f"      ! {w[:160]}")

        if human_checkpoint:
            reqs = intake.human_checkpoint(reqs)

        result.requirements = reqs
        reqs.save(run_dir / "requirements.json")
        (run_dir / "requirements.md").write_text(reqs.to_markdown(), encoding="utf-8")

        # -- stage 3: generate candidates --------------------------------
        self.log("\n[3/7] Generating the agent definition")
        tools, tool_rationale, tool_decisions = generator.select_tools(reqs, docs, self.client)
        self.log(f"      tools attached: {tools or '(none)'}")
        for d in tool_decisions:
            self.log(f"        - {d}")

        styles = generator.candidate_styles(cfg)
        candidates = []
        for style in styles:
            definition = generator.build_definition(
                reqs, docs, base_chunks, self.client, cfg,
                style=style, tools=tools, tool_rationale=tool_rationale,
                tool_decisions=tool_decisions,
            )
            candidates.append(definition)
            self.log(f"      candidate '{style[0]}' -> {definition.fingerprint()}")

        # -- stage 4: eval set -------------------------------------------
        self.log("\n[4/7] Generating the eval set from the documents")
        eval_set = evalgen.generate_eval_set(
            reqs, base_chunks, self.client, cfg, agent_id=candidates[0].agent_id
        )
        result.eval_set = eval_set
        eval_set.save(run_dir / "evalset.json")
        self.log(f"      {len(eval_set.cases)} cases: {eval_set.counts()}")
        for note in eval_set.notes:
            self.log(f"      ! {note}")
        if len(eval_set.cases) < cfg.min_eval_cases:
            self.log(
                f"      ! WARNING: only {len(eval_set.cases)} cases "
                f"(floor is {cfg.min_eval_cases})"
            )

        # -- stretch: prompt tournament ----------------------------------
        definition = candidates[0]
        if len(candidates) > 1:
            self.log("\n[4b] Prompt tournament")
            # Judge candidates on a stratified subset to keep the budget sane,
            # then run the winner on the full set.
            subset = EvalSet(
                agent_id=eval_set.agent_id,
                cases=_stratified(eval_set, 6),
                generation_prompt_files=eval_set.generation_prompt_files,
            )
            best_score = -1.0
            for cand in candidates:
                if self.ledger.exhausted():
                    self.log("      budget exhausted; tournament cut short")
                    break
                agent = self._build_agent(cand, docs, corpus_dir)
                card, _ = run_evaluation(
                    agent, subset, self.client, iteration=-1, seed=cfg.seed
                )
                style = cand.agent_id.rsplit("-", 1)[-1]
                result.tournament.append(
                    {
                        "style": style,
                        "agent_id": cand.agent_id,
                        "fingerprint": cand.fingerprint(),
                        "subset_size": len(subset.cases),
                        "score": card.score,
                        "passed": card.passed,
                        "by_kind": card.by_kind,
                    }
                )
                self.log(f"      {style:<14} score {card.score:.3f} "
                         f"({card.passed}/{card.total} on {len(subset.cases)}-case subset)")
                if card.score > best_score:
                    best_score, definition = card.score, cand
            winner = definition.agent_id.rsplit("-", 1)[-1]
            self.log(f"      winner: {winner}")
            for entry in result.tournament:
                entry["winner"] = entry["agent_id"] == definition.agent_id

        definition.changelog.append(
            {"event": "selected", "reason": "prompt tournament winner"
             if len(candidates) > 1 else "only candidate"}
        )

        # -- stage 5: baseline evaluation --------------------------------
        self.log("\n[5/7] Baseline evaluation")
        agent = self._build_agent(definition, docs, corpus_dir)
        card, results = run_evaluation(
            agent, eval_set, self.client, iteration=0, seed=cfg.seed, verbose=self.verbose
        )
        result.scorecards.append(card)
        self._write_results(run_dir, 0, card, results)
        self.log(f"      iteration 0: score {card.score:.3f}  "
                 f"({card.passed}/{card.total} passed)  failures {card.failures_by_category}")

        best_definition, best_card, best_results = definition, card, results
        # Publish the best definition on the result as soon as one exists, and
        # again whenever it improves. The agent *is* the product, and a run
        # that stops on a budget - which is a documented, expected way for
        # this factory to end - used to reach its exit handler holding a
        # perfectly good scored agent and write everything except that.
        result.definition = best_definition

        # -- stage 6: critic-optimizer loop ------------------------------
        self.log("\n[6/7] Critic-optimizer loop (max "
                 f"{cfg.max_iterations} iterations)")
        stop_reason = f"completed {cfg.max_iterations} iteration(s)"
        attempted: set[str] = set()

        for iteration in range(1, cfg.max_iterations + 1):
            # The aggregate score is a shipping gate, but it must not be the
            # only one. A document-grounded agent that fabricates is not
            # shippable at any average: "0.889 overall" and "it invents a
            # figure once in twenty-two questions" are very different claims
            # to the person who has to answer for the wrong figure. So the
            # happy stop requires a passing score *and* a clean hallucination
            # count. Other failure classes are survivable - a wrong refusal
            # annoys a user, a fabricated entitlement misleads one - and the
            # loop is allowed to leave those on the table once the score is
            # good enough.
            blocking = best_card.failures_by_category.get("hallucination", 0)
            if best_card.score >= cfg.pass_threshold and not blocking:
                stop_reason = (
                    f"pass threshold reached ({best_card.score:.3f} >= {cfg.pass_threshold})"
                )
                self.log(f"      stopping: {stop_reason}")
                break
            if best_card.score >= cfg.pass_threshold:
                self.log(
                    f"      score {best_card.score:.3f} clears the "
                    f"{cfg.pass_threshold} threshold, but {blocking} "
                    "hallucination(s) remain - not stopping on that"
                )
            if self.ledger.exhausted():
                stop_reason = "budget exhausted before iteration " + str(iteration)
                self.log(f"      stopping: {stop_reason}")
                break

            plans = critic.plan_fixes(
                best_results, best_definition, exclude=attempted
            )
            if not plans:
                stop_reason = "no classified failures left to fix"
                self.log(f"      stopping: {stop_reason}")
                break
            if all(p.kind == "none" for p in plans):
                stop_reason = (
                    "every applicable fix has already been tried without improving "
                    "the score"
                )
                self.log(f"      stopping: {stop_reason}")
                result.iterations.append(
                    IterationLog(
                        iteration=iteration,
                        score_before=best_card.score,
                        score_after=best_card.score,
                        failures_addressed=[
                            f"{p.category}:{len(p.case_ids)} case(s)" for p in plans
                        ],
                        changes=[
                            {
                                "event": "no_fix_available",
                                "iteration": iteration,
                                "category": p.category,
                                "why": p.reason,
                                "targeted_cases": p.case_ids,
                            }
                            for p in plans
                        ],
                        reasoning="; ".join(p.reason for p in plans),
                        accepted=False,
                        note="Loop ended here: no untried targeted fix remained.",
                    )
                )
                break
            for p in plans:
                attempted.add(critic.fix_key(p.category, p.kind, p.target))

            self.log(f"\n      --- iteration {iteration} ---")
            candidate_def = best_definition
            records: list[dict] = []
            rechunk = False
            for plan in plans:
                candidate_def, record, needs = critic.apply_fix(
                    candidate_def, plan, best_results, reqs, self.client, iteration
                )
                rechunk = rechunk or needs
                records.append(record)
                self.log(
                    f"      fix: {plan.category} -> "
                    f"{plan.kind}:{plan.target or 'n/a'} "
                    f"({len(plan.case_ids)} case(s))"
                )

            cand_agent = self._build_agent(candidate_def, docs, corpus_dir)
            new_card, new_results = run_evaluation(
                cand_agent, eval_set, self.client, iteration=iteration, seed=cfg.seed
            )
            gain = new_card.score - best_card.score
            accepted = gain > 0
            self.log(
                f"      score {best_card.score:.3f} -> {new_card.score:.3f} "
                f"({gain:+.3f})  {'ACCEPTED' if accepted else 'ROLLED BACK'}"
            )

            result.scorecards.append(new_card)
            self._write_results(run_dir, iteration, new_card, new_results)
            result.iterations.append(
                IterationLog(
                    iteration=iteration,
                    score_before=best_card.score,
                    score_after=new_card.score,
                    failures_addressed=[
                        f"{p.category}:{len(p.case_ids)} case(s)" for p in plans
                    ],
                    changes=records,
                    reasoning="; ".join(p.reason for p in plans),
                    accepted=accepted,
                    note=(
                        "" if accepted else
                        "Change lowered or did not move the score; the previous "
                        "definition was kept."
                    ),
                )
            )

            if accepted:
                best_definition, best_card, best_results = (
                    candidate_def, new_card, new_results
                )
                result.definition = best_definition
                if 0 < gain < cfg.budget.min_gain_per_iteration:
                    stop_reason = (
                        f"iteration stopped paying for itself (gain {gain:+.3f} < "
                        f"{cfg.budget.min_gain_per_iteration})"
                    )
                    self.log(f"      stopping: {stop_reason}")
                    break
            # A rolled-back iteration still counts against the cap.

        result.stop_reason = stop_reason
        result.definition = best_definition

        # -- stage 7: audit + report -------------------------------------
        self.log("\n[7/7] Auditing the judge and writing the report")
        try:
            result.judge_audit = audit_judge(eval_set, self.client, seed=cfg.seed)
        except Exception as exc:
            result.judge_audit = {"ran": False, "reason": f"audit failed: {exc}"}
        if result.judge_audit.get("ran"):
            self.log(
                f"      judge: recall {result.judge_audit['recall_on_known_good']:.2f}, "
                f"specificity {result.judge_audit['specificity_on_known_bad']:.2f}, "
                f"verdict {result.judge_audit['verdict']}"
            )

        self._persist(result, brief, corpus_dir, started)
        self.log(f"\nDone. Final score {best_card.score:.3f} "
                 f"({best_card.passed}/{best_card.total}). Artifacts in {run_dir}")
        return result

    # ------------------------------------------------------------------
    def _write_results(
        self, run_dir: Path, iteration: int, card: Scorecard, results: list[CaseResult]
    ) -> None:
        d = run_dir / "iterations"
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "scorecard": card.to_dict(),
            "cases": [r.to_dict() for r in results],
        }
        (d / f"iteration-{iteration}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _persist(
        self, result: RunResult, brief: str, corpus_dir: Path, started: float
    ) -> None:
        run_dir = result.run_dir
        meta: dict[str, Any] = {
            "brief": brief,
            "corpus_dir": str(corpus_dir),
            "model": self.cfg.model.describe(),
            "provider": self.cfg.model.provider,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_seconds": round(time.time() - started, 1),
            "stop_reason": result.stop_reason,
            "budget": self.ledger.summary(),
            "scorecards": [s.to_dict() for s in result.scorecards],
            "tournament": result.tournament,
            "judge_audit": result.judge_audit,
            "iterations": [i.to_dict() for i in result.iterations],
        }
        if result.error:
            meta["error"] = result.error

        # The agent definition is the deliverable, so it is written on every
        # exit path - success, budget stop, spent quota, Ctrl-C - not only the
        # happy one. A ship-spares run proved why: it scored three iterations,
        # kept the best, then hit its latency budget during the fourth and
        # wrote the scorecards, the eval set and the report while discarding
        # the one artifact a reviewer actually has to run.
        if result.definition is not None:
            try:
                result.definition.save(run_dir / "agent.json")
                (run_dir / "agent_prompt.md").write_text(
                    f"# System prompt - {result.definition.name}\n\n"
                    f"_fingerprint {result.definition.fingerprint()}, version "
                    f"{result.definition.version}_\n\n```\n"
                    + result.definition.render_prompt()
                    + "\n```\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # never lose the rest over this
                self.log(f"!! could not write the agent definition: {exc}")

        (run_dir / "run.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if result.iterations or result.scorecards:
            (run_dir / "iteration_log.json").write_text(
                json.dumps(
                    [i.to_dict() for i in result.iterations], indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )
        try:
            write_report(result, meta, run_dir)
        except Exception as exc:  # a report bug must not destroy the artifacts
            (run_dir / "REPORT.md").write_text(
                f"# Report generation failed\n\n{exc}\n", encoding="utf-8"
            )


def _stratified(eval_set: EvalSet, n: int) -> list:
    """A small subset that still contains every case kind."""
    by_kind: dict[str, list] = {}
    for c in eval_set.cases:
        by_kind.setdefault(c.kind, []).append(c)
    out = []
    i = 0
    while len(out) < min(n, len(eval_set.cases)):
        added = False
        for kind in sorted(by_kind):
            if i < len(by_kind[kind]) and len(out) < n:
                out.append(by_kind[kind][i])
                added = True
        if not added:
            break
        i += 1
    return out
