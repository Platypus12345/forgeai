"""ForgeAI command line.

    forgeai build  --brief-file demos/hr_policy/brief.txt --docs demos/hr_policy/docs
    forgeai chat   runs/<run>/agent.json --docs demos/hr_policy/docs
    forgeai regress --run runs/<run> --docs demos/hr_policy/docs_v2
    forgeai tools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent_runtime import RunnableAgent
from .config import FactoryConfig
from .docs import chunk_documents, load_documents
from .factory import Factory
from .llm import Ledger, LLMClient
from .schemas import AgentDefinition, EvalSet
from .tools import REGISTRY


def _read_brief(args: argparse.Namespace) -> str:
    if args.brief_file:
        return Path(args.brief_file).read_text(encoding="utf-8").strip()
    if args.brief:
        return args.brief.strip()
    print("Paste the brief, then Ctrl-Z+Enter (Windows) / Ctrl-D (Unix):")
    return sys.stdin.read().strip()


# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    brief = _read_brief(args)
    if not brief:
        print("error: empty brief", file=sys.stderr)
        return 2

    cfg = FactoryConfig()
    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    if args.cases is not None:
        cfg.min_eval_cases = args.cases
    if args.no_tournament:
        cfg.prompt_tournament = False

    factory = Factory(cfg, verbose=not args.quiet)
    result = factory.build(
        brief,
        args.docs,
        run_name=args.name or Path(args.docs).name,
        interactive=args.interactive,
        human_checkpoint=args.checkpoint,
        out_dir=args.out,
    )
    print(f"\nArtifacts: {result.run_dir}")
    print(f"Report:    {result.run_dir / 'REPORT.md'}")
    if result.definition:
        print(
            f"Chat:      python -m forgeai.cli chat "
            f"\"{result.run_dir / 'agent.json'}\" --docs \"{args.docs}\""
        )
    return 0 if not result.error else 1


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = FactoryConfig()
    client = LLMClient(cfg.model, Ledger(cfg.budget))
    agent = RunnableAgent.from_files(args.definition, args.docs, client)
    d = agent.definition

    print("=" * 70)
    print(f"{d.name}   [{cfg.model.describe()}]")
    print(f"{d.description[:200]}")
    print(f"tools: {d.tools or 'none'} | top_k={d.retrieval.top_k} "
          f"| chunks={len(agent.chunks)}")
    print("=" * 70)
    print("Ask a question. '/prompt' shows the system prompt, '/quit' exits.\n")

    if args.ask:
        reply = agent.answer(args.ask)
        print(json.dumps(reply.to_dict(), indent=2, ensure_ascii=False))
        return 0

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            return 0
        if question == "/prompt":
            print(d.render_prompt())
            continue

        reply = agent.answer(question)
        print()
        if reply.refused:
            print(f"agent > [REFUSED] {reply.answer}")
        else:
            print(f"agent > {reply.answer}")
        if reply.citations:
            print(f"        sources: {', '.join(reply.citations)}")
        if reply.tool_calls:
            for call in reply.tool_calls:
                print(f"        tool {call['name']}({call['arguments']})")
        if not reply.contract_ok:
            print(f"        [contract] {'; '.join(reply.contract_notes)}")
        print()


def cmd_regress(args: argparse.Namespace) -> int:
    """Stretch goal: documents changed - re-run the old eval set and report breakage."""
    run_dir = Path(args.run)
    old_eval = EvalSet.load(run_dir / "evalset.json")
    old_def = AgentDefinition.load(run_dir / "agent.json")
    old_results_path = run_dir / "iterations"
    old_scores: dict[str, bool] = {}
    best = sorted(old_results_path.glob("iteration-*.json"))
    if best:
        payload = json.loads(best[-1].read_text(encoding="utf-8"))
        old_scores = {c["case_id"]: c["passed"] for c in payload.get("cases", [])}

    cfg = FactoryConfig()
    client = LLMClient(cfg.model, Ledger(cfg.budget))

    docs, warnings = load_documents(args.docs)
    chunks = chunk_documents(
        docs, old_def.retrieval.chunk_size, old_def.retrieval.chunk_overlap
    )
    agent = RunnableAgent(old_def, chunks, client, corpus_dir=args.docs)

    from .stages.judge import run_evaluation

    print(f"Re-running {len(old_eval.cases)} stored cases against the updated corpus")
    print(f"  documents: {[d.name for d in docs]}")
    for w in warnings:
        print(f"  ! {w}")

    card, results = run_evaluation(
        agent, old_eval, client, iteration=99, seed=cfg.seed, verbose=not args.quiet
    )

    broke = [r for r in results if r.passed is False and old_scores.get(r.case_id) is True]
    fixed = [r for r in results if r.passed is True and old_scores.get(r.case_id) is False]

    lines = [
        "# Regression report",
        "",
        f"- Agent: `{old_def.agent_id}` fingerprint `{old_def.fingerprint()}`",
        f"- New corpus: {args.docs}",
        f"- Score now: **{card.score:.3f}** ({card.passed}/{card.total})",
        f"- Newly broken: **{len(broke)}**",
        f"- Newly fixed: {len(fixed)}",
        "",
    ]
    if broke:
        lines += ["## Cases that used to pass and now fail", ""]
        for r in broke:
            lines += [
                f"- **{r.case_id}** ({r.kind}): {r.question}",
                f"  - now: {(r.answer or '(empty)')[:200]}",
                f"  - category: {r.failure_category}, gold doc retrieved="
                f"{r.gold_chunk_retrieved}, grounded={r.grounded}",
            ]
        lines += [""]
    if fixed:
        lines += ["## Cases that now pass", ""] + [
            f"- {r.case_id}: {r.question}" for r in fixed
        ] + [""]

    out = Path(args.out) if args.out else run_dir / "REGRESSION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nScore {card.score:.3f} ({card.passed}/{card.total}); "
          f"{len(broke)} newly broken, {len(fixed)} newly fixed")
    print(f"Written: {out}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    for name, tool in REGISTRY.items():
        print(f"{name}")
        print(f"  {tool.description}")
        print(f"  use when: {tool.when_to_use}")
        print(f"  args: {tool.args_schema}")
        print(f"  network: {tool.requires_network}")
        print()
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forgeai", description="The agent that builds agents.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="run the factory end to end")
    b.add_argument("--brief", help="the brief, inline")
    b.add_argument("--brief-file", help="file containing the brief")
    b.add_argument("--docs", required=True, help="the document folder")
    b.add_argument("--name", help="run name (defaults to the folder name)")
    b.add_argument("--out", help="output directory (defaults to runs/<name>-<stamp>)")
    b.add_argument("--interactive", action="store_true",
                   help="put the clarifying questions to a human")
    b.add_argument("--checkpoint", action="store_true",
                   help="require human approval of the requirements before generating")
    b.add_argument("--max-iterations", type=int, help="override the iteration cap")
    b.add_argument("--cases", type=int, help="minimum number of eval cases")
    b.add_argument("--no-tournament", action="store_true",
                   help="skip the prompt tournament (one candidate only)")
    b.add_argument("--quiet", action="store_true")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("chat", help="talk to a produced agent")
    c.add_argument("definition", help="path to agent.json")
    c.add_argument("--docs", required=True, help="the document folder it was built on")
    c.add_argument("--ask", help="ask one question and exit (JSON output)")
    c.set_defaults(func=cmd_chat)

    r = sub.add_parser("regress", help="re-run a stored eval set against changed documents")
    r.add_argument("--run", required=True, help="an existing run directory")
    r.add_argument("--docs", required=True, help="the updated document folder")
    r.add_argument("--out", help="where to write the regression report")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_regress)

    t = sub.add_parser("tools", help="list the tool registry")
    t.set_defaults(func=cmd_tools)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
