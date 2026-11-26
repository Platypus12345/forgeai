"""Serverless ASGI front end for ForgeAI, for hosts like Vercel.

Why this exists alongside `app.py`:

`app.py` is Streamlit, and Streamlit is a long-lived server holding a
WebSocket per viewer. A serverless platform gives you a function that must
answer one HTTP request and exit, so it looks for a top-level ASGI callable
and fails the build when it finds a Streamlit script instead. The two cannot
be reconciled by configuration; they are different shapes of program.

So the split is by capability, not by preference:

  * Streamlit (`app.py`, local) can *run the factory* - a build takes tens of
    minutes and streams progress, which is what a long-lived server is for.
  * This app can *show what the factory built and talk to it* - reading the
    committed artifacts and answering one question per request, which is what
    a serverless function is for.

A build is deliberately not exposed here. It would exceed any serverless
execution limit, and a public endpoint that spends the host's model quota on
demand is a bill, not a demo.

Without an API key the chat still answers, through the same deterministic
extractive path the CLI falls back to, so the deployment is never a dead page.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from api.render import markdown, page  # noqa: E402
from forgeai.agent_runtime import RunnableAgent  # noqa: E402
from forgeai.config import BudgetConfig, ModelConfig  # noqa: E402
from forgeai.llm import Ledger, LLMClient  # noqa: E402

RUNS = REPO / "runs"
_AGENTS: dict[str, RunnableAgent] = {}


# -- artifact access --------------------------------------------------------

def run_dirs() -> list[Path]:
    if not RUNS.is_dir():
        return []
    return sorted(
        (p for p in RUNS.iterdir() if (p / "run.json").is_file()),
        key=lambda p: p.name,
        reverse=True,
    )


def load_run(name: str) -> dict[str, Any] | None:
    """Read one run's metadata. `name` is untrusted, so it is never joined."""
    for path in run_dirs():
        if path.name == name:
            data = json.loads((path / "run.json").read_text(encoding="utf-8"))
            data["_dir"] = path
            return data
    return None


def corpus_for(run: dict[str, Any]) -> Path:
    """Resolve the corpus the run was built from, as this checkout sees it.

    run.json stores the path as it was given on the machine that produced it,
    which on Windows means backslashes. Normalising here is what lets a run
    produced on a laptop be served from a Linux function.
    """
    raw = str(run.get("corpus_dir") or "").replace("\\", "/")
    candidate = (REPO / raw) if raw and not Path(raw).is_absolute() else Path(raw)
    return candidate if candidate.is_dir() else REPO / "demos" / "hr_policy" / "docs"


def client() -> LLMClient:
    return LLMClient(ModelConfig.from_env(), Ledger(BudgetConfig()))


def agent_for(name: str) -> RunnableAgent | None:
    """Build (and keep) a runnable agent for a run.

    Cached in module state so a warm function reuses the parsed corpus; a cold
    start pays the PDF parse once.
    """
    if name in _AGENTS:
        return _AGENTS[name]
    run = load_run(name)
    if not run:
        return None
    definition = run["_dir"] / "agent.json"
    if not definition.is_file():
        return None
    agent = RunnableAgent.from_files(definition, corpus_for(run), client())
    _AGENTS[name] = agent
    return agent


def has_model() -> bool:
    return ModelConfig.from_env().provider != "heuristic"


# -- fragments --------------------------------------------------------------

def model_notice() -> str:
    if has_model():
        cfg = ModelConfig.from_env()
        return (
            "<div class='notice'><div><b>Live model.</b> Answers on this "
            f"deployment are written by <code>{html.escape(cfg.model)}</code>, "
            "grounded in the run's documents.</div></div>"
        )
    return (
        "<div class='notice'><div><b>Running without a model key.</b> Answers "
        "come from the deterministic extractive path the factory falls back to "
        "&mdash; real retrieval over the real documents, quoting them rather "
        "than writing prose, so the phrasing is seam-y on purpose. What it "
        "still does correctly is refuse questions the documents cannot "
        "support, which is the part worth testing.</div></div>"
    )


def score_block(cards: list[dict]) -> str:
    """Baseline, final and the delta between them."""
    if not cards:
        return (
            "<div class='chips'><span class='chip no'>no scorecard</span></div>"
        )
    first, last = cards[0]["score"], cards[-1]["score"]
    delta = last - first
    iters = len(cards) - 1
    if iters:
        cls = "delta" if delta > 0 else "delta flat"
        sign = "+" if delta > 0 else ""
        middle = (
            f"<span class='arrow'>&rarr;</span>"
            f"<div class='score'><b>{last:.3f}</b><span>final</span></div>"
            f"<span class='{cls}'>{sign}{delta:.3f} over "
            f"{iters} iteration{'s' if iters != 1 else ''}</span>"
        )
    else:
        middle = (
            "<span class='delta flat'>stopped at baseline</span>"
        )
    bar = int(round(last * 100))
    good = " good" if last >= 0.85 else ""
    return (
        f"<div class='scores'><div class='score'><b>{first:.3f}</b>"
        f"<span>baseline</span></div>{middle}</div>"
        f"<div class='bar'><i class='{good.strip()}' style='width:{bar}%'></i></div>"
    )


def case_chips(card: dict) -> str:
    kinds = card.get("by_kind") or {}
    label = {
        "answerable": "answerable",
        "out_of_scope": "refusals",
        "edge": "edge cases",
    }
    chips = []
    for key, text in label.items():
        k = kinds.get(key)
        if not k:
            continue
        ok = k["passed"] == k["total"]
        chips.append(
            f"<span class='chip {'ok' if ok else ''}'>{text} "
            f"{k['passed']}/{k['total']}</span>"
        )
    for cat, count in sorted((card.get("failures_by_category") or {}).items()):
        chips.append(f"<span class='chip no'>{html.escape(cat)} &times;{count}</span>")
    return f"<div class='chips'>{''.join(chips)}</div>" if chips else ""


# -- routes -----------------------------------------------------------------

async def index(_: Request) -> HTMLResponse:
    runs = run_dirs()
    body = (
        "<div class='hero'><h1>The machine that builds the chatbot</h1>"
        "<p class='lede'>ForgeAI takes a one-paragraph brief and a folder of "
        "documents and produces a working, tested, document-grounded agent "
        "&mdash; then writes its own eval set from those documents, "
        "<strong>classifies every failure</strong>, applies targeted fixes, "
        "re-tests, and reports what still does not work.</p></div>"
    ) + model_notice()

    if not runs:
        body += (
            "<div class='card empty'><h2>No runs committed yet</h2>"
            "<p class='sub'>Produce one with "
            "<code class='mono'>python -m forgeai.cli build --brief-file "
            "demos/hr_policy/brief.txt --docs demos/hr_policy/docs</code></p>"
            "</div>"
        )
        return HTMLResponse(page("ForgeAI", body))

    for path in runs:
        try:
            data = json.loads((path / "run.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        cards = data.get("scorecards") or []
        definition: dict[str, Any] = {}
        try:
            definition = json.loads(
                (path / "agent.json").read_text(encoding="utf-8")
            )
        except Exception:
            pass
        tools = definition.get("tools") or []
        tool_chips = "".join(
            f"<span class='chip key'>{html.escape(t)}</span>" for t in tools
        ) or "<span class='chip'>no tools attached</span>"
        model = str(data.get("model", "")).split(":", 1)[-1]
        runnable = (path / "agent.json").is_file()
        chat_btn = (
            f"<a class='btn primary' href='/chat/{path.name}'>Ask it something</a>"
            if runnable
            else "<span class='btn' title='this run stopped before producing an "
                 "agent' aria-disabled='true'>No agent produced</span>"
        )
        body += (
            f"<div class='card pad-lg'><div class='run-head'>"
            f"<h2>{html.escape(definition.get('name') or path.name)}</h2>"
            f"<span class='chip'>{html.escape(model)}</span></div>"
            f"<p class='sub'>{html.escape(path.name)}</p>"
            f"{score_block(cards)}"
            f"{case_chips(cards[-1]) if cards else ''}"
            f"<h3>Tools attached</h3><div class='chips'>{tool_chips}</div>"
            f"<div class='actions'>{chat_btn}"
            f"<a class='btn' href='/run/{path.name}'>Read the report</a>"
            f"</div></div>"
        )
    return HTMLResponse(page("ForgeAI", body))


async def run_page(request: Request) -> HTMLResponse:
    run = load_run(request.path_params["run"])
    if not run:
        return HTMLResponse(
            page("Not found", "<div class='card empty'><h2>No such run</h2>"
                              "<p class='sub'><a href='/'>Back to all runs</a>"
                              "</p></div>"),
            status_code=404,
        )
    name = run["_dir"].name
    report = run["_dir"] / "REPORT.md"
    text = report.read_text(encoding="utf-8") if report.is_file() else "(no report)"
    return HTMLResponse(
        page(
            name,
            f"<div class='card pad-lg'><div class='md'>{markdown(text)}</div></div>",
            nav_extra=(
                f"<a href='/chat/{name}'>Chat</a><a href='/'>All runs</a>"
            ),
        )
    )


SUGGESTIONS = {
    "hr": [
        "How many annual leave days do shore staff get?",
        "What is the notice period for resigning?",
        "What is my shipboard rotation entitlement?",
    ],
    "spares": [
        "How many AX-3310 units are in stock?",
        "What is the warranty on the turbocharger?",
        "What is the weather in Mumbai today?",
    ],
}


async def chat_page(request: Request) -> HTMLResponse:
    name = request.path_params["run"]
    run = load_run(name)
    if not run:
        return HTMLResponse(
            page("Not found", "<div class='card empty'><h2>No such run</h2>"
                              "<p class='sub'><a href='/'>Back to all runs</a>"
                              "</p></div>"),
            status_code=404,
        )
    # A run stopped by a budget or a spent quota keeps whatever it earned, but
    # it may never have reached the stage that writes agent.json. There is
    # nothing to talk to, and saying so is better than a 500.
    definition_path = run["_dir"] / "agent.json"
    if not definition_path.is_file():
        return HTMLResponse(
            page(
                "Nothing to chat with",
                "<div class='card empty'><h2>This run has no agent</h2>"
                "<p class='sub'>It stopped before the generator finished &mdash; "
                f"<em>{html.escape(str(run.get('stop_reason', 'unknown')))}</em>. "
                f"The <a href='/run/{html.escape(name)}'>report</a> still shows "
                "what it managed to produce.</p></div>",
            ),
            status_code=404,
        )
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    tools = definition.get("tools") or []
    chips = "".join(
        f"<span class='chip key'>{html.escape(t)}</span>" for t in tools
    ) or "<span class='chip'>no tools attached</span>"
    picks = SUGGESTIONS["spares" if "spare" in name else "hr"]
    buttons = "".join(
        f"<button onclick=\"fill({json.dumps(q)})\">{html.escape(q)}</button>"
        for q in picks
    )
    body = (
        f"<div class='hero'><h1>{html.escape(definition.get('name') or name)}</h1>"
        "<p>Every answer is grounded in this run's document folder. Ask "
        "something the documents do not cover and a correctly built agent "
        "refuses rather than guessing &mdash; try the last suggestion.</p></div>"
        + model_notice()
        + "<div class='card pad-lg'>"
        f"<div class='chips'>{chips}</div>"
        "<div class='ask'>"
        "<input type='text' id='q' placeholder='Ask a question'"
        " autocomplete='off' onkeydown='if(event.key===\"Enter\")ask()'>"
        "<button class='btn primary' id='go' onclick='ask()'>Ask</button></div>"
        f"<div class='suggest'>{buttons}</div>"
        "<div id='out'></div></div>"
    )
    script = (
        "<script>const RUN=" + json.dumps(name) + ";"
        "function esc(s){const d=document.createElement('div');"
        "d.textContent=s==null?'':String(s);return d.innerHTML}"
        "function fill(q){document.getElementById('q').value=q;ask()}"
        "async function ask(){"
        "const el=document.getElementById('q'),q=el.value.trim();if(!q)return;"
        "const out=document.getElementById('out'),go=document.getElementById('go');"
        "go.disabled=true;"
        "out.innerHTML=\"<div class='msg'><p class='q'>\"+esc(q)+\"</p>"
        "<p class='a'><span class='spin'></span>Retrieving and answering...</p></div>\";"
        "try{"
        "const r=await fetch('/api/chat',{method:'POST',"
        "headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({run:RUN,question:q})});"
        "const d=await r.json();"
        "if(d.error){out.innerHTML=\"<div class='msg'><p class='q'>\"+esc(q)+"
        "\"</p><p class='a'>\"+esc(d.error)+'</p></div>';}"
        "else{"
        "const foot=(d.refused?\"<span class='chip no'>refused</span>\":"
        "\"<span class='chip ok'>answered</span>\")+"
        "(d.citations||[]).map(function(c){"
        "return \"<span class='chip key'>\"+esc(c)+'</span>'}).join('')+"
        "(d.contract_ok?'':\"<span class='chip no'>contract violation</span>\");"
        "const ret=(d.retrieved||[]).map(function(c){return esc(c)}).join('<br>');"
        "out.innerHTML=\"<div class='msg'><p class='q'>\"+esc(q)+\"</p>\"+"
        "\"<p class='a'>\"+esc(d.answer)+'</p>'+"
        "\"<div class='foot'>\"+foot+'</div>'+"
        "(ret?\"<details class='detail'><summary>Chunks retrieved for this \"+"
        "\"question</summary><div class='body'>\"+ret+'</div></details>':'')+"
        "'</div>';}"
        "}catch(e){out.innerHTML=\"<div class='msg'><p class='a'>\"+"
        "esc(String(e))+'</p></div>';}"
        "go.disabled=false;el.select();}"
        "</script>"
    )
    return HTMLResponse(
        page(
            f"Chat - {name}",
            body + script,
            nav_extra=f"<a href='/run/{name}'>Report</a><a href='/'>All runs</a>",
        )
    )


async def api_chat(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "expected a JSON body"}, status_code=400)
    question = str(body.get("question") or "").strip()
    name = str(body.get("run") or "")
    if not question:
        return JSONResponse({"error": "no question given"}, status_code=400)
    if len(question) > 2000:
        return JSONResponse({"error": "question too long"}, status_code=413)
    agent = agent_for(name)
    if agent is None:
        return JSONResponse({"error": "no such run"}, status_code=404)
    try:
        reply = agent.answer(question)
    except Exception as exc:  # a hosted demo reports, it does not 500
        return JSONResponse({"error": f"the agent could not answer: {exc}"})
    return JSONResponse(
        {
            "answer": reply.answer,
            "citations": reply.citations,
            "refused": reply.refused,
            "retrieved": reply.retrieved,
            "contract_ok": reply.contract_ok,
        }
    )


async def healthz(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


app = Starlette(
    routes=[
        Route("/", index),
        Route("/healthz", healthz),
        Route("/run/{run}", run_page),
        Route("/chat/{run}", chat_page),
        Route("/api/chat", api_chat, methods=["POST"]),
    ]
)

# Some Python hosts look for `handler` or `application` instead of `app`.
handler = app
application = app
