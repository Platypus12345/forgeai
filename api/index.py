"""Serverless ASGI front end for ForgeAI, for hosts like Vercel.

Why this exists alongside `app.py`:

`app.py` is Streamlit, and Streamlit is a long-lived server holding a
WebSocket per viewer. A serverless platform gives you a function that must
answer one HTTP request and exit, so it looks for a top-level ASGI callable
and fails the build when it finds a Streamlit script instead. The two cannot
be reconciled by configuration; they are different shapes of program.

So the split is by capability, not by preference:

  * Streamlit (`app.py`, local or Streamlit Community Cloud) can *run the
    factory* - a build takes tens of minutes and streams progress, which is
    exactly what a long-lived server is for.
  * This app can *show what the factory built and talk to it* - reading the
    committed artifacts and answering one question per request, which is
    exactly what a serverless function is for.

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

from forgeai.agent_runtime import RunnableAgent  # noqa: E402
from forgeai.config import ModelConfig  # noqa: E402
from forgeai.llm import Ledger, LLMClient  # noqa: E402
from forgeai.config import BudgetConfig  # noqa: E402

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
    cfg = ModelConfig.from_env()
    return LLMClient(cfg, Ledger(BudgetConfig()))


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


# -- rendering --------------------------------------------------------------

STYLE = """
:root{--bg:#0b0d10;--card:#13171c;--line:#242b33;--fg:#e8edf2;--dim:#9aa7b4;
--accent:#6ea8fe;--good:#4ec9a0;--bad:#f07178}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 72px}
header h1{font-size:26px;margin:0 0 4px}
header p{color:var(--dim);margin:0 0 28px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin:0 0 16px}
.card h2{font-size:17px;margin:0 0 10px}
.meta{color:var(--dim);font-size:13px}
.row{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0}
.stat{min-width:120px}
.stat b{display:block;font-size:22px}
.stat span{color:var(--dim);font-size:12px;text-transform:uppercase;
letter-spacing:.06em}
pre{background:#0e1216;border:1px solid var(--line);border-radius:8px;
padding:14px;overflow:auto;font-size:13px;white-space:pre-wrap;word-break:break-word}
input,button{font:inherit}
input[type=text]{width:100%;padding:11px 13px;border-radius:8px;
border:1px solid var(--line);background:#0e1216;color:var(--fg)}
button{padding:11px 18px;border-radius:8px;border:0;background:var(--accent);
color:#06101f;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.note{border-left:3px solid var(--accent);padding:8px 0 8px 14px;color:var(--dim);
font-size:14px;margin:14px 0}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;
border:1px solid var(--line);color:var(--dim);margin:0 6px 6px 0}
.answer{border-top:1px solid var(--line);margin-top:16px;padding-top:16px}
footer{color:var(--dim);font-size:13px;margin-top:34px}
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body><div class='wrap'>{body}"
        "<footer>ForgeAI &mdash; the factory runs from the CLI or the Streamlit "
        "app; this deployment serves what it produced."
        "</footer></div></body></html>"
    )


def model_note() -> str:
    cfg = ModelConfig.from_env()
    if cfg.provider == "heuristic":
        return (
            "<div class='note'>No model key is configured on this deployment, so "
            "answers come from the deterministic extractive path &mdash; real "
            "retrieval over the real documents, without a model writing the prose. "
            "Set <code>FORGEAI_API_KEY</code>, <code>FORGEAI_BASE_URL</code> and "
            "<code>FORGEAI_MODEL</code> in the host's environment to use a model."
            "</div>"
        )
    return f"<div class='note'>Model: {html.escape(cfg.describe())}</div>"


async def index(_: Request) -> HTMLResponse:
    runs = run_dirs()
    if not runs:
        cards = (
            "<div class='card'><h2>No runs committed yet</h2><p class='meta'>"
            "Produce one with <code>python -m forgeai.cli build --brief-file "
            "demos/hr_policy/brief.txt --docs demos/hr_policy/docs</code>.</p></div>"
        )
    else:
        cards = ""
        for path in runs:
            try:
                data = json.loads((path / "run.json").read_text(encoding="utf-8"))
            except Exception:
                continue
            cards_scores = data.get("scorecards") or []
            first = cards_scores[0]["score"] if cards_scores else None
            last = cards_scores[-1]["score"] if cards_scores else None
            stats = ""
            if first is not None:
                stats = (
                    f"<div class='row'><div class='stat'><b>{first:.3f}</b>"
                    "<span>baseline</span></div>"
                    f"<div class='stat'><b>{last:.3f}</b><span>final</span></div>"
                    f"<div class='stat'><b>{len(cards_scores) - 1}</b>"
                    "<span>iterations</span></div></div>"
                )
            cards += (
                f"<div class='card'><h2>{html.escape(path.name)}</h2>"
                f"<p class='meta'>{html.escape(str(data.get('model', '')))} &middot; "
                f"{html.escape(str(data.get('stop_reason', '')))}</p>"
                f"{stats}"
                f"<p style='margin-top:14px'><a href='/run/{path.name}'>Report</a>"
                f" &nbsp;&middot;&nbsp; <a href='/chat/{path.name}'>Chat with it</a></p>"
                "</div>"
            )
    return page(
        "ForgeAI",
        "<header><h1>ForgeAI</h1><p>A factory that turns a one-paragraph brief and "
        "a folder of documents into a tested, document-grounded agent &mdash; then "
        "measures it, diagnoses its failures and repairs them.</p></header>"
        + model_note()
        + cards,
    )


async def run_page(request: Request) -> HTMLResponse:
    run = load_run(request.path_params["run"])
    if not run:
        return page("Not found", "<div class='card'><h2>No such run</h2>"
                                 "<p><a href='/'>Back</a></p></div>")
    report = run["_dir"] / "REPORT.md"
    text = report.read_text(encoding="utf-8") if report.is_file() else "(no report)"
    name = run["_dir"].name
    return page(
        name,
        f"<header><h1>{html.escape(name)}</h1>"
        f"<p><a href='/'>&larr; All runs</a> &nbsp;&middot;&nbsp; "
        f"<a href='/chat/{name}'>Chat with this agent</a></p></header>"
        f"<div class='card'><pre>{html.escape(text)}</pre></div>",
    )


async def chat_page(request: Request) -> HTMLResponse:
    name = request.path_params["run"]
    run = load_run(name)
    if not run:
        return page("Not found", "<div class='card'><h2>No such run</h2>"
                                 "<p><a href='/'>Back</a></p></div>")
    definition = json.loads((run["_dir"] / "agent.json").read_text(encoding="utf-8"))
    tools = definition.get("tools") or []
    tag_html = "".join(f"<span class='tag'>{html.escape(t)}</span>" for t in tools)
    if not tools:
        tag_html = "<span class='tag'>no tools attached</span>"
    return page(
        f"Chat - {name}",
        f"<header><h1>{html.escape(definition.get('name') or name)}</h1>"
        f"<p><a href='/'>&larr; All runs</a> &nbsp;&middot;&nbsp; "
        f"<a href='/run/{name}'>Report</a></p></header>"
        + model_note()
        + f"<div class='card'><p class='meta'>Tools: {tag_html}</p>"
        "<p class='meta'>Answers are grounded in this run's document folder. "
        "Ask something outside it and a correctly built agent will refuse.</p>"
        "<div style='display:flex;gap:10px;margin-top:14px'>"
        "<input type='text' id='q' placeholder='Ask a question'"
        " onkeydown='if(event.key===\"Enter\")ask()'>"
        "<button id='go' onclick='ask()'>Ask</button></div>"
        "<div id='out'></div></div>"
        f"<script>const RUN={json.dumps(name)};"
        "async function ask(){"
        "const q=document.getElementById('q').value.trim();if(!q)return;"
        "const out=document.getElementById('out'),go=document.getElementById('go');"
        "go.disabled=true;out.innerHTML=\"<div class='answer meta'>Thinking...</div>\";"
        "try{const r=await fetch('/api/chat',{method:'POST',"
        "headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({run:RUN,question:q})});"
        "const d=await r.json();"
        "if(d.error){out.innerHTML=\"<div class='answer'>\"+esc(d.error)+'</div>';}"
        "else{out.innerHTML=\"<div class='answer'><p>\"+esc(d.answer)+'</p>'+"
        "\"<p class='meta'>\"+(d.refused?'refused &middot; ':'')+"
        "'cited: '+(d.citations.length?esc(d.citations.join(', ')):'none')+"
        "'</p></div>';}"
        "}catch(e){out.innerHTML=\"<div class='answer'>\"+esc(String(e))+'</div>';}"
        "go.disabled=false;}"
        "function esc(s){const d=document.createElement('div');"
        "d.textContent=s;return d.innerHTML;}</script>",
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
