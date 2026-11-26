"""Streamlit UI for ForgeAI: run the factory, then chat with what it built.

    streamlit run app.py

Everything here is a thin shell over the same `Factory` and `RunnableAgent`
the CLI uses - the UI cannot produce an agent the CLI could not, and it reads
the same artifacts from `runs/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from forgeai.agent_runtime import RunnableAgent
from forgeai.config import RUNS_DIR, FactoryConfig
from forgeai.docs import chunk_documents, load_documents
from forgeai.factory import Factory
from forgeai.llm import Ledger, LLMClient
from forgeai.schemas import AgentDefinition

st.set_page_config(page_title="ForgeAI", page_icon="=", layout="wide")

REPO = Path(__file__).resolve().parent
UPLOADS = REPO / "uploads"


def apply_runtime_credentials() -> None:
    """Make the hosted app usable without spending the author's own quota.

    A public deployment is a shop window, not a free compute service. The key
    is never committed and never baked into the deployment: it comes either
    from Streamlit secrets, if whoever hosts this chooses to fund the demo, or
    from a visitor's own key typed into the sidebar and held for that session
    only. With no key at all the app still does something honest - it reads
    the committed runs and answers through the deterministic extractive path -
    which is the same no-model degradation the CLI falls back to.
    """
    for key in (
        "FORGEAI_PROVIDER", "FORGEAI_BASE_URL", "FORGEAI_MODEL", "FORGEAI_API_KEY",
    ):
        try:
            if key in st.secrets:  # type: ignore[operator]
                os.environ.setdefault(key, str(st.secrets[key]))
        except Exception:
            pass  # no secrets.toml at all, which is normal when run locally

    typed = st.sidebar.text_input(
        "API key (optional, this session only)",
        type="password",
        help=(
            "A Groq or other OpenAI-compatible key. It is kept for this "
            "browser session, never written to disk and never logged. Without "
            "one, the Runs and Chat pages still work on the committed "
            "artifacts using the deterministic fallback."
        ),
    )
    if typed.strip():
        os.environ["FORGEAI_API_KEY"] = typed.strip()
        os.environ.setdefault("FORGEAI_PROVIDER", "openai")
        os.environ.setdefault("FORGEAI_BASE_URL", "https://api.groq.com/openai/v1")
        os.environ.setdefault("FORGEAI_MODEL", "openai/gpt-oss-20b")


def model_banner() -> FactoryConfig:
    cfg = FactoryConfig()
    if cfg.model.provider == "heuristic":
        st.sidebar.warning(
            "No model configured, so the factory would run on its "
            "deterministic fallbacks only. Paste a key in the sidebar, or set "
            "FORGEAI_API_KEY and FORGEAI_BASE_URL in .env when running "
            "locally. Reading past runs needs no key."
        )
    else:
        st.sidebar.success(f"Model: {cfg.model.describe()}")
    return cfg


def page_build() -> None:
    st.header("Build an agent")
    cfg = model_banner()

    brief = st.text_area(
        "The brief (one paragraph)",
        height=140,
        placeholder="I need an assistant that answers HR policy questions for shore staff...",
    )

    source = st.radio(
        "Documents", ["Use a demo corpus", "Point at a folder", "Upload files"],
        horizontal=True,
    )
    docs_dir: Path | None = None

    if source == "Use a demo corpus":
        demos = sorted(p for p in (REPO / "demos").iterdir() if (p / "docs").is_dir())
        if demos:
            choice = st.selectbox("Demo", [d.name for d in demos])
            docs_dir = REPO / "demos" / choice / "docs"
            if not brief:
                brief_file = REPO / "demos" / choice / "brief.txt"
                if brief_file.exists():
                    brief = st.text_area(
                        "The brief (loaded from the demo)",
                        value=brief_file.read_text(encoding="utf-8").strip(),
                        height=140,
                        key="demo_brief",
                    )
    elif source == "Point at a folder":
        typed = st.text_input("Folder path", value=str(REPO / "demos" / "hr_policy" / "docs"))
        # On a public deployment this box would otherwise read any directory
        # the container can see, so it is confined to the checkout. Locally
        # that is no restriction worth minding; on a hosted instance it is the
        # difference between a demo and a file browser.
        if typed:
            candidate = Path(typed).expanduser()
            try:
                resolved = candidate.resolve()
                resolved.relative_to(REPO.resolve())
                docs_dir = resolved
            except (ValueError, OSError):
                st.error(
                    "For safety this box only opens folders inside the "
                    "ForgeAI checkout. Use the upload option for anything "
                    "else, or run the CLI locally."
                )
                docs_dir = None
    else:
        uploaded = st.file_uploader(
            "Documents (pdf, csv, md, txt)", accept_multiple_files=True
        )
        if uploaded:
            UPLOADS.mkdir(exist_ok=True)
            target = UPLOADS / "session"
            target.mkdir(exist_ok=True)
            for f in uploaded:
                (target / f.name).write_bytes(f.getbuffer())
            docs_dir = target
            st.caption(f"{len(uploaded)} file(s) staged in {target}")

    col1, col2, col3 = st.columns(3)
    iterations = col1.slider("Max iterations", 0, 3, 3)
    min_cases = col2.slider("Minimum eval cases", 20, 40, 20)
    tournament = col3.checkbox("Prompt tournament", value=True)

    if docs_dir and docs_dir.exists():
        try:
            docs, warnings = load_documents(docs_dir)
            st.caption(
                f"{len(docs)} document(s) detected: "
                + ", ".join(f"{d.name} ({d.kind})" for d in docs)
            )
            for w in warnings:
                st.warning(w)
        except Exception as exc:
            st.error(f"Could not read that folder: {exc}")

    if st.button("Run the factory", type="primary", disabled=not (brief and docs_dir)):
        cfg.max_iterations = iterations
        cfg.min_eval_cases = min_cases
        cfg.prompt_tournament = tournament
        with st.status("Running...", expanded=True) as status:
            log = st.empty()
            lines: list[str] = []

            class UILogFactory(Factory):
                def log(self, message: str) -> None:  # type: ignore[override]
                    lines.append(message)
                    log.code("\n".join(lines[-28:]))

            factory = UILogFactory(cfg, verbose=True)
            result = factory.build(brief, docs_dir, run_name=Path(docs_dir).name)
            status.update(
                label=f"Done - score {result.final_score:.3f}",
                state="error" if result.error else "complete",
            )
        st.session_state["last_run"] = str(result.run_dir)
        st.success(f"Artifacts written to {result.run_dir}")
        if (result.run_dir / "REPORT.md").exists():
            st.markdown((result.run_dir / "REPORT.md").read_text(encoding="utf-8"))


def _runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        (p for p in RUNS_DIR.iterdir() if (p / "agent.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def page_runs() -> None:
    st.header("Past runs")
    runs = _runs()
    if not runs:
        st.info("No completed runs yet.")
        return
    choice = st.selectbox("Run", [p.name for p in runs])
    run_dir = RUNS_DIR / choice

    tabs = st.tabs(["Report", "Requirements", "Agent", "Eval set", "Iterations"])
    with tabs[0]:
        path = run_dir / "REPORT.md"
        st.markdown(path.read_text(encoding="utf-8") if path.exists() else "_none_")
    with tabs[1]:
        path = run_dir / "requirements.md"
        st.markdown(path.read_text(encoding="utf-8") if path.exists() else "_none_")
    with tabs[2]:
        definition = AgentDefinition.load(run_dir / "agent.json")
        st.code(definition.render_prompt())
        st.json(
            {
                "tools": definition.tools,
                "retrieval": definition.retrieval.to_dict(),
                "output_contract": definition.output_contract.to_dict(),
            }
        )
    with tabs[3]:
        path = run_dir / "evalset.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            st.caption(f"{data['total']} cases: {data['counts']}")
            st.dataframe(
                [
                    {
                        "id": c["case_id"],
                        "kind": c["kind"],
                        "question": c["question"],
                        "refuse?": c["should_refuse"],
                        "expected facts": ", ".join(c["expected_facts"]),
                    }
                    for c in data["cases"]
                ],
                use_container_width=True,
            )
    with tabs[4]:
        path = run_dir / "run.json"
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            st.dataframe(meta.get("scorecards", []), use_container_width=True)
            st.json(meta.get("iterations", []))


def page_chat() -> None:
    st.header("Chat with a produced agent")
    cfg = model_banner()
    runs = _runs()
    if not runs:
        st.info("Build an agent first.")
        return

    choice = st.selectbox("Agent", [p.name for p in runs])
    run_dir = RUNS_DIR / choice
    meta_path = run_dir / "run.json"
    default_docs = ""
    if meta_path.exists():
        default_docs = json.loads(meta_path.read_text(encoding="utf-8")).get("corpus_dir", "")
    docs_dir = st.text_input("Document folder", value=default_docs)

    if not docs_dir or not Path(docs_dir).exists():
        st.warning("Point at the document folder this agent was built on.")
        return

    key = f"chat::{choice}"
    if st.session_state.get("chat_key") != key:
        st.session_state["chat_key"] = key
        st.session_state["messages"] = []

    definition = AgentDefinition.load(run_dir / "agent.json")
    docs, _ = load_documents(docs_dir)
    chunks = chunk_documents(
        docs, definition.retrieval.chunk_size, definition.retrieval.chunk_overlap
    )
    agent = RunnableAgent(
        definition, chunks, LLMClient(cfg.model, Ledger(cfg.budget)), corpus_dir=docs_dir
    )
    st.caption(
        f"{definition.name} | tools: {definition.tools or 'none'} | "
        f"top_k {definition.retrieval.top_k} | {len(chunks)} chunks"
    )

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    if question := st.chat_input("Ask the agent something"):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            reply = agent.answer(question)
            body = (
                f"**[REFUSED]** {reply.answer}" if reply.refused else reply.answer
            )
            st.markdown(body)
            bits = []
            if reply.citations:
                bits.append("sources: " + ", ".join(reply.citations))
            if reply.tool_calls:
                bits.append("tools: " + ", ".join(c["name"] for c in reply.tool_calls))
            if not reply.contract_ok:
                bits.append("contract: " + "; ".join(reply.contract_notes))
            meta = " | ".join(bits)
            if meta:
                st.caption(meta)
        st.session_state["messages"].append(
            {"role": "assistant", "content": body, "meta": meta}
        )


st.sidebar.title("ForgeAI")
st.sidebar.caption("The agent that builds agents")
apply_runtime_credentials()
PAGE = st.sidebar.radio("", ["Build", "Runs", "Chat"], label_visibility="collapsed")
st.sidebar.divider()

if PAGE == "Build":
    page_build()
elif PAGE == "Runs":
    page_runs()
else:
    page_chat()
