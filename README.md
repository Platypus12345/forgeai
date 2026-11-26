# ForgeAI - the agent that builds agents

ForgeAI takes a one-paragraph brief and a folder of documents, and produces a
working, tested, document-grounded agent: a system prompt, retrieval
configuration, tool selection, an output contract, and an evaluation set it
generated from the documents themselves. It then tests the agent it just built,
classifies every failure, applies targeted fixes, re-tests, and ships a report
that says what still does not work.

It is built for the case where **every stage runs on a small, imperfect open
model** - so the interesting part is not the prompting, it is what happens when
the model returns garbage, and how you can tell whether the resulting score
means anything.

---

## Quickstart

```bash
git clone <this repo> && cd marine_assign
python -m pip install -r requirements.txt

cp .env.example .env     # then set FORGEAI_API_KEY (see "Configuring the model")

# Run the factory end to end on a demo brief
python -m forgeai.cli build \
    --brief-file demos/hr_policy/brief.txt \
    --docs       demos/hr_policy/docs

# Talk to what it built
python -m forgeai.cli chat runs/<run-dir>/agent.json --docs demos/hr_policy/docs
```

Or use the web UI for both:

```bash
streamlit run app.py
```

### Running an unseen brief at the review

One command, unattended, no code changes:

```bash
python -m forgeai.cli build --brief "<your paragraph>" --docs /path/to/your/folder
```

The factory asks its clarifying questions, answers them itself from the brief and
the documents (every auto-answer is recorded as such), and writes everything to
`runs/<name>-<timestamp>/`. Add `--interactive` to answer the questions yourself,
and `--checkpoint` to approve or edit the requirements before it generates.

Supported document types: `.pdf`, `.csv`, `.tsv`, `.md`, `.txt`. Unknown file
types are skipped with a warning rather than crashing the run, and a PDF with no
extractable text is reported as such instead of silently becoming an empty
document.

### Configuring the model

Every stage uses one model. Set it in `.env`:

| Provider | `FORGEAI_BASE_URL` | Example `FORGEAI_MODEL` |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `openai/gpt-oss-20b` |
| Together | `https://api.together.xyz/v1` | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| OpenRouter | `https://openrouter.ai/api/v1` | `qwen/qwen-2.5-7b-instruct` |
| Ollama (local) | `http://localhost:11434` | `qwen2.5:7b-instruct` |

Any OpenAI-compatible `/chat/completions` endpoint works. The committed demo
runs were produced on **`openai/gpt-oss-20b` via Groq** - a small open-weights
model, which is the regime the brief asks for.

**Rate limiting.** Free tiers meter tokens per minute (Groq allowed 8,000 TPM
while this was built). `FORGEAI_TPM` enforces that ceiling client-side with a
sliding-window throttle, so the factory paces itself instead of collecting 429s
and burning retries; when a 429 does arrive, the provider's own "try again in
Xs" hint is honoured rather than guessed at. Set `FORGEAI_TPM=0` for a local
Ollama or a paid tier. At 8,000 TPM a full demo run takes roughly 30-50 minutes,
nearly all of it spent waiting on the throttle.

Note that open reasoning models such as `gpt-oss` spend hidden reasoning tokens
out of the same output budget. The client sends `reasoning_effort: "low"` to
those models, detects replies cut off at the limit (`finish_reason: "length"`)
and retries them with a larger budget, because a truncated reply is almost
never parseable JSON.

**If no model is configured**, the factory still runs end to end on its
deterministic fallbacks and the produced agent answers extractively. This is how
the test suite runs in CI, and it is a useful smoke test - but every artifact and
the report itself are stamped `heuristic` so nobody mistakes those numbers for a
real evaluation.

---

## What it produces

Every run writes a directory of evidence:

```
runs/<name>-<timestamp>/
├── requirements.json / requirements.md   scope, audience, tone, must-answer,
│                                          must-refuse, the Q&A, groundability table
├── agent.json                             the complete declarative agent definition
├── agent_prompt.md                        its system prompt, rendered
├── evalset.json                           every generated case + expected grounding
├── iterations/iteration-N.json            per-case results for each iteration
├── iteration_log.json                     what changed, why, and the score either side
├── run.json                               scorecards, tournament, budget, judge audit
└── REPORT.md                              the honest report
```

---

## How the nine core requirements are met

| # | Requirement | Where | Notes |
|---|---|---|---|
| 1 | Intake interviewer | `forgeai/stages/intake.py` | Asks at most 5 questions, interactive or self-answered; writes the structured requirements document. |
| 2 | Groundability guard | `forgeai/stages/groundability.py` | Retrieval evidence first, model only in the ambiguous band. Unsupported topics are **moved into must-refuse**, not just annotated. |
| 3 | Agent generator | `forgeai/stages/generator.py` | Sectioned system prompt, corpus-derived retrieval parameters, vetted tool selection, output contract. |
| 4 | Eval generator | `forgeai/stages/evalgen.py` | >=20 cases from the documents; answerable (with expected facts + source doc), out-of-scope, and edge (ambiguous + multi-document). Prompts in `prompts/`. |
| 5 | Judge | `forgeai/stages/judge.py` | Deterministic-first, with explicit defences against position, verbosity and self-preference bias, plus a self-audit. |
| 6 | Critic-optimizer | `forgeai/stages/critic.py` | Classifies every failure, applies one targeted fix, max 3 iterations, full change log, rollback on regression. |
| 7 | Honest reporting | `forgeai/report.py` | Per-iteration scorecard, what still fails with the actual answers, judge audit, standing limitations. |
| 8 | Runnable product | `forgeai/agent_runtime.py`, `cli.py chat`, `app.py` | A generic runner that loads any produced definition. |
| 9 | Two demonstrations | `demos/`, `runs/` | HR policy (3 PDFs) and ship spares procurement (CSV + 2 PDFs), committed in full. |

---

## The four design decisions that matter

### 1. The system prompt is sections, not a string

`AgentDefinition.system_prompt` is an ordered dict of seven named sections
(`role`, `scope`, `grounding_rules`, `refusal_policy`, `tone`, `output_format`,
`edge_cases`). The brief requires "a targeted fix, not a full regeneration", and
you cannot target anything inside an opaque blob of text. Sections make a fix
addressable - *rewrite `refusal_policy`, leave the other six byte-identical* -
and they make the iteration log readable, because the diff is one section wide.

### 2. Tool selection is model-proposed and deterministically vetoed

The brief makes "an HR agent with a weather tool" an explicit factory defect. A
small model asked "which of these four tools do you need?" is strongly biased
toward "all of them" - it reads the list as a menu to satisfy rather than a
constraint to justify. So the model proposes, and `tools/vet_selection()` checks
the world instead of the model:

- a CSV tool requires an actual CSV in the corpus,
- a weather tool requires an actual weather/voyage signal in the brief or the
  derived requirements,

and every accept and reject is written into the agent definition as an audit
trail. On the two demos this produces **no overlap in tool selection**: the HR
agent gets what its leave-accrual and notice-period questions need, the spares
agent gets `csv_lookup` + `calculator` + `current_date`, and neither gets
`weather`.

### 3. Judging is deterministic-first, and the judge is itself audited

Using a small model to grade a small model is circular, so the model is kept out
of every judgement that code can make:

| Check | How |
|---|---|
| Did it refuse? | Read the structured `refused` field. No model. |
| Did it cite real excerpts? | Set membership against what was retrieved. No model. |
| Was the gold document retrieved? | Set membership. No model. |
| Is it grounded? | Overlap of the answer's numbers and terms with the retrieved text. No model. |
| Does it contain the expected facts? | Literal/normalised matching first. **The model is asked only about facts the matcher did not find** - i.e. only to catch paraphrase. |

The three named biases are handled concretely:

- **Position bias** - facts are shuffled and the judge runs **twice with the
  order reversed**; a fact counts as present only if both passes agree.
  Disagreements are counted and reported, not averaged away.
- **Verbosity bias** - the judge grades each fact independently and is told
  length is not quality; and since facts the matcher already found never reach
  the judge, padding cannot buy credit for facts that are genuinely present.
- **Self-preference** - the judge never sees the agent's prompt or identity.

Then `audit_judge()` scores the judge itself against planted answers whose grade
is known in advance: a known-good answer containing every fact, a known-bad
answer about something else, and the same bad answer padded to five times the
length. Recall, specificity and verbosity-resistance appear in the report
**above** the scorecard they produced. If the judge cannot tell a good answer
from an unrelated one, the reader learns that before believing the number.

### 4. Every stage degrades instead of crashing

Every structured call to the model goes through `LLMClient.structured()`, which
strips code fences, extracts the first balanced JSON value, repairs trailing
commas, and on failure sends one repair turn handing the model its own broken
output back. If that also fails, the stage takes a **deterministic fallback**
that produces a valid, useful result - a real extractive eval case, a real
serviceable prompt - and the fallback is counted. The report prints how many
stages fell back, because a run that quietly used heuristics for half its
pipeline is not a run you should trust.

This is also why an empty folder, a scanned PDF or a brief the documents cannot
support produce a *reported bad outcome* rather than a traceback. At the review
this has to run unattended on a folder nobody has seen.

---

## Stretch goals

All five are implemented.

| Goal | How to run it |
|---|---|
| **Prompt tournament** | On by default. Three candidate prompts (terse-rules, explanatory, procedural) are drafted and scored on a stratified subset of the generated eval set; the eval set picks the winner. Comparison table is in `REPORT.md`. Disable with `--no-tournament`. |
| **Budget enforcement** | Token and latency budgets per run (`FORGEAI_MAX_TOKENS`, `FORGEAI_MAX_SECONDS`), enforced in `llm.Ledger`, with early stopping when an iteration's gain falls below `FORGEAI_MIN_GAIN`. Per-stage spend is in the report. |
| **Human checkpoint** | `--checkpoint` prints the requirements and lets the requester approve or edit scope / must-answer / must-refuse before generation proceeds. |
| **Factory vs human baseline** | `python scripts/baseline_compare.py --run runs/<run> --baseline demos/hr_policy/handwritten_agent.json --docs demos/hr_policy/docs` - scores a hand-written agent and the factory's on the same eval set and reports where each wins. |
| **Regression mode** | `python -m forgeai.cli regress --run runs/<run> --docs demos/hr_policy/docs_v2` - re-runs the stored eval set against a changed corpus and reports exactly which cases broke. `docs_v2` is a revision-8 leave policy with changed entitlements and the parental leave section removed. |
| **Web UI** | `streamlit run app.py` - build, browse past runs and artifacts, and chat with any produced agent. |

---

## The two demonstrations

Chosen to be as unlike each other as possible, because the factory has to work
on a brief it has never seen.

**Demo 1 - HR policy assistant** (`demos/hr_policy/`): three prose PDFs for
shore staff. The brief deliberately asks for two things the documents do not
contain - *shipboard rotation entitlement* and *sea-service days for certificate
revalidation* - because the leave policy is explicitly scoped to shore staff.
A factory that generates an agent which answers those anyway has failed core
requirement 2. This is the groundability test.

**Demo 2 - Ship spares procurement** (`demos/ship_spares/`): a 24-row CSV
catalogue plus two interlocking policy PDFs. Tabular, arithmetic-heavy and
date-sensitive, with facts that span documents by construction (the warranty on
a turbocharger needs the catalogue's supplier column *and* the supplier terms).
This is the tool-selection and multi-document test.

Both corpora are LLM-synthesised; the synthesis prompts and the reasoning behind
them are committed in `demos/SYNTHESIS.md`, and the plain-text sources sit beside
the PDFs in `demos/<run>/source/` so the content is reviewable as a diff.
`scripts/make_demo_pdfs.py` regenerates the PDFs.

---

## Tests

```bash
python -m pytest tests -q
```

52 tests, no network and no model required. They concentrate on the
deterministic machinery - JSON salvage, chunking, retrieval, the tool veto,
contract enforcement, failure classification, the full artifact set - because
that is the layer the factory leans on when the model misbehaves, and its bugs
would *inflate a score* rather than crash a run.

One of them caught exactly that: `groundedness()` originally awarded its numeric
component to answers containing no numbers at all, giving every number-free
answer a floor of 0.6 - above the pass threshold - so a fluent, wholly fabricated
prose answer scored as grounded. The test is
`test_groundedness_of_unrelated_answer_is_low`.

---

## Repository layout

```
forgeai/
├── config.py          model + budget configuration
├── llm.py             the one model client: JSON salvage, repair, fallbacks, ledger
├── schemas.py         every artifact as a dataclass
├── docs.py            loading + chunking (CSV chunks repeat their header)
├── retrieval.py       BM25, pure standard library
├── prompts.py         loader for the committed stage prompts
├── agent_runtime.py   the produced agent, with contract enforcement
├── factory.py         the orchestrator
├── report.py          the honest report
├── cli.py             build / chat / regress / tools
├── stages/            intake, groundability, generator, evalgen, judge, critic
└── tools/             csv_lookup, calculator, current_date, weather + the veto
prompts/               all 12 stage prompts, committed
demos/                 the two briefs, corpora, sources and synthesis prompts
scripts/               make_demo_pdfs.py, baseline_compare.py
tests/                 the test suite
app.py                 Streamlit UI
```

---

## Known limitations

Stated here as well as in every generated report, because they bound what any
score from this system means:

1. **The tests and the agent share an author.** The same model writes the eval
   questions and the agent's prompt, so they share blind spots. A question the
   model would not think to ask is a weakness neither the agent nor the score
   will show.
2. **Expected facts are the model's opinion of what matters.** A wrong expected
   fact marks a correct answer as a failure, and the reverse.
3. **Groundedness is lexical.** A fluent sentence that recombines real terms into
   a false claim can score as grounded.
4. **Twenty-odd cases is a small sample.** A few points between iterations is
   noise, not progress. The loop's rollback rule mitigates acting on that noise;
   it does not eliminate it.
5. **Refusal is binary.** An over-cautious agent can bank easy points on the
   out-of-scope third of the set, which is why the report always breaks the score
   down by case kind rather than leading with the single number.
6. **No reranking or embeddings.** Retrieval is BM25 only, which is a deliberate
   trade for zero heavy dependencies and a clone-and-run review, but it will lose
   to a dense retriever on vocabulary-mismatch questions. The optimizer's
   `query_expansion` rung is a partial mitigation.
