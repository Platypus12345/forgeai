# ForgeAI run report - Shore Staff Policy Guide

## Headline

- **Final score:** 0.889 (17/22 cases passed)
- **Winning iteration:** 0
- **Model (every stage):** `openai:qwen/qwen3.8-27b`
- **Wall clock:** 4076.8s
- **Stop reason:** pass threshold reached (0.889 >= 0.85)

## The brief

> I need an assistant that answers HR policy questions for our shore staff. People
constantly ask the HR team the same things about annual leave, sick leave, pay
dates, overtime, notice periods and the hybrid working rules, and we want them to
self-serve. It should also be able to tell someone their shipboard rotation
entitlement and how many sea-service days they need for their next certificate
revalidation. Keep answers short and plain, and it must never give anyone advice
about their individual case or make a decision for a manager.

## Per-iteration scorecard

| Iteration | Score | Passed | Answerable | Out-of-scope | Edge | Failures |
|---|---|---|---|---|---|---|
| 0 | **0.889** | 17/22 | 9/12 | 6/6 | 2/4 | hallucination:1, prompt_defect:3, wrong_refusal:1 |

## The agent that was produced

- **Name:** Shore Staff Policy Guide
- **Fingerprint:** `6aa5d40ec3a5` (version 1)
- **Lineage:** initial (style: explanatory, prompt source: fallback)
- **Retrieval:** chunk 900/150 overlap, top_k 5, min_score 0.05, per-doc cap 2, query expansion off
- **Output contract:** json, fields ['answer', 'citations', 'refused'], max 160 words

### Tools

- **None.** Nothing in this brief requires a tool; the agent answers purely from retrieved documents.

Selection audit trail:

- model proposed: [] (source: repair)

## Groundability guard

- supported: 8
- partial: 0
- **unsupported: 0**

## The eval set

22 cases generated from the documents themselves: 12 answerable, 4 edge, 6 out_of_scope.

Generation prompts (committed): `prompts/evalgen_answerable.md`, `prompts/evalgen_refusal.md`, `prompts/evalgen_edge.md`

Case provenance: 12 from fallback, 8 from model, 2 from repair.

- :warning: 12 of 22 cases came from the deterministic generator because the model's output was unusable for that seed. These are mechanically derived and test retrieval more than comprehension.

## Prompt tournament

Three candidate system prompts were drafted in different styles and
scored on a stratified subset of the generated eval set. The eval set
picked the winner, not the author.

| Style | Score | Passed | Winner |
|---|---|---|---|
| rules | 0.829 | 5/6 | no |
| explanatory | 0.830 | 5/6 | **yes** |
| procedural | 0.829 | 5/6 | no |

## What the optimizer changed

_No iteration ran: the baseline already met the threshold, or the loop stopped before the first fix._

## What still fails, and why

5 of 22 cases still fail at iteration 0.

### prompt_defect (3) - the agent had the evidence but its instructions led it astray

- **ans-003** (answerable): What do the documents say about minimum cent pensionable meridian?
  - agent said: Meridian contributes a minimum of 8 per cent of pensionable pay to the defined contribution pension scheme. Employees contributing 7 per cent or more receive an enhanced employer contribution of 10 per cent.
  - refused=False, grounded=1.0, fact recall=0.5, gold doc retrieved=True, contract ok=True
- **ans-012** (answerable): A supplier offered me a gift worth about 300 pounds — can I accept it?
  - agent said: No. Gifts above £250 may not be accepted. You should decline the gift.
  - refused=False, grounded=0.9, fact recall=0.5, gold doc retrieved=True, contract ok=True
- **edge-021** (edge): How do meridian maritime service and above receive overtime relate to each other?
  - agent said: Assuming you're asking how grades 6 and above handle overtime: they receive no overtime payment. Time off in lieu may be agreed instead.
  - refused=False, grounded=0.867, fact recall=0.5, gold doc retrieved=True, contract ok=True

### hallucination (1) - the agent asserted something the retrieved text does not say

- **ans-011** (answerable): Can I use my company laptop for personal stuff, like streaming or personal email?
  - agent said: Company devices are for business use; incidental personal use is permitted. The policy does not define what counts as 'incidental,' so it does not specifically confirm or prohibit streaming or personal email. You should check with your line
  - refused=False, grounded=0.364, fact recall=0.667, gold doc retrieved=True, contract ok=True

### wrong_refusal (1) - the agent declined a question its documents could answer

- **edge-019** (edge): What about meridian maritime?
  - agent said: I can't answer that from the documents I've been given. The question is too broad to map to a specific policy. I can help with annual leave, sick leave, pay dates, overtime, hybrid working, home working equipment, or acceptable use of IT fo
  - refused=True, grounded=1.0, fact recall=0.0, gold doc retrieved=True, contract ok=True

## Can you trust the judge?

The judge was tested against planted answers on 3 case(s):

- **Recall on known-good answers:** 1.00 (it should mark every fact present)
- **Specificity on known-bad answers:** 1.00 (it should mark every fact absent)
- **Verbosity resistance:** 1.00 (the same wrong answer, padded five times longer)
- **Verdict:** trustworthy

### How judging actually works here

Scoring is deterministic-first. Refusal correctness, citation validity,
whether the gold document was retrieved, and the contract check are pure
code with no model involved. Expected facts are matched literally first;
the model is asked only about facts the matcher did not find, i.e. only
to catch paraphrase, and then twice with the fact order reversed - a fact
counts as present only if both passes agree.

## Budget

- model calls: 181
- tokens (estimated): 202,109 of 900,000
- wall clock: 4076.8s of 18000s
- JSON repair turns needed: 11
- deterministic fallbacks taken: 8
- API calls that failed outright: 64
- replies truncated at the token limit: 18

| Stage | Calls | Tokens | Seconds | Repairs | Fallbacks | API errors |
|---|---|---|---|---|---|---|
| agent.answer | 65 | 99,699 | 31.86 | 0 | 0 | 25 |
| judge | 38 | 22,322 | 16.43 | 1 | 0 | 11 |
| evalgen.answerable | 21 | 20,077 | 18.61 | 2 | 1 | 9 |
| groundability | 10 | 11,674 | 6.04 | 0 | 0 | 3 |
| intake.requirements | 3 | 8,467 | 5.96 | 0 | 0 | 1 |
| evalgen.edge.multi_document | 6 | 7,135 | 4.74 | 1 | 1 | 2 |
| intake.questions | 6 | 7,059 | 6.34 | 1 | 1 | 2 |
| generator.prompt | 12 | 6,271 | 7.76 | 3 | 3 | 3 |
| intake.auto_answer | 3 | 5,846 | 5.56 | 0 | 0 | 1 |
| evalgen.edge.ambiguous | 6 | 5,193 | 4.77 | 1 | 1 | 2 |
| generator.tools | 5 | 4,663 | 2.79 | 1 | 0 | 3 |
| evalgen.refusal | 6 | 3,703 | 4.42 | 1 | 1 | 2 |

> :warning: 64 model call(s) failed at the transport
> level (rate limit, timeout or provider error) and exhausted their
> retries. Each one pushed its stage onto a deterministic fallback, so
> this run is partly a measurement of the provider, not only of the
> factory. Sample error(s):

> - `429 from model endpoint: {"error":{"message":"Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_01m8ae5p65e538bpqsjz0ac8ax` service tier `on_demand` on output tokens per minute (OTP`
> - `429 from model endpoint: {"error":{"message":"Request too large for model `qwen/qwen3.8-27b` in organization `org_01m8ae5p65e538bpqsjz0ac8ax` service tier `on_demand` on output tokens per minute (OTPM`

> 8 stage call(s) fell back to
> deterministic logic because the model's output could not be parsed or
> validated. Those parts of the pipeline did not benefit from the model.

## Limitations of this report

These apply to every run, including the good-looking ones:

1. **The tests and the agent share an author.** The same model wrote the
   eval questions and the agent's prompt, so they share blind spots. A
   question the model would not think to ask is a weakness neither the
   agent nor the score will show.
2. **Expected facts are the model's opinion of what matters.** The grading
   key was extracted from the documents by the same small model. A wrong
   expected fact marks a correct answer as a failure, and the reverse.
3. **Groundedness is lexical.** It measures whether the answer's numbers and
   terms appear in the retrieved text. A fluent sentence that recombines
   real terms into a false claim can score as grounded.
4. **The eval set is small.** Twenty-odd cases put a wide confidence
   interval on any score; a few points of difference between iterations is
   noise, not progress.
5. **Refusal is scored as a binary.** An agent that refuses politely and an
   agent that refuses rudely score identically, and an over-cautious agent
   can bank easy points on the out-of-scope third of the set.
