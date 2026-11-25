# Stage 6 - Optimizer: rewrite ONE section of the agent's system prompt

The agent failed part of its evaluation. The diagnosis says the cause is
**{{category}}**. Rewrite exactly one section of its system prompt -
`{{section}}` - to fix it.

## The agent's job
- Scope: {{scope}}
- Audience: {{audience}}
- Must refuse: {{must_refuse}}

## Current text of the `{{section}}` section
{{current_section}}

## The failures this must fix
{{failures_block}}

## Diagnosis
{{diagnosis}}

## Rules
- Fix the named cause and nothing else. Do not restate the whole prompt. Do not
  add instructions that belong in another section.
- Be concrete: name the actual topics, documents and behaviours involved in the
  failures above. A generic instruction ("be accurate", "be helpful") will not
  change the model's behaviour and counts as no fix at all.
- Do not over-correct. Tightening refusal until the agent refuses everything
  trades one failure category for another, and the next evaluation will catch it.
- Keep it under 160 words.

## Output
{
  "section": "{{section}}",
  "new_text": "the rewritten section",
  "what_changed": "one sentence on the behavioural difference this should make"
}
