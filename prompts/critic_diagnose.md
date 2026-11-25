# Stage 6a - Critic: explain a cluster of failures

Deterministic checks have already classified these failures as **{{category}}**
using hard evidence (whether the gold passage was retrieved, whether the agent
refused, whether its claims appear in the retrieved text). Do not re-classify
them.

Your job is to say, in two or three sentences, what the agent is doing wrong
that produces this cluster, specifically enough that a targeted edit can fix it.

## Category
{{category}} - {{category_meaning}}

## Failing cases
{{failures_block}}

## The agent's current instructions for the relevant section (`{{section}}`)
{{current_section}}

## Output
{"diagnosis": "two or three sentences naming the concrete defect"}
