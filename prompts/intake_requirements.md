# Stage 1b - Intake interviewer: write the requirements document

You are the intake interviewer of an agent factory. Turn the brief, the document
evidence and the requester's answers into a precise, structured requirements
document. Everything downstream - the agent's prompt, its tools, its test set -
is generated from what you write here, so be concrete and avoid restating the
brief in fancier words.

## The brief
{{brief}}

## What the documents actually contain
{{corpus_summary}}

## Excerpts from the documents
{{corpus_preview}}

## Clarifying questions and answers
{{qa_block}}

## Rules
- `must_answer_topics`: 4-8 specific topics, each phrased as the kind of thing a
  user would actually ask about. Each one must be plausibly answerable from the
  documents above. Prefer "notice period for resignation" over "HR policies".
- `must_refuse_topics`: 4-8 topics that are adjacent and tempting but out of
  scope, or that the documents cannot support, or that are unsafe for this
  audience. Include the obvious near-misses a real user would try.
- `scope`: two or three sentences a new engineer could use to decide whether any
  given question is in or out.
- `requested_capabilities`: anything the brief implies beyond retrieval, in plain
  words - for example "arithmetic on prices", "needs to know today's date",
  "look up rows in the catalogue". Leave the array empty if retrieval is enough.
- `agent_name`: three or four words, no vendor names, no "AI".

## Output
Return exactly this JSON object:

{
  "agent_name": "...",
  "scope": "...",
  "audience": "...",
  "tone": "...",
  "must_answer_topics": ["..."],
  "must_refuse_topics": ["..."],
  "requested_capabilities": ["..."]
}
