# Stage 1c - Unattended intake: answer the factory's own questions

The factory is running unattended: there is no human to answer the intake
questions. Answer them yourself, honestly, using only the brief and the document
evidence. Where the evidence does not settle a question, say so plainly and give
the most conservative reasonable default - a narrower scope and a stricter
refusal policy are the safe directions.

## The brief
{{brief}}

## What the documents actually contain
{{corpus_summary}}

## Excerpts from the documents
{{corpus_preview}}

## The questions
{{questions_block}}

## Output
Return a JSON array, one entry per question, in the same order:

[
  {"question": "...", "answer": "the answer, and the evidence or assumption behind it"}
]
