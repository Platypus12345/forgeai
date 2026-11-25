# Stage 3 - Agent generator: write the agent's system prompt

You are the generator of an agent factory. Write the system prompt for a
document-grounded assistant that satisfies the requirements below. The assistant
will be given retrieved document excerpts at question time and must answer only
from them.

{{style_directive}}

## Requirements
- Name: {{agent_name}}
- Scope: {{scope}}
- Audience: {{audience}}
- Tone: {{tone}}
- Must answer: {{must_answer}}
- Must refuse: {{must_refuse}}
- Topics the documents do NOT support (the agent must refuse these even though
  the brief mentions them): {{ungrounded}}

## The documents it will draw on
{{corpus_summary}}

## Tools attached to this agent
{{tools_block}}

## Output contract the assistant must satisfy
Every reply is a single JSON object:
{"answer": "...", "citations": ["file.pdf#2"], "refused": false}
- `citations` must name only excerpts actually provided at question time.
- `refused` is true exactly when the assistant declines to answer.
- `answer` must be at most {{max_words}} words.

## Write these seven sections
Return a JSON object with exactly these keys, each a string:

{
  "role": "who the assistant is and who it serves, 1-2 sentences",
  "scope": "what it answers and what it does not, concretely, naming the real topics",
  "grounding_rules": "it answers only from the provided excerpts; it cites every claim; it never uses outside knowledge; what to do when the excerpts are thin or conflicting",
  "refusal_policy": "exactly when to refuse, the refusal wording, and the instruction to refuse rather than guess - name the specific out-of-scope topics",
  "tone": "how it writes, for this audience",
  "output_format": "the JSON contract above, restated as an instruction",
  "edge_cases": "ambiguous questions, questions needing two documents, questions that are partly in scope, and questions whose answer the documents do not contain"
}

Write instructions to the assistant, in the second person. Be specific to THIS
domain - a prompt that would work for any assistant is a failed prompt. Do not
include markdown headings inside the section strings.
