# Stage 4a - Eval generator: answerable cases from a passage

You are the QA engineer of an agent factory. Below is one real passage from the
agent's document corpus. Write {{n}} test questions that this passage genuinely
answers.

## Source document
{{doc_name}} (citation label: {{cite}})

## Passage
{{passage}}

## Topics the agent must handle
{{must_answer}}

## Rules
- Ask what a real {{audience}} would ask, in their words - not "what does
  paragraph 3 state". Never refer to "the passage", "the document" or "above";
  the agent answering will not see this passage, it will have to retrieve it.
- `expected_facts` are the specific things a correct answer MUST contain:
  numbers, durations, amounts, names, conditions. Copy them from the passage
  verbatim where you can - "21 days", "GBP 450.00", "chief engineer".
  Two to four facts per question, each a short phrase, not a sentence.
- Do not write a question whose answer is not literally in the passage. If the
  passage only supports {{n}} minus one good questions, return fewer.
- Vary the form: a lookup, a condition ("what happens if..."), a boundary.

## Output
[
  {
    "question": "...",
    "expected_facts": ["...", "..."],
    "rationale": "which line of the passage answers it"
  }
]
