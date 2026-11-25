# Stage 4c - Eval generator: edge cases

You are the QA engineer of an agent factory. Write {{n}} hard-but-fair test
cases of type **{{edge_type}}**.

{{edge_instruction}}

## Context
- Scope: {{scope}}
- Audience: {{audience}}

## Source material
{{passages}}

## Output
[
  {
    "question": "...",
    "expected_facts": ["the specific things a good answer must contain"],
    "expected_source_docs": ["file names the answer must draw on"],
    "rationale": "what makes this an edge case and what a good answer does"
  }
]
