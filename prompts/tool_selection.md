# Stage 3b - Agent generator: choose tools

Choose which tools this agent needs. Attaching a tool the agent does not need is
a defect, not a bonus: it widens the attack surface, invites the agent to use it
when it should be reading documents, and makes the agent slower. Most
document-grounded assistants need NO tools at all. Attach a tool only when a
question the agent is REQUIRED to answer cannot be answered without it.

## The agent
- Scope: {{scope}}
- Must answer: {{must_answer}}
- Capabilities the requester implied: {{capabilities}}

## The documents
{{corpus_summary}}

## The registry
{{catalogue}}

## Test yourself before attaching each tool
For each tool you select, name a specific question from "must answer" that the
agent CANNOT answer correctly without it. If you cannot name one, do not select
it.

## Output
{
  "tools": ["only the tool names you can justify, possibly empty"],
  "rationale": {"tool_name": "the specific question that requires it"},
  "rejected": {"tool_name": "why this agent does not need it"}
}
