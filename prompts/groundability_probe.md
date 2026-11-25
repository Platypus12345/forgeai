# Stage 2 - Groundability guard: judge the evidence for one topic

A retrieval search was run over the document corpus for a topic the agent is
supposed to answer. Decide whether the documents genuinely support it.

## Topic
{{topic}}

## Best matching passages found
{{evidence}}

## Decide
- "supported": the passages contain the substance needed to answer questions on
  this topic.
- "partial": the topic is touched on but a user would hit gaps quickly.
- "unsupported": the passages are about something else, or merely mention the
  words without the content. Keyword overlap is NOT support.

Be strict. The purpose of this check is to stop the factory from shipping an
agent that will invent answers for a topic its documents never covered. A false
"supported" is the expensive mistake here.

## Output
{"status": "supported|partial|unsupported", "reason": "one sentence, citing what is or is not present"}
