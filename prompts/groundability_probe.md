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

Be strict about substance, but note what you are looking at: these are the
best-matching passages, not the whole corpus, and a passage marked truncated
continues beyond what you can see. If the topic asks about one specific row,
code or identifier and these passages are a sample of a larger table, treat the
*kind* of information as the question - if the table clearly carries that column
for its other rows, the topic is supported. Do not answer "unsupported" merely
because one identifier is missing from an excerpt.

A false "supported" ships an agent that invents answers. A false "unsupported"
pushes a topic the documents do cover into the refusal policy, and the agent
then declines its own subject. Both are real failures; weigh the evidence rather
than defaulting either way.

## Output
{"status": "supported|partial|unsupported", "reason": "one sentence, citing what is or is not present"}
