# Stage 5 - Judge: score one answer against its expected facts

You are grading one answer from a document-grounded assistant. Deterministic
checks have already verified citation validity and refusal behaviour. Your only
job is semantic: does the answer actually contain each expected fact?

## Question
{{question}}

## Expected facts (order is randomised and carries no meaning)
{{facts_block}}

## The answer under test
{{answer}}

## Grading rules - read these carefully
- Judge each expected fact INDEPENDENTLY. For each one, the answer either
  conveys it or it does not.
- A fact counts as present if the answer states it in different words or an
  equivalent form ("three weeks" for "21 days", "450 pounds" for "GBP 450.00").
- A fact does NOT count as present if the answer merely mentions the topic, or
  hedges, or says where to find it.
- **Length is not quality.** A one-line answer containing every fact scores
  better than three paragraphs containing most of them. Do not reward
  elaboration, confidence, formatting or politeness.
- Extra correct detail is neither rewarded nor punished. Extra detail that
  CONTRADICTS an expected fact makes that fact absent.
- You are not grading the writing. You are checking facts.

## Output
{
  "facts": [{"fact": "...", "present": true}],
  "contradiction": false,
  "note": "one short sentence; name any fact you were unsure about"
}
