"""BM25 retrieval over the chunked corpus.

Pure standard library on purpose. A reviewer clones the repo and runs it on an
unseen folder; "pip install torch" is a way to fail that demo. BM25 over
lightly normalised tokens is strong enough for the document sizes in play, and
every parameter the critic-optimizer is allowed to tune (`top_k`,
`chunk_size`, `chunk_overlap`, `min_score`, `query_expansion`, `per_doc_cap`)
is a parameter of *this* module, so a "retrieval miss" diagnosis has somewhere
concrete to go.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .schemas import Chunk, RetrievalConfig

STOPWORDS = {
    "a", "about", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "been", "but", "by", "can", "did", "do", "does", "doing", "for",
    "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
    "its", "may", "me", "more", "most", "must", "my", "no", "not", "of", "on",
    "or", "our", "out", "over", "shall", "should", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "up", "was", "we", "were", "what", "when", "where", "which",
    "who", "whom", "why", "will", "with", "would", "you", "your",
}

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-/\.]*")


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN.findall((text or "").lower())
    out = []
    for t in tokens:
        t = t.strip("-/.")
        if not t:
            continue
        if not keep_stopwords and t in STOPWORDS:
            continue
        # Crude singularisation: "policies"->"policy", "days"->"day".
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


@dataclass
class Hit:
    chunk: Chunk
    score: float

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class BM25Index:
    """Classic Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tokens = [tokenize(c.text) for c in chunks]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.freqs = [Counter(t) for t in self.tokens]
        self.df: Counter[str] = Counter()
        for tf in self.freqs:
            self.df.update(tf.keys())
        self.n = len(chunks)
        self._idf = {
            term: math.log(1 + (self.n - d + 0.5) / (d + 0.5))
            for term, d in self.df.items()
        }
        self._max_possible = sum(sorted(self._idf.values(), reverse=True)[:8]) or 1.0

    def _raw_score(self, query_tokens: list[str], i: int) -> float:
        tf, length = self.freqs[i], self.lengths[i]
        score = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self._idf.get(term, 0.0)
            denom = f + self.k1 * (
                1 - self.b + self.b * (length / self.avg_len if self.avg_len else 1)
            )
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(
        self, query: str, top_k: int = 4, *, expansion_terms: list[str] | None = None
    ) -> list[Hit]:
        qt = tokenize(query)
        if expansion_terms:
            qt = qt + [t for e in expansion_terms for t in tokenize(e)]
        if not qt:
            return []
        scored = [(self._raw_score(qt, i), i) for i in range(self.n)]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        # Normalise to a roughly 0..1 band so `min_score` means something
        # comparable across corpora of different sizes.
        return [
            Hit(self.chunks[i], min(1.0, s / self._max_possible))
            for s, i in scored[: max(top_k * 4, top_k)]
        ][:top_k]


class Retriever:
    """A configured view over the index: applies min_score and per-doc spread."""

    def __init__(self, chunks: list[Chunk], config: RetrievalConfig):
        self.config = config
        self.chunks = chunks
        self.index = BM25Index(chunks)
        self.by_id = {c.chunk_id: c for c in chunks}

    def expansion_terms(self, query: str) -> list[str]:
        """Cheap pseudo-relevance feedback: the top terms of the best first hit.

        Enabled only when the optimizer turns `query_expansion` on, which it
        does in response to an actual retrieval miss - not by default, because
        on a small corpus it just as often drags in noise.
        """
        hits = self.index.search(query, top_k=1)
        if not hits:
            return []
        counts = Counter(tokenize(hits[0].chunk.text))
        qt = set(tokenize(query))
        return [t for t, _ in counts.most_common(12) if t not in qt][:5]

    def retrieve(self, query: str) -> list[Hit]:
        cfg = self.config
        expansion = self.expansion_terms(query) if cfg.query_expansion else None
        pool = self.index.search(
            query, top_k=max(cfg.top_k * 3, cfg.top_k), expansion_terms=expansion
        )
        pool = [h for h in pool if h.score >= cfg.min_score]

        if cfg.per_doc_cap and cfg.per_doc_cap > 0:
            per_doc: dict[str, int] = {}
            spread: list[Hit] = []
            for h in pool:
                used = per_doc.get(h.chunk.doc_id, 0)
                if used >= cfg.per_doc_cap:
                    continue
                per_doc[h.chunk.doc_id] = used + 1
                spread.append(h)
            pool = spread
        return pool[: cfg.top_k]

    def evidence_score(self, text: str) -> tuple[float, str]:
        """Best single piece of evidence for `text`, as (score, short snippet)."""
        hits = self.index.search(text, top_k=3)
        if not hits:
            return 0.0, ""
        best = hits[0]
        snippet = " ".join(best.chunk.text.split())[:160]
        return best.score, f"{best.chunk.cite()}: {snippet}"

    def evidence_bundle(
        self, text: str, k: int = 3, chars: int = 700
    ) -> tuple[float, str]:
        """Several substantial passages for `text`, for the groundability guard.

        The guard originally judged each topic on the single best chunk cut to
        160 characters. On a real corpus that is not enough to judge anything:
        "Annual Leave Entitlement" matched the title page of the leave policy,
        the model correctly observed that the title page says nothing about
        entitlement, and a core topic was wrongly marked unsupported and pushed
        into the refusal policy. Showing the top `k` passages at a readable
        length fixes the input to that judgement rather than the threshold.
        """
        hits = self.index.search(text, top_k=k)
        if not hits:
            return 0.0, ""
        blocks = []
        for hit in hits:
            body = " ".join(hit.chunk.text.split())[:chars]
            blocks.append(f"[{hit.chunk.cite()}] (match {hit.score:.2f})\n{body}")
        return hits[0].score, "\n\n".join(blocks)


def format_context(hits: list[Hit]) -> str:
    """Render retrieved chunks for the agent prompt, with citable labels."""
    if not hits:
        return "(no documents retrieved)"
    blocks = []
    for h in hits:
        blocks.append(f"[{h.chunk.cite()}]\n{h.chunk.text}")
    return "\n\n---\n\n".join(blocks)
