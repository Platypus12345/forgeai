"""Loading and chunking the document folder.

The factory is handed a folder it has never seen, so this has to be tolerant:
unknown extensions are skipped with a note rather than crashing the run, and a
PDF that yields no text (scanned images) is reported as such - that is a
groundability fact the requirements stage needs to know about, not an error to
swallow.

CSV files are chunked row-wise with the header line repeated on every chunk.
A naive character chunker splits a table mid-row and the retriever then returns
rows whose column meanings are gone, which looks exactly like a model
hallucinating numbers. Keeping the header attached is the single highest-value
piece of chunking logic here.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from .schemas import Chunk, Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}
CSV_SUFFIXES = {".csv", ".tsv"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED = TEXT_SUFFIXES | CSV_SUFFIXES | PDF_SUFFIXES


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _read_pdf(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        return "", "pypdf not installed; PDF skipped"
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return "", f"unreadable PDF: {exc}"
    pages: list[str] = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"[page {i}]\n{text.strip()}")
    body = "\n\n".join(pages)
    note = "" if body.strip() else "PDF produced no extractable text (likely scanned)"
    return body, note


def _read_csv(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
    except Exception as exc:
        return raw, f"CSV parsed as plain text: {exc}"
    if not rows:
        return "", "empty CSV"
    header = rows[0]
    lines = [" | ".join(header)]
    for row in rows[1:]:
        if any(cell.strip() for cell in row):
            lines.append(" | ".join(row))
    return "\n".join(lines), ""


def load_documents(folder: str | Path) -> tuple[list[Document], list[str]]:
    """Load every supported file in `folder`. Returns (documents, warnings)."""
    folder = Path(folder)
    warnings: list[str] = []
    if not folder.exists():
        raise FileNotFoundError(f"document folder not found: {folder}")

    docs: list[Document] = []
    paths = sorted(p for p in folder.rglob("*") if p.is_file())
    if not paths:
        warnings.append(f"document folder {folder} is empty")

    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED:
            warnings.append(f"skipped unsupported file: {path.name}")
            continue
        note = ""
        if suffix in PDF_SUFFIXES:
            text, note = _read_pdf(path)
            kind = "pdf"
        elif suffix in CSV_SUFFIXES:
            text, note = _read_csv(path)
            kind = "csv"
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            kind = "md" if suffix in {".md", ".markdown"} else "txt"
        if note:
            warnings.append(f"{path.name}: {note}")
        if not text.strip():
            warnings.append(f"{path.name}: no usable text, excluded from the corpus")
            continue
        docs.append(
            Document(
                doc_id=_slug(path.stem),
                name=path.name,
                path=str(path),
                kind=kind,
                text=text,
                note=note,
            )
        )
    return docs, warnings


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # A very long paragraph is split on sentence boundaries so the chunker
        # never has to cut mid-sentence.
        if len(part) > 1200:
            out.extend(
                s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", part) if s.strip()
            )
        else:
            out.append(part)
    return out


def _chunk_csv(doc: Document, size: int) -> list[str]:
    lines = [ln for ln in doc.text.splitlines() if ln.strip()]
    if not lines:
        return []
    header, rows = lines[0], lines[1:]
    if not rows:
        return [header]
    chunks, current = [], [header]
    length = len(header)
    for row in rows:
        if length + len(row) > size and len(current) > 1:
            chunks.append("\n".join(current))
            current, length = [header], len(header)
        current.append(row)
        length += len(row) + 1
    if len(current) > 1:
        chunks.append("\n".join(current))
    return chunks


def chunk_documents(
    docs: list[Document], chunk_size: int = 900, overlap: int = 150
) -> list[Chunk]:
    """Paragraph-packing chunker with character overlap; row-wise for CSVs."""
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks: list[Chunk] = []

    for doc in docs:
        if doc.kind == "csv":
            texts = _chunk_csv(doc, chunk_size)
        else:
            texts = []
            current = ""
            for para in _split_paragraphs(doc.text):
                if not current:
                    current = para
                elif len(current) + len(para) + 2 <= chunk_size:
                    current = f"{current}\n\n{para}"
                else:
                    texts.append(current)
                    tail = current[-overlap:] if overlap else ""
                    # Resume from a clean word boundary inside the overlap.
                    if tail:
                        cut = tail.find(" ")
                        tail = tail[cut + 1 :] if cut != -1 else tail
                    current = f"{tail}\n\n{para}".strip() if tail else para
            if current:
                texts.append(current)

        for ordinal, text in enumerate(texts, 1):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal}",
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    ordinal=ordinal,
                    text=text.strip(),
                )
            )
    return chunks


def corpus_summary(docs: list[Document], chunks: list[Chunk]) -> str:
    if not docs:
        return "No readable documents."
    per_doc = {}
    for c in chunks:
        per_doc[c.doc_name] = per_doc.get(c.doc_name, 0) + 1
    lines = [
        f"{len(docs)} document(s), {sum(d.n_chars for d in docs):,} characters, "
        f"{len(chunks)} chunks:"
    ]
    for d in docs:
        lines.append(
            f"- {d.name} ({d.kind}, {d.n_chars:,} chars, {per_doc.get(d.name, 0)} chunks)"
        )
    return "\n".join(lines)


def preview_corpus(docs: list[Document], per_doc_chars: int = 1200) -> str:
    """A compact corpus preview to put in a prompt without blowing the context."""
    out = []
    for d in docs:
        body = re.sub(r"\n{3,}", "\n\n", d.text.strip())
        out.append(f"### {d.name} ({d.kind})\n{body[:per_doc_chars]}")
    return "\n\n".join(out)
