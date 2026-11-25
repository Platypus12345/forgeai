"""Render the committed demo source documents (.md) into the PDFs the factory reads.

The plain-text sources live in `demos/<run>/source/` and the PDFs the factory
actually ingests live in `demos/<run>/docs/`. Both are committed: the source so
the content is reviewable as a diff, the PDF so the demo corpus exercises real
PDF text extraction rather than a convenient text file.

    python scripts/make_demo_pdfs.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
JOBS = [
    ("demos/hr_policy/source", "demos/hr_policy/docs"),
    ("demos/ship_spares/source", "demos/ship_spares/docs"),
]

LEFT, TOP, LEADING, MAX_WIDTH = 20 * mm, 277 * mm, 13, 170 * mm


def wrap(text: str, c: canvas.Canvas, font: str, size: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if c.stringWidth(trial, font, size) <= MAX_WIDTH:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def render(src: Path, dst: Path) -> None:
    c = canvas.Canvas(str(dst), pagesize=A4)
    y = TOP
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line:
            y -= LEADING * 0.6
            continue
        heading = line.isupper() or (line[:2].strip().rstrip(".").isdigit() and line.isupper())
        font, size = ("Helvetica-Bold", 11) if heading else ("Helvetica", 10)
        c.setFont(font, size)
        for piece in wrap(line, c, font, size):
            if y < 20 * mm:
                c.showPage()
                y = TOP
                c.setFont(font, size)
            c.drawString(LEFT, y, piece)
            y -= LEADING
    c.save()


def main() -> None:
    for src_dir, dst_dir in JOBS:
        src_path, dst_path = ROOT / src_dir, ROOT / dst_dir
        dst_path.mkdir(parents=True, exist_ok=True)
        for md in sorted(src_path.glob("*.md")):
            pdf = dst_path / (md.stem + ".pdf")
            render(md, pdf)
            print(f"{md.relative_to(ROOT)} -> {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
