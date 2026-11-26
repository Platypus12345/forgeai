"""Page rendering for the serverless front end: styles, chrome, Markdown.

Kept apart from `index.py` so the routing stays readable. The Markdown
renderer here is deliberately small and deliberately not a dependency: the
only Markdown this site ever renders is `REPORT.md`, which the factory writes
itself, so the input is known and a 12-rule renderer covers it exactly. The
reports are mostly tables, and a table rendered as preformatted text is the
difference between a report someone reads and one they scroll past.
"""

from __future__ import annotations

import html
import re

# -- styles -----------------------------------------------------------------
#
# One stylesheet, no framework, no webfont. A page that must answer a cold
# request in under a second should not spend 200ms of it fetching a font, and
# the system stack looks native on every platform anyway.

STYLE = """
:root{
  --bg:#0a0c10; --surface:#111419; --surface-2:#161a21; --line:#232833;
  --fg:#e9edf3; --fg-dim:#98a2b3; --fg-faint:#6b7585;
  --accent:#7aa2f7; --accent-soft:rgba(122,162,247,.12);
  --good:#4ec9a0; --good-soft:rgba(78,201,160,.14);
  --warn:#e0af68; --bad:#f7768e; --bad-soft:rgba(247,118,142,.14);
  --radius:14px;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Inter,sans-serif;
}
@media (prefers-color-scheme: light){
  :root:not([data-theme="dark"]){
    --bg:#f7f8fa; --surface:#fff; --surface-2:#f1f3f7; --line:#e2e6ed;
    --fg:#161a21; --fg-dim:#5a6472; --fg-faint:#8a94a3;
    --accent:#2f6fd0; --accent-soft:rgba(47,111,208,.09);
    --good:#1a8f6a; --good-soft:rgba(26,143,106,.12);
    --warn:#9a6b16; --bad:#c43d55; --bad-soft:rgba(196,61,85,.1);
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.65 var(--sans);
  -webkit-font-smoothing:antialiased;
}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:900px;margin:0 auto;padding:0 16px 96px}

/* top bar */
.top{
  position:sticky;top:0;z-index:10;backdrop-filter:blur(12px);
  background:var(--bg);
  background:color-mix(in srgb,var(--bg) 86%,transparent);
  border-bottom:1px solid var(--line);margin-bottom:40px;
}
.top-in{
  max-width:900px;margin:0 auto;padding:14px 16px;
  display:flex;align-items:center;gap:14px;flex-wrap:wrap;
}
.brand{font-weight:650;letter-spacing:-.015em;font-size:16px;color:var(--fg)}
.brand:hover{text-decoration:none}
.brand .dot{color:var(--accent)}
.top nav{margin-left:auto;display:flex;gap:18px;font-size:14px}
.top nav a{color:var(--fg-dim)}
.top nav a:hover{color:var(--fg)}

/* hero */
.hero{padding:16px 0 30px}
.hero h1{
  font-size:clamp(28px,5vw,40px);line-height:1.12;margin:0 0 14px;
  letter-spacing:-.03em;font-weight:680;
}
.hero p{font-size:17px;color:var(--fg-dim);margin:0;max-width:62ch}
.hero .lede strong{color:var(--fg);font-weight:600}

/* cards */
.card{
  background:var(--surface);border:1px solid var(--line);
  border-radius:var(--radius);padding:22px;margin:0 0 18px;
}
.card.pad-lg{padding:26px 26px 22px}
.card h2{font-size:18px;margin:0 0 6px;letter-spacing:-.01em;font-weight:620}
.card h3{font-size:15px;margin:22px 0 8px;color:var(--fg-dim);
  text-transform:uppercase;letter-spacing:.07em;font-weight:600}
.sub{color:var(--fg-dim);font-size:14px;margin:0}
.mono{font-family:var(--mono);font-size:13px}

/* run card */
.run-head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.run-head h2{margin:0}
.scores{display:flex;align-items:center;gap:14px;margin:18px 0 0;flex-wrap:wrap}
.score{display:flex;flex-direction:column;gap:2px}
.score b{font-size:26px;font-weight:650;letter-spacing:-.02em;line-height:1.1}
.score span{font-size:11px;color:var(--fg-faint);text-transform:uppercase;
  letter-spacing:.08em}
.arrow{color:var(--fg-faint);font-size:20px;align-self:center;padding:0 2px}
.delta{
  font-size:13px;font-weight:600;padding:3px 9px;border-radius:999px;
  background:var(--good-soft);color:var(--good);align-self:center;
}
.delta.flat{background:var(--surface-2);color:var(--fg-dim)}
.bar{height:6px;border-radius:999px;background:var(--surface-2);
  margin:16px 0 0;overflow:hidden;position:relative}
.bar i{display:block;height:100%;border-radius:999px;background:var(--accent)}
.bar i.good{background:var(--good)}

/* chips */
.chips{display:flex;gap:7px;flex-wrap:wrap;margin:14px 0 0}
.chip{
  font-size:12px;padding:3px 10px;border-radius:999px;
  border:1px solid var(--line);color:var(--fg-dim);background:var(--surface-2);
  font-family:var(--mono);
}
.chip.ok{border-color:transparent;background:var(--good-soft);color:var(--good)}
.chip.no{border-color:transparent;background:var(--bad-soft);color:var(--bad)}
.chip.key{border-color:transparent;background:var(--accent-soft);color:var(--accent)}

.actions{display:flex;gap:10px;margin:20px 0 0;flex-wrap:wrap}
.btn{
  display:inline-block;padding:9px 16px;border-radius:9px;font-size:14px;
  font-weight:550;border:1px solid var(--line);color:var(--fg);
  background:var(--surface-2);cursor:pointer;
}
.btn:hover{text-decoration:none;border-color:var(--fg-faint)}
.btn.primary{background:var(--accent);border-color:transparent;color:#07101f}
.btn.primary:hover{opacity:.9}
.btn:disabled{opacity:.45;cursor:default}

/* notice */
.notice{
  display:flex;gap:12px;padding:14px 16px;border-radius:var(--radius);
  background:var(--accent-soft);border:1px solid transparent;
  color:var(--fg-dim);font-size:14px;margin:0 0 18px;line-height:1.55;
}
.notice b{color:var(--fg);font-weight:600}
.notice code{font-family:var(--mono);font-size:12.5px;color:var(--fg)}

/* chat */
.ask{display:flex;gap:10px;margin:18px 0 0}
.ask input{
  flex:1;padding:12px 14px;border-radius:10px;border:1px solid var(--line);
  background:var(--bg);color:var(--fg);font:15px var(--sans);
}
.ask input:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
.suggest{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 0}
.suggest button{
  font-size:13px;padding:6px 12px;border-radius:999px;border:1px dashed var(--line);
  background:transparent;color:var(--fg-dim);cursor:pointer;font-family:var(--sans);
}
.suggest button:hover{color:var(--fg);border-color:var(--fg-faint)}
.msg{margin:20px 0 0;padding:18px 0 0;border-top:1px solid var(--line)}
.msg .q{font-weight:600;margin:0 0 10px}
.msg .a{margin:0;white-space:pre-wrap}
.msg .foot{margin:14px 0 0;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
details.detail{margin:12px 0 0}
details.detail summary{
  cursor:pointer;color:var(--fg-faint);font-size:12.5px;list-style:none;
}
details.detail summary::-webkit-details-marker{display:none}
details.detail summary:before{content:"▸ ";font-size:10px}
details.detail[open] summary:before{content:"▾ "}
details.detail .body{margin:9px 0 0;font-family:var(--mono);font-size:12px;
  color:var(--fg-dim);line-height:1.8}
.spin{display:inline-block;width:13px;height:13px;border-radius:50%;
  border:2px solid var(--line);border-top-color:var(--accent);
  animation:sp .7s linear infinite;vertical-align:-2px;margin-right:7px}
@keyframes sp{to{transform:rotate(360deg)}}

/* markdown (the report) */
.md{font-size:15px}
.md h1{font-size:23px;margin:0 0 18px;letter-spacing:-.02em;font-weight:660}
.md h2{
  font-size:17px;margin:34px 0 12px;padding-top:18px;
  border-top:1px solid var(--line);font-weight:620;letter-spacing:-.01em;
}
.md h2:first-of-type{border-top:0;padding-top:0;margin-top:22px}
.md h3{font-size:14.5px;margin:22px 0 8px;color:var(--fg);font-weight:620;
  text-transform:none;letter-spacing:0}
.md p{margin:0 0 13px}
.md ul,.md ol{margin:0 0 13px;padding-left:22px}
.md li{margin:0 0 5px}
.md blockquote{
  margin:0 0 15px;padding:2px 0 2px 16px;border-left:3px solid var(--accent);
  color:var(--fg-dim);
}
.md code{
  font-family:var(--mono);font-size:13px;background:var(--surface-2);
  padding:1.5px 6px;border-radius:5px;
}
.md pre{
  background:var(--surface-2);border:1px solid var(--line);border-radius:10px;
  padding:14px 16px;overflow-x:auto;margin:0 0 15px;
}
.md pre code{background:none;padding:0;font-size:12.5px;line-height:1.6}
.md hr{border:0;border-top:1px solid var(--line);margin:26px 0}
.md strong{color:var(--fg);font-weight:620}
.table-scroll{overflow-x:auto;margin:0 0 16px;
  border:1px solid var(--line);border-radius:10px}
.md table{border-collapse:collapse;width:100%;font-size:13.5px}
.md th,.md td{padding:9px 13px;text-align:left;border-bottom:1px solid var(--line)}
.md th{
  background:var(--surface-2);font-weight:600;font-size:12px;
  text-transform:uppercase;letter-spacing:.05em;color:var(--fg-dim);
  white-space:nowrap;
}
.md tr:last-child td{border-bottom:0}
.md td code{font-size:12px}

footer{
  margin:50px 0 0;padding:22px 0 0;border-top:1px solid var(--line);
  color:var(--fg-faint);font-size:13px;line-height:1.7;
}
footer a{color:var(--fg-dim)}
.empty{text-align:center;padding:40px 20px;color:var(--fg-dim)}
.empty h2{color:var(--fg)}
@media (max-width:560px){
  .wrap{padding:0 14px 72px}
  .scores{gap:10px}
  .score b{font-size:22px}
}
"""


def page(title: str, body: str, *, nav_extra: str = "") -> str:
    """Wrap page content in the shared chrome."""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        "<meta name='description' content='ForgeAI - a factory that builds, "
        "tests and repairs document-grounded agents.'>"
        f"<style>{STYLE}</style></head><body>"
        "<div class='top'><div class='top-in'>"
        "<a class='brand' href='/'>Forge<span class='dot'>AI</span></a>"
        f"<nav>{nav_extra}"
        "<a href='https://github.com/Platypus12345/forgeai' rel='noopener'>"
        "Source</a></nav>"
        "</div></div>"
        f"<div class='wrap'>{body}"
        "<footer>"
        "The factory runs from the CLI or the Streamlit control panel; a build "
        "takes tens of minutes, which is why this deployment serves what it "
        "produced rather than producing it. "
        "<a href='https://github.com/Platypus12345/forgeai'>Read the source</a>."
        "</footer></div></body></html>"
    )


# -- Markdown ---------------------------------------------------------------

_INLINE = (
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)"), r"<a href='\2'>\1</a>"),
)


def _inline(text: str) -> str:
    out = html.escape(text)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    return out


def _is_divider(row: str) -> bool:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    return bool(cells) and all(set(c) <= set("-: ") and "-" in c for c in cells)


def _split_row(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def markdown(text: str) -> str:
    """Render the subset of Markdown the factory's own reports use.

    Headings, paragraphs, lists, blockquotes, fenced code, horizontal rules,
    GitHub-style tables, and inline code/bold/italic/links. Anything outside
    that is escaped and shown as text, which is the safe failure: a report is
    data this page displays, never markup it trusts.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            block: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(
                "<pre><code>" + html.escape("\n".join(block)) + "</code></pre>"
            )
            continue

        if not stripped:
            i += 1
            continue

        if set(stripped) <= {"-", "*", "_"} and len(stripped) >= 3:
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(
                f"<h{min(level, 4)}>{_inline(stripped[level:].strip())}"
                f"</h{min(level, 4)}>"
            )
            i += 1
            continue

        # table: a header row followed by a |---|---| divider
        if (
            stripped.startswith("|")
            and i + 1 < n
            and _is_divider(lines[i + 1])
        ):
            head = _split_row(stripped)
            i += 2
            body: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                body.append(_split_row(lines[i].strip()))
                i += 1
            cells = "".join(f"<th>{_inline(c)}</th>" for c in head)
            rows = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                for r in body
            )
            out.append(
                "<div class='table-scroll'><table><thead><tr>"
                f"{cells}</tr></thead><tbody>{rows}</tbody></table></div>"
            )
            continue

        if stripped.startswith(">"):
            quote: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(quote))}</blockquote>")
            continue

        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if bullet or ordered:
            tag = "ul" if bullet else "ol"
            items: list[str] = []
            while i < n:
                s = lines[i].strip()
                m = re.match(
                    r"^[-*+]\s+(.*)$" if bullet else r"^\d+[.)]\s+(.*)$", s
                )
                if not m:
                    break
                items.append(f"<li>{_inline(m.group(1))}</li>")
                i += 1
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        para: list[str] = []
        while i < n and lines[i].strip() and not re.match(
            r"^(#|\||>|```|[-*+]\s|\d+[.)]\s)", lines[i].strip()
        ):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
        else:
            i += 1

    return "".join(out)
