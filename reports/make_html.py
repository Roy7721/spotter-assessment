"""Turn REPORT.md into a single self-contained HTML file for printing to PDF.

Images are embedded as base64 so the file can be moved or emailed on its own.
Open reports/REPORT.html in a browser, then Ctrl+P -> Save as PDF.
"""

import base64
import mimetypes
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SRC = HERE / "REPORT.md"
OUT = HERE / "REPORT.html"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 11.5pt; line-height: 1.55; color: #1a1a1a;
  max-width: 720px; margin: 2rem auto; padding: 0 1rem;
}
h1 { font-size: 23pt; margin: 0 0 .2em; letter-spacing: -.01em; }
h2 { font-size: 15pt; margin: 2.2em 0 .6em; padding-bottom: .25em;
     border-bottom: 1px solid #d8d8d8; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin: 1.6em 0 .4em; page-break-after: avoid; }
h1 + p { color: #555; margin-top: .2em; }
p, li { orphans: 3; widows: 3; }
strong { color: #000; }
hr { border: none; border-top: 1px solid #e2e2e2; margin: 2em 0; }
code {
  font-family: Consolas, "DejaVu Sans Mono", monospace;
  font-size: 9.5pt; background: #f2f2f0; padding: .1em .35em; border-radius: 3px;
}
pre {
  background: #f7f7f5; border: 1px solid #e4e4e0; border-left: 3px solid #999;
  padding: .85em 1em; border-radius: 4px; overflow-x: auto;
  page-break-inside: avoid; line-height: 1.45;
}
pre code { background: none; padding: 0; font-size: 9pt; }
table {
  border-collapse: collapse; width: 100%; margin: 1.1em 0;
  font-size: 10pt; page-break-inside: avoid;
}
th, td { border: 1px solid #ddd; padding: .5em .7em; text-align: left;
         vertical-align: top; }
th { background: #f0f0ee; font-weight: 600; }
tr:nth-child(even) td { background: #fbfbfa; }
img { max-width: 100%; display: block; margin: 1.2em auto;
      border: 1px solid #e4e4e0; border-radius: 4px; page-break-inside: avoid; }
blockquote { margin: 1em 0; padding-left: 1em; border-left: 3px solid #ccc;
             color: #555; }
ul, ol { padding-left: 1.4em; }
li { margin: .35em 0; }
"""


def embed(match):
    """Replace a <img src="..."> path with an inline base64 data URI."""
    prefix, src, suffix = match.group(1), match.group(2), match.group(3)
    if src.startswith(("data:", "http://", "https://")):
        return match.group(0)
    path = (HERE / src).resolve()
    if not path.exists():
        print(f"  WARNING: image not found, left as a link -> {src}")
        return match.group(0)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    print(f"  embedded {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return f'{prefix}data:{mime};base64,{data}{suffix}'


def main():
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    body = re.sub(r'(<img[^>]*\ssrc=")([^"]+)(")', embed, body)

    title = re.search(r"^#\s+(.+)$", text, re.M)
    title = title.group(1).strip() if title else "Report"

    OUT.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{title} - Rana Roy</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n",
        encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print("open it in a browser, then Ctrl+P -> Save as PDF")


if __name__ == "__main__":
    main()
