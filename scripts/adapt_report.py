#!/usr/bin/env python3
"""Fit a standalone report into the blog's look. Idempotent.

Usage: python3 scripts/adapt_report.py [static/reports/<slug>/index.html ...]
       (no arguments: every report under static/reports/)

- drops the inlined ET Book font; CSS and canvas text use the blog's sans
  stack (script font stacks naming et-book become RS("--serif"))
- links /reports/report.css and /reports/report.js, which Hugo publishes
- adds the site header placeholder that report.js fills in
- routes every hex color in the report's scripts through the theme:
  neutral tokens (paper, ink, line, ...) and white become RC("--token"),
  which follows the blog's light/dark colors; other colors become RD("#hex"),
  which keeps the hue and lightens it on dark paper; rgba() strings become
  RA(color, alpha) with the same mapping
- adds dark-mode variants for the report's own accent variables
- adds Open Graph tags and the site favicon when missing
"""
import html
import pathlib
import re
import sys

SITE = "https://njx-njx.github.io"
NEUTRAL = ["--paper", "--paper-hi", "--ink", "--ink-md", "--ink-lo", "--line", "--line-lo"]
NEUTRALIZED_ACCENTS = {"--red", "--red-hi", "--blue"}  # report.css maps these to the blog's text color
HEX = r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})"
MARK = "<!-- njx-report-theme -->"


def full_hex(value):
    value = value.lower()
    if len(value) == 4:
        value = "#" + "".join(c * 2 for c in value[1:])
    return value


def lighten(value):
    """Same mix as RD() in report.js."""
    value = full_hex(value)
    channels = [int(value[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * 0.4):02x}" for c in channels)


def root_variables(page):
    css = "\n".join(re.findall(r"<style>(.*?)</style>", page, re.S))
    variables = {}
    for block in re.findall(r":root\s*\{(.*?)\}", css, re.S):
        for name, value in re.findall(rf"(--[\w-]+)\s*:\s*({HEX})\b", block):
            variables.setdefault(name, full_hex(value))
    return variables


def theme_scripts(page, neutral):
    def ref(value):
        """'"--token"' for neutral colors and white, '"#hex"' for everything else."""
        value = full_hex(value)
        if value in neutral:
            return f'"{neutral[value]}"'
        return '"--paper"' if value == "#ffffff" else f'"{value}"'

    def solid(value):
        target = ref(value)
        return f"RC({target})" if target.startswith('"--') else f"RD({target})"

    def rgb_hex(match):
        return "#" + "".join(f"{int(match.group(i)):02x}" for i in (2, 3, 4))

    def rgba_literal(match):
        if match.group(5) is None:
            return solid(rgb_hex(match))
        return f"RA({ref(rgb_hex(match))}, {match.group(5)})"

    def rgba_template(match):
        value = "#" + "".join(f"{int(match.group(i)):02x}" for i in (1, 2, 3))
        return f"RA({ref(value)}, ({match.group(4)}))"

    def script(match):
        code = match.group(2)
        code = re.sub(r"`rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\$\{([^{}]+)\}\s*\)`", rgba_template, code)
        code = re.sub(r"([\"'`])rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)\1", rgba_literal, code)
        code = re.sub(rf"(?<!RD\()(?<!RA\()([\"'])({HEX})\1", lambda m: solid(m.group(2)), code)
        code = re.sub(r"'[^'\n]*et-book[^'\n]*'", 'RS("--serif")', code)
        return match.group(1) + code + match.group(3)

    return re.sub(r"(<script>)(.*?)(</script>)", script, page, flags=re.S)


def head_block(page, variables, slug):
    accents = {name: value for name, value in variables.items()
               if name not in NEUTRAL and name not in NEUTRALIZED_ACCENTS}
    dark = "".join(f"{name}:{lighten(value)};" for name, value in sorted(accents.items()))
    lines = [MARK, f'<style>:root[data-theme="dark"]{{{dark}}}</style>']
    title = re.search(r"<title>(.*?)</title>", page, re.S).group(1).strip()
    description = re.search(r'<meta name="description" content="([^"]*)"', page)
    meta = {
        "og:type": "article",
        "og:site_name": "Njx'Log",
        "og:title": html.unescape(title),
        "og:url": f"{SITE}/reports/{slug}/",
        "og:image": f"{SITE}/blog-cover.jpg",
    }
    if description:
        meta["og:description"] = html.unescape(description.group(1))
    for prop, value in meta.items():
        if f'property="{prop}"' not in page:
            lines.append(f'<meta property="{prop}" content="{html.escape(value)}">')
    if 'name="twitter:card"' not in page:
        lines.append('<meta name="twitter:card" content="summary_large_image">')
    if 'rel="icon"' not in page:
        lines.append('<link rel="icon" href="/favicon-32x32.png" sizes="32x32">')
        lines.append('<link rel="apple-touch-icon" href="/apple-touch-icon.png">')
    lines.append('<link rel="stylesheet" href="/reports/report.css">')
    lines.append('<script src="/reports/report.js"></script>')
    lines.append(MARK)
    return "\n".join(lines) + "\n"


def adapt(path):
    page = path.read_text(encoding="utf-8")
    slug = path.parent.name
    page = re.sub(rf"\n?{re.escape(MARK)}.*?{re.escape(MARK)}\n?", "\n", page, flags=re.S)
    page = re.sub(r"<style>\s*/\*[^*]*ET Book.*?</style>\n?", "", page, flags=re.S)
    page = page.replace('<link rel="icon" href="data:,">\n', "")
    variables = root_variables(page)
    neutral = {value: name for name, value in variables.items() if name in NEUTRAL}
    page = theme_scripts(page, neutral)
    page = page.replace("</head>", head_block(page, variables, slug) + "</head>", 1)
    if 'id="njx-bar"' not in page:
        page = re.sub(r"(<body[^>]*>)", r'\1\n<header id="njx-bar"></header>', page, count=1)
    path.write_text(page, encoding="utf-8")
    code = "".join(re.findall(r"<script>(.*?)</script>", page, re.S))
    raw_hex = len(re.findall(rf"(?<!RD\()(?<!RA\()[\"']{HEX}[\"']", code))
    raw_rgba = len(re.findall(r"rgba?\(", code))
    print(f"{slug}: {len(page) // 1024} KB, unconverted colors in scripts: {raw_hex} hex, {raw_rgba} rgba")


def main(paths):
    targets = [pathlib.Path(p) for p in paths] or sorted(pathlib.Path("static/reports").glob("*/index.html"))
    for path in targets:
        adapt(path)


if __name__ == "__main__":
    main(sys.argv[1:])
