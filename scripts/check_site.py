#!/usr/bin/env python3
"""Check the built site. Run after `hugo`.

Usage: python3 scripts/check_site.py [public]

- every internal link resolves to a page, and its #fragment to an id on it
- every search-index record (both languages) points at an existing anchor
- every report has a companion post in both languages and has been passed
  through scripts/adapt_report.py; the shared report assets exist
"""
import html
import json
import pathlib
import re
import sys
from urllib.parse import unquote, urlsplit

SITE = "https://njx-njx.github.io"
REPORT_MARK = "<!-- njx-report-theme -->"


def page_file(public, path):
    target = public / unquote(path).lstrip("/")
    if path.endswith("/") or target.is_dir():
        target = target / "index.html"
    return target


def ids_of(file, cache):
    if file not in cache:
        text = file.read_text(encoding="utf-8", errors="replace")
        cache[file] = set(re.findall(r'\bid=["\']?([^"\'\s>]+)', text))
    return cache[file]


def internal_links(text):
    for href in re.findall(r'\bhref=["\']?([^"\'\s>]+)', text):
        href = html.unescape(href)
        if href.startswith(SITE):
            href = href[len(SITE):] or "/"
        if href.startswith("/") and not href.startswith("//"):
            yield href


def main(public):
    public = pathlib.Path(public)
    errors = []
    ids = {}

    for file in sorted(public.rglob("*.html")):
        text = file.read_text(encoding="utf-8", errors="replace")
        if 'http-equiv="refresh"' in text or "http-equiv=refresh" in text:
            continue
        for href in set(internal_links(text)):
            parts = urlsplit(href)
            target = page_file(public, parts.path)
            if not target.exists():
                errors.append(f"{file.relative_to(public)}: broken link {href}")
            elif parts.fragment and target.suffix == ".html" and unquote(parts.fragment) not in ids_of(target, ids):
                errors.append(f"{file.relative_to(public)}: missing anchor {href}")

    for index in ("index.json", "en/index.json"):
        for record in json.loads((public / index).read_text(encoding="utf-8")):
            path = urlsplit(record["permalink"]).path
            if record["anchor"] and record["anchor"] not in ids_of(page_file(public, path), ids):
                errors.append(f"{index}: missing anchor {path}#{record['anchor']}")

    for asset in ("reports/report.css", "reports/report.js"):
        if not (public / asset).exists():
            errors.append(f"missing {asset}")
    for report in sorted(public.glob("reports/*/index.html")):
        slug = report.parent.name
        if REPORT_MARK not in report.read_text(encoding="utf-8"):
            errors.append(f"reports/{slug}: not adapted, run scripts/adapt_report.py")
        for post in (f"posts/{slug}/index.html", f"en/posts/{slug}/index.html"):
            if not (public / post).exists():
                errors.append(f"reports/{slug}: no companion page {post}")

    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("OK   links, anchors, search index and reports")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "public")
