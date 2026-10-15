#!/usr/bin/env python3
"""Check that every post exists in both languages and that each pair stays in sync.

Usage: python3 scripts/check_translations.py [content/posts]

A post is a pair of files: <slug>.md (Chinese) and <slug>.en.md (English).
Within a pair, everything except the translated prose must match: front matter
other than title/summary, heading levels, math, code fences and external links.
Internal links must point at pages in the same language.
"""
import pathlib
import re
import sys

TRANSLATED_KEYS = {"title", "summary"}


def split(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return None, None
    front = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        front[key.strip()] = value.strip()
    return front, m.group(2)


def strip_code(body):
    return re.sub(r"```.*?```", "", body, flags=re.S)


def math_segments(body):
    segments = re.findall(r"\$\$.+?\$\$|\$[^$\n]+?\$", strip_code(body), flags=re.S)
    return [re.sub(r"\\text\{[^}]*\}", r"\\text{}", s) for s in segments]


def heading_levels(body):
    return [len(h) for h in re.findall(r"^(#{1,6})\s", strip_code(body), flags=re.M)]


def links(body):
    return re.findall(r"\]\(([^)\s]+)\)", strip_code(body))


def compare(zh, en):
    errors = []
    zfront, zbody = split(zh)
    efront, ebody = split(en)
    if zfront is None or efront is None:
        return ["missing front matter"]
    if list(zfront) != list(efront):
        errors.append(f"front matter keys differ: {list(zfront)} vs {list(efront)}")
    for key in zfront:
        if key not in TRANSLATED_KEYS and zfront.get(key) != efront.get(key):
            errors.append(f"front matter '{key}' differs: {zfront.get(key)!r} vs {efront.get(key)!r}")
    if heading_levels(zbody) != heading_levels(ebody):
        errors.append("heading structure differs")
    zmath, emath = math_segments(zbody), math_segments(ebody)
    if zmath != emath:
        if len(zmath) != len(emath):
            errors.append(f"math segment count differs: {len(zmath)} vs {len(emath)}")
        else:
            for a, b in zip(zmath, emath):
                if a != b:
                    errors.append(f"math differs:\n      {a}\n      {b}")
    if zbody.count("```") != ebody.count("```"):
        errors.append("code fence count differs")
    zext = sorted(l for l in links(zbody) if not l.startswith("/"))
    eext = sorted(l for l in links(ebody) if not l.startswith("/"))
    if zext != eext:
        errors.append(f"external links differ: {sorted(set(zext) ^ set(eext))}")
    errors += [f"Chinese post links to English page {l}" for l in links(zbody) if l.startswith("/en/")]
    errors += [f"English post links to Chinese post {l}" for l in links(ebody) if l.startswith("/posts/")]
    return errors


def main(root):
    root = pathlib.Path(root)
    slugs = {p.name.removesuffix(".en.md").removesuffix(".md")
             for p in root.glob("*.md") if not p.name.startswith("_")}
    failed = False
    for slug in sorted(slugs):
        zh, en = root / f"{slug}.md", root / f"{slug}.en.md"
        missing = [p.name for p in (zh, en) if not p.exists()]
        errors = [f"missing {', '.join(missing)}"] if missing else compare(zh, en)
        if errors:
            failed = True
            print(f"FAIL {slug}")
            for error in errors:
                print(f"  - {error}")
    if failed:
        sys.exit(1)
    print(f"OK   {len(slugs)} posts, both languages in sync")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "content/posts")
