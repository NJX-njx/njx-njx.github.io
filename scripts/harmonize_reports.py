#!/usr/bin/env python3
"""Apply the shared Njx'Log shell to the standalone interactive reports.

The transformation is idempotent: report content and visualization scripts
stay in place; only the duplicated embedded ET Book font block, favicon
placeholder, font constants and site-chrome hooks are normalized.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "static" / "reports"

BLOG_FONT_STACK = (
    '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen,Ubuntu,'
    'Cantarell,"Open Sans","Helvetica Neue","PingFang SC","Hiragino Sans GB",'
    '"Microsoft YaHei",sans-serif'
)

# CSS custom properties are written unquoted; keep one space after each comma.
BLOG_FONT_STACK_CSS = BLOG_FONT_STACK.replace(",", ", ")

THEME_INIT = """<script>
(function () {
  var saved;
  try {
    saved = localStorage.getItem('pref-theme');
  } catch (_) {
    // 禁用站点存储时回退到系统主题。
  }
  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme =
    saved === 'dark' || (saved !== 'light' && prefersDark) ? 'dark' : 'light';
})();
</script>"""

SITE_HEADER = """<header class="rpt-site-header" data-report-shell>
  <a class="rpt-home" href="/" aria-label="返回 Njx'Log 首页">
    <img src="/blog-icon.png" alt="" width="28" height="28">
    <span>Njx'Log</span>
  </a>
  <nav class="rpt-nav" aria-label="站点导航">
    <a href="/posts/">文章</a>
    <a href="/reports/" aria-current="page">报告</a>
    <a href="/archives/">归档</a>
    <a href="/search/">搜索</a>
    <a href="/about/">关于</a>
  </nav>
  <div class="rpt-actions">
    <button class="rpt-theme" type="button" data-report-theme-toggle aria-label="切换浅色或深色主题" title="切换主题">
      <svg class="rpt-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
      <svg class="rpt-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
    </button>
  </div>
</header>"""

SHELL_LINKS = """<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="stylesheet" href="../report-shell.css">"""

SHELL_SCRIPT = '<script src="../report-shell.js" defer></script>'


def remove_previous_shell(html: str) -> str:
    html = re.sub(
        r"\n?<script>\s*\(function \(\) \{\s*var saved(?: = localStorage\.getItem\('pref-theme'\))?;.*?</script>\n?",
        "\n",
        html,
        flags=re.S,
    )
    html = re.sub(
        r'\n?<link rel="(?:icon|apple-touch-icon)"[^>]*href="/(?:favicon-32x32\.png|apple-touch-icon\.png)"[^>]*>',
        "",
        html,
    )
    html = re.sub(
        r'\n?<link rel="stylesheet" href="\.\./report-shell\.css">',
        "",
        html,
    )
    html = re.sub(
        r'\n?<header class="rpt-site-header" data-report-shell>.*?</header>',
        "",
        html,
        flags=re.S,
    )
    html = re.sub(
        r'\n?<script src="\.\./report-shell\.js" defer></script>',
        "",
        html,
    )
    body_class = re.search(r"<body([^>]*)>", html)
    if body_class:
        attrs = body_class.group(1)
        attrs = re.sub(r'\s+class="([^"]*)"', lambda m: ' class="' + " ".join(x for x in m.group(1).split() if x != "rpt-shell") + '"', attrs)
        attrs = attrs.replace(' class=""', "")
        html = html[:body_class.start(1)] + attrs + html[body_class.end(1):]
    return html


def remove_embedded_et_book(html: str) -> str:
    return re.sub(
        r"<style>\s*/\*\s*ET Book.*?</style>",
        "<!-- Inlined ET Book removed; reports use the shared blog font stack. -->",
        html,
        count=1,
        flags=re.S | re.I,
    )


def normalize_css_fonts(html: str) -> str:
    # Point the report's --serif/--sans custom properties (including the CJK
    # fallback overrides some reports declare later) at the blog system stack.
    def replace_style(match: re.Match[str]) -> str:
        attrs, body = match.group(1), match.group(2)
        body = re.sub(
            r"(--serif|--sans)\s*:\s*[^;]+;",
            lambda m: f"{m.group(1)}: {BLOG_FONT_STACK_CSS};",
            body,
        )
        return f"<style{attrs}>{body}</style>"

    return re.sub(r"<style([^>]*)>(.*?)</style>", replace_style, html, flags=re.S | re.I)


def normalize_script_fonts(html: str) -> str:
    def replace_script(match: re.Match[str]) -> str:
        attrs, body = match.group(1), match.group(2)
        replacement = "'" + BLOG_FONT_STACK + "'"
        body = re.sub(r"'\s*\"et-book\"[^']*'", replacement, body)
        body = re.sub(
            r'"et-book"(?:(?!";)[^"\n])*"',
            '"' + BLOG_FONT_STACK + '"',
            body,
        )
        return f"<script{attrs}>{body}</script>"

    return re.sub(r"<script([^>]*)>(.*?)</script>", replace_script, html, flags=re.S | re.I)


def harmonize(path: Path) -> None:
    html = path.read_text()
    original = html
    html = remove_previous_shell(html)
    html = re.sub(r'\s*<link rel="icon" href="data:,">\s*', "\n", html)
    html = remove_embedded_et_book(html)
    html = normalize_css_fonts(html)
    html = normalize_script_fonts(html)

    if 'name="viewport"' not in html:
        raise RuntimeError(f"{path}: viewport meta not found")
    html = html.replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + THEME_INIT,
        1,
    )
    # Collapse any blank run left by a previous pass before re-inserting.
    html = re.sub(r"\n[ \t]*(?:\n[ \t]*)+(?=</head>)", "\n", html, count=1)
    html = html.replace("</head>", f"\n{SHELL_LINKS}\n</head>", 1)

    body = re.search(r"<body([^>]*)>", html)
    if not body:
        raise RuntimeError(f"{path}: body tag not found")
    attrs = body.group(1)
    class_match = re.search(r'class="([^"]*)"', attrs)
    classes = class_match.group(1).split() if class_match else []
    if "rpt-shell" not in classes:
        classes.append("rpt-shell")
    new_class = ' class="' + " ".join(classes) + '"'
    if class_match:
        attrs = attrs[:class_match.start()] + new_class + attrs[class_match.end():]
    else:
        attrs += new_class
    html = html[:body.start(1)] + attrs + html[body.end(1):]
    html = re.sub(r"(<body[^>]*>)", r"\1\n" + SITE_HEADER, html, count=1)
    html = re.sub(r"\n[ \t]*(?:\n[ \t]*)+(?=</body>)", "\n", html, count=1)
    html = html.replace("</body>", f"\n{SHELL_SCRIPT}\n</body>", 1)

    if html != original:
        path.write_text(html)


def main() -> None:
    paths = sorted(REPORTS.glob("*/index.html"))
    if not paths:
        raise RuntimeError(f"no reports found under {REPORTS}")
    for path in paths:
        harmonize(path)
        print(f"harmonized {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
