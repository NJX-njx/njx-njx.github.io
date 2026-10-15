# Njx'Log

Personal technical blog on the engineering side of LLM training, built with
[Hugo](https://gohugo.io/) and the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme and hosted on
GitHub Pages。中文为默认语言，文章提供 `/en/` 下的英文版本；交互报告正文保持中文。

Production: <https://njx-njx.github.io/>

## Local setup

Requires Hugo Extended (the version pinned in `.github/workflows/hugo.yaml`),
Node 22 and Python 3.

```bash
brew install hugo                         # other platforms: https://gohugo.io/installation/
git submodule update --init --recursive   # PaperMod theme
hugo server -D                            # http://localhost:1313, drafts included
```

## Checks

The same checks run on every pull request:

```bash
hugo --gc --minify                        # build into public/
python3 scripts/check_posts.py            # zh/en post pairs in sync, summary length
python3 scripts/check_site.py public      # links, anchors, search index, reports
node scripts/search_quality.mjs public    # search ranking regression test
```

## Writing

- Posts live in `content/posts/` as `<slug>.md` (Chinese) and `<slug>.en.md`
  (English).
- 中文交互报告放在 `static/reports/<slug>/index.html`，与文章同名。
  运行 `python3 scripts/adapt_report.py` 接入共享层（可重复执行）。
  样式和主题脚本位于 `assets/css/report.css` 与 `assets/js/report.js`，
  由 Hugo 发布到 `/reports/`；不是完全自包含 HTML。
- Conventions, layout overrides and gotchas are in [AGENTS.md](AGENTS.md).

## Deployment

Pushing to `main` builds, checks and deploys the site through
`.github/workflows/hugo.yaml`.
