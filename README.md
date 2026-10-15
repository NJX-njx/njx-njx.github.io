# Njx'Log

Personal technical blog on the engineering side of LLM training, built with
[Hugo](https://gohugo.io/) and the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme and hosted on
GitHub Pages. Chinese is the default language; every page also has an English
version under `/en/`.

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
- An interactive report goes in `static/reports/<slug>/index.html`, next to a
  post with the same slug; run `python3 scripts/adapt_report.py` on it once.
- Conventions, layout overrides and gotchas are in [AGENTS.md](AGENTS.md).

## Deployment

Pushing to `main` builds, checks and deploys the site through
`.github/workflows/hugo.yaml`.
