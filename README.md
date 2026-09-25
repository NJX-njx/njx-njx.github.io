# Njx'Log

Personal technical blog built with [Hugo](https://gohugo.io/), the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, and GitHub
Pages.

Production site: <https://njx-njx.github.io/>

## Project Layout

```text
content/posts/              Blog posts in Markdown
content/*.md                Special pages such as About, FAQ, Search, Archive, Reports
hugo.yaml                   Main Hugo configuration
layouts/_default/           Local layout overrides (reports, search-assistant, rss, baseof)
assets/css/extended/        Custom CSS auto-loaded by PaperMod after theme styles
assets/js/                  Custom JavaScript processed by Hugo Pipes
static/reports/             Standalone interactive report pages (one folder per report)
static/*.png|jpg            Favicons, logo, OG image
workers/openrouter-proxy/   Cloudflare Worker proxy for the AI assistant
themes/PaperMod/            PaperMod theme submodule
.github/workflows/hugo.yaml GitHub Pages deployment workflow
```

Do not edit files under `themes/PaperMod/` directly. Override theme behavior in
the root `layouts/` and `assets/` directories instead.

## Local Setup

This site requires Hugo Extended.

macOS (Homebrew):

```bash
brew install hugo
hugo version
```

Windows (Scoop):

```powershell
scoop install hugo-extended
hugo version
```

Other install options are documented at <https://gohugo.io/installation/>.

After cloning, initialize the theme submodule if needed:

```bash
git submodule update --init --recursive
```

## Development Commands

```powershell
hugo server -D
```

Starts a local development server with draft posts enabled.

```powershell
hugo server
```

Starts a local server with production-visible content only.

```powershell
hugo --gc --minify
```

Builds the production site into `public/`.

## Writing Posts

Create a new post with:

```powershell
hugo new posts/my-new-post.md
```

Post conventions:

- Use kebab-case filenames, for example `my-new-post.md`.
- Use ISO 8601 dates with the `+08:00` timezone.
- Keep drafts as `draft: true` until they are ready to publish.
- Use lowercase tags in YAML frontmatter.
- Put static images in `static/` and reference them from site-root paths.

Example frontmatter:

```yaml
---
title: "Post Title"
date: 2026-02-07T08:00:00+08:00
draft: false
tags: ["hugo", "blog"]
categories: ["Tech"]
summary: "Short description for lists and search results."
---
```

## Interactive Reports

Long posts can have a companion interactive report in `static/reports/<slug>/`
(a self-contained HTML page, linked from the matching blog post). To publish
one:

1. Drop the report folder into `static/reports/<slug>/`.
2. Register it in the front matter of `content/reports.md` (title, url, date,
   desc) so it appears on the Reports page (`/reports/`).

The Reports page is rendered by `layouts/_default/reports.html` and reuses
PaperMod's `post-entry` card styles, so it always matches the site theme.

## Search And AI Assistant

The search page uses a custom layout, `search-assistant`, at `/search/`. It
loads a custom browser script from `assets/js/search-assistant.js` and reads
Hugo's generated `/index.json` search index. The project overrides PaperMod's
default JSON index in `layouts/_default/index.json` so the index is chunk-based:
each Markdown heading section becomes a searchable record with `title`,
`section`, `content`, `permalink`, `anchor`, `summary`, `tags`, and `date`.
Search results link directly to the matched section anchor.

The Ask box uses the same chunk search layer. It sends the top matching chunks
to the Cloudflare Worker as contextual excerpts before the Worker calls
OpenRouter.

This custom layout intentionally does not use PaperMod's built-in `search`
layout name, because PaperMod automatically injects its own `fastsearch` script
for pages whose layout is exactly `search`. Keeping this page on
`search-assistant` prevents two scripts from binding to the same search input.

The AI assistant is enabled in `hugo.yaml` and points at the deployed
Cloudflare Worker endpoint:
`https://njx-log-ai-assistant.njx-log.workers.dev/ask`. Keep
`OPENROUTER_API_KEY` as a Cloudflare secret; never commit it or place it in
Hugo/JavaScript frontend code.

See `workers/openrouter-proxy/README.md` for local development, secret setup,
and deployment commands.

## Deployment

Pushing to `main` triggers `.github/workflows/hugo.yaml`.

The workflow:

1. Checks out the repository and submodules.
2. Installs Hugo Extended.
3. Runs `hugo --gc --minify --baseURL "https://njx-njx.github.io/"`.
4. Uploads `public/` as a GitHub Pages artifact.
5. Deploys it to GitHub Pages.

You can also run the workflow manually from the GitHub Actions tab.

## Quick Checks

```bash
node --check assets/js/search-assistant.js
node --check scripts/search_quality.js
python -m py_compile scripts/llm_model.py
hugo --gc --minify
node scripts/search_quality.js public/index.json
```

## For AI Agents

See [AGENTS.md](AGENTS.md) for repository conventions, structure, and
workflows when working on this project with an AI coding agent.
