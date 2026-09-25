# Copilot Instructions for Njx'Log

## Project Overview

Hugo static site (PaperMod theme) deployed to GitHub Pages at
`https://njx-njx.github.io/`. A field notebook on LLM training engineering:
harness/rollout systems, model architecture, mid/post-training, evals and
benchmarks, reasoning RL and agentic RL, RSI, data synthesis. Posts are mixed
Chinese/English.

## Architecture

```text
content/posts/    → Blog posts (Markdown + YAML frontmatter)
content/*.md      → Special pages (about, faq, search, archives, reports)
hugo.yaml         → Site configuration (YAML, NOT toml; locale: "zh")
layouts/_default/ → Layout overrides: reports.html, search-assistant.html,
                    rss.xml, baseof.html, index.json, partials/
assets/css/extended/ → Extra CSS auto-loaded after theme styles
assets/js/        → search-assistant.js, toc-drawer.js
static/reports/   → Standalone interactive report pages
workers/openrouter-proxy/ → Cloudflare Worker (Ask feature backend)
themes/PaperMod/  → Theme submodule (do NOT modify)
.github/workflows/hugo.yaml → GitHub Pages deployment
```

## Creating New Posts

Use the Hugo CLI or follow this frontmatter pattern:

```markdown
---
title: "Post Title"              # keep short enough for one line in list cards
date: 2026-01-15T10:00:00+08:00  # Use +08:00 timezone
draft: false                     # Set true to hide from production
tags: ["tag1", "tag2"]           # lowercase
categories: ["Tech"]             # Optional
summary: "Short description for lists and search results."
---
```

**Key conventions:**

- Dates use ISO 8601 with `+08:00` timezone
- Filenames use kebab-case: `my-new-post.md`
- Tags are lowercase arrays
- Keep post titles concise (~46 display units or fewer) so list cards render
  on a single line

## Special Pages

| Page     | File                 | Layout            |
|----------|----------------------|-------------------|
| Archives | `content/archives.md` | `layout: "archives"` |
| Search   | `content/search.md`  | `layout: "search-assistant"` |
| Reports  | `content/reports.md` | `layout: "reports"` |
| FAQ      | `content/faq.md`     | (default)         |
| About    | `content/about.md`   | (default)         |

The Reports page reads its card list from the page's front matter
(`reports:` entries with title/url/date/desc) and renders PaperMod
`post-entry` cards. New interactive reports must be registered there after
being added under `static/reports/<slug>/`.

## Configuration Notes (hugo.yaml)

- `locale` / `defaultContentLanguage`: `"zh"` (theme only ships `zh.yaml`;
  `zh-cn` silently loses all translations)
- PaperMod home-info mode (not profile mode)
- `disableAnchoredHeadings: true` (no anchor icons on headings)
- `assets.favicon` / `label.icon` point to `static/blog-icon.png`
- Custom social icons live in `layouts/partials/social_icons.html`
  (X logo overrides PaperMod's old Twitter bird)
- Ask feature: `params.aiAssistant.endpoint` → Cloudflare Worker
  `https://njx-log-ai-assistant.njx-log.workers.dev/ask`

## Local Development Commands

```bash
brew install hugo                # macOS; requires Hugo Extended
hugo server -D                   # Dev server with drafts
hugo server                      # Dev server, production content only
hugo new posts/my-post.md        # New post from archetype
hugo --gc --minify               # Production build into public/
```

## Deployment

- Push to `main` triggers `.github/workflows/hugo.yaml` (Hugo Extended +
  `--baseURL "https://njx-njx.github.io/"`), deployed to GitHub Pages
- Live in ~1 minute; verify with `gh run list`

## Do NOT

- Modify files in `themes/PaperMod/` (override in root `layouts/` / `assets/`)
- Use `hugo.toml` (this project uses `hugo.yaml`)
- Commit `OPENROUTER_API_KEY` or any secret (Cloudflare secret only)
- Commit draft posts (`draft: true`) unless intentional
