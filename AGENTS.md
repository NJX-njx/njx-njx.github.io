# AGENTS.md

Guidelines for AI coding agents working on **Njx'Log** (`njx-njx.github.io`).

## What This Is

A personal technical blog (Hugo + PaperMod → GitHub Pages) about the
engineering side of LLM training: harness and rollout systems, model
architecture, mid-training and post-training, evals and benchmarks,
reasoning RL and agentic RL, RSI, and data synthesis. Posts are mixed
Chinese/English; the site language is Chinese (`locale: "zh"`).

Production: <https://njx-njx.github.io/>

## Quick Facts

- Hugo **Extended** required; config is `hugo.yaml` (never `hugo.toml`)
- Theme is a git submodule at `themes/PaperMod/` — **never edit it**
- Push to `main` → GitHub Actions deploys to Pages in ~1 minute
- Verify deploys with `gh run list`; verify live CSS/HTML with `curl`

## Repository Layout

```text
content/posts/              Blog posts (Markdown + YAML frontmatter)
content/about.md            About page
content/faq.md              FAQ page
content/search.md           Search + Ask page (layout: search-assistant)
content/archives.md         Archive page (layout: archives)
content/reports.md          Reports listing (layout: reports) + report registry
hugo.yaml                   Site configuration
layouts/_default/           Layout overrides:
  reports.html              Reports card listing (reads .Params.reports)
  search-assistant.html     Chunk search + Ask UI
  index.json                Chunk-based search index (heading-section records)
  baseof.html, single.html, rss.xml
  partials/                 social_icons.html (X logo), toc.html, ...
assets/css/extended/        Auto-loaded after theme styles:
  pagination.css            Pagination button overrides
  side-toc.css              Side TOC layout + drawer (desktop/mobile)
assets/js/                  search-assistant.js, toc-drawer.js
static/reports/<slug>/      Standalone interactive report pages (self-contained HTML)
static/blog-icon.png        Site logo + favicon source (avatar from cover art)
static/favicon-16x16.png, favicon-32x32.png, apple-touch-icon.png
static/blog-cover.jpg       OG share image
workers/openrouter-proxy/   Cloudflare Worker proxying the Ask feature to OpenRouter
.github/workflows/hugo.yaml Deploy workflow
.github/copilot-instructions.md  Lighter-weight agent hints (keep in sync)
```

## Common Commands

```bash
# Setup (macOS)
brew install hugo
git submodule update --init --recursive

# Develop
hugo server -D                      # drafts visible
hugo server                         # production content only

# Build & check
hugo --gc --minify                  # output in public/
node --check assets/js/search-assistant.js
node scripts/search_quality.js public/index.json

# Deploy = push to main; then watch
gh run list --limit 3
```

## Workflows

### Adding a blog post

1. `hugo new posts/<kebab-case-slug>.md` (or copy an existing post).
2. Frontmatter conventions: `date` in `+08:00`, lowercase `tags`, a
   one-line `summary`, `draft: true` until ready.
3. Keep `title` short — list cards should render on one line
   (~46 display units; CJK chars count ~2, ASCII ~1).
4. Build locally and confirm the card renders correctly.

### Adding an interactive report

1. Place the self-contained report at `static/reports/<slug>/index.html`.
2. Register it in `content/reports.md` front matter under `reports:` with
   `title` (match the blog post title), `url: /reports/<slug>/`, `date`
   (match the post date), and a `desc` (~100 chars) for the card.
3. The card appears on `/reports/` automatically; entries are sorted by the
   site owner by ordering them newest-first in the file.

### Styling changes

- Put all CSS overrides in `assets/css/extended/*.css`; PaperMod loads them
  after theme styles. Do not edit theme CSS.
- Match the established visual language: translucent dark surfaces
  (`rgba(255,255,255,0.06)`), 1px `var(--border)` borders, small radii
  (10–12px), `--secondary` text / `--primary` on hover.

### Ask feature (AI assistant) operations

- Backend: Cloudflare Worker `njx-log-ai-assistant`
  (`workers/openrouter-proxy/`); the secret `OPENROUTER_API_KEY` lives only
  in Cloudflare.
- If the UI shows raw upstream errors (e.g. `User not found.` from
  OpenRouter), the API key is invalid — rotate it:
  1. Create a new key at <https://openrouter.ai/keys>.
  2. `cd workers/openrouter-proxy && npx wrangler login && npm run secret:set`
  3. Paste the key; verify by asking a question on `/search/`.
- Never write the key into any committed file.

## Configuration Gotchas

- `locale` / `defaultContentLanguage` must be `"zh"` — the theme only ships
  `zh.yaml`; `zh-cn` silently loses every translation (buttons like
  「下一页」render empty).
- `disableAnchoredHeadings: true` is intentional — no `#` anchor icons on
  headings.
- The Search page must keep `layout: "search-assistant"` (not `search`) so
  PaperMod's built-in fastsearch script does not double-bind the input.
- Social icons are customized in `layouts/partials/social_icons.html`;
  the `x`/`twitter` name renders the current X logo.
- Favicon/logo/OG image are generated from the cover artwork; regenerate
  with a Pillow script (center-square crop, top offset 60px on the
  1312×1792 source) rather than hand-editing.

## Hard Rules

1. Never modify `themes/PaperMod/` — override via root `layouts/` and
   `assets/css/extended/`.
2. Never commit secrets (Worker API key, `.dev.vars`).
3. Never convert the site to `hugo.toml` or change the theme.
4. Keep generated artifacts out of git (`public/` is gitignored; don't
   force-add it).
5. After any change: `hugo --gc --minify` must succeed, and any HTML/CSS
   claims must be verified against `public/` output before committing.
