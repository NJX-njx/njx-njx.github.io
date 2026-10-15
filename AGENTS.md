# AGENTS.md

Guidelines for AI coding agents working on **Njx'Log** (`njx-njx.github.io`).

## What This Is

A personal technical blog (Hugo + PaperMod → GitHub Pages) about the
engineering side of LLM training: harness and rollout systems, model
architecture, mid-training and post-training, evals and benchmarks,
reasoning RL and agentic RL, RSI, and data synthesis.

The site is bilingual: Chinese is the default language and lives at `/`;
English lives under `/en/`. Every page exists in both languages as a pair of
files with the same base name: `<name>.md` (Chinese) and `<name>.en.md`
(English). The header's language link jumps to the current page's
translation.

Production: <https://njx-njx.github.io/>

## Quick Facts

- Hugo **Extended** required; config is `hugo.yaml` (never `hugo.toml`)
- Theme is a git submodule at `themes/PaperMod/` — **never edit it**
- Push to `main` → GitHub Actions deploys to Pages in ~1 minute
- Verify deploys with `gh run list`; verify live CSS/HTML with `curl`

## Repository Layout

```text
content/posts/              Blog posts: <slug>.md (zh) + <slug>.en.md (en)
content/about(.en).md       About page
content/faq(.en).md         FAQ page
content/search(.en).md      Search page (layout: site-search)
content/archives(.en).md    Archive page (layout: archives)
content/reports(.en).md     Reports listing (layout: reports)
hugo.yaml                   Site configuration; per-language menus/params
i18n/zh.yaml, i18n/en.yaml  Strings used by our own templates
layouts/partials/header.html  Theme header + page-level language switch
layouts/_default/           Layout overrides:
  reports.html              Reports cards: posts that have static/reports/<slug>/
  site-search.html          Chunk-based search UI
  index.json                Chunk-based search index (heading-section records)
  _markup/render-passthrough.html  Build-time math rendering (KaTeX)
  single.html               Theme single page + side TOC and drawer
  partials/                 header.html, social_icons.html (X logo), toc.html, ...
assets/css/extended/        Auto-loaded after theme styles:
  pagination.css            Pagination button overrides
  side-toc.css              Side TOC layout + drawer (desktop/mobile)
assets/js/                  site-search.js (search page, bundled with js.Build),
                            search-core.js (ranking, shared with the test),
                            toc-drawer.js, report.js (report theme helpers)
assets/css/report.css       Report skin; published with report.js to /reports/
scripts/                    search_quality.mjs (ranking test),
                            check_translations.py (zh/en pairs in sync),
                            adapt_report.py (fits a report into the blog theme)
static/reports/<slug>/      Standalone interactive report pages (self-contained HTML)
static/blog-icon.png        Site logo + favicon source (avatar from cover art)
static/favicon-16x16.png, favicon-32x32.png, apple-touch-icon.png
static/blog-cover.jpg       OG share image
.github/workflows/hugo.yaml Deploy workflow
.github/copilot-instructions.md  Pointer to this file (do not duplicate content there)
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
node scripts/search_quality.mjs      # needs a fresh public/
python3 scripts/check_translations.py

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
4. Add the translation as `<kebab-case-slug>.en.md` (or `.md` if the original
   is English). Keep `date`, `tags` and the heading structure identical;
   translate `title` and `summary`. Links to other posts point at the same
   language (`/posts/...` in Chinese, `/en/posts/...` in English); report
   links stay `/reports/<slug>/` because reports are Chinese-only.
5. Build locally and confirm the card renders correctly in both languages.

### Adding an interactive report

1. Place the self-contained report at `static/reports/<slug>/index.html`,
   where `<slug>` is the companion post's file name
   (`content/posts/<slug>.md`).
2. Run `python3 scripts/adapt_report.py static/reports/<slug>/index.html`
   (idempotent). It removes the report's own font, links
   `/reports/report.css` + `/reports/report.js`, adds the site header and OG
   tags, and routes every color in the report's scripts through the blog's
   light/dark theme: `RC("--token")` for neutral tokens, `RD("#hex")` for
   fixed colors (lightened on dark paper), `RA(color, alpha)` for rgba().
3. That is the whole registration: `/reports/` lists every post whose report
   folder exists, using the post's own title, summary and date (newest
   first). There is no separate registry to keep in sync.
4. Check the report in both themes. Charts read their colors once at load,
   so the header's theme toggle reloads the page.

### Styling changes

- Put all CSS overrides in `assets/css/extended/*.css`; PaperMod loads them
  after theme styles. Do not edit theme CSS.
- Match the established visual language: translucent dark surfaces
  (`rgba(255,255,255,0.06)`), 1px `var(--border)` borders, small radii
  (10–12px), `--secondary` text / `--primary` on hover.

## Configuration Gotchas

- The Chinese language key and `defaultContentLanguage` must be `"zh"` — the
  theme only ships `zh.yaml`; `zh-cn` silently loses every translation
  (buttons like 「下一页」render empty).
- Do not set `url:` in content front matter. Default paths already give
  `/about/` and `/en/about/`; a `url:` would pin both languages to one path.
- `layouts/partials/header.html` is a copy of the theme header with one
  change: the language link targets the current page's translation (falling
  back to the other language's home) and uses `.Language.Label`. Re-apply that
  block if the theme header is ever re-synced. `layouts/_default/single.html`
  is the theme's single template plus the side TOC; re-sync it the same way.
- Builds print two deprecation warnings (`LanguageDirection`, `LanguageCode`)
  from PaperMod's own templates. They do not change the output; do not copy
  theme files into `layouts/` just to silence them.
- `disableAnchoredHeadings: true` is intentional — no `#` anchor icons on
  headings.
- Math is written as `$...$` / `$$...$$` and rendered to HTML at build time
  (Goldmark passthrough + `transform.ToMath`), so Markdown never touches the
  TeX. Invalid TeX fails the build with its file position. Pages with math
  load the KaTeX stylesheet automatically; there is no `math:` front matter
  and no client-side KaTeX script.
- The Search page must keep `layout: "site-search"` (not `search`) so
  PaperMod's built-in fastsearch script does not double-bind the input.
- Social icons are customized in `layouts/partials/social_icons.html`;
  the `x`/`twitter` name renders the current X logo.
- Favicon/logo/OG image are generated from the cover artwork; regenerate
  with a Pillow script (center-square crop, top offset 60px on the
  1312×1792 source) rather than hand-editing.

## Hard Rules

1. Never modify `themes/PaperMod/` — override via root `layouts/` and
   `assets/css/extended/`.
2. Never commit secrets or credentials.
3. Never convert the site to `hugo.toml` or change the theme.
4. Keep generated artifacts out of git (`public/` is gitignored; don't
   force-add it).
5. After any change: `hugo --gc --minify` must succeed, and any HTML/CSS
   claims must be verified against `public/` output before committing.
