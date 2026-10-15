# AGENTS.md

Guidelines for AI coding agents working on **Njx'Log** (`njx-njx.github.io`).

## What This Is

A personal technical blog (Hugo + PaperMod → GitHub Pages) about the
engineering side of LLM training: harness and rollout systems, model
architecture, mid-training and post-training, evals and benchmarks,
reasoning RL and agentic RL, RSI, and data synthesis.

站点支持双语：中文默认位于 `/`，英文位于 `/en/`。Hugo 内容页使用
`<name>.md`（中文）与 `<name>.en.md`（英文）配对，header 跳转当前页译文。
`static/reports/` 交互报告保持中文，两种语言的报告目录链接到同一份报告。

Production: <https://njx-njx.github.io/>

## Quick Facts

- Hugo **Extended** required; config is `hugo.yaml` (never `hugo.toml`)
- Theme is a git submodule at `themes/PaperMod/` — **never edit it**
- Pull requests → GitHub Actions builds and runs every check; push to `main`
  → the same workflow also deploys to Pages in ~1 minute
- The Hugo version is pinned once, as `HUGO_VERSION` in
  `.github/workflows/hugo.yaml`; the Cloud Agent environment's install
  script pins the same version and must be bumped with it
- Verify deploys with `gh run list`; verify live CSS/HTML with `curl`

## Repository Layout

```text
content/posts/              Blog posts: <slug>.md (zh) + <slug>.en.md (en)
content/about(.en).md       About page（包含 FAQ）
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
                            check_posts.py (zh/en pairs in sync, summary length),
                            check_site.py (links, anchors, search index, reports),
                            adapt_report.py (fits a report into the blog theme)
static/reports/<slug>/      中文交互报告 HTML（依赖共享 report.css/report.js）
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

# Build & check (CI runs the same steps on every pull request)
hugo --gc --minify                  # output in public/
python3 scripts/check_posts.py      # zh/en pairs in sync, summary length
python3 scripts/check_site.py public    # links, anchors, search index, reports
node scripts/search_quality.mjs public  # search ranking regression test

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
5. Refer to other posts by name and link, never by time ("昨天那篇", "this
   week", "today's post"); posts outlive their publishing week.
6. Multi-part series: add `series: ["<key>"]` to every part (both
   languages). The series title lives in `content/series/<key>/_index.md` and
   `_index.en.md`; the series box at the end of each part
   (`layouts/partials/extend_post_content.html`) lists the parts by date.
7. Build locally and confirm the card renders correctly in both languages.

### Adding an interactive report

1. 将中文报告放在 `static/reports/<slug>/index.html`，`<slug>` 与
   `content/posts/<slug>.md` 对应。报告依赖共享资源，并非完全自包含文件。
2. Run `python3 scripts/adapt_report.py static/reports/<slug>/index.html`
   (idempotent). It removes the report's own font, links
   `/reports/report.css` + `/reports/report.js`, adds the site header and OG
   tags, and routes every color in the report's scripts through the blog's
   light/dark theme: `RC("--token")` for neutral tokens, `RD("#hex")` for
   fixed colors (lightened on dark paper), `RA(color, alpha)` for rgba().
3. That is the whole registration: `/reports/` lists every post whose report
   folder exists, using the post's own title, summary and date (newest
   first). There is no separate registry to keep in sync.
4. 检查桌面、320/390px 窄屏和两种主题，包括章节跳转、滑杆、下钻卡与刷新保留。
   图表在加载时解析配色，因此切换主题会重载页面；存储不可用时使用页面参数
   `report-theme` 保持本页配色，链接和 hash 不受影响，不保证跨页保存。
5. 共享层只有 `assets/css/report.css` 和 `assets/js/report.js`，经 Hugo 发布到
   `/reports/`。不要叠加另一套头部或 Canvas/SVG 反色滤镜。
   `adapt_report.py` 应幂等，重新运行不得修改正文、数据或产生重复引用。

### Styling changes

- 博客 CSS 覆盖放在 `assets/css/extended/*.css`，报告共享样式放在
  `assets/css/report.css`；不要编辑主题子模块。
- The site follows the reader's light/dark preference, so colors come from
  PaperMod's theme tokens (`--theme`, `--entry`, `--code-bg`, `--border`,
  `--primary`, `--secondary`, `--tertiary`), never hard-coded light or dark
  values. The visual language: `var(--entry)` surfaces, 1px `var(--border)`
  borders, small radii (`var(--radius)` to 12px), `--secondary` text that
  turns `--primary` on hover.

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
6. 只有用户明确授权后才可推送；不得强制覆盖远端已有提交。
