# AGENTS.md

Guidelines for AI coding agents working on **Njx'Log** (`njx-njx.github.io`).

## What This Is

A personal technical blog (Hugo + PaperMod → GitHub Pages) about the
engineering side of LLM training: harness and rollout systems, model
architecture, mid-training and post-training, evals and benchmarks,
reasoning RL and agentic RL, RSI, and data synthesis.

站点支持双语：中文默认位于 `/`，英文位于 `/en/`。
Hugo 内容页使用成对文件：`<name>.md` 为中文、`<name>.en.md` 为英文；
header 的语言入口跳转到当前页面的译文。`static/reports/` 下的交互报告是例外：
正文保持中文，不创建英文副本，双语报告目录指向同一报告。

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
  baseof.html, single.html, rss.xml
  partials/                 social_icons.html (X logo), toc.html, ...
assets/css/extended/        Auto-loaded after theme styles:
  pagination.css            Pagination button overrides
  side-toc.css              Side TOC layout + drawer (desktop/mobile)
assets/js/                  site-search.js, toc-drawer.js
static/reports/<slug>/      交互报告 HTML（中文，依赖共享站点层）
static/reports/report-shell.css  报告导航、主题和样式适配
static/reports/report-shell.js   主题切换与章节导航偏移
scripts/harmonize_reports.py     幂等接入共享站点层
scripts/check_translations.py    检查中英文文章配对
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
node --check assets/js/site-search.js
node --check static/reports/report-shell.js
python3 scripts/check_translations.py
node scripts/search_quality.js public/en/index.json

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

1. 将报告放在 `static/reports/<slug>/index.html`，`<slug>` 与
   `content/posts/<slug>.md` 对应；报告正文保持中文，不翻译。
2. 运行 `python3 scripts/harmonize_reports.py`，接入共享头部、favicon、主题初始化，
   移除内嵌 ET Book 并使用博客字体栈。脚本应幂等：连续运行两次，第二次不应产生差异。
3. 报告依赖 `static/reports/report-shell.css` 和 `report-shell.js`，不是完全自包含文件。
   共享层使用 `html[data-theme]` 和 `pref-theme`，与博客主题一致；存储不可用时仍能切换。
   Canvas/SVG 使用共享暗色滤镜，正文布局和图表数据由报告自己维护。
4. 共享样式与行为直接改 CSS/JS；修改头部或早期主题模板后重新运行接入脚本。
   不要给报告加载整套 PaperMod 样式，避免与原有图表类名冲突。
5. `/reports/` 与 `/en/reports/` 自动列出有对应报告目录的文章，使用各自语言的
   标题、摘要和日期，无需额外注册；卡片均打开中文报告。
6. 构建后核验桌面和窄屏、两种主题、导航跳转避让、主题持久化及滑杆、提示框和下钻卡。

### Styling changes

- 博客样式放在 `assets/css/extended/*.css`，报告共享样式放在
  `static/reports/report-shell.css`；不要修改主题子模块。
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
  block if the theme header is ever re-synced.
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
6. 未获用户明确授权不得推送；本地提交也不得携带构建产物或凭据。
