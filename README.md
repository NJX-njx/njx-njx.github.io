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
layouts/_default/           Local layout overrides (reports, site-search, rss, baseof)
assets/css/extended/        Custom CSS auto-loaded by PaperMod after theme styles
assets/js/                  Custom JavaScript processed by Hugo Pipes
static/reports/             中文交互报告及 report-shell.css / report-shell.js
scripts/                   报告接入、文章配对和搜索质量检查
static/*.png|jpg            Favicons, logo, OG image
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

## 双语页面

中文为默认语言，位于 `/`；英文位于 `/en/`。文章使用 `<slug>.md` 与
`<slug>.en.md` 配对，保持日期、标签、章节层级、公式和代码一致，仅翻译正文、标题与摘要。
header 的语言入口优先跳转到当前页译文，无对应译文时回退到另一语言首页。
新增或修改文章后运行 `python3 scripts/check_translations.py`。

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

长文可以配套 `static/reports/<slug>/index.html` 交互报告。
这是独立 HTML 页面，但依赖上一级目录的共享站点样式和脚本，不再完全自包含。
`<slug>` 必须与 `content/posts/<slug>.md` 的文章文件名对应。

报告正文保持中文，中文和英文报告目录均链接到 `/reports/<slug>/`。共享的
`static/reports/report-shell.css` 和 `report-shell.js` 提供 Njx'Log 头部、
站点导航、无衬线字体和浅色／深色切换，使用与博客一致的 `pref-theme` 存储键。
报告仍保留自己的图表、数据和交互；暗色适配覆盖 Canvas、SVG、提示框与下钻卡。

新增报告后运行 `python3 scripts/harmonize_reports.py`，接入共享层并移除旧的
内嵌 ET Book 字体。脚本可重复运行，不应产生重复头部、资源引用或额外差异。
修改共享 CSS/JS 即可更新全部报告；修改脚本中的头部或初始化模板后，需重新运行脚本。
验收时检查桌面和窄屏、两种主题、章节跳转、滑杆和下钻卡，并确认刷新后主题保留。
浏览器禁用站点存储时，主题仍可在当前页面切换，但不保证跨页面保存。

报告目录页（`layouts/_default/reports.html`）自动列出存在对应报告目录的文章，
使用文章自身的标题、摘要和日期，无需维护额外注册表。添加报告并完成共享层接入后，
正常构建即可发布；目录卡片复用 PaperMod 的 `post-entry` 样式。

## Search

The search page uses a custom layout, `site-search`, at `/search/`. It
loads a custom browser script from `assets/js/site-search.js` and reads
Hugo's generated `/index.json` search index. The project overrides PaperMod's
default JSON index in `layouts/_default/index.json` so the index is chunk-based:
each Markdown heading section becomes a searchable record with `title`,
`section`, `content`, `permalink`, `anchor`, `summary`, `tags`, and `date`.
Search results link directly to the matched section anchor.

This custom layout intentionally does not use PaperMod's built-in `search`
layout name, because PaperMod automatically injects its own `fastsearch` script
for pages whose layout is exactly `search`. Keeping this page on
`site-search` prevents two scripts from binding to the same search input.

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
node --check assets/js/site-search.js
node --check scripts/search_quality.js
node --check static/reports/report-shell.js
python3 scripts/check_translations.py
hugo --gc --minify
node scripts/search_quality.js public/en/index.json
```

## For AI Agents

See [AGENTS.md](AGENTS.md) for repository conventions, structure, and
workflows when working on this project with an AI coding agent.
