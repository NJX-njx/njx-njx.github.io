# Njx'Log

Personal technical blog built with [Hugo](https://gohugo.io/), the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, and GitHub
Pages.

Production site: <https://njx-njx.github.io/>

## Project Layout

```text
content/posts/              Blog posts in Markdown
content/*.md                Special pages such as About, FAQ, Search, Archive
hugo.yaml                   Main Hugo configuration
layouts/_default/           Local layout overrides
assets/css/                 Custom CSS processed by Hugo Pipes
assets/js/                  Custom JavaScript processed by Hugo Pipes
static/                     Static files copied as-is
workers/openrouter-proxy/   Cloudflare Worker proxy for the AI assistant
themes/PaperMod/            PaperMod theme submodule
.github/workflows/hugo.yaml GitHub Pages deployment workflow
```

Do not edit files under `themes/PaperMod/` directly. Override theme behavior in
the root `layouts/` and `assets/` directories instead.

## Local Setup

This site requires Hugo Extended. On Windows with Scoop:

```powershell
scoop install hugo-extended
hugo version
```

Other install options are documented at <https://gohugo.io/installation/>.

After cloning, initialize the theme submodule if needed:

```powershell
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

## Search And AI Assistant

The search page uses a custom layout, `search-assistant`, at `/search/`. It
loads a custom browser script from `assets/js/search-assistant.js` and reads
Hugo's generated `/index.json` search index.

This custom layout intentionally does not use PaperMod's built-in `search`
layout name, because PaperMod automatically injects its own `fastsearch` script
for pages whose layout is exactly `search`. Keeping this page on
`search-assistant` prevents two scripts from binding to the same search input.

The AI assistant should call the Cloudflare Worker in
`workers/openrouter-proxy/` instead of calling OpenRouter directly from the
browser. Keep `OPENROUTER_API_KEY` as a Cloudflare secret; never commit it or
place it in Hugo/JavaScript frontend code.

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

```powershell
node --check assets/js/search-assistant.js
python -m py_compile scripts/llm_model.py
hugo --gc --minify
```
