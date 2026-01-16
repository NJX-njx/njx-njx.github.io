# Copilot Instructions for Njx'Log

## Project Overview

This is a Hugo static site using the **PaperMod** theme, deployed to GitHub Pages at `njx-njx.github.io`. The blog documents learning notes in Chinese/English.

## Architecture

```
content/posts/    → Blog posts (Markdown with YAML frontmatter)
content/*.md      → Special pages (archives, search, faq)
hugo.yaml         → Site configuration (NOT toml)
themes/PaperMod/  → Theme (git submodule, do NOT modify)
.github/workflows/hugo.yaml → GitHub Actions deployment
```

## Creating New Posts

Use Hugo CLI or follow this frontmatter pattern:

```markdown
---
title: "Post Title"
date: 2024-01-15T10:00:00+08:00  # Use +08:00 timezone
draft: false                     # Set true to hide from production
tags: ["tag1", "tag2"]
categories: ["Tech"]             # Optional
cover:
    image: "url-or-path"         # Optional cover image
    alt: "Description"
    caption: "Caption text"
---
```

**Key conventions:**
- Dates use ISO 8601 with `+08:00` timezone
- File names use kebab-case: `my-new-post.md`
- Tags are lowercase arrays

## Local Development Commands

```bash
hugo server -D          # Start dev server with drafts
hugo server             # Start dev server (production content only)
hugo new posts/my-post.md  # Create new post from archetype
hugo                    # Build site to ./public/
```

## Configuration (hugo.yaml)

- Theme: PaperMod with home-info mode (not profile mode)
- Features enabled: TOC, reading time, word count, breadcrumbs, code copy buttons
- Search: Uses JSON output (see `outputs.home` includes JSON)
- Comments: Disabled (`comments: false`)

## Special Pages Setup

| Page | File | Layout |
|------|------|--------|
| Archives | `content/archives.md` | `layout: "archives"` |
| Search | `content/search.md` | `layout: "search"` |

## Deployment

- **Automatic**: Push to `main` branch triggers GitHub Actions
- **Hugo version**: Uses `peaceiris/actions-hugo@v3` with extended version
- **Output**: Built to `./public/`, uploaded as Pages artifact

## Do NOT

- Modify files in `themes/PaperMod/` (use Hugo's override system in root `layouts/` instead)
- Use `hugo.toml` (this project uses `hugo.yaml`)
- Commit draft posts (`draft: true`) unless intentional
