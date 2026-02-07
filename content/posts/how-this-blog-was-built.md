---
title: "Building Njx'Log: The Full Stack of Hugo, PaperMod, and GitHub Pages"
date: 2026-02-07T14:30:00+08:00
draft: false
tags: ["hugo", "papermod", "blog", "cicd"]
categories: ["Tech"]
summary: "A comprehensive guide on how I built this technical blog using Hugo, the PaperMod theme, and automated deployment via GitHub Actions."
---

## The Motivation

After years of using fragmented note-taking tools, I decided to build a dedicated space for my technical reflections. My requirements were simple: **speed, minimalism, Markdown native, and zero maintenance overhead**.

I landed on [Hugo](https://gohugo.io/) paired with the [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme. Hugo's Go-based engine provides blazing-fast build times, while PaperMod offers a clean, distraction-free aesthetic.

## Architecture Overview

The site is hosted on GitHub and utilizes GitHub Actions for automated building and deployment.

*   **Static Site Generator**: Hugo (Extended version)
*   **Theme**: PaperMod (Home-info mode)
*   **Deployment**: GitHub Pages
*   **Automation**: GitHub Actions
*   **Configuration**: YAML (`hugo.yaml`)

## Key Technical Setup

### 1. Configuration (`hugo.yaml`)

While many Hugo sites use TOML, I preferred YAML for its readability. I configured PaperMod in `home-info` mode to serve as a minimalist landing page rather than a heavy profile view.

```yaml
params:
  homeInfoParams:
    Title: "👋 Welcome to Njx'Log"
    Content: >
        Hi, this is Jiaxing Ni. I share my AI/ML learning notes here.
```

### 2. Special Pages & Search

PaperMod supports local search out-of-the-box using `Fuse.js`. To enable this, I ensured the build output includes JSON:

```yaml
outputs:
  home:
    - HTML
    - RSS
    - JSON # Required for search
```

I also set up dedicated layouts for archives and search by creating specific Markdown files in the `content/` directory with `layout` parameters.

### 3. CI/CD Workflow

The deployment is fully automated. Every push to the `main` branch triggers a workflow in `.github/workflows/hugo.yaml`. It sets up the Hugo environment, builds the static assets, and uploads them to GitHub Pages.

```yaml
- name: Setup Hugo
  uses: peaceiris/actions-hugo@v3
  with:
    hugo-version: '0.153.0'
    extended: true # Necessary for SCSS compilation
```

## My Content Workflow

To maintain consistency, I follow these self-imposed rules:
1.  **Creation**: Use `hugo new posts/your-post.md` to scaffold new content.
2.  **Timezone**: Always use `+08:00` in frontmatter.
3.  **Assets**: Store images in `static/` and link them via relative paths.

## Conclusion

> "The faintest ink is better than the best memory."

This blog is more than just a repository of notes; it's a laboratory for my engineering thoughts. Moving forward, I’ll be sharing more deep dives into LLM fine-tuning, RAG pipelines, and MLOps.

Stay tuned for more updates.
