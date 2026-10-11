---
title: "搭建 Njx'Log：Hugo、PaperMod、GitHub Pages"
date: 2026-02-07T08:00:00+08:00
draft: false
tags: ["hugo", "papermod", "blog", "cicd"]
categories: ["Tech"]
summary: "一份完整的指南：我如何使用 Hugo、PaperMod 主题以及基于 GitHub Actions 的自动化部署，搭建起这个技术博客。"
---

## 动机

用了多年零散的笔记工具之后，我决定为自己的技术思考打造一个专属空间。我的要求很简单：**速度快、极简、原生支持 Markdown、零维护开销**。

最终我选定了 [Hugo](https://gohugo.io/)，搭配 [PaperMod](https://github.com/adityatelange/hugo-PaperMod) 主题。Hugo 基于 Go 的引擎带来极快的构建速度，而 PaperMod 提供简洁、无干扰的视觉风格。

## 架构概览

网站托管在 GitHub 上，并使用 GitHub Actions 自动完成构建与部署。

*   **静态站点生成器**：Hugo（Extended 版本）
*   **主题**：PaperMod（Home-info 模式）
*   **部署**：GitHub Pages
*   **自动化**：GitHub Actions
*   **配置**：YAML（`hugo.yaml`）

## 关键技术配置

### 1. 配置（`hugo.yaml`）

虽然很多 Hugo 站点使用 TOML，但出于可读性考虑，我更喜欢 YAML。我把 PaperMod 配置成 `home-info` 模式，让它充当一个极简的落地页，而不是厚重的个人资料视图。

```yaml
params:
  homeInfoParams:
    Title: "👋 Welcome to Njx'Log"
    Content: >
        Hi, this is Jiaxing Ni. I share my AI/ML learning notes here.
```

### 2. 特殊页面与搜索

PaperMod 借助 `Fuse.js` 开箱即用地支持本地搜索。为了启用它，我确保构建输出中包含 JSON：

```yaml
outputs:
  home:
    - HTML
    - RSS
    - JSON # Required for search
```

我还在 `content/` 目录下创建了带有 `layout` 参数的特定 Markdown 文件，为归档和搜索设置了专门的布局。

### 3. CI/CD 工作流

部署是完全自动化的。每次推送到 `main` 分支，都会触发 `.github/workflows/hugo.yaml` 中的工作流。它会配置 Hugo 环境、构建静态资源，并把它们上传到 GitHub Pages。

```yaml
- name: Setup Hugo
  uses: peaceiris/actions-hugo@v3
  with:
    hugo-version: '0.153.0'
    extended: true # Necessary for SCSS compilation
```

## 我的内容工作流

为了保持一致性，我遵循以下几条给自己定下的规则：
1.  **创建**：使用 `hugo new posts/your-post.md` 生成新内容的骨架。
2.  **时区**：frontmatter 中始终使用 `+08:00`。
3.  **资源**：图片存放在 `static/` 中，并通过相对路径引用。

## 结语

> “好记性不如烂笔头。”

这个博客不只是一个笔记仓库，更是我工程思考的实验室。接下来，我会分享更多关于 LLM 微调、RAG 流水线和 MLOps 的深度解析。

敬请期待后续更新。
