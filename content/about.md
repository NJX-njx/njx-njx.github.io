---
title: "关于"
summary: "关于 Jiaxing Ni"
aliases: ["/faq/"]
hidemeta: true
ShowToc: true
ShowReadingTime: false
ShowWordCount: false
ShowBreadCrumbs: false
ShowShareButtons: false
---
## Jiaxing Ni

19 岁，做 LLM 训练的工程这一面——harness、评测、后训练，以及它们交界处的所有管线。

好的系统表面上很安静；底下是 rollout 管线、reward 轨迹、评测切片，还有几个守夜班的脚本。我的大部分工作，是让这些东西说实话。

---

## 我在做的事

一条主线：把真实世界的使用痕迹，变成训练能听懂的信号。

- 从线上 session 和用户反馈里挖模型的坏模式，再把它们做成 bench 与多轮数据
- 给 agent 任务造可执行的 verifier：分层判分、变异测试自检，判分噪声要能被剥出来
- 评测不只要交出一个分数，还要回答它为什么能泛化
- RL 环境、rollout 系统、SFT 数据合成——让后训练的每一克信号都有出处

---

## 痕迹

几段经历绕的都是同一件事：做过 agent harness 和 swarm 评测，从十几万条多轮会话里归纳过模型的通用毛病；搭过预训练的数据基座，也试过让基座模型自己长出品鉴训练方案的「品味」；研究过评测机制本身——让评分锚定证据，用对话的几何结构把满意度变成可解释的量；再往前，给散落在 macOS 各处的个人上下文造过一个可信任、可授权、可行动的系统。

具体在哪些牌子底下做的就不列了——做过的事比待过的地方重要，细节留给线下聊。

---

## 坐标

- **身份：** 研究者、建造者、工程师。
- **所在地：** 北京 / 上海。
- **口味：** 原型要快，测量要诚实，只留下经得起真实边界检验的东西。

---

## 关于这个博客

一本关于 LLM 训练背后机制的田野笔记：一个旋钮到底做了什么，拧动它时什么会坏，指标又说了什么。

话题围绕 **harness 与 rollout 系统、模型架构、mid-training 与后训练、评测与 benchmark、推理 RL 与 agentic RL、RSI、数据合成** 展开。较长的文章通常配有交互报告版，公式变成滑杆，曲线变成探针，见[报告](/reports/)。

博客用 [Hugo](https://gohugo.io/) 和 [PaperMod](https://github.com/adityatelange/hugo-PaperMod) 搭建，托管在 [GitHub Pages](https://pages.github.com/)，源码公开在 [GitHub](https://github.com/NJX-njx/njx-njx.github.io)。

---

## 常见问题

**哪些内容不会写在这里？** 我尽量不写泛泛的 AI 评论。如果一篇文章没有代码、没有实验痕迹、没有评测细节，也没有诚实交代的不确定性，它大概属于别的地方。

**可以引用、翻译或转发文章吗？** 可以。请保留原文链接，也欢迎告诉我它流传到了哪里。我喜欢看到笔记出现在另一张书桌上。

**愿意合作或交流吗？** 愿意，尤其是 RL 基础设施（rollout 与 serving）、后训练配方、评测设计、agent 系统，以及那些让研究更好用的奇怪小工具。

---

## 在网上找到我

- [GitHub](https://github.com/NJX-njx)
- [Hugging Face](https://huggingface.co/NJX-njx)
- [X](https://x.com/NJX_njx_ai)
- 邮箱：[njxnjxai@gmail.com](mailto:njxnjxai@gmail.com)
