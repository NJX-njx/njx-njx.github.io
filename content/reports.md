---
title: "Reports"
layout: "reports"
description: "交互式研报：把博客文章里的机制拆解做成可交互的数据可视化报告"
reports:
  - title: "「模型自己把自己练强」：RSI 与三个闭环"
    url: /reports/rsi-self-improvement-loops/
    date: 2026-09-01T12:30:00+08:00
    desc: "RSI 与自我改进的层次：数据闭环、奖励闭环、系统闭环——各自的成立条件、兑现程度和失效模式并不相同。"
  - title: "7 轮交互失败了该怪谁：奖励归因怎么进 GRPO/GSPO"
    url: /reports/turn-level-reward-credit-assignment/
    date: 2026-08-31T12:55:00+08:00
    desc: "多轮 agentic RL 的奖励归因：轨迹概率、三层信号、ORM/PRM 对照实验、三条中间路线、turn-level 反向折扣，以及 GSPO-token 为多轮留的那个口子。"
  - title: "一票还是十票：token-mean 与 sample-mean 在争什么"
    url: /reports/token-mean-vs-sample-mean/
    date: 2026-08-30T12:05:00+08:00
    desc: "同一个 batch 里，500 token 的回答和 5000 token 的回答，谁对梯度的贡献大？GRPO 说一样大，DAPO 说长的大十倍。损失聚合方式之争：sample-mean 每条回答一票…"
  - title: "截断的回答该怎么判：DAPO Overlong 在修什么"
    url: /reports/dapo-overlong-reward-shaping/
    date: 2026-08-30T12:05:00+08:00
    desc: "RL 训练必须有生成长度上限，被截断的回答默认按答错处理——这会把「无法判定」的样本错判成「错」样本，把奖励噪声经组内归一化放大注入训练。DAPO 的两个方案：Overlong Filtering 承…"
  - title: "“答错就写长一点”：GRPO 归一化项在扭曲什么"
    url: /reports/dr-grpo-length-and-difficulty-bias/
    date: 2026-08-29T12:05:00+08:00
    desc: "R1-Zero 的「长度与奖励齐涨」被当成推理涌现的标志性曲线，Dr.GRPO 论文指出其中一半涨幅可能是优化器自己的偏好：1/|y| 让模型答错时倾向写长，逐题 std 让极端难度的题拿走更多权重。…"
  - title: "一道题采几条回答：GRPO 组大小 G 买来了什么"
    url: /reports/grpo-group-size-tradeoffs/
    date: 2026-08-29T12:00:00+08:00
    desc: "DeepSeekMath 每题采 64 条回答，教学实现默认 8 条，最新论文 2 条也声称够用——同一个旋钮差出 32 倍。G 是 GRPO 唯一的 baseline 来源和最大的算力开关，它到底买…"
  - title: "解剖极简 SFT 训练器：每个决策，教科书和框架怎么说"
    url: /reports/forge-sft-anatomy/
    date: 2026-08-28T20:12:53+08:00
    desc: "一个极简 SFT 训练器的 7 个工程决策，对照两套教材、四个生产框架与六路信息源。"
  - title: "同一个模型，两个概率：训练推理不一致的代价"
    url: /reports/tis-train-inference-mismatch/
    date: 2026-08-27T13:00:00+08:00
    desc: "rollout 用推理引擎、训练用训练引擎，同一个模型在两边算出两个概率：名义上的 on-policy 悄悄变成 off-policy。本报告拆解差异来源、有偏梯度与部署差距两个后果、算法补丁谱系，以…"
  - title: "比率用错了单位：GSPO 把重要性采样提到序列级"
    url: /reports/gspo-sequence-level-importance-sampling/
    date: 2026-08-26T17:40:00+08:00
    desc: "GRPO 沿用 PPO 的 token 级重要性比率，Qwen 团队认定这是对重要性采样的根本误用。GSPO 把比率、裁剪、优化全部提到序列级：被裁掉的 token 多两个数量级，训练反而更高效。"
  - title: "熵是怎么塌掉的：DAPO 的 Clip-Higher 在修什么"
    url: /reports/dapo-clip-higher-entropy-collapse/
    date: 2026-08-26T14:30:00+08:00
    desc: "naive GRPO 复现 R1 时熵快速坍缩，DAPO 的 Clip-Higher 只把裁剪上界从 0.2 抬到 0.28。这一个数背后，是 PPO-Clip 在概率轴上的结构性不对称。"
---

<!-- 报告列表由 layouts/_default/reports.html 渲染 -->
