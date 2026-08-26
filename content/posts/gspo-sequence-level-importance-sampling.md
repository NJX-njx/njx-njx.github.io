---
title: "比率用错了单位：GSPO 为什么把重要性采样提到序列级"
date: 2026-08-26T17:40:00+08:00
draft: false
math: true
tags: ["rl", "post-training", "grpo", "gspo", "moe"]
categories: ["AI"]
summary: "GRPO 沿用 PPO 的 token 级重要性比率，Qwen 团队认定这是对重要性采样的根本误用，会在 MoE 长回答训练中导致不可逆崩溃。GSPO 把比率、裁剪、优化全部提到序列级，被裁掉的 token 多两个数量级，训练反而更高效。本文拆解这条论证链，以及论文没点破的前提与代价。"
---

> 本文有配套的[交互报告](/reports/gspo-sequence-level-importance-sampling/)，含可下钻的公式拆解、裁剪带对比与稀释效应计算器。

## 一个补丁暴露的问题

2025 年中之前，用 GRPO 训 MoE 模型需要一个专门的补丁。

Qwen 团队在 GSPO 论文（arXiv:2507.18071）里交代了他们此前的办法：**Routing Replay**。采样时把旧策略 $\pi_{\theta_{old}}$ 激活的专家路由缓存下来，计算重要性比率时在新策略 $\pi_\theta$ 上"重放"这套路由，让比率的分子分母跑同一张激活网络。之所以需要这个补丁，是因为他们观察到一个数字：48 层的 Qwen3-30B-A3B，每做一次梯度更新，同一条 rollout 样本在新策略下激活的专家约有 10% 与旧策略不同。路由一变，逐 token 的概率跟着变，token 级重要性比率剧烈波动，GRPO 直接不收敛。

补丁能跑，但代价实打实：额外的显存和通信开销，而且强制新策略走旧路由等于限制模型的实际容量。更麻烦的是另一件事——即便有补丁，大模型长回答任务上的 GRPO 仍会发生**不可逆的模型崩溃**。论文的措辞很重：崩溃一旦发生，回退 checkpoint、细调裁剪范围、延长生成长度、换 RL query，全都救不回来。

token 级重要性比率从 PPO 时代就是标配，用了八年。为什么在 LLM 的 RL 里突然成了病灶？GSPO 的回答是：它不是突然生病的，它从一开始就用错了地方——这是对重要性采样的误用，只是在小模型、短回答、近 on-policy 的年代里，错误被掩盖了。

## 重要性采样为什么会"失效"

先把重要性采样（IS）本身写出来。要用行为分布 $\pi_{beh}$ 的样本，估计目标分布 $\pi_{tar}$ 下函数 $f$ 的期望：

$$\mathbb{E}_{z\sim\pi_{tar}}[f(z)] = \mathbb{E}_{z\sim\pi_{beh}}\left[\frac{\pi_{tar}(z)}{\pi_{beh}(z)} f(z)\right]$$

这个恒等式成立不靠比率本身，靠的是**大量样本的平均**：$N \gg 1$ 时，随机比率 $w(z)=\pi_{tar}(z)/\pi_{beh}(z)$ 的涨落被平均掉，估计才收敛到真值。单个样本乘上 $w$，起不到任何校正作用，只是给 $f(z)$ 乘了一个高方差的随机数。

现在看 GRPO 在做什么。它对每个 token 算比率：

$$w_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}$$

其中 $y_{i,t}$ 是第 $i$ 条回答的第 $t$ 个 token，$y_{i,<t}$ 是它前面的上下文。$\pi_{\theta_{old}}(\cdot|x,y_{i,<t})$ 是第 $t$ 步的 next-token 分布，从里面**只采了一个样本** $y_{i,t}$。也就是说，每个比率都是拿单个样本对单步分布做"校正"——按上面的判据，这根本不构成 IS 校正，它只是往梯度里注入了一个逐 token 的随机权重。序列越长，累积的随机权重越多；模型越大越稀疏（MoE），单个权重的方差越大。clipping 再把超出范围的 token 梯度清零，噪声就被放大成"随机抹掉一部分 token 的学习信号"。RLHF Book 第 6 章对失效现场有两个具体描述：单个大比率 token 可以主导整个更新；同一条 response 内部的 token 被各自独立地裁剪，学习信号被切得粉碎。

GSPO 论文把症结提炼成一句话：**优化目标的单位应当匹配奖励的单位**。RLVR 的奖励是给整条序列的（答对 1 分答错 0 分），而 GRPO 的校正和裁剪都发生在 token 级。单位错配，校正越"认真"，伤害越精确。

这里要诚实补一句：GRPO 这么写并非没有理论来源。把每个 token 看作一步 action、整条轨迹的 IS 比率分解为逐步比率，是标准 RL 的推导路径。问题出在实现层面把逐 token 比率当作独立的校正权重逐个使用、逐个裁剪——这一步近似在大词表、长轨迹、单样本的条件下不再无害。这是 GSPO 论文的批评角度，不是公理，读者可以保留自己的判断。

## GSPO：比率、裁剪、优化全部换到序列级

GSPO 的目标函数：

$$\mathcal{J}_{GSPO}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\{y_i\}_{i=1}^G\sim\pi_{\theta_{old}}}\left[\frac{1}{G}\sum_{i=1}^G \min\left(s_i(\theta)\hat{A}_i,\ \mathrm{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_i\right)\right]$$

和 GRPO 的差别一眼可见：min-clip 外面没有了 $\frac{1}{|y_i|}\sum_t$ 的逐 token 求和，一条 response 整体只占目标函数里的一项。advantage 还是组内归一化的 $\hat{A}_i$（用组内奖励的均值和标准差标准化），整条 response 共享。

核心是序列级比率 $s_i$ 的定义：

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)$$

即整条序列似然比的几何平均，等价于逐 token 对数比率的算术平均再取 exp。$1/|y_i|$ 这个长度归一化不能省：不归一化，几千个比率连乘，少数几个 token 的似然变化就能让总比率剧烈波动，而且不同长度的 response 需要不同的裁剪范围，$\varepsilon$ 根本没法设。归一化之后，$s_i$ 的数值范围与序列长度解耦，一个 $\varepsilon$ 通吃。（本地教材 hands-on-modern-rl 第 16.4 节介绍 GSPO 时公式漏了这个 $1/|y_i|$，以论文公式 7 为准——漏掉它，整条论证链条就断了。）

裁剪的含义也随之改变。GRPO 的裁剪是把单个 token 的梯度清零——信号被切碎，但样本还在；GSPO 的裁剪是把整条 response 从梯度估计中剔除——样本要么整个用，要么整个不用。这和"奖励给整条序列"在粒度上对齐了。

## 梯度视角：不等权 vs 等权

把两个目标的梯度展开（论文公式 8–12），根本差别落在一个地方：逐 token 的 $\nabla_\theta\log\pi_\theta$ 前面乘的是什么。

GRPO 的梯度里，token $t$ 的 $\nabla\log\pi$ 乘的是**它自己的比率** $w_{i,t}$。经过 min-clip 后，这些逐 token 权重分布在 $(0, 1+\varepsilon]$（$\hat{A}_i>0$ 时）或 $[1-\varepsilon, +\infty)$（$\hat{A}_i<0$ 时）——同一条 response 内部，不同 token 拿到的梯度权重可以差出量级，且随训练步数累积，后果难以预测。

GSPO 的梯度里，同一条 response 的所有 token 共享**同一个标量权重** $s_i$：

$$\nabla_\theta\mathcal{J}_{GSPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G s_i(\theta)\hat{A}_i \cdot \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]$$

每个 token 对更新的贡献只相差它们自身 $\nabla\log\pi$ 的大小，不再有逐 token 的随机权重调制。论文把这称为消除 GRPO 的不稳定因素。

## 证据：三个实验事实

论文的实验设置本身就值得注意：冷启动模型（Qwen3-30B-A3B-Base 经 SFT 得到），每个 rollout batch 切成 4 个 mini-batch 做梯度更新——这是标准的 off-policy 场景，也正是上一篇写 Clip-Higher 时说过的"clip 真正开始工作"的场景（比率才会偏离 1）。GRPO 基线的裁剪范围 0.2/0.27 是精心调过的，GSPO 用的是 3e-4 / 4e-4。

事实一：**ε 差三个数量级，不是笔误**。token 级比率单个就能冲到 1.2，序列级几何平均把逐 token 的涨落平均掉了，$s_i$ 天然贴着 1 小幅波动，裁剪带必须窄才有意义。换算一下：3e-4 的容忍带对应平均每个 token 只允许约 0.03% 的对数漂移。反过来读更有意思——逐 token 各自漂移 20% 在 GRPO 里是家常便饭，几何平均之后只剩万分之三，方差缩减的数量级就是这么来的。

事实二：**被裁的 token 多一百倍，效率反而更高**。GSPO 被裁剪的 token 比例比 GRPO 高约两个数量级（调 $\varepsilon$ 改变不了这个量级差），用于训练的 token 更少，训练效率和 benchmark 表现却更好（AIME'24 / LiveCodeBench / CodeForces 三条曲线都赢）。论文的解读：GRPO 的 token 级梯度估计本来就噪声大，被随机保留下来的那部分并不比被裁掉的部分更有用；GSPO 的序列级筛选提供了更可靠的学习信号——少而干净，胜过 多而脏。

事实三：**MoE 不再需要 Routing Replay**。序列似然 $\pi_\theta(y_i|x)$ 对路由抖动不敏感——10% 的专家换了，单个 token 的概率可能明显变化，但整条序列的似然（模型语言建模能力的体现）不会剧变。GSPO 只关心后者，所以直接正常算比率就能收敛，补丁的显存、通信开销全省了，模型也不再被旧路由束缚。

顺带一个基建红利（论文 §5.4）：工程上训练引擎（Megatron）和推理引擎（SGLang、vLLM）存在精度差异，GRPO 必须用训练引擎把旧策略的似然重算一遍才敢用；序列级似然对精度差异的容忍度高得多，GSPO 可以直接用推理引擎返回的似然做优化，省掉重算。对 partial rollout、多轮 RL、训推分离架构，这是实打实的吞吐红利。

## 我的三个保留意见

一、**"单样本"批评同样适用于 GSPO 自己**。$s_i$ 也是从单条样本 $y_i$ 算出的比率，同样没有 $N \gg 1$ 的平均。所以 GSPO 的改进不宜理解为"恢复了重要性采样的理论正确性"——它没有，论文的 IS 论证是直觉性的，不是形式化的。它真正做的是三件事的合流：单位对齐（被裁的是整个"坏样本"而非随机 token）、方差归一（$1/|y|$ 把数值范围压到统一）、等权梯度（消除逐 token 不等权的累积）。这三点都是工程论证，扎实，但不是定理。

二、**几何平均是稀释器，好坏同体**。一条一万 token 的 CoT 里，某个关键 token 的概率漂移 50%，对 $s_i$ 的贡献只有 $e^{\log 1.5 / 10000} \approx 1.00004$，几乎不可见。对训练-推理引擎的精度噪声，稀释是特性（这正是基建红利的来源）；对"关键 token 被策略更新改掉"的检测，稀释是漏洞。GSPO 能正常工作，恰恰因为它配了万分之三量级的极窄裁剪带——靠平均漂移而不是单点漂移来做信任域判断。这套组合在超长 CoT 上是否会对某些失效模式失明（比如策略只在一小段推理上大幅偏离，平均后被放过），论文没有回答。

三、**序列级裁剪放弃了"部分正确"样本的利用方式**。一条 response 90% 的推理都对、最后一步算错，GSPO 把它整个用或整个扔。论文给的出路是 GSPO-token 变体：用 stop-gradient 把序列级权重 $s_i$（只取数值、不回传梯度）与逐 token 项 $\pi_\theta / \mathrm{sg}[\pi_\theta]$ 拼起来，得到的目标、裁剪条件、理论梯度与 GSPO 完全等价，但允许逐 token 定制 advantage——为多轮 RL 和细粒度奖励留的口子。这条路线的另一端是 VAPO：干脆把 value model 请回来做 token 级信用分配。序列级与 token 级不是对错之争，而是"信用分配粒度"这个老问题的两种工业回答。

## 和 Clip-Higher 的关系，以及一个没人点破的细节

上一篇写 DAPO 的 Clip-Higher：病灶同为 off-policy 下 token 级 clip 的结构性缺陷，DAPO 的解法是调边界（上界 0.2 抬到 0.28，放探索 token 一马），GSPO 的解法是换单位（整个 response 共用一个比率）。两者不互斥——Qwen 的实践就是 GSPO 叠加 Dr.GRPO 式的去归一化偏差修正。

还有个细节值得点破：GSPO 论文的裁剪范围 3e-4 / 4e-4 本身就是**不对称的**——右界比左界宽，给强化方向（$\hat{A}>0$ 时比率向上突破）多留了 33% 的空间。这不就是序列级版本的 Clip-Higher 吗。论文没有讨论这个不对称的来历，但方向与 DAPO 的逻辑一致：好的样本，让它走得更远一点。两个团队在同一病灶上独立开出了同方向的药，这比任何单篇论文的消融都更能说明问题。

## 未决问题

- GSPO 的极窄 $\varepsilon$（3e-4 量级）与 clip-higher 式不对称边界叠加时，最优范围在哪，目前没有公开消融；
- 几何平均的稀释效应在超长 CoT（32k token 以上）上是否会系统性放过"局部剧变"，需要构造性实验验证；
- GSPO-token 在多轮 agentic RL 里与 turn 级奖励配合的实际效果，目前只有 Qwen 内部数据。

## 参考来源

- GSPO 论文：[arXiv:2507.18071v2](https://arxiv.org/abs/2507.18071)，本文公式编号与实验数据均出自其 v2 版
- RLHF Book 第 6 章（重要性采样背景、GSPO/CISPO 小节）：[rlhfbook.com](https://rlhfbook.com/)，本地仓库 `book-zh/chapters/06-policy-gradients.md`
- hands-on-modern-rl 第 16.4 节（GRPO 改进家族对比；注意其 GSPO 公式遗漏长度归一化）：本地仓库 `docs/chapter18_grpo/grpo-family.md`
- 前篇：[熵是怎么塌掉的：DAPO 的 Clip-Higher 在修什么](/posts/dapo-clip-higher-entropy-collapse/)
