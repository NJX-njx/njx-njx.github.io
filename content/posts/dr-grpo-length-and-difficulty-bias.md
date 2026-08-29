---
title: "“答错就写长一点”：GRPO 的两个归一化项是如何扭曲优化方向的"
date: 2026-08-29T12:05:00+08:00
draft: false
math: true
tags: ["rl", "post-training", "grpo", "dr-grpo", "normalization"]
categories: ["AI"]
summary: "R1-Zero 的‘长度与奖励齐涨’被当成了推理涌现的标志性曲线，Sea AI Lab 的 Dr.GRPO 论文指出其中一半涨幅可能是优化器自己的偏好：1/|y| 让模型在答错时倾向于写长，除以 std 让极端难度的题拿走更多权重。两个归一化项、两种偏差的机制拆解，外加一个藏在 masked_mean 里的实现层偏差。"
---

> 本文有配套的[交互报告](/reports/dr-grpo-length-and-difficulty-bias/)，含可下钻的偏差算例、masked_mean 实现对照与整改选型矩阵。

## 一条被过度信任的曲线

R1-Zero 训练里最有名的一张图，是回答长度和训练奖励同步上涨。社区很快给它编好了故事：模型学会了想更久，长 CoT 通过 RL 不断形成。

Sea AI Lab 的 Dr.GRPO 论文（arXiv:2503.20783）提出了一个更简单的疑问：在奖励已停止上涨的平台期，长度为什么仍在继续增加？论文经逐一定位后得出结论：这不是模型学会了更多推理，而是**优化器会持续推动模型生成更长的回答**。动机来自 GRPO 目标函数中的两个归一化项：在正确和错误两种情况下，它们会分别以相反的方式调整模型的优化方向，其中对错误回答的影响指向同一个结果：**答错时，写得越长，越不容易受到惩罚。**

## 偏差一：1/|y|，按 token 平均会把惩罚变稀

GRPO 目标函数（论文公式 3）里有一个不显眼的分母：

$$\mathcal{J}_{GRPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\left(w_{i,t}\hat{A}_i,\ \mathrm{clip}(w_{i,t},1-\varepsilon,1+\varepsilon)\hat{A}_i\right)\right]$$

$1/|y_i|$ 的意思是：一条回答内部，先对所有 token 的梯度项取平均，再把 $G$ 条回答加起来。也就是说，**每个 token 实际拿到的梯度权重是 $\hat{A}_i/|y_i|$**。举个具体算例：同一道题采出两条回答，一条 100 token、一条 1000 token，两条回答 advantage 的绝对值都是 1（方向相反）：

- 答对（$\hat{A}=+1$）：短回答每个 token 被强化 $1/100 = 0.01$，长回答每个 token 被强化 $1/1000 = 0.001$。**正确回答里，越短越被器重**；
- 答错（$\hat{A}=-1$）：短回答每个 token 被惩罚 0.01，长回答每个 token 只被惩罚 0.001。**错误回答里，越长越不受惩罚**——相差十倍。

这不是数学错误造成的异常，而是目标函数写进每个更新步骤里的内容：答对必须简洁，答错不妨冗长。训练后期奖励进入平台期后，模型在上半部分的提升空间已经用尽，下半部分仍在推动模型生成更长的回答——尤其是错误回答继续变长。论文 Figure 4 里的 Fig.5 Plot 2 清楚地展示了这一阶段：GRPO 的长度曲线在奖励进入平台期后继续上升。

## 偏差二：除以 std，极端难度的题拿走更多权重

第二个归一化项在 advantage 里：

$$\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)}{\mathrm{std}(\{R_j\}_{j=1}^G)}$$

除以组内奖励标准差，本意是让 advantage 的量纲统一（这是 RL 的常规 trick）。但论文指出一个关键差异：经典 RL 的 advantage normalization 是在**整个 batch**上算 std，所有样本共享一个除数；GRPO 是**每道题单独算 std**，每道题有自己的除数——除数不同，权重就不同。

再算一笔具体算例：题 1 采 8 条全对（std≈0.35，$\hat{A}$≈+2.9 按 z-score 的尺度），题 2 采 8 条四对四错（std≈0.52，$\hat{A}$≈±1.9）。越极端的题目——全对的送分题、全错的死题——std 越小，除完之后的权重越大。模型会把最多的更新预算花在已经全对的题（容易做题）和完全不会的题（一次都答对不了）上，而“部分对部分错、正好处于学习价值区间”的题目权重最小。这就是论文说的**难度偏差（difficulty bias）**。

需要说明边界：std 归一化并不是只有坏处，它在小方差场景下放大信号的设计初衷是让模型也能从全对的题里学到东西。论文的批评在于**逐题归一化**造成的权重不等，不是归一化本身。

## 偏差三：藏在实现里，不在公式里

第三个发现对工程师可能更有冲击力。PPO 的目标公式（论文公式 2）是对 response 的 token **求和**，没有除以长度的步骤。但论文检查了主流开源实现，发现 trl、verl 等框架的 loss 里都有一句 `masked_mean`：用 response 的实际 token 数当除数算平均（`loss.sum(dim) / mask.sum(dim)`）。也就是说，**公式里没有长度偏差，实现却把它加上了**。

为什么会出现这种情况？论文的推测源于历史实践：预训练阶段所有数据打包成定长 context，`loss.mean(-1)` 按 context 长度平均既方便数值又稳定，这是常数除法，没有引入偏差；这套写法延伸到 RL 阶段后，除数悄悄从“定长 context”变成了“每条回答的实际长度”——从常数变变量，效果也随之改变。Dr.GRPO 在实现上的修正只改一行：把 `masked_mean` 的除数换成全局常数（如生成长度上限 MAX_TOKENS）。

公式作者和框架作者可能做的是同一套测试，却得到两套梯度。这类“公式与实现不一致”的问题，比论文层面的偏差更隐蔽：读十遍论文都发现不了，只有读代码才看见。

## Dr.GRPO 的修正与依据

修正方法很简单：把 $1/|y_i|$ 和 std 两项归一化都去掉，advantage 只减组内均值：

$$\tilde{A}_i = R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)$$

论文 Appendix A 给出三点理论依据：去掉两项后目标函数**恢复为 PPO 目标（公式 2）**，advantage 就是无 baseline 的 Monte Carlo return 加无偏 baseline；**减均值不引入偏差**——baseline 对梯度的期望贡献恒等于零，这是策略梯度里的标准结论；修正后的 advantage 和 RLOO（Leave-One-Out）只差一个常数因子 $G/(G-1)$，这个因子可以纳入学习率，不影响训练动力学。

实验在 Oat 框架、Qwen2.5-1.5B 上做对照。结果分三层：

- **训练动力学**：GRPO 和 Dr.GRPO 前期都出现 R1-Zero 式的奖励、长度一起上涨，但 GRPO 在奖励平台期后长度继续攀爬，Dr.GRPO 的长度稳定下来（Fig.5 Plot 1&2）；
- **行为层面**：Dr.GRPO 在 benchmark 上错误回答的长度显著缩短——这次改动同时缓解了过度思考（Fig.5 Plot 4）；
- **可复现性**：Llama-3.2-3B 数学预训练后同样出现 GRPO 的长度与奖励一起上涨，Dr.GRPO 同样能抑制长度的持续增长（Fig.7）。这套方法还能让 7B 的 Qwen2.5-Math-7B 使用 MATH level 3–5，在 8 张 A100 上训练 27 小时，AIME 2024 达到 43.3%——这是当年 7B 的 SOTA，整套方法极其简单。

## 一次教材勘误，以及和前两天的分工

本地教材 hands-on-modern-rl 的 16.4 节也讲到了这组偏差，但表述值得修正：它把“std 趋近于零时优势被放大”标成了“长度偏差”，并把长度膨胀归因于“除以 std 鼓励模型增加组内方差”。对照论文原文，这个说法有两层错位：std 造成的是**难度偏差**（不同题之间的权重不等），不是长度偏差；而长度偏差的直接来源是 $1/|y_i|$，与 std 无关，作用在回答内部每个 token 的权重上。两项归一化、两种偏差各自独立存在——勘误的价值在于，整改方案的选择依赖这个区分：去掉 std 解决不了长度问题。

本周已发的几篇文章，恰好分别从三个层面回应了这件事：

- **DAPO 的 token-mean**（§3.2）：把“样本内先平均再对 G 条求平均”改成“batch 内所有 token 统一按总 token 数平均”，也就是 $\frac{1}{\sum_i|y_i|}\sum_i\sum_t$。同时所有 token 都按 token 总数加权，从形式上消除长度偏差——但 DAPO 的 advantage 保留 z-score（含 std），**只解决了一半**；
- **GSPO**：在序列级使用一个比率，且不再在目标函数里逐项对 token 求和，因此并不存在“每个 token 除以 |y|”这一平均操作——它从结构上绕开了这一偏差；
- **Dr.GRPO**（本篇）：不改目标函数结构，只从归一化项中去掉两个变量除数。

## 我的三个判断

一、**RL 训练曲线必须先排除优化器引入的虚假信号**。当“长度持续上涨”同时被学术界引用为“RL 生效”的现象级证据，又被工程队引用为“要干预”的异常指标时，追根溯源就格外重要。论文同时观察到，使用 template 的 Qwen2.5-Math 模型在 RL 前的能力会先被 template 破坏，再被 RL 重建（§3.3）——这是另一类需要排除的因素：一部分“RL 增益”其实就是原本就存在的能力。分析 RL 曲线前，先逐项排除优化器引入的偏差和评估口径造成的假象。

二、**Dr.GRPO 和 dynamic sampling 解决的是不同问题，不构成替代关系**。去掉 std 归一化修正的是权重**失真**（不该被放大的样本得到了过大的权重）；DAPO 的 dynamic sampling（std=0 的题直接丢弃重采）处理的是**没有梯度信号**（全对/全错/全零时，减均值后 advantage 照样是 0，什么也没学到）。注意，去掉 std 后，零信号的题还是零信号——Dr.GRPO 没有为这类题带来任何梯度。两者一个保证样本内的相对权重正确，一个保证样本整体有可学习的信号，实际训练系统里需要同时配置。

三、**检查归一化，先查“除的是变量还是常数”**。这次的三项偏差有一个共同模式：表面上是常规操作的归一化，仔细检查后会发现，用作分母的并不是常数——$1/|y_i|$ 的除数是每条回答的长度，std 的除数是每道题的组内波动，`masked_mean` 的除数是当前 batch 的掩码和。查训练代码里的每个 mean/divide，问一句“这个除数会不会随样本变化”，是成本最低的审美审计。

## 未决问题

- Dr.GRPO 去掉 std 后，“极端难度题目”的权重变化对超长训练（数万步）的影响，论文的 1.5B/7B 实验时间窗口较短；
- token-mean（DAPO）与去掉变量除数（Dr.GRPO）在同一训练栈叠加时的有效学习率换算，缺少公开的统一口径；
- 对“预训练打包阶段的 `loss.mean(-1)` 习惯还有多少残留在 RL 框架”的系统性检查，目前只有论文对 trl/verl 的抽查。

## 参考来源

- Sea AI Lab / NUS：[Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)（本文公式、实验与偏差定义的出处；v2）
- 本地教材 hands-on-modern-rl：第 16.4 节（采用勘误后的表述）与附录 DAPO cheatsheet（`docs/en/appendix_code_cheatsheet/dapo.md`，DAPO 的 z-score advantage 与 token-mean 损失粒度对照）
- RLOO 参照：[Back to Basics: Revisiting REINFORCE Style Optimization](https://arxiv.org/abs/2402.14740)
- 前篇：[熵是怎么塌掉的：DAPO 的 Clip-Higher 在修什么](/posts/dapo-clip-higher-entropy-collapse/)、[比率用错了单位：GSPO 为什么把重要性采样提到序列级](/posts/gspo-sequence-level-importance-sampling/)、[同一个模型，两个概率：训练推理不一致怎么把 on-policy 变成 off-policy](/posts/tis-train-inference-mismatch/)
