---
title: "一票还是十票：token-mean 与 sample-mean 在争什么"
date: 2026-08-30T12:05:00+08:00
draft: false
math: true
tags: ["rl", "post-training", "grpo", "dapo", "dr-grpo"]
categories: ["AI"]
summary: "同一个 batch 里，500 token 的回答和 5000 token 的回答，谁对梯度的贡献大？GRPO 说一样大，DAPO 说长的大十倍，两家都能给出理由。这是损失聚合方式之争——sample-mean 每条回答一票，token-mean 每个 token 一票。DAPO 论文 §3.3 说长 CoT 场景必须换 token 级，但换完会把长度倾斜从 response 级赶到 question 级。本文拆解三个公式、两家的论证和一个少有人讲的 tradeoff。"
---

## 先算一票

同一个训练 batch 里有两条回答：一条 500 token，一条 5000 token。它们对这一步梯度的贡献各是多少？

- **GRPO（sample-mean）**：一样大。先在每条回答内部对 token 求平均，再对 $G$ 条回答求平均——每条回答无论长短，都是平等的一票；
- **DAPO（token-mean）**：长的那条大十倍。batch 内所有 token 拉平，统一除以总 token 数——每个 token 一票，5000 token 的回答拿 5000 票；
- **Dr.GRPO**：形式上也是每条回答一票，但除数从"每条回答的实际长度"换成全局常数，票的含义变了（这是昨天那篇的主题）。

三种答案都有论文背书，而且三家处理的是**同一个式子里的同一个位置**——损失的聚合除数。这个旋钮和昨天讲的 advantage 归一化（std 那半）是正交的：昨天的问题是"advantage 怎么算"，今天的问题是"算完之后，token 们的损失按什么规则加在一起"。

## 三个公式并排

GRPO 的目标函数（DeepSeekMath 公式 3 的省略 KL 版）：

$$\mathcal{J}_{GRPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}\right], \quad \ell_{i,t}=\min\big(w_{i,t}\hat{A}_i,\ \mathrm{clip}(w_{i,t},1-\varepsilon,1+\varepsilon)\hat{A}_i\big)$$

其中 $o_i$ 是第 $i$ 条回答，$|o_i|$ 是它的长度，$w_{i,t}$ 是新旧策略对该 token 的概率比，$\hat{A}_i$ 是整条回答共享的 advantage。注意求和的结构：$\frac{1}{|o_i|}\sum_t$ 先把每条回答**自己**平均成一个数，$\frac{1}{G}\sum_i$ 再对 $G$ 个数求平均。两条回答各出一票，长短无关。

DAPO 的改动只有一个地方（论文公式 12，红字标注的部分）：

$$\mathcal{J}_{DAPO} = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\ell_{i,t}\right]$$

内层的 $\frac{1}{|o_i|}$ 没了，两层求和拉平，除数换成整个 group 的总 token 数。每个 token 权重完全相等，都是 $1/\sum_i|o_i|$；一条回答的总权重正比于它的长度。

第三个是 Dr.GRPO：聚合结构与 DAPO 相同（token 拉平），但除数用全局常数（如长度上限），且 advantage 里连 std 也去掉了——昨天展开过，今天只看除数这一项。

## DAPO 为什么非要换：两个罚不到

DAPO 论文 §3.3 给的论证很具体，针对长 CoT 场景。sample-mean 下，长样本"对整体损失的贡献不成比例地低"（原文：disproportionately lower contribution），引出两个恶果：

一、**高质量长样本里的推理模式学不到**。一条 5000 token 的正确推理，每个 token 分到的梯度权重是 $1/(G\cdot 5000)$，只有 500 token 短样本的十分之一。长样本承载的恰恰是复杂推理，按 sample-mean，越重要的内容单位权重越低。

二、**超长样本里的坏模式罚不掉**。论文报告了一个观察：过长的样本里常出现 gibberish（乱码）和重复词这类低质模式。这些坏 token 的惩罚同样被 $1/|o_i|$ 摊薄——样本越长，每个坏 token 被罚得越轻。结果是"熵和回答长度的不健康上涨"（论文图 4a/4b 的实测曲线）。

换成 token-mean 后，单 token 不论身处长样本还是短样本，权重相同。DAPO 对效果直说不讳：长序列由此"在整体梯度更新中比短序列有更大影响"（原文：longer sequences can have more influence）。这不是副作用，是设计目标——长 CoT 时代，梯度本来就该主要花在长回答上。

## 一个少有人讲的 tradeoff

但"长样本拿更多票"这句话值得停下来多看一眼，它有两层含义，DAPO 只讲了第一层。

第一层是 response 级：同一条回答内部，token 不再被长度摊薄。这是 Dr.GRPO 也赞同的方向——Dr.GRPO 批的"response-level length bias"（答对时短者被器重、答错时长者更安全）和 DAPO 批的"坏 token 罚不到"，是同一个 $1/|o_i|$ 的两副面孔，两家开的药殊途同归：per-token 等权。

第二层是 question 级：整个 batch 里，**回答长的题拿走更多总权重**。一道模型的回答平均 6000 token 的题，和一道平均 600 token 的题，在 token-mean 下的 batch 权重差 10 倍；sample-mean 下两者相等。换句话说，token-mean 把长度倾斜从 response 级赶到了 question 级——长题主导梯度。这在长 CoT 场景或许正是想要的（难题、推理重的题往往回答也长），但它是个真实的偏向：题目权重不再均匀，而是正比于该题当前的长度高估。这层 tradeoff 我未在任何论文里见到量化（此点为推断，欢迎打脸）。

还有第二个二阶效应：DAPO 的除数 $\sum_i|o_i|$ 是**随 batch 变化的随机变量**，Dr.GRPO 的常数除数不是。除数随机的直接后果是梯度尺度随 batch 的长度构成波动；间接后果更有意思——训练后期回答普遍变长，除数变大，每个 token 的有效学习率自动衰减。也就是说 token-mean 自带一个与长度膨胀联动的隐式学习率衰减，而 sample-mean 和常数除数都没有这个性质。这是机制层面的推断，没有论文测过，但它是选聚合方式时真实存在的变量：你选的不只是权重分配，还是梯度尺度的动力学。

## 两家都说"回到 PPO"，回的不是同一个 PPO

一个容易被忽略的史实：PPO 的原始目标函数本来就是 per-token 聚合的（对整个 batch 的 token 求和或求均值，advantage 由 GAE 逐 token 给出）。**GRPO 的 sample-mean 才是那个偏离者**——因为 GRPO 的 advantage 是"一条 response 一个数"，把每条 response 先平均成一个数再平均，语义上最整齐。

然后 DAPO 和 Dr.GRPO 都宣称自己"恢复了 PPO 目标"：DAPO 恢复的是 **PPO 的聚合方式**（token 拉平，但 advantage 保留 z-score，std 还在）；Dr.GRPO 恢复的是 **PPO 的目标函数形式**（常数除数，advantage 退化为 $r-\text{mean}$，std 去掉）。两个旋钮——聚合粒度、advantage 归一化——是独立的，可以自由组合：verl 等实现里这两项确实是分开配置的（昨天那篇的实现层讨论：trl/verl 的 `masked_mean` 默认把公式没有的 per-response 平均又加回去了，审计每个除数"是变量还是常数"是成本最低的检查）。这也意味着"GRPO vs DAPO vs Dr.GRPO"不是三个算法排排坐，而是一个 2×2 以上的配置空间，目前已知好的点都还在边界上试出来。

## 什么时候哪票该大

把场景拆开，选择并不玄：

- **短 CoT、长度方差小**：两种聚合近似等价（长度差不多，$1/|o_i|$ 和 $1/\sum|o_i|$ 差一个近似常数），不用纠结；
- **长 CoT、坏模式藏在长样本里**（重复、乱码、无效啰嗦）：token-mean 关键，否则坏 token 永远罚不到位——这是 DAPO 的核心场景；
- **想要题目间的均匀权重**（比如题目难度方差很大、担心长题霸榜）：sample-mean 或常数除数更可控；
- **想要梯度尺度稳定、学习率行为可预测**：常数除数（Dr.GRPO 路线）最干净，token-mean 的随机除数会把长度动力学混进学习率。

我的判断：聚合方式之争的本质是"梯度预算按什么单位分"。sample-mean 按条分，隐含假设是每条回答信息量相等——短答案时代成立；token-mean 按 token 分，隐含假设是每个 token 信息量相等——长 CoT 时代更接近真相，但代价是把题间权重交给长度决定。真正缺的论文是把这个预算分配**显式化**的：权重既不按条也不按 token，而是按估计的信息量或不确定性分。在那之前，DAPO 的改法是长 CoT 下最务实的默认。

## 未决问题

- token-mean 的 question 级长度倾斜（题间权重 ∝ 长度）对训练终局的影响，没有公开量化；
- 随机除数带来的隐式学习率衰减，与显式 LR schedule 叠加时的相互作用，没有人测过；
- 聚合粒度与 advantage 归一化两个旋钮的完整 2×2 消融（sample/token × 有 std/无 std），公开文献里只有对角线上的点。

## 参考来源

- DAPO 论文：[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)，token-level loss 见 §3.3（公式 12），熵与长度曲线见图 4
- Dr.GRPO 论文：[arXiv:2503.20783](https://arxiv.org/abs/2503.20783)，response-level length bias 与 question-level difficulty bias 原文表述
- DeepSeekMath / GRPO：[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)，sample-mean 目标函数（公式 3）
- 前篇（昨日，advantage 归一化那一半）：[Dr.GRPO 两个归一化项的两种偏差](/posts/dr-grpo-length-and-difficulty-bias/)（其中 token-mean 小节号误写为 §3.2，正确为 §3.3，已顺带更正）
- question 级长度倾斜、随机除数的隐式 LR 衰减为本文推断，文中已逐处标注
