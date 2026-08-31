---
title: "7 轮交互失败了该怪谁：多轮轨迹的奖励归因怎么进 GRPO/GSPO"
date: 2026-08-31T12:55:00+08:00
draft: false
math: true
tags: ["rl", "post-training", "agentic-rl", "grpo", "credit-assignment"]
categories: ["AI"]
summary: "单轮 RL 里 advantage 整条序列共享，从没人为难过；到了多轮 agentic RL，同一条轨迹里第 1 轮搜索做对了、第 2 轮 query 跑偏、最后答错——单一 trajectory reward 该怎么分摊，变成一个真实的设计决策。本文拆解形式化层（action mask、轨迹概率）、归因谱系（ORM/PRM/turn 折扣/三条中间路线）、以及 GSPO-token 为多轮留的那个口子。"
---

> 本文有配套的[交互报告](/reports/turn-level-reward-credit-assignment/)，含可切换的 ORM/PRM 归因对照、γ 滑杆反向折扣与 GSPO-token 公式拆解。

## 一个单轮训练里不存在的难题

看一条五轮交互的轨迹：第 1 轮搜索命中了关键证据，第 2 轮改写 query 时跑偏带回了无关内容，第 3 轮纠错找回一半，第 4 轮计算出错，第 5 轮给出错误答案。环境只回了一个数：失败，reward = 0。

单轮 RL 里没有这个问题——整个 response 共享一个 advantage，"全对全错"天经地义。多轮下这个粗暴逻辑立刻出问题：第 1 轮明明做对了，凭什么和第 4 轮一起被罚？而且工程上还另有一层：轨迹里混着两类 token，模型自己生成的（思考、工具调用参数、最终答案）和环境返回的（搜索结果、代码输出），后者根本不该进梯度。所以"turn 级 reward 怎么进 GRPO/GSPO"实际是三个问题的合体：**哪些 token 算动作、每个 turn 分多少 advantage、这个 advantage 乘到谁的头上**。本文按这个顺序拆。

## 先把对象说清楚：轨迹概率与 action mask

多轮轨迹的形式化（教材采用 AppWorld 论文的 POMDP 写法，arXiv:2504.11536）里，完整状态是 $z_t = [s_0, c, x_{1:t}]$：$s_0$ 是隐藏的初始环境状态（数据库快照、REPL 状态），$c$ 是任务上下文，$x_{1:t}$ 是到当前为止的全部 token 历史——包括模型生成的，也包括环境返回的。模型只看到文本部分，所以叫部分可观测。

轨迹概率的链式分解有一个关键细节：

$$\rho_\theta(x \mid s_0, c) = \mathbb{I}(s_0, x) \prod_{t \in a(x)} p_\theta(x_t \mid c, x_{1:t-1})$$

连乘**只在模型生成的 token 位置集合 $a(x)$ 上进行**；环境返回的 observation token 由环境动力学决定，不在连乘里。翻译成 loss 就是 action mask：与轨迹等长的 0/1 向量，模型 token 标 1、prompt 和工具返回标 0。SearchR1 的实现把这件事做得非常直白：`<think>` 和 `<search>` 的内容参与训练，检索器返回的 `<information>` 段整体 mask。如果让环境 token 也参与梯度，等于逼模型"学习预测搜索引擎返回什么网页"，策略梯度被污染。

这一点决定了今天的答案骨架：turn 级 reward 进入训练的方式，**不是给环境 token 加权，而是给每个 turn 的模型 token 构造各自的 advantage**。

## 三层信号：reward 怎么从轨迹走到权重

教材把信号流拆成三层，这个分层是理解所有方案谱系的地图：

1. **Trajectory reward** $R(\tau_i)$：环境给的原始标量，"这一局成没成"；
2. **Step advantage** $A_{i,t}$：把最终结果拆成每轮的得分，回答"第 $t$ 轮该不该被奖励"——这是信用分配（credit assignment）的主战场；
3. **Token gradient**：$A_{i,t}$ 乘到该轮 action token 的 log-prob 上，$\Delta\theta = A_{i,t} \sum_{k \in a_{i,t}} \nabla \log \pi_\theta(y_k \mid \cdot)$。

所谓"turn 级 reward 进 GRPO"，落点在第 2 层：GRPO 的组内比较天然工作在 trajectory 层（同一道题的多条轨迹比优劣），要落到 turn 级，就得自定义 $A_{i,t}$ 再喂进去。

## 朴素方案为什么会失效：一个对照实验

教材的 Mini Agent Loop 实验给了最有说服力的一组数字。任务：查地球半径 → 算赤道周长 → 验证 → 作答。构造一条"坏"轨迹：**只有第 2 步错了**（把 π 取成 3），第 1、3 步都正确，最终答案错。

- **ORM（只看最终结果）**：失败时所有步骤的 credit 全是 0——因为 $0 \times \gamma = 0$，零信号反向传播后每一步都"无信号"；如果改成"失败给 -1"，正确的第 1 步搜索会拿到 -0.857 的惩罚。错怪好人，或者干脆全员沉默。
- **PRM（每步独立评分）**：正确步骤正分、错误步骤负分，区分度是 ORM 的 19 倍，同等步数下任务成功率高约 30 个百分点。代价是每步都要评估——真实场景意味着标注成本或再训一个打分模型。

PRM 太贵，ORM 太糙，2025–2026 年的中间路线按"信号从哪来"分成三类，本质都是在两者之间找折中。

## 归因谱系：ORM 和 PRM 之间的三条路

**路线一：同状态下比较动作（state-anchored）。** GiGPO 的思路是组内多条 rollout 若在某个状态 $s$ 汇合，就比较"同一局面下不同动作各自的后续回报"，组内相对得分就是 step advantage。它的修正版 HGPO 指出同页面不等于同上下文（前面漏没漏约束，同一个"加入购物车"含义完全不同），用 $k$-step 历史把比较组分层，各层加权融合。这条路不需要任何额外标注，但依赖状态可碰撞——网页类任务天然友好，开放对话类几乎碰不上。

**路线二：把最终 reward 摊回每步（progress redistribution）。** SPA-RL 训一个 progress estimator，要求每步贡献之和还原最终 reward（$\hat R = \sum_t \hat c_t \approx R$），成功轨迹贡献和近 1、失败近 0，模型被迫把正贡献分给真正推任务的步骤。IGPO 更便宜：直接拿"模型对 ground-truth 答案的 log 概率增量"当 turn 级 reward——第 $t$ 轮交互后模型对正确答案更确信了（log-prob 从 -4.0 升到 -1.5），记 $r^{IG} = +2.5$；被带偏了就记负。它给出的是现成的 turn 级标量，可以直接进折扣累积，不需要 Monte Carlo 估值；代价是依赖高质量标准答案，多答案问题会被误伤。

**路线三：换优化粒度（step-aligned 范式）。** StepPO 和 Turn-PPO 的观点最激进：现有算法继承了 RLHF 的 token-centric 范式，但 agent 的决策天然是 turn 粒度的，token 级优化和 turn 级决策之间存在 granularity mismatch。Turn-PPO 用 PPO 取代 GRPO 做多轮，理由是 trajectory 级组内 advantage 方差太大，改用 turn 级 advantage 估计——每个 turn 有自己的 advantage，在 turn 内部 token 间共享。这条路线和前面所有方法正交：别人改 $A_{i,t}$ 从哪来，它改优化的基本单元。

无论走哪条路，多轮 RL 还多一个纯时间维度的问题：**越早犯的错责任越大**（第 1 步走错，后面每步都在错误上展开）。教材和工程实现都用反向折扣处理：从最终结果往回推，越早的步骤折扣越大——7 轮交互只有最后一轮有 reward 1.0、$\gamma = 0.9$ 时，各轮分到的回报是 $[0.531, 0.590, 0.656, 0.729, 0.810, 0.900, 1.000]$。

## GRPO/GSPO 家族在这张地图上的位置

上周 GSPO 那篇留下的尾巴今天兑现。GSPO 的"优化单位应匹配奖励单位"原则在多轮场景会遇到真实张力：GSPO 的目标函数把整条 response 当一个 action，适合 outcome 级奖励；多轮 RL 想要 turn 级归因，等于要求比序列更细的粒度。GSPO 论文 §4.3 为此设计的 GSPO-token 变体，公式很能说明问题：

$$s_{i,t}(\theta) = \mathrm{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\mathrm{sg}[\pi_\theta(y_{i,t}|x,y_{i,<t})]}$$

$\mathrm{sg}[\cdot]$ 是 stop-gradient（只取数值不回传梯度）。这个构造的妙处：数值上 $s_{i,t}$ 恒等于序列级比率 $s_i$，所以目标、裁剪条件、理论梯度与 GSPO 完全等价；但梯度表达式里出现了一个"逐 token 可定制的 advantage 位"。把同一 turn 的所有 token 设成同一个 $A_{i,t}$，就精确得到 turn 级归因——这是算法层为多轮留的正式口子，而不是工程 hack。

GRPO 家族走对称的另一条路：组内比较多条轨迹的机制不动，把 $A_{i,t}$ 的构造外包出去——你给它 trajectory 级比较（默认）还是 turn 级分数（ORM + 折扣 / PRM / IGPO / SALT），它照乘不误。所以**"turn 级 reward 进 GRPO"在实现上不是改算法，是改 advantage 构造器 + 确认 action mask 干净**。两条路线在论文层面的组合（GSPO-token × turn 级 $A_{i,t}$）目前没有公开消融，是个实在的空白。

## 我的三个判断

一、**轮数决定档位，算法决定不了。** 教材的选型建议按轮数分档：3–5 轮纯 ORM/GRPO 就够（episode 短，稀疏性不致命）；5–15 轮上里程碑奖励或 SALT/GiGPO 这类无标注方法；15 轮以上必须 PRM 或 progress reward 加树搜索。这个分档比任何单项技术都重要——reward 密度撑不起学习时，换更强的 RL 算法收益为零。今天那条 5 轮轨迹，其实正处在第一档和第二档的边界上。

二、**归因质量和熵的关系比想象中直接。** ORM 下失败轨迹全员受罚，模型学到的最省事的对策是降低行为多样性（少做动作、早点给答案）——熵坍缩在多轮下不是副产品，是错误归因的直接后果。反过来，PRM 或 IGPO 这类细粒度信号允许模型"保住对的步骤、只改错的"，探索才得以保留。观察多轮训练时 agent 的 turn 数分布和工具调用多样性，比看 reward 曲线更早暴露归因问题。

三、**"能跑"和"归因对"之间隔着四个 reward 类型。** 教材把 agentic reward 分四类：Outcome（最终对不对）、Format（动作能否被解析执行）、Cost（轮数、API 花费）、Process（步骤是否推进任务）。入门时通常只实现前两类保证训练跑通，Process 正是归因层要 dense 化的对象。但很多工程把 Format 惩罚也混进 outcome 信号里一起归因——格式错误的 turn 和推理错误的 turn 在轨迹里承担同样的 blame，这类混淆在公开报告里几乎无人拆开分析。

## 未决问题

- GSPO-token 与 turn 级 $A_{i,t}$ 的组合在多轮 agentic 任务上的效果，没有公开消融——Qwen 内部是否验证过无从得知；
- turn-level discounting 的 $\gamma$ 在多轮 LLM 训练里没有系统性扫描（教材示例取 0.9，来源是传统 RL 惯例而非消融）；
- 四类 reward（outcome/format/cost/process）在归因层的配比与隔离，缺公开的对照实验——大多数报告只报最终指标，不报归因结构。

## 参考来源

- 本地教材 hands-on-modern-rl：22.2 多轮 RL 形式化（AppWorld POMDP、action mask、reward 四类）与 22.3 轨迹信用分配（三层信号、ORM/PRM 实验、三条中间路线、turn-level discounting、选型分档），本地仓库 `docs/chapter22_agentic/formulation.md` 与 `docs/chapter22_agentic/credit-assignment.md`
- GSPO 论文：[arXiv:2507.18071](https://arxiv.org/abs/2507.18071) §4.3（GSPO-token 公式与等价性论证）
- AppWorld：[Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2404.14394)；ORM/PRM 原始框架：[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)（Lightman et al., 2023）
- 中间路线：SALT [2510.20022]、GiGPO/HGPO [2604.18401]、SPA-RL [2505.20732]、IGPO [2504.05678]、Turn-PPO [2512.17008]、StepPO [2604.18401]（arXiv id 均转引自教材）
- 前篇：[比率用错了单位：GSPO 为什么把重要性采样提到序列级](/posts/gspo-sequence-level-importance-sampling/)、[熵是怎么塌掉的：DAPO 的 Clip-Higher 在修什么](/posts/dapo-clip-higher-entropy-collapse/)
