---
title: "熵是怎么塌掉的：DAPO 的 Clip-Higher 在修什么"
date: 2026-08-26T14:30:00+08:00
draft: false
math: true
tags: ["rl", "post-training", "grpo", "dapo", "rlvr"]
categories: ["AI"]
summary: "naive GRPO 复现 R1 时熵快速坍缩、组内样本趋同，DAPO 的 Clip-Higher 只把裁剪上界从 0.2 抬到 0.28。这一个数背后，是 PPO-Clip 在概率轴上的结构性不对称。本文拆解失效链条、机制、证据与边界。"
---

> 本文有配套的[交互报告](/reports/dapo-clip-higher-entropy-collapse/)，含可下钻的公式拆解与消融数据图表。

## 一个训练现场

2025 年初，很多团队都在做同一件事：用 GRPO 复现 DeepSeek-R1 的训练。DAPO 论文（arXiv:2503.14476）报告了他们自己的起点——用 Qwen2.5-32B base model 跑 naive GRPO，AIME 2024 只拿到 30 分，而 DeepSeek 的 R1-Zero-Qwen-32B 是 47 分。

差距不是慢慢拉开的，而是一种很具体的死法：训练中 actor 模型的熵快速下降，同一道题采出来的一组回答越来越像，最后几乎一模一样。论文把这叫 **entropy collapse（熵坍缩）**。采样失去多样性，组内奖励趋同，GRPO 的 advantage 全部趋近于零，梯度信号自我灭绝。训练还在跑，但已经学不到东西了。

DAPO 给出的第一味药叫 **Clip-Higher**：把 PPO-Clip 的裁剪上界 ε 从默认的 0.2 抬到 0.28，下界保持 0.2 不动。改动只有一个数。为什么这一个数能拦住坍缩？答案藏在 PPO-Clip 一个很少被摊开讲的结构性质里。

## 先把"熵"说清楚

整篇文章都绕着一个"熵"字转，先把它定义死。这里的熵就是信息论里那个熵，作用对象是**模型每生成一个 token 时输出的、覆盖整个词表的概率分布**：

$$H = -\sum_{x \in \text{词表}} p(x) \log p(x)$$

分布越平（很多 token 概率差不多），熵越高，模型"拿不定主意"；分布越尖（一个 token 占 0.99，其余瓜分 0.01），熵越低，模型"很确定"。举个数值例子：假设词表只有 3 个 token，分布 [0.9, 0.05, 0.05] 的熵约 0.39 nat，均匀分布 [1/3, 1/3, 1/3] 的熵约 1.10 nat。从前者采样 10 次有 9 次拿到同一个 token，从后者采样结果均匀散开——**熵就是"采样多样性"的数学度量**。DAPO 论文监控的"actor 模型的熵"，就是 rollout 生成过程中这些逐步分布熵的平均值。

为什么它是 RL 训练的命门：GRPO 的全部训练信号来自组内奖励方差，而方差来自采样多样性。熵高，同一道题采 16 条回答各不相同，有对有错，advantage 非零；熵坍缩，16 条趋同，奖励全同，advantage 全零。

两个边界，别理解过头：一，**熵不是越高越好**——熵过高对应 gibberish、重复词等失控模式，论文只说熵要维持在"合适区间"，没给区间怎么定（本文末尾的未决问题之一）。二，**别把熵和采样温度混为一谈**：熵是分布的内在形状，温度是你怎么从分布里取；rollout 时用 temperature 1.0 这类参数才把熵兑现成多样性。Clip-Higher 影响的是前者——防止训练本身把分布越压越尖。

## 先把 PPO-Clip 的账算清楚

PPO/GRPO 的目标函数，逐 token 看是：

$$J(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\,\hat{A}_t\right)\right]$$

符号逐个说清：

- $r_t(\theta) = \pi_\theta(o_t \mid q, o_{<t}) / \pi_{\theta_{old}}(o_t \mid q, o_{<t})$，**重要性采样比率**：当前策略对第 $t$ 个 token 的概率，除以采样时旧策略的概率。$r=1$ 表示策略没变。
- $\hat{A}_t$，**advantage**：这个 token 比平均水平好多少。GRPO 里它是组内归一化算出来的：$\hat{A}_{i,t} = (R_i - \text{mean}(\{R\})) / \text{std}(\{R\})$。
- $\varepsilon$，**裁剪范围**：比率允许偏离 1 的幅度，绝大多数实现的默认值是 0.2——这个默认值来自 2017 年 PPO 原论文的 Atari 和机器人控制实验。

advantage 从哪来，PPO 和 GRPO 是两条路，读材料时别混：

- **PPO**：靠一个学出来的价值函数（critic），用 GAE 逐 token 估计：$\hat{A}_t = \sum_{l}(\gamma\lambda)^l \delta_{t+l}$，其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。critic 准，advantage 方差就低；critic 歪，整个训练跟着歪。
- **GRPO**：不要 critic，对同一道题采 G 条回答，拿组内奖励做归一化。本质是用组内均值当 baseline——这是 REINFORCE 的方差缩减技巧，只不过基线不用学，直接拿同组样本算。

GRPO 这条路有三个直接后果，后面都会用到。一，**整组同奖则全零**：G 条全对或全错时，mean 等于每条自己的奖励，advantage 全是 0，这道题不贡献任何梯度。二，**除 std 有隐含加权**：std 小的组（比如 16 条只对 1 条）会被归一化放大权重，Dr.GRPO 批评的难度偏置就来自这一项。三，**advantage 按整条回答算，同一条里每个 token 共用**：答错的回答里前 80% 推理步骤可能是对的，但它们和最后算错那一步吃同一个负 advantage——GRPO 本身没有 token 级信用分配。

$\min$ 的行为取决于 $\hat{A}$ 的符号，这是理解一切的钥匙：

- **$\hat{A} > 0$（好 token，要强化）**：当 $r > 1+\varepsilon$，clipped 支被选中，目标函数变成常数，**梯度为零**。也就是说，每一步更新最多把一个 token 的概率抬到采样时的 $1+\varepsilon$ 倍，再往上就不给梯度了。
- **$\hat{A} < 0$（坏 token，要压制）**：对称地，每步最多把概率压到采样时的 $1-\varepsilon$ 倍。

这里有个容易误读的点，值得展开：**clip 对每种符号的 advantage 都是单边的**。看 $\hat{A} > 0$ 且 $r < 1-\varepsilon$ 的情形：$\min$ 选中的仍然是 $r \cdot \hat{A}$（因为它更小）。目标函数的值虽然小，但**梯度是活的**——$dJ/d\pi > 0$，优化器照样以全额斜率 $\hat{A}$ 把这个好 token 的概率往回拉。值小不等于抑制：**值是结果，梯度才是动作**。$\min$ 的作用是**撤掉激励**，从来不是**施加反向激励**。所以完整结构是：好 token 只封顶上涨激励（右侧平台），坏 token 只封顶压制激励（左侧平台），反方向的纠偏永远满梯度放行。

交互报告里那张核心机制图就是按这个逻辑画的，读法：横轴是比率 $r$，纵轴是这个 token 对目标函数的贡献 $J(r)$，$\hat{A}$ 归一化为 1。$r < 1+\varepsilon$ 时 $J(r)=r$ 是一条上升的直线（梯度活着），$r > 1+\varepsilon$ 后是水平平台（梯度到此为止）。

RLHF Book 第 6 章还特别提醒：clip 不是一直在工作。只有当比率真的出界、且 $\min$ 恰好选中 clipped 支时，梯度才被掐断。裁剪分数（clip fraction）就是监控这件事的指标。记住这一点，后面会用上。

## 对称裁剪，不对称的效果

现在做一道算术题。设 $\varepsilon = 0.2$，$\hat{A} > 0$，看两个 token：

- 一个 $\pi_{\theta_{old}} = 0.9$ 的 token：概率上界是 $0.9 \times 1.2 = 1.08$。概率不可能超过 1，所以**它实际上不受限**——一轮就能从 0.9 推到 0.99 以上。
- 一个 $\pi_{\theta_{old}} = 0.01$ 的 token：上界是 $0.01 \times 1.2 = 0.012$。**绝对增量 0.002**，几乎原地不动。

先把"一轮"说准确，它指一个"采样→更新"循环：采完样 $\pi_{\theta_{old}}$ 冻结成快照，之后在这批数据上跑多次梯度更新（DAPO 是 16 次，论文 §4.1），这期间 $\pi_\theta$ 逐渐偏离快照，$r$ 从 1 开始漂移；一旦 $r$ 越过 $1+\varepsilon$，这个 token 的梯度就断了，本轮剩下的更新不再推它。所以"上界 0.012"的准确含义是**激励消失点**，不是硬钳制——步长迈大了概率可能一步冲过去，只是冲过去之后激励归零，没人继续推，也没人拉回来。

而预算是**每轮刷新**的：下一轮重新采样，$\pi_{\theta_{old}}$ 更新，$r$ 重置为 1，又拿到一份 $\times 1.2$ 预算。所以低概率 token 的成长路线是乘法的：$0.01 \to 0.012 \to 0.0144 \to \cdots$，比率上是指数增长，前半段绝对值却爬得非常慢；高概率 token 一两轮就顶到 1.0 饱和。压制方向对称但更狠：$\hat{A} < 0$ 时每轮最多砍到 $\times 0.8$，五轮后剩 0.33，低概率 token 很快被打到接近 0——**下行的乘法衰减比上行的乘法增长快得多**。

问题就出在这里：**裁剪以"比率"为刻度，但学习效果发生在"概率"轴上。**同样容忍 20% 的比率变化，高概率 token 拿到的是绝对概率上接近 0.2 的空间，低概率 token 拿到的是 0.002。两种 token 的可动范围差了快一百倍。

而 RL 里的"探索"恰恰由低概率 token 承载——那些当前策略不太倾向选、但可能通向新解法的 token。DAPO 的实测印证了这一点：训练中真正被上界裁住的 token，平均概率小于 0.2（论文图 3a）。换句话说，**被裁剪机制反复按住头的，正是探索 token**；exploitation token（模型已经很确定的 token）几乎从不被约束。

长 CoT 场景把这个偏差放大了：一条推理链几千个 token，关键转折点上的探索 token 每轮只能挪 0.2%，还没等它长起来，分布其他部分已经进一步尖锐化了。

## 坍缩的闭环

把上面这个微观偏差放进 GRPO 的训练动力学里，就得到一条完整的失效链：

1. 探索 token 概率长不起来，高概率 token 越长越高，分布变尖，**熵下降**；
2. 分布变尖意味着同一道题采一组（比如 16 条）回答**越来越像**；
3. GRPO 的 advantage 是组内归一化算的，组内奖励趋同时 $\text{std} \to 0$，所有 advantage 趋近于零；
4. **梯度信号消失**，模型失去改进方向，只能继续强化已经确定的模式——回到第 1 步。

RLVR 场景里这个循环尤其致命，因为它的全部训练信号就是组内奖励的方差。奖励是规则的 0/1（答对 +1，答错 -1），没有奖励模型提供连续信号。多样性一旦枯竭，信号源就断了。熵坍缩不是"一个监控指标不好看"，而是训练在烧掉自己的燃料。

这也解释了 DAPO 四个技巧的顺序感：Clip-Higher 防坍缩，Dynamic Sampling（过滤掉全对或全错的 prompt）是在坍缩已经造成零梯度组时止血。一个治本，一个治标，论文把 Clip-Higher 放在第一节。

## Clip-Higher：只抬上界，是有讲究的

Clip-Higher 的做法是把对称裁剪拆成两个超参数：

$$\text{clip}\left(r_t(\theta),\ 1-\varepsilon_{low},\ 1+\varepsilon_{high}\right),\quad \varepsilon_{low}=0.2,\ \varepsilon_{high}=0.28$$

上界从 1.2 抬到 1.28，低概率探索 token 的单轮上涨空间从 +20% 变成 +28%，配合多轮预算的乘法积累，效果可观。

值得问的是：为什么不动下界？论文说得很直白——**$\varepsilon_{low}$ 如果放宽（比如允许压到 0.6 倍），负 advantage 的梯度会以同样的比率逻辑把低概率 token 直接压到接近 0**，采样空间里这些 token 等于被删掉了，坍缩只会来得更快。前面算过乘法衰减的账：下行方向本来就比上行快，压制方向的裁剪反而要收紧，不能放松。

所以 Clip-Higher 不是"把探索空间的门开大一点"这种模糊操作，它是一次精确的不对称手术：**上涨方向松绑，下跌方向保持警惕**。

效果上，论文图 2 显示应用 Clip-Higher 后熵曲线明显回升、AIME 准确率同步上升。消融链（Qwen2.5-32B，AIME 2024 avg@32）：

| 配置 | AIME 分数 |
|---|---|
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling（完整 DAPO） | 50 |

要诚实地说：Clip-Higher 单项只贡献了 +2 分，不是链上最大的（Overlong Filtering +6，最后加 Dynamic Sampling +8）。但它的角色不在加分——它修的是"训练能不能继续 scale"的失速点。熵塌了，后面三个技巧全都无从发挥。

## 两个少有人摊开讲的前提

**前提一：clip 只在 off-policy 时才激活。** RLHF Book 第 6 章指出，语言模型的 RL 实践里很多算法每个 batch 只跑一步梯度更新——此时比率恒等于 1，clip 永远不触发，真正起正则作用的是 KL 惩罚。而 DAPO 每个 rollout 跑 16 个梯度步（mini-batch 512，论文 §4.1），策略在 batch 内多次更新，比率才会偏离 1，clip 才真的开始工作。**所以 Clip-Higher 这个旋钮的价值，和训练的 off-policy 程度是耦合的**：如果你是严格的单步 on-policy 训练，这个旋钮根本不存在。反过来，异步 RL（rollout 和训练重叠、样本天然 off-policy）才是它发挥作用的土壤。

**前提二：DAPO 同时去掉了 KL 项。** 先把 KL 惩罚本身说清楚。KL 散度衡量当前策略 $\pi_\theta$ 和参考策略 $\pi_{ref}$（通常是冻结的 SFT/base 模型）差多远：

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{x \sim \pi_\theta}\left[\log \pi_\theta(x) - \log \pi_{ref}(x)\right]$$

经典 RLHF 把它作为惩罚项放进目标函数（$-\beta D_{KL}}$），动机有二：一是**防 reward hacking**——奖励模型是有漏洞的代理，策略跑得够偏就能找到 RM 没见过、瞎打高分的输出，KL 把搜索拴在 RM 见过的分布范围内；二是保住初始模型的语言能力，防止格式崩坏。工程代价是显存里要常驻一个冻结参考模型，每个 token 算两边的前向。

DAPO 把它整个去掉了（$\beta=0$，参考模型不用加载），论文 §2.3 的理由：KL 惩罚的初衷是"对齐但不要偏离初始模型太远"，而长 CoT 训练的目的**恰恰是大幅偏离初始模型**——base 模型本来不会写长思维链、不会自我反思，这些行为全靠 RL 重塑分布，拴着锚等于不让船开。还有一个没明说的配套条件：RLVR 的奖励是规则判定，**没有 RM 可以 hack**，KL 防 hacking 的那一半动机直接消失。

把两个前提合起来看，能看出 DAPO 的一次隐含换挡：经典配置里防漂移靠 KL——**绝对锚**，不许离初始模型太远；DAPO 去掉 KL 后只剩 clip——**相对锚**，每步不许离上一步太远，但允许持续走远。信任区域的角色从一个锚换到了另一个锚。同时注意去 KL 和抬上界方向一致——都在给探索松绑：KL 惩罚是逐 token 把分布往参考模型拉，去掉它，策略才敢往新区域走。论文没有明说两者是协同设计，这是我的推断；但从"防坍缩"的目标看，它们确实是同一战壕的。当然，锚拆了不是没有代价：训练失控时没有任何东西把模型拉回来，所以 DAPO 才那么依赖监控仪表盘（熵、长度、平均概率），也才需要 clip、动态采样这些替代性稳定器。

## 实现层对照

纸面机制之外，我核对了三个实现，细节基本对齐：

- **rlhf-book** 的 `code/policy_gradients/loss.py` 里 `DAPOLoss`：解耦 clip（`clip_eps_lo=0.2 / clip_eps_hi=0.28`，见 `configs/dapo.yaml`），token-level 损失归一化（损失求和后除以全 batch 的有效 token 总数，而不是 GRPO 原来的逐序列平均再平均），不加 KL。教学和论文一致。
- **verl 生产实现**（`verl/trainer/ppo/core_algos.py`）：`clip_ratio_low` / `clip_ratio_high` 是独立配置项，未设置时回落到对称的 `clip_ratio`——也就是说 DAPO 的非对称裁剪已经是一等公民，但默认行为仍是经典 PPO。文件里还有 dual-clip 的 `clip_ratio_c=3.0`，防的是负 advantage 样本的比率爆炸，和 Clip-Higher 解决的是不同方向的问题。
- **行业佐证**：MiniMax 的 CISPO 同样允许非对称裁剪界；VAPO 直接沿用了 clip-higher。这个 insight 已经不是 DAPO 独有，而是长 CoT RL 的工程共识。

也提一个甄别案例：中文课程 hands-on-modern-rl 在讲 Clip-Higher 时，说"0.8 的下限意味着低概率动作最多被降到 0.008，几乎被彻底压制"，把下界的压制作用和上界的上涨空间混在一起讲，容易让读者以为改的是下界。论文的实际机制恰恰相反：改的是上界；下界保持 0.2，正是因为再放宽会把低概率 token 压死。该课程自声明 AI 协助生成、尚未全面审稿，读二手材料务必回对原文——这次回对就抓到了一处。

## 补遗：RLVR 和 GRPO 不是同一层的东西

读这篇的时候很容易把 RLVR 和 GRPO 连着念，其实它们是两根正交的轴：

| | 回答的问题 | 属于哪一层 | 同层的其他选项 |
|---|---|---|---|
| **RLVR** | 奖励从哪来？ | 奖励来源 / 训练范式 | 人类偏好（RLHF 原点）、学出来的奖励模型、可验证规则奖励 |
| **GRPO** | 拿到奖励后怎么更新模型？ | 策略优化算法 | REINFORCE、PPO、DAPO、GSPO |

两者可以自由组合：RLVR + PPO 完全可以（VAPO 就是 value 路线跑可验证奖励）；RM + GRPO 也成立（GRPO 出自 DeepSeekMath 时并不绑定规则奖励，组内归一化对任何标量奖励都适用）。RLVR + GRPO 之所以成为主流配方，是互相成就：RLVR 的规则奖励便宜、客观、不怕 reward hacking，但它是稀疏的 0/1 信号；GRPO 的组内归一化恰好适合 0/1——同一道题采一组，对错比例天然形成 baseline，不需要再训一个 critic 去估计"这条推理值多少分"。反过来，PPO 的 critic 在稀疏 0/1 奖励加几千 token 长序列上很难训准。

放回本文的失效链里，两者的分工也清楚：RLVR 决定**信号源有多脆**（0/1 奖励只有全对、全错、有对有错三种形态，前两种直接零梯度），GRPO 决定**信号依赖什么**（组内奖励方差，也就是采样多样性）。DAPO 的四个技巧全是在 GRPO 这个算法层打补丁，没动 RLVR 的奖励来源；另一个方向的改进——给 RLVR 造更多可验证任务、把奖励做细到过程级（PRM）——那是在奖励范式层动刀子，和算法层的补丁是正交的两条路。

## 我的判断与未决问题

**判断**：Clip-Higher 的本质，是把"探索节奏"从一个历史遗留的隐式默认值（ε=0.2，来自 2017 年连续控制场景）变成面向长 CoT 推理的显式旋钮。0.28 没有理论推导，是经验值，但它背后的诊断框架——比率裁剪在概率轴上的不对称——是有普适解释力的。以后看到任何"训练好好的突然熵崩了"，先查被 clip 按住的 token 的概率分布，应该成为标准动作。

**未决问题**，按我在乎的程度排序：

1. **熵没有目标区间。** 论文的经验结论是"熵缓慢上升有利"，但同样承认熵过高对应 gibberish 和重复。那么 ε_high 是否应该跟着熵曲线自适应调整，而不是固定 0.28？我没看到有人系统做过。
2. **消融的顺序效应。** 表 1 是累加链，Clip-Higher 排在 Overlong Filtering 之后测，+2 分是在已有改动的底座上量的。它干净的独立贡献是多少，未知。
3. **和序列级裁剪的兼容性。** GSPO 把重要性采样比率提到序列级（解决 MoE 模型 token 级比率的数值不稳），那时"抬 token 级上界"的逻辑还成不成立？本周后续选题会专门拆 GSPO。
4. **异步训练下的稳健性。** 异步 RL 的比率分布漂移更大、更偏，固定 0.28 在不同 off-policy 程度下是否都合适，论文没有覆盖。

## 参考来源

- DAPO 论文：[arXiv:2503.14476](https://arxiv.org/abs/2503.14476)（机制、数值例子、消融数据、训练细节的一手来源）
- RLHF Book 第 6 章及配套实现：`code/policy_gradients/loss.py`、`configs/dapo.yaml`（本地仓库，含 [book-zh 中文对照](https://github.com/natolambert/rlhf-book)）
- verl 生产实现：[`verl/trainer/ppo/core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py)（clip_ratio_low/high、dual-clip）
- B 站视频：[GRPO loss/objective 分析及 biases 分析（DAPO，Dr. GRPO），BV1LgXbY5EFD](https://www.bilibili.com/video/BV1LgXbY5EFD)（注意：该集主要讲动态采样与长度 bias、Dr. GRPO 的标准差问题，Clip-Higher 着墨很少）
- hands-on-modern-rl `docs/chapter18_grpo/deepseek-dapo.md`（中文讲解，Clip-Higher 一节有表述混乱，正文已标注）
- Notion「共同语言」笔记页（GRPO 三轮工业改造线：DAPO / Dr.GRPO / GSPO）

---

> 更新记录：2026-08-26 初版发布；同日根据读者问答补充——"熵"的定义与边界、PPO 与 GRPO 两种 advantage 来源及其后果、min 的单边裁剪读法（值 vs 梯度）、"单轮预算"的准确含义与乘法积累、RLVR 与 GRPO 的分层、KL 惩罚详解与绝对锚/相对锚的换挡。
