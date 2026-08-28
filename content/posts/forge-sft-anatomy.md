---
title: "解剖一个极简 SFT 训练器：每个决策，教科书和生产框架各怎么说"
date: 2026-08-28T20:12:53+08:00
draft: false
math: true
tags: ["sft", "post-training", "forge", "engineering", "llm"]
categories: ["AI"]
summary: "TRL 的 SFTTrainer 一行就能训，为什么还有人手写 SFT 循环？我把 forge 仓库 Stage 1 的每个决策（chat template、prompt masking、collate 与 loss、bf16、LR schedule、数据、验证）逐一摆到台面上，对照两套教材（rlhf-book、hands-on-modern-rl）、四个生产框架（alignment-handbook、trl、open-instruct、OpenRLHF）和六路信息源：哪些决策有共识，哪些有真实的配方之争，哪些两本教材各自糊弄过去了。"
---

> 本文有配套的[交互报告](/reports/forge-sft-anatomy/)，含决策对照矩阵与可下钻的出处索引。

## 为什么手写一个只有 511 行的训练器

我在搭一个随学习路径逐阶段生长的后训练仓库 [forge](https://github.com/NJX-njx)（Stage 1：SFT），原则只有一条：成熟的工程件直接拿来用，训练-数据核心链路每一行都自己写、能讲清为什么。

写 SFT 有三条路线。 hands-on-modern-rl 课程的 [`sft_pipeline.py`](https://github.com/walkinglabs/hands-on-modern-rl) 是 TRL `SFTTrainer` 一行调用，训练循环、loss、padding 全在库内；Nathan Lambert《RLHF Book》的 [`instruction_tuning`](https://github.com/natolambert/rlhf-book) 是手写循环；生产框架（alignment-handbook、open-instruct、OpenRLHF）则在 Trainer 之上叠 packing、flash-attention、去污染一整套。

forge 选了手写。动机不是性能——0.5B 模型怎么训都快——而是**每个决策必须显式可见**，因为后面所有阶段（RM、DPO、GRPO、agentic RL）都要复用这个循环，RL 阶段还要在循环里塞采样和 reward，黑箱迟早要拆开。这篇文章把 forge 的每个决策和两套教材、四个生产框架、六路信息源（Notion 笔记、知乎、飞书业界动态群、arXiv、本地视频课）摆在一起对照。结论先行：大部分决策有强共识，少数是真实的配方之争，还有几个坑，两本教材各自用不同方式糊弄过去了。

## 决策一：chat template，字符串拼接里的三个坑

原始数据是结构化的 messages（role + content），模型只认纯文本，所以第一步是渲染。forge 手写 ChatML：每条消息包成 `<|im_start|>{role}\n{content}<|im_end|>\n`。我验证过，这和 Qwen 官方 tokenizer 自带的 `chat_template` 渲染结果**逐字符一致**。

看似没有决策含量，三个信息源各自给了一个坑：

1. **base 模型没有模板**。OLMo-2-1B 是 base 模型，tokenizer 里没住 chat template，rlhf-book 的做法是从姊妹模型 `allenai/OLMo-2-0425-1B-SFT` 的 tokenizer 整段拷贝 `chat_template` 属性（`utils.py:82-88`）。真实世界里模板不一定现成。
2. **`apply_chat_template` 有隐藏行为**。Qwen 的模板在 messages 无 system 段时会注入默认 system prompt（"You are Qwen, created by Alibaba Cloud..."）。hands-on 课程的数据没有 system 段，训练文本里实际带着这句默认 system——用 tokenizer 自带模板，可能引入你没审过的文本。forge 只渲染样本里已有的段，不注入任何东西。
3. **模板错是 SFT 的头号隐形问题**。HuggingFace 官方博客《Chat Templates: An End to the Silent Performance Killer》（2024-01，我的 Notion 材料库把它标为"SFT 最高频的坑"）；中文社区的 Trainer 排错清单第一条就是"loss 不降 → chat template 错 / labels mask 错"。

我的取舍：先手写模板把结构看懂（模板就是字符串拼接，特殊 token 在词表里注册过，`<|im_start|>` 编码出来是完整一个 token，不会碎裂），之后对接官方模板时才知道自己在对接什么。

## 决策二：prompt masking，全篇最大的分歧点

渲染好的序列，每个位置都要决定算不算 loss。规则是：**每个位置的 logits 预测下一个 token，预测目标是 assistant token 就计分，否则置 -100**。forge 的选择是所有 assistant 段都计（含包裹标记），user/system 段全 mask。

这个决策在信息源里的分布很有意思：

- **rlhf-book 只训最后一轮 assistant**（多轮对话中间的 assistant 回复也 mask），连 `<|assistant|>` 生成头本身也 mask（`utils.py:138,142-148`）。它的教材 Chapter 4 明确把两种约定并列：只训最后一轮，或只 mask user 段。forge 选了后者。
- **hands-on 课程的代码完全不 mask**：用 "text" 列喂 SFTTrainer，对整个序列计 loss。而课程自己的文档（`imitation-learning-pipeline.md:236-238`）写道，把 user/system 的 token 算进 loss 是"最常见的坑……这不是洁癖，这是在保护你训练出来的模型不去背诵提示词"。**同一仓库，代码和文档互相矛盾**——这是"教学实现 ≠ 最佳实践"的最直接样本。
- **trl 现在的方案**是 `assistant_only_loss` + 模板里的 `{% generation %}` 标记，并且有一句明确的警告：如果 assistant 轮的结束 token 不在 loss mask 内，"the model may never learn to stop"。这佐证了 forge 把 `<|im_end|>` 训进去的必要性——模型必须学会停止。
- **边界 token 的归属**：生产模板（trl/OpenRLHF）通常只把 assistant content 包进 generation 标记，即 `<|im_start|>assistant\n` 头不计 loss；forge 把头也算进去了。一两 token 的差异，属于可以争论的细节。
- **masking 本身被研究挑战过**：《Instruction Tuning with Loss over Instructions》（arXiv [2405.14394](https://arxiv.org/abs/2405.14394)）发现对 instruction 部分也算 loss（IM）在 21 个 benchmark 的许多场景下优于只在 output 上算 loss（OM）。也就是说 completion-only 是当前框架默认，不是被证明的正确——当目标是让模型内化指令分布时，masking 会丢信号。

用公式写清楚 forge 的 loss：

$$L = -\frac{1}{|A|}\sum_{t:\,x_{t+1}\in A}\log p_\theta(x_{t+1}\mid x_{\le t})$$

其中 $A$ 是 assistant 段的 token 集合，$|A|$ 是有效 token 数。

## 决策三：collate 与 loss，逐行同构，加一个两书都没处理好的尾巴

训练侧的核心三连：right-pad collate（pad 填 input、-100 填 labels、attention_mask 标 0）、shift（`logits[:, :-1]` 对 `labels[:, 1:]`）、`F.cross_entropy(ignore_index=-100)`。这套实现和 rlhf-book 逐行同构——两处独立写出的代码长得一样，是最强的"这就是标准做法"证据。

但对照研究暴露了两个连教材都没处理干净的问题：

1. **grad accumulation 的尾部**。一个 epoch 的 micro-batch 数不一定整除 accumulation 步数。rlhf-book 的做法是尾部残余组直接丢弃——backward 白算，那几个样本不贡献更新；forge 的做法是残余组也 step，但 loss 仍除以标称的 accum 数——残余组的梯度按偏小的尺度更新。两本书各糊弄了一半。真实生产里更常见的做法是按残余组实际数量归一，或者干脆 `drop_last=True` 并把调度器步数对齐。
2. **micro-batch 平均的口径**。forge 和 rlhf-book 都是"每个 micro-batch 先取平均，再除以 accum"。当各样本 trainable token 数差很多时，这和"全部 token 加权平均"有细微差别。绝大多数框架接受这个妥协，但如果你要抠训练细节（长度差异大的混合数据），这里值得回来重看。生产侧的对应物是 trl 的 `chunked_nll`：先把 -100 位置丢掉再算投影，数学等价，省峰值显存。

OpenRLHF 还有一个值得抄的细节：**序列强制以 EOS 收尾**——截断后 `input_ids[-1] = eos_token_id`（`sft_dataset.py`）。forge 不做这个保证，被截断样本的最后一个 token 是任意值，而截断恰好切断 assistant 尾部的 `<|im_end|>` 时，等于在教模型"说完一半就停"。

## 决策四：精度，纯 bf16 和一层多余的 autocast

两本教材都是纯 bf16 加载权重：不用 autocast、没有 fp32 master weights、没有 GradScaler。教学场景图省事，0.5B 模型也确实犯不上。

forge 的做法暴露了一个自我修正点：权重已经 bf16 加载（`model.py`），forward 外面又包了一层 `torch.autocast(bf16)`（`train.py:92`）——权重已是 bf16 时这层 autocast 基本是冗余的。真正的混合精度（fp32 master + loss scaling）在 Megatron/DeepSpeed 那一层，Stage 1 不需要，但博客里值得记下这笔账。Tulu 3 的 `--mixed_precision bf16` 和社区的"BF16 比 FP32 快 2 倍"（hands-on 课程文档）给了这个默认值生产侧的背书。

## 决策五：LR schedule，cosine vs linear 是配方之争

forge 用 `get_cosine_schedule_with_warmup`，warmup 3%。生产侧的答案并不统一：Tulu 3 的 8B SFT 配方是 **lr 5e-6 + linear decay + warmup 0.03**（open-instruct `finetune_8b.sh`，可复现配置比论文转述更硬）；Zephyr 是 2e-5 + cosine + warmup 0.1；OpenRLHF 默认 cosine_with_min_lr。rlhf-book 干脆手写线性衰减——教学书自己也不用 cosine。

两个有价值的观察：

- **默认 scheduler 是个陷阱**。不显式指定时，HF/TRL 默认 linear + warmup 0（hands-on 课程的代码就是这样跑出来的）。你以为没做决策，其实已经做了。
- **lr 和 packing 强耦合**。教材引用的 OLMo 3 实践：sequence packing 会显著增大有效 token batch，lr 要跟着上调。离开吞吐配置谈 lr 没有意义。

forge 的数字和主流配方逐项吻合：lr 1e-5（落在 5e-6~2e-5 生产区间，且"模型越小 lr 越大"）、warmup 0.03、weight decay 0.0、grad clip 1.0、epochs 2。有效 batch 32（8×4）对 0.5B 是教学缩放——rlhf-book 巧合地也是 32（4×8），生产是 128-256。epochs=2 有研究侧的支持：SFT 的 scaling 研究显示重复数据的收益约在前 4 个 epoch 内（arXiv [2402.17193](https://arxiv.org/abs/2402.17193)），《SFT Memorizes, RL Generalizes》（arXiv [2501.17161](https://arxiv.org/abs/2501.17161)）进一步指出多训 epoch 主要强化记忆而非泛化——2 个 epoch 在安全区，也预示了纯 SFT 切片的天花板。

## 决策六：数据，forge 最大的简化，正好照出生产的全貌

forge 用 `HuggingFaceTB/smoltalk` 的 `everyday-conversations` 子集 1 万条，单源，零质量过滤。这是全仓库最大的简化点，信息源给它的定位很精确：

SmolTalk 数据集卡片（已核实）写明：全集 1.1M 条、14 个子集，核心新数据是 Smol-Magpie-Ultra 400K（Magpie 管线 + Llama-3.1-405B 生成）；**everyday-conversations 是多轮日常对话，只是"基础对话行为"切片**，数学、代码、长上下文由其他子集承担。也就是说，用它单独训练，产出的是完整配方里的对话层，不是完整能力。

数据规模的光谱两端都有名字：LIMA（arXiv [2305.11206](https://arxiv.org/abs/2305.11206)）用 1000 条人工精选让 65B 模型接近当时 GPT-4，提出"表面对齐假说"——预训练已储备能力，SFT 只做浅层对齐；Tulu 3（arXiv [2411.15124](https://arxiv.org/abs/2411.15124)）用 94 万条做严格的多技能**配比消融**，并配去污染工具链。中间是 AlpaGasus（52K 里 90% 低质，筛出 9K 反超全量）和 Deita（按复杂性、质量、多样性打分，6-10K 超越更大的混合集）。我的 Notion 笔记里有一句话可以给这个光谱定调：LIMA 与工业百万级并存，硬能力每条都要数据支撑；数据工作只有四个决策——要什么、留什么、配多少、不够怎么办。

生产侧数据工程的真实形态，中文社区和飞书群给出了两端：知乎专栏把 curation 链路（去重 → 质量门 → 剔除与评测集重合 → 保证多轮完整）称为"整条链路里删得最多、也最不透明的一环"；配比上混 10%-20% 通用数据防灾难性遗忘是社区共识度最高的经验（也有 1:1 和 1:10~1:100 经验重放两种相差十倍的说法——配比本身没有定论）。"SFT 效果 90% 取决于数据质量"这类说法在社区流传很广，但给不出实验出处，只能当情绪指标。

更新的信号来自业界动态群：上海 AI Lab 的 SKT、AutoCompact 的 judge-guided SFT、MindForge 的轨迹微调——生产 SFT 数据越来越多是"强模型轨迹 + 验证器"，forge 直接加载的 smoltalk 本质上也是他人蒸馏好的轨迹。另一篇被转发的论文《SFT Conflicts, RL Coexists》（arXiv [2608.03573](https://arxiv.org/abs/2608.03573)）给出了反向注脚：多阶段多任务 SFT 平均退化 23.1%，而同样的多阶段 RL 提升 24.9%——单任务单阶段的最小 SFT 避开的坑不在循环本身，在数据组合。

我的判断：单源 1 万条是合理的教学起点——先把链路验证正确，配比和质量门是 Stage 2 的事。一个实际约束要记牢：现在 `max_len=1024`，长推理链数据动辄 2000-8000 token，换数据时上下文长度必须一起动，否则大量样本会被截成"问题加半个思考过程"，还顺手切掉了 `<|im_end|>`。

## 决策七：验证，val loss 是低方差选择，但只是一半

forge 的验证三件套：1% 数据留作 val split、训练后算 val loss、8 个固定 prompt 贪心解码直观看效果。

我的 Notion 笔记里有一条方法论直接给 val loss 提供了背书：小模型上准确率类指标方差大，**优先用分领域验证 loss，多 seed 重复**。0.5B 模型上跑 MMLU 式评测，噪声会淹没信号；loss 至少稳定。

课程文档（未经审稿，但这条与一线经验一致）列了 SFT 应看的五项指标：train_loss、eval_loss、格式通过率、重复率、人工抽样，并警告"只看 loss 是新手误区——loss 会很好看，但模型会变成格式正确、内容贫血的助手"。rlhf-book 的做法补上了另一半：训练中定期用固定 4 个 prompt 生成样本，step 0 先打印 base 模型的输出——"base 答非所问 → SFT 后答完就停"的定性对比，是检查点是否生效的最直观手段。讽刺的是它全模块不存模型（grep 验证），tracker 和过程生成都有，checkpoint 没有。

forge 目前缺：训练过程中的定期生成、格式通过率这类质量指标、wandb  tracker。以及生产实践的一组吞吐 cut：packing + padding-free + flash-attention（Netflix 的分享给出 4.7 倍吞吐，IBM 的 arXiv [2407.09105](https://arxiv.org/abs/2407.09105) 是落地权威说明）、按长度分桶组 batch、训练侧梯度检查点。这些都不进 Stage 1——极简的意义就是让你看清没有它们时系统在干什么。

## 没做的清单，就是 Stage 2 的路线图

对照完六个信息源，forge 刻意没做和应该尽快补的，按优先级：

- **P0：packing + padding-free + FA2**。四个生产框架全有，是 forge 与生产的第一工程差距；Netflix 的 4.7 倍吞吐说明这不是细节。
- **P1：mask 边界对齐生产口径**（assistant 头不计 loss、结束 token 必计）、**截断样本强制 EOS 收尾**（OpenRLHF 一行可修）。
- **P1：accum 尾部按实际组数归一**；顺手去掉那层冗余的 autocast。
- **P2：chunked loss、grad checkpointing、去污染**——大模型阶段再说；去污染属于数据准备层，不进训练器。
- **不跟进：NEFTune**（框架支持、主流 recipe 没用）、**cosine 换 linear**（两派并存的配方之争，forge 站多数派）。

回头看，手写 511 行买到的不是性能，是一张决策清单：每个旋钮为什么在这个位置，教材怎么选、生产怎么选、研究怎么说。SFTTrainer 一行能跑，但那一行里藏着十几个默认决策，其中三个（全序列 loss、linear schedule 零 warmup、静默截断）恰好是社区排错清单的前几名。把默认拆开看，才知道自己在选什么。

## 参考与出处

**教材代码**：rlhf-book `code/instruction_tuning/`（[仓库](https://github.com/natolambert/rlhf-book)，关键行号见正文）；hands-on-modern-rl `code/chapter08_rlhf/sft_pipeline.py` 与 `docs/chapter15_rlhf/`（[仓库](https://github.com/walkinglabs/hands-on-modern-rl)，该课程自声明未经全面审稿）。

**生产框架**（均为 main 分支源码快照）：[alignment-handbook](https://github.com/huggingface/alignment-handbook) `recipes/zephyr-7b-beta/sft/config_full.yaml`；[trl](https://github.com/huggingface/trl) `sft_trainer.py`/`sft_config.py`；[open-instruct](https://github.com/allenai/open-instruct) `finetune.py` 与 `scripts/train/tulu3/finetune_8b.sh`；[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) `openrlhf/datasets/sft_dataset.py`。

**论文**：LIMA [2305.11206](https://arxiv.org/abs/2305.11206) · AlpaGasus [2307.08701](https://arxiv.org/abs/2307.08701) · Deita [2312.15685](https://arxiv.org/abs/2312.15685) · Tulu 3 [2411.15124](https://arxiv.org/abs/2411.15124) · SmolLM2 [2502.02737](https://arxiv.org/abs/2502.02737) · Magpie [2406.08464](https://arxiv.org/abs/2406.08464) · SFT scaling [2402.17193](https://arxiv.org/abs/2402.17193) · SFT Memorizes, RL Generalizes [2501.17161](https://arxiv.org/abs/2501.17161) · Instruction Modelling [2405.14394](https://arxiv.org/abs/2405.14394) · Packing with FlashAttention [2407.09105](https://arxiv.org/abs/2407.09105) · SFT Conflicts, RL Coexists [2608.03573](https://arxiv.org/abs/2608.03573)。

**社区与业界**：[HuggingFace · Chat Templates: An End to the Silent Performance Killer](https://huggingface.co/blog/chat-templates) · [知乎 · SFT 数据 curation](https://zhuanlan.zhihu.com/p/2073360093089883686) · [知乎 · SFT 与 RL 训练数据准备](https://zhuanlan.zhihu.com/p/2023522185143173700) · [Netflix 工程博客 · Scaling LLM Post-Training](https://netflixtechblog.com/)（经 Notion 材料库索引，2026-02）· 飞书业界动态群（SKT、AutoCompact、MindForge 分享，2026-05 至 2026-08）。

**forge 仓库**：`src/forge/data.py`、`src/forge/train.py`、`src/forge/model.py`、`configs/sft.yaml`、`tests/test_data.py`（本地，尚未开源推送；文中行号以 2026-08-28 工作区为准）。
