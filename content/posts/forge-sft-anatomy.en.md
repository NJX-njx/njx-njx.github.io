---
title: "Anatomy of a Minimal SFT Trainer: Every Decision"
date: 2026-08-28T20:12:53+08:00
draft: false
tags: ["sft", "post-training", "forge", "engineering", "llm"]
summary: "TRL's SFTTrainer can train in a single line, so why does anyone still hand-write an SFT loop? I put every Stage 1 decision in the forge repo on the table one by one (chat template, prompt masking, collate and loss, bf16, LR schedule, data, validation) and check each against two textbooks (rlhf-book, hands-on-modern-rl), four production frameworks (alignment-handbook, trl, open-instruct, OpenRLHF), and six information sources: which decisions have consensus, which are genuine recipe disputes, and which the two textbooks each gloss over."
---

> This post comes with a companion [interactive report](/reports/forge-sft-anatomy/) (in Chinese), which includes a decision comparison matrix and a drill-down index of sources.

## Why Hand-Write a Trainer of Only 511 Lines

I'm building forge, a post-training repo that grows stage by stage along my learning path (Stage 1: SFT; local, not yet pushed as open source). It has only one principle: use mature engineering components off the shelf, but write every line of the core training-data path myself and be able to explain why.

There are three routes to writing SFT. The hands-on-modern-rl course's [`sft_pipeline.py`](https://github.com/walkinglabs/hands-on-modern-rl) is a one-line call to TRL's `SFTTrainer`, with the training loop, loss, and padding all inside the library; the [`instruction_tuning`](https://github.com/natolambert/rlhf-book) code from Nathan Lambert's "RLHF Book" is a hand-written loop; production frameworks (alignment-handbook, open-instruct, OpenRLHF) stack a whole suite of packing, flash-attention, and decontamination on top of the Trainer.

forge went with hand-writing. The motivation isn't performance—a 0.5B model trains fast no matter how you train it—but that **every decision must be explicitly visible**, because every later stage (RM, DPO, GRPO, agentic RL) will reuse this loop, and the RL stages will also have to stuff sampling and reward into it; the black box has to be opened sooner or later. This post lines up each of forge's decisions against two textbooks, four production frameworks, and six information sources (Notion notes, Zhihu, a Feishu industry-news group, arXiv, local video courses). Conclusion first: most decisions have strong consensus, a few are genuine recipe disputes, and then there are a few pitfalls that the two textbooks each paper over in different ways.

## Decision 1: Chat Template—Three Pitfalls in String Concatenation

The raw data is structured messages (role + content), but the model only understands plain text, so the first step is rendering. forge hand-writes ChatML: each message is wrapped as `<|im_start|>{role}\n{content}<|im_end|>\n`. I verified that this is **character-for-character identical** to the output of the `chat_template` bundled with Qwen's official tokenizer.

It looks like there's no real decision in this, but three sources each turned up a pitfall:

1. **Base models have no template**. OLMo-2-1B is a base model, and its tokenizer has no chat template in it; rlhf-book's approach is to copy the entire `chat_template` attribute from the tokenizer of the sibling model `allenai/OLMo-2-0425-1B-SFT` (`utils.py:82-88`). In the real world, a template isn't necessarily ready-made.
2. **`apply_chat_template` has hidden behavior**. When the messages have no system turn, Qwen's template injects a default system prompt ("You are Qwen, created by Alibaba Cloud..."). The hands-on course's data has no system turn, so its training text actually carries this default system line—using the tokenizer's bundled template can bring in text you never reviewed. forge renders only the turns already present in the sample and injects nothing.
3. **Template errors are SFT's number-one silent problem**. HuggingFace's official blog published "Chat Templates: An End to the Silent Performance Killer" (2024-01; my Notion materials library tags it as "the most frequent SFT pitfall"); and the first item on the Chinese community's Trainer troubleshooting checklist is "loss not going down → wrong chat template / wrong labels mask".

My trade-off: hand-write the template first to understand its structure (a template is just string concatenation; the special tokens are registered in the vocabulary, so `<|im_start|>` encodes to exactly one whole token and doesn't fragment), so that later, when I hook into the official template, I know what I'm hooking into.

## Decision 2: Prompt Masking—The Biggest Point of Contention in This Post

For the rendered sequence, every position needs a decision on whether it counts toward the loss. The rule is: **the logits at each position predict the next token; if the prediction target is an assistant token, it is scored, otherwise it is set to -100**. forge's choice is to count every assistant turn (including the wrapping markers) and mask all user/system turns.

How this decision is distributed across the sources is interesting:

- **rlhf-book trains only on the final assistant turn** (assistant replies in the middle of a multi-turn conversation are masked too), and even masks the `<|assistant|>` generation header itself (`utils.py:138,142-148`). Chapter 4 of its textbook explicitly lays out both conventions side by side: train only on the final turn, or mask only the user turns. forge chose the latter.
- **The hands-on course's code doesn't mask at all**: it feeds SFTTrainer a "text" column and computes the loss over the entire sequence. Yet the course's own docs (`imitation-learning-pipeline.md:236-238`) say that counting user/system tokens in the loss is "the most common pitfall... this isn't pedantry; it's protecting the model you train from reciting the prompts". **Same repo, and the code and the docs contradict each other**—the most direct example of "teaching implementation ≠ best practice".
- **trl's current approach** is `assistant_only_loss` + `{% generation %}` markers in the template, with an explicit warning: if an assistant turn's end token isn't inside the loss mask, "the model may never learn to stop". This corroborates the need for forge to train on `<|im_end|>`—the model must learn to stop.
- **Ownership of the boundary tokens**: production templates (trl/OpenRLHF) usually wrap only the assistant content in the generation markers, i.e., the `<|im_start|>assistant\n` header doesn't count toward the loss; forge counts the header too. A difference of a token or two—a detail open to debate.
- **Masking itself has been challenged by research**: "Instruction Tuning with Loss over Instructions" (arXiv [2405.14394](https://arxiv.org/abs/2405.14394)) found that also computing the loss on the instruction part (IM) outperforms computing it only on the output (OM) in many settings across 21 benchmarks. In other words, completion-only is the current framework default, not something proven correct—when the goal is for the model to internalize the instruction distribution, masking throws away signal.

Written out as a formula, forge's loss is:

$$L = -\frac{1}{|A|}\sum_{t:\,x_{t+1}\in A}\log p_\theta(x_{t+1}\mid x_{\le t})$$

where $A$ is the set of tokens in assistant turns and $|A|$ is the number of valid tokens.

## Decision 3: Collate and Loss—Line-for-Line Isomorphic, Plus a Tail Neither Book Handles Well

The core trio on the training side: right-pad collate (pad tokens fill the inputs, -100 fills the labels, attention_mask is marked 0), shift (`logits[:, :-1]` against `labels[:, 1:]`), and `F.cross_entropy(ignore_index=-100)`. This implementation is line-for-line isomorphic to rlhf-book's—two independently written pieces of code that look the same are the strongest evidence that "this is the standard practice".

But the side-by-side comparison exposed two problems that not even the textbooks handle cleanly:

1. **The grad accumulation tail**. The number of micro-batches in an epoch isn't necessarily divisible by the number of accumulation steps. rlhf-book simply discards the leftover group at the tail—the backward passes are wasted, and those few samples contribute nothing to the update; forge steps on the leftover group as well, but still divides the loss by the nominal accum count—so the leftover group's gradient is applied at a scale that's too small. Each of the two books fudges half of it. What's more common in real production is to normalize by the leftover group's actual size, or simply set `drop_last=True` and align the scheduler's step count.
2. **The micro-batch averaging convention**. Both forge and rlhf-book "average within each micro-batch first, then divide by accum". When samples differ greatly in their number of trainable tokens, this differs subtly from "a weighted average over all tokens". The vast majority of frameworks accept this compromise, but if you want to pick at training details (mixed data with large length differences), this is worth coming back to. The production-side counterpart is trl's `chunked_nll`: drop the -100 positions first, then compute the projection—mathematically equivalent, and it saves peak GPU memory.

OpenRLHF has another detail worth copying: **sequences are forced to end with EOS**—after truncation, `input_ids[-1] = eos_token_id` (`sft_dataset.py`). forge makes no such guarantee: the last token of a truncated sample is arbitrary, and when truncation happens to cut off the `<|im_end|>` at the end of an assistant turn, that amounts to teaching the model to "stop halfway through".

## Decision 4: Precision—Pure bf16 and a Redundant Layer of Autocast

Both textbooks load the weights in pure bf16: no autocast, no fp32 master weights, no GradScaler. Teaching code goes for convenience, and a 0.5B model really doesn't call for more anyway.

forge's approach exposes something I need to correct: the weights are already loaded in bf16 (`model.py`), yet the forward pass is additionally wrapped in a `torch.autocast(bf16)` (`train.py:92`)—when the weights are already bf16, this autocast layer is essentially redundant. Real mixed precision (fp32 master + loss scaling) lives at the Megatron/DeepSpeed level; Stage 1 doesn't need it, but it deserves a note in the blog. Tulu 3's `--mixed_precision bf16` and the community's "BF16 is 2x faster than FP32" (hands-on course docs) give this default production-side backing.

## Decision 5: LR Schedule—Cosine vs. Linear Is a Recipe Dispute

forge uses `get_cosine_schedule_with_warmup` with 3% warmup. Production gives no single answer: Tulu 3's 8B SFT recipe is **lr 5e-6 + linear decay + warmup 0.03** (open-instruct `finetune_8b.sh`; a reproducible config is harder evidence than a paraphrase of the paper); Zephyr uses 2e-5 + cosine + warmup 0.1; OpenRLHF defaults to cosine_with_min_lr. rlhf-book simply hand-writes a linear decay—the teaching book itself doesn't use cosine either.

Two worthwhile observations:

- **The default scheduler is a trap**. If you don't specify one explicitly, HF/TRL default to linear + warmup 0 (that's exactly how the hands-on course's code was run). You think you didn't make a decision, but you already did.
- **lr and packing are strongly coupled**. The OLMo 3 practice cited in the textbook: sequence packing significantly increases the effective token batch, so the lr has to go up with it. Talking about lr in isolation from the throughput configuration is meaningless.

forge's numbers match mainstream recipes item by item: lr 1e-5 (within the 5e-6~2e-5 production range, and in line with "the smaller the model, the larger the lr"), warmup 0.03, weight decay 0.0, grad clip 1.0, epochs 2. An effective batch of 32 (8×4) is teaching-scale for 0.5B—rlhf-book coincidentally also uses 32 (4×8), while production uses 128-256. epochs=2 has research-side support: SFT scaling research shows that the gains from repeated data come roughly within the first 4 epochs (arXiv [2402.17193](https://arxiv.org/abs/2402.17193)), and "SFT Memorizes, RL Generalizes" (arXiv [2501.17161](https://arxiv.org/abs/2501.17161)) further points out that training for more epochs mainly reinforces memorization rather than generalization—2 epochs is in the safe zone, and it also foreshadows the ceiling of a pure SFT slice.

## Decision 6: Data—forge's Biggest Simplification Reveals the Full Production Picture

forge uses 10,000 examples from the `everyday-conversations` subset of `HuggingFaceTB/smoltalk`: a single source, zero quality filtering. This is the biggest simplification in the entire repo, and the sources pin down its place precisely:

The SmolTalk dataset card (verified) states: the full set has 1.1M examples in 14 subsets, and the core new data is Smol-Magpie-Ultra 400K (Magpie pipeline + Llama-3.1-405B generation); **everyday-conversations is multi-turn everyday dialogue, merely a "basic conversational behavior" slice**, with math, code, and long context covered by other subsets. In other words, training on it alone yields the conversational layer of the full recipe, not full capability.

Both ends of the data-scale spectrum have names: LIMA (arXiv [2305.11206](https://arxiv.org/abs/2305.11206)) used 1,000 hand-curated examples to bring a 65B model close to the GPT-4 of its day and proposed the "Superficial Alignment Hypothesis"—pretraining has already stored up the capabilities, and SFT only does shallow alignment; Tulu 3 (arXiv [2411.15124](https://arxiv.org/abs/2411.15124)) used 940K examples for rigorous multi-skill **mixture ablations**, backed by a decontamination toolchain. In between sit AlpaGasus (90% of the 52K was low quality; the 9K filtered out of it beat the full set) and Deita (scoring on complexity, quality, and diversity; 6-10K beat larger mixed sets). One line in my Notion notes sets the tone for this spectrum: LIMA coexists with industrial million-scale datasets, and every hard capability needs data behind it; data work comes down to just four decisions—what to get, what to keep, how much to mix in, and what to do when it isn't enough.

As for what production data engineering actually looks like, the Chinese community and the Feishu group supplied the two ends: a Zhihu column calls the curation pipeline (dedup → quality gate → remove overlap with eval sets → ensure multi-turn completeness) "the link in the whole pipeline that deletes the most and is the least transparent"; on mixture ratios, mixing in 10%-20% general data to prevent catastrophic forgetting is the highest-consensus rule of thumb in the community (there are also two claims a factor of ten apart, 1:1 and 1:10~1:100 experience replay—the ratio itself is unsettled). Claims like "90% of SFT performance depends on data quality" circulate widely in the community, but no experimental source can be given for them, so they can only be treated as a sentiment indicator.

Newer signals come from the industry-news group: Shanghai AI Lab's SKT, AutoCompact's judge-guided SFT, MindForge's trajectory fine-tuning—production SFT data is increasingly "strong-model trajectories + verifiers", and the smoltalk that forge loads directly is, in essence, also trajectories that someone else has already distilled. Another paper shared in the group, "SFT Conflicts, RL Coexists" (arXiv [2608.03573](https://arxiv.org/abs/2608.03573)), supplies a footnote from the opposite direction: multi-stage multi-task SFT degrades by 23.1% on average, while the same multi-stage RL improves by 24.9%—the pitfall a minimal single-task, single-stage SFT sidesteps isn't in the loop itself but in the data composition.

My judgment: 10,000 examples from a single source is a reasonable teaching starting point—first verify that the pipeline is correct; mixtures and quality gates are Stage 2's business. One practical constraint to keep firmly in mind: right now `max_len=1024`, while long reasoning-chain data routinely runs 2000-8000 tokens, so when switching data the context length has to change along with it; otherwise a large number of samples will be truncated into "the question plus half a thinking process", with `<|im_end|>` cut off in the bargain.

## Decision 7: Validation—Val Loss Is the Low-Variance Choice, but Only Half the Story

forge's validation trio: hold out 1% of the data as a val split, compute val loss after training, and greedy-decode 8 fixed prompts to eyeball the results.

A methodological rule in my Notion notes directly backs val loss: accuracy-type metrics have high variance on small models, so **prefer per-domain validation loss, repeated across multiple seeds**. Run an MMLU-style eval on a 0.5B model and the noise will drown out the signal; loss is at least stable.

The course docs (unreviewed, but this point matches frontline experience) list five metrics to watch in SFT: train_loss, eval_loss, format pass rate, repetition rate, and manual spot checks, and warn that "looking only at loss is a beginner's mistake—the loss will look great, but the model will turn into an assistant with correct formatting and anemic content". rlhf-book's approach supplies the other half: during training it periodically generates samples from 4 fixed prompts, and at step 0 it first prints the base model's outputs—the qualitative contrast of "base misses the question → after SFT it answers and then stops" is the most intuitive way to check whether a checkpoint is taking effect. Ironically, the entire module never saves the model (verified with grep): it has a tracker and in-training generation, but no checkpoint.

What forge currently lacks: periodic generation during training, quality metrics like format pass rate, and a wandb tracker. Plus a set of throughput optimizations from production practice: packing + padding-free + flash-attention (Netflix's write-up reports 4.7x throughput, and IBM's arXiv [2407.09105](https://arxiv.org/abs/2407.09105) is the authoritative account of putting it into production), length-bucketed batching, and gradient checkpointing on the training side. None of these go into Stage 1—the point of minimalism is to let you see clearly what the system is doing without them.

## The Not-Done List Is the Stage 2 Roadmap

Having compared against all six sources, here's what forge deliberately left out and what it should add soon, in order of priority:

- **P0: packing + padding-free + FA2**. All four production frameworks have it, and it's the number-one engineering gap between forge and production; Netflix's 4.7x throughput shows this is no mere detail.
- **P1: align the mask boundaries with the production convention** (assistant header excluded from the loss, end token always included), and **force truncated samples to end with EOS** (a one-line fix, as in OpenRLHF).
- **P1: normalize the accum tail by its actual group size**; and while at it, remove that redundant autocast layer.
- **P2: chunked loss, grad checkpointing, decontamination**—these can wait for the large-model stage; decontamination belongs to the data-preparation layer and doesn't go into the trainer.
- **Not pursuing: NEFTune** (frameworks support it, mainstream recipes don't use it), **switching cosine to linear** (a recipe dispute with two coexisting camps; forge sides with the majority).

Looking back, what hand-writing 511 lines bought me isn't performance but a decision checklist: why each knob sits where it does, how the textbooks choose, how production chooses, and what research says. SFTTrainer runs in one line, but that one line hides a dozen-plus default decisions, three of which (full-sequence loss, linear schedule with zero warmup, silent truncation) happen to be among the top entries on the community's troubleshooting checklists. Only by taking the defaults apart do you know what you're choosing.

## References and Sources

**Textbook code**: rlhf-book `code/instruction_tuning/` ([repo](https://github.com/natolambert/rlhf-book); key line numbers are given in the main text); hands-on-modern-rl `code/chapter08_rlhf/sft_pipeline.py` and `docs/chapter15_rlhf/` ([repo](https://github.com/walkinglabs/hands-on-modern-rl); the course states that it has not been comprehensively reviewed).

**Production frameworks** (all main-branch source snapshots): [alignment-handbook](https://github.com/huggingface/alignment-handbook) `recipes/zephyr-7b-beta/sft/config_full.yaml`; [trl](https://github.com/huggingface/trl) `sft_trainer.py`/`sft_config.py`; [open-instruct](https://github.com/allenai/open-instruct) `finetune.py` and `scripts/train/tulu3/finetune_8b.sh`; [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) `openrlhf/datasets/sft_dataset.py`.

**Papers**: LIMA [2305.11206](https://arxiv.org/abs/2305.11206) · AlpaGasus [2307.08701](https://arxiv.org/abs/2307.08701) · Deita [2312.15685](https://arxiv.org/abs/2312.15685) · Tulu 3 [2411.15124](https://arxiv.org/abs/2411.15124) · SmolLM2 [2502.02737](https://arxiv.org/abs/2502.02737) · Magpie [2406.08464](https://arxiv.org/abs/2406.08464) · SFT scaling [2402.17193](https://arxiv.org/abs/2402.17193) · SFT Memorizes, RL Generalizes [2501.17161](https://arxiv.org/abs/2501.17161) · Instruction Modelling [2405.14394](https://arxiv.org/abs/2405.14394) · Packing with FlashAttention [2407.09105](https://arxiv.org/abs/2407.09105) · SFT Conflicts, RL Coexists [2608.03573](https://arxiv.org/abs/2608.03573).

**Community and industry**: [HuggingFace · Chat Templates: An End to the Silent Performance Killer](https://huggingface.co/blog/chat-templates) · [Zhihu · SFT data curation](https://zhuanlan.zhihu.com/p/2073360093089883686) · [Zhihu · Training data preparation for SFT and RL](https://zhuanlan.zhihu.com/p/2023522185143173700) · [Netflix engineering blog · Scaling LLM Post-Training](https://netflixtechblog.com/) (indexed via the Notion materials library, 2026-02) · Feishu industry-news group (posts on SKT, AutoCompact, MindForge, 2026-05 to 2026-08).

**forge repo**: `src/forge/data.py`, `src/forge/train.py`, `src/forge/model.py`, `configs/sft.yaml`, `tests/test_data.py` (local, not yet pushed as open source; line numbers in the text refer to the 2026-08-28 working tree).
