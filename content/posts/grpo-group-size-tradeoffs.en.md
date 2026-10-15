---
title: "Samples per Prompt: What GRPO's Group Size Buys"
date: 2026-08-29T12:00:00+08:00
draft: false
tags: ["rl", "post-training", "grpo", "rlvr", "dapo"]
categories: ["AI"]
summary: "DeepSeekMath samples 64 responses per question, a teaching implementation defaults to 8, and the latest paper claims even 2 is enough — a 32x spread on the same knob. G is GRPO's only source of a baseline and its biggest compute lever, so what does it actually buy? This post unpacks G's three roles: the sample size of the control variate, the coverage of groups with effective gradients, and the quality of the std-normalization denominator, plus why dynamic sampling can replace a larger G with filtering."
---

> This post has a companion [interactive report](/reports/grpo-group-size-tradeoffs/) (in Chinese), with a zero-gradient-group probability calculator, a dual-curve baseline chart, and a side-by-side comparison of the three camps' choices.

## A Knob With a 32x Spread

On every training step, GRPO samples a group of $G$ responses for each prompt. How large this $G$ should be is a question on which different camps give wildly different answers:

- DeepSeekMath (the original GRPO paper, arXiv:2402.03300 §4.2): **64**. With a batch size of 1024, that means a single step covers only 16 prompts;
- The RLHF Book's companion teaching code (`code/policy_gradients/configs/`): **8**, the default across the whole lineup, from GRPO to GSPO and CISPO;
- The October 2025 paper "It Takes Two: Your GRPO Is Secretly DPO" (arXiv:2510.00977): **2** is enough, and the paper argues specifically that the "large groups are necessary" view is a misreading of why GRPO succeeds.

A 32x gap. Worse, $G$ is two things at once: it is GRPO's only source of a baseline (there is no critic; the group mean is the baseline), and it is the biggest multiplier on training cost (rollouts are the main compute expense in RL training, and doubling $G$ doubles that cost). One knob governs both signal quality and the compute bill, yet people have gotten everything from 2 to 64 to work — which suggests that the question "what does $G$ actually buy?" is itself worth taking apart.

## Role One: The Baseline's Sample Size

Having dropped the critic, GRPO uses the group mean as its baseline: $\hat{A}_i = r_i - \text{mean}(\{r_j\}_{j=1}^G)$ (setting aside the division by std for now). Subtracting a baseline is the standard variance-reduction technique for policy gradients, and the validity condition the textbook (RLHF Book, Chapter 6) gives is that the baseline must not depend on the current action, because $\mathbb{E}_{a\sim\pi}[b(s)\nabla_\theta\log\pi_\theta(a|s)] = 0$ holds for any $b(s)$ that depends only on the state — the baseline leaves the expected gradient unchanged and only suppresses variance.

There is a detail here that few people mention: **GRPO's mean includes the very response being evaluated**, so strictly speaking the baseline depends on the action, and the unbiasedness condition is not strictly satisfied. Compare RLOO (REINFORCE Leave-One-Out, same chapter of rlhf-book): its baseline is the mean of the **other** $K-1$ responses, deliberately excluding the response itself, to "keep the gradient estimator fully unbiased." The relationship between the two advantages can be written out directly:

$$\hat{A}_i^{\text{RLOO}} = r_i - \frac{1}{G-1}\sum_{j\neq i}r_j = \frac{G}{G-1}\left(r_i - \frac{1}{G}\sum_{j}r_j\right) = \frac{G}{G-1}\hat{A}_i^{\text{GRPO(no std)}}$$

The directions are identical; the two differ only by a scale factor of $G/(G-1)$. At $G=64$ the factor is 1.016, which hardly matters; at $G=8$ it is 1.14; **at $G=2$ it is 2** — the GRPO-form advantage is only half as large as the leave-one-out form. This bias is not directional, just a rescaling that grows as $G$ shrinks, and in practice its effect can be absorbed by the learning rate (this is an inference, not an experimental finding from any paper), but it explains why every property of small $G$ gets amplified.

The second entry in the ledger is the baseline's estimation noise. The group mean is an estimate of the true expectation $\mathbb{E}[r]$ from $G$ samples, with noise of magnitude roughly $\sigma_r/\sqrt{G}$ ($\sigma_r$ is the standard deviation of a single response's reward). When $G$ is small, this noise leaks straight into every response's advantage: a correct response can end up with a negative advantage because "everyone else in its group happened to be strong too." The probability that baseline noise randomly flips the gradient direction rises as $G$ decreases — this is the standard "large $G$ reduces variance" argument, and also what 2510.00977 sets out to challenge; more on that later.

## Role Two: Coverage of Groups With Effective Gradients

Rewards in RLVR are binary (right or wrong; DAPO uses +1/−1, DeepSeekMath uses 1/0, and the two are equivalent). If a group's $G$ responses are all right or all wrong, there is no reward variance within the group and every advantage is zero — those $G$ rollouts were run for nothing and contribute no gradient at all.

The probability that a group is uniform can be computed directly (what follows is my own calculation): let the model's per-sample accuracy on the question be $p$, with samples independent; then

$$P(\text{zero-gradient group}) = p^G + (1-p)^G$$

Plugging in numbers to see the trend ($p$ is the per-sample accuracy; the table gives the probability that the group has zero gradient):

| $p$ | $G=8$ | $G=64$ |
|---|---|---|
| 0.5 | 0.8% | ≈0 |
| 0.8 | 16.8% | ≈0 |
| 0.9 | 43.0% | 0.1% |
| 0.95 | 66.3% | 3.8% |
| 0.99 | 92.3% | 52.6% |

How to read it: for medium-difficulty questions ($p$ in the 0.3–0.7 range), even a small $G$ is basically safe; what really consumes $G$ is **questions at the difficulty boundary** — at $p=0.95$, with $G=8$ two-thirds of the groups are all-correct and gradient-free, and only $G=64$ pushes that down to 4%. And the training trajectory pushes questions precisely toward the boundaries: for questions the model has learned, $p \to 1$; for questions it can't learn, $p \to 0$; and the later in training, the fewer questions remain in the "gradient-bearing range." This is one source of the "training efficiency and stability" problems that the DAPO paper describes in its motivation.

Faced with the same table, there are two remedies. One is to increase $G$ and brute-force $p^G$ down — with costs growing multiplicatively. The other is DAPO's **Dynamic Sampling**: instead of increasing $G$, simply rule that zero-gradient groups are not accepted. DAPO's objective includes a hard constraint (the s.t. condition of Equation 8 in the paper):

$$0 < \left|\{o_i \mid \texttt{is\_equivalent}(a, o_i)\}\right| < G$$

That is, the number of correct responses in a group must lie strictly between 0 and $G$ — all-correct and all-wrong groups are filtered out and replaced through oversampling, guaranteeing that every prompt in the batch has an effective gradient. For the same effective-group coverage, the price of filtering is additive oversampling rather than a multiplicative increase in $G$.

But there is another layer of coupling between filtering and $G$ that is easy to miss: **the filtering decision is itself based on an estimate from $G$ samples**. At $G=4$, "all 4 wrong" does not mean the model can't solve the question — a question with $p=0.3$ has a 24% chance of getting all 4 wrong and being mistakenly thrown on the scrap heap. Pairing a small $G$ with dynamic sampling will misjudge a whole set of medium-difficulty questions that "actually carry gradient value" as dead groups; the misjudgment rate falls exponentially with $G$ ($ (1-p)^G $, the same table). So dynamic sampling does not eliminate $G$'s coverage problem; it merely converts it from "zero gradient" into "mistaken discards" (this point is an inference; the DAPO paper does not discuss the misjudgment rate).

## Role Three: The Quality of the std-Normalization Denominator

GRPO's advantage is also divided by the group std: $\hat{A}_i = (r_i - \text{mean})/\text{std}$. This std is likewise an estimate from $G$ samples and is very noisy when $G$ is small; for a uniform group the std is exactly 0, so implementations either add an eps or skip the group (setting a divide-by-zero trap).

Dr.GRPO's critique of the std (which gets a dedicated post later in this series) — that when the std of all-correct or all-wrong groups approaches zero, the advantage is inflated to unreasonable magnitudes — is at its worst with small $G$: when $G$ is small, the std is more likely to land on a small value, and extreme advantages show up frequently. Extreme advantages in turn push up how far the ratio moves, indirectly raising the clip fraction (this chain is an inference: abnormally large $\hat{A}$ → $\pi_\theta$ moves more within a single step → $w_t$ hits the clipping bound sooner). In other words, part of the claim that "large $G$ stabilizes training" is really not about the baseline but about the std — two ledgers that the literature often lumps together.

One more theoretical result worth knowing (arXiv:2502.18548, "What is the Alignment Objective of GRPO?"): the authors give a framework characterizing GRPO's stationary policies and point out that GRPO's way of aggregating preferences differs fundamentally from the logarithmic pooling of standard RLHF — $G$ affects not only the variance of the estimate but also the fixed point of the normalized objective itself. This is analytical work, and the conditions under which its conclusions hold (stationarity, infinitely many steps) are some distance from engineering practice, but it is a reminder of one thing: understanding $G$ purely as a "sampling budget" is incomplete.

## Counterpoint: Are 2 Responses Enough?

The argument of 2510.00977 is worth restating on its own terms: GRPO's effectiveness comes from an **implicit contrastive objective** — comparing just two responses already yields a learning signal, and the group mean is merely a control variate for variance reduction that does not need to be a precise statistical estimate, so 2-GRPO with $G=2$ also works and, structurally, even ends up connecting to DPO. Their theoretical analysis and experiments (at small to medium scale) both support this conclusion.

Does this contradict DeepSeekMath's $G=64$? Not necessarily. Of the things a large $G$ buys, baseline precision is only one, and possibly the least important. The others are coverage of boundary-difficulty questions (the second role), the reliability of the filtering decision (the companion to dynamic sampling), and exploration coverage — a group needs at least one correct response for the model to have a "good sample" to learn from; for a hard question with $p=0.1$, there is an 81% chance at $G=2$ that the group contains no positive sample, versus only 0.1% at $G=64$. These gains depend strongly on model scale and on the task difficulty distribution: for a small model grinding through GSM8K-level problems, the $p$ distribution is concentrated in the middle range, so a small $G$ costs little; for a large model attacking AIME-level problems, $p$ is pinned around 0.1, every positive sample is precious, and $G$ is the exploration budget itself. Whether the 2-GRPO conclusion extends to large models in long-CoT settings is something nobody can answer yet (the paper's experiments are limited in scale; this is a boundary, not a criticism).

## The Cost Ledger, and My Take

First, the accounting: rollouts per training step = number of prompts × $G$. DeepSeekMath's configuration (batch 1024, $G=64$) gets through only 16 prompts per step — prompt diversity and response diversity are traded against each other on the same budget sheet. One more configuration detail deserves attention: DeepSeekMath performs only **one** policy update per exploration stage (strictly on-policy, §4.2), so the ratio is always 1 and clipping never triggers, and $G$'s entire effect shows up in the baseline and in coverage; whereas with DAPO/GSPO-style multi-mini-batch updates, $G$ also indirectly affects the clip fraction through std noise. The same $G$ does not have the same marginal return in the two training regimes.

My take: **there is no universally optimal $G$, because it spends money on three budget sheets at once** — baseline precision ($\sigma_r/\sqrt{G}$, with the fastest-diminishing returns), effective-group coverage ($p^G$, with the most durable returns at boundary difficulty), and the exploration budget (the probability that a group contains a positive sample, almost linear on hard questions). The practical default (8–16) is the balance point under "a medium-difficulty distribution + dynamic sampling as a safety net"; 64 was the naive choice of the DeepSeekMath era, before dynamic sampling and before any clip modifications; 2 is the theoretical limit from the "contrastive objective" perspective, which in engineering terms nobody has yet dared to use at production scale. Before picking $G$, first ask yourself which of the three sheets is the bottleneck: if the questions are too hard (the exploration budget is tight), increase $G$ or add dynamic sampling; if the questions are too varied (prompt diversity is tight), give the budget to the number of prompts; if training is unstable, check the std and the clip fraction first, and don't rush to increase $G$.

The first four posts in this series happen to form a single thread: Clip-Higher fixes the token-level clipping bounds, GSPO changes the unit of the ratio, TIS plugs the numerical gap between training and inference, and this post goes back to the source of the signal — how to allocate the sampling budget. Every "self-evident" default configuration in GRPO, once taken apart, turns out to have a ledger of its own.

## Open Questions

- There is no public systematic ablation of the optimal coupling between $G$, the learning rate, and the clip $\varepsilon$; whether small $G$ + a large learning rate can match the signal-to-noise ratio of a large $G$ is backed only by scattered folk wisdom;
- No paper has quantified the actual impact on data efficiency of dynamic sampling's "mistaken discard" rate ($(1-p)^G$) on medium-difficulty questions;
- Whether 2-GRPO holds for large models, long CoT, and boundary-difficulty tasks is one of the most worthwhile public replications to attempt right now.

## References

- DeepSeekMath / original GRPO paper: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300); $G=64$, batch 1024, and the single-step on-policy update are in §4.2
- DAPO paper: [arXiv:2503.14476](https://arxiv.org/abs/2503.14476); the Dynamic Sampling constraint is in Equation 8, the ±1 reward in Equation 7
- It Takes Two: Your GRPO Is Secretly DPO: [arXiv:2510.00977](https://arxiv.org/abs/2510.00977)
- What is the Alignment Objective of GRPO?: [arXiv:2502.18548](https://arxiv.org/abs/2502.18548)
- RLHF Book, Chapter 6 (the baseline unbiasedness condition, the RLOO leave-one-out formula): local repo `book-zh/chapters/06-policy-gradients.md`; teaching-code default `num_rollouts: 8`: `code/policy_gradients/configs/`
- The zero-gradient-group probability table, the mistaken discards, and the clip-fraction chain are this post's own calculations/inferences, each flagged where it appears in the text
- Previous post: [The Ratio's Wrong Unit: GSPO Goes Sequence-Level](/en/posts/gspo-sequence-level-importance-sampling/)
