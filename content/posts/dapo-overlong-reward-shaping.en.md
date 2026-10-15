---
title: "Judging Truncated Answers: DAPO's Overlong Fix"
date: 2026-08-30T12:05:00+08:00
draft: false
tags: ["rl", "post-training", "grpo", "dapo", "rlvr"]
summary: "RL training needs a generation length cap, and truncated responses are treated as wrong by default—which misjudges 'undecidable' samples as 'wrong' ones and injects reward noise straight into the group-relative advantage. DAPO's two fixes: Overlong Filtering acknowledges that such samples are undecidable and masks their loss; Soft Overlong Punishment builds a linear buffer ramp before the cliff, turning length control into a continuous signal. This post unpacks the mechanism, the formula, and the configuration values, and fits it together with Dr.GRPO and token-mean into a complete picture of the length problem."
---

> This post comes with a companion [interactive report](/reports/dapo-overlong-reward-shaping/) (in Chinese), featuring a drill-down length ruler, a three-piece breakdown of Equation 13, and a switchable comparison of the three handling approaches.

## An Engineering Setting You Can't Avoid

Rollouts in RL training must have a generation length cap. GPU memory has to be reserved for the longest sequence, the batch has to wait for the slowest response to finish, and a single runaway response can drag down the throughput of the whole batch. DAPO's configuration: an expected maximum length of 16,384 tokens, an additional 4,096-token buffer, and a hard generation cap of 20,480 (paper §4.1).

Once a cap is set, some responses will inevitably be truncated. What should happen to these samples? The default practice is common: assign a punitive reward and treat them as wrong answers (reward recorded as 0 or a negative score).

The DAPO paper (arXiv:2503.14476) points out in §3.4 that this default causes problems: **improper handling of truncated samples introduces reward noise, and that noise is injected into the entire training process, amplified by within-group normalization**. The paper's reasoning is straightforward: when a reasoning process that is sound in itself gets penalized merely for running too long, the penalty confuses the model—it can't tell whether its reasoning was valid.

## How the Noise Gets into the Advantage

GRPO's advantage is a within-group relative ranking: sample $G$ responses for the same problem and subtract the group mean (dividing by the std). This means that **once the reward of any single response is wrong, it affects the whole group**.

Take a concrete example: 8 responses are sampled for a problem, and the true split is 4 right and 4 wrong; among them, 2 "wrong" responses are actually high-quality reasoning that got truncated (had they been finished, they would likely have been right).

- By true quality: the group mean is 0.5, the correct responses get positive advantages, and the wrong ones get negative ones;
- Scoring truncation as 0: the truncated samples are also counted as wrong answers and the group mean is dragged down (6 counted as wrong, mean 0.25), so the advantage of every correct sample is inflated and the negative advantage of every wrong sample is softened—including the two that genuinely got it wrong.

As if one layer of distortion weren't enough, there is a second: those two truncated high-quality reasoning traces themselves get reinforced as negative samples. What the model learns from them is not "write a bit shorter" but "my way of reasoning is wrong"—which is exactly where the paper says the model gets confused. Once these two kinds of penalty are lumped together, the model throws out the good way of reasoning along with "writing long".

## Approach 1: Overlong Filtering—Deem Truncated Samples "Undecidable" and Mask Their Loss

DAPO's first approach: rather than treating a truncated sample as a wrong sample, simply refuse to recognize it as a sample at all. Overlong Filtering masks the entire loss of a truncated response—the sample takes no part in advantage estimation and produces no gradient.

The effect shown in Figure 5 of the paper: after applying Overlong Reward Shaping, both AIME accuracy and the actor model's entropy become more stable, and training stability improves markedly. This is worth noting: the stabilized entropy means truncation noise itself is interfering with the exploration dynamics—it doesn't merely affect the reward numbers on paper.

The costs are just as obvious. The paper doesn't go into them, so I'll add two points:

1. **Wasted compute**. The rollout cost of a truncated sample has already been paid (it is often the longest response, which happens to consume the most compute), and once it is masked, not a single gradient comes back in return. This is the same logic as dynamic sampling—both "trade compute for signal quality", which is the orientation of the whole DAPO paper;
2. **Hidden survivorship bias**. Which samples are more likely to be truncated? Long problems, hard problems, problems that need long CoT. Systematically masking truncated samples amounts to systematically down-weighting hard problems in the training distribution. In the short term, training becomes more stable; the long-term cost is that the model may reveal weaknesses on exactly the problems that most need long reasoning. The paper doesn't discuss this side effect; this reservation is my own.

## Approach 2: Soft Overlong Punishment—Build a Buffer Ramp Before the Cliff

Masking everything loses information: the model can't learn that length itself matters. DAPO's second approach is Soft Overlong Punishment, which turns length itself into a **continuous, discriminable reward dimension** (Equation 13 in the paper):

$$R_{length}(y) = \begin{cases} 0, & |y| \le L_{max} - L_{cache} \\ \dfrac{(L_{max} - L_{cache}) - |y|}{L_{cache}}, & L_{max}-L_{cache} < |y| \le L_{max} \\ -1, & L_{max} < |y| \end{cases}$$

The three pieces, one at a time:

- $|y| \le L_{max}-L_{cache}$ (i.e., ≤ 12,288 tokens under DAPO's configuration): no penalty. Length is completely free in this regime;
- Entering the cache interval (12,288 to 16,384): a linearly ramping penalty, falling smoothly from 0 to -1. Every extra token written increases the penalty by $1/L_{cache} = 1/4096$;
- Beyond $L_{max}$ (truncated): the penalty is capped at -1. Note that this term is **added to the original correctness reward**—a response that is correct but runs over can still have its total score pulled down.

The most ingenious thing about this design: **the cliff is not removed; a buffer ramp is built in front of it**. Going past $L_{max}$ is still a hard -1 penalty, just as harsh as the naive approach; but the model starts receiving a continuous "almost over budget" signal four thousand tokens before the cliff, and the signal strength is monotonically related to how far over budget it is. After within-group normalization, two responses—"12,500 tokens, nearly over" and "16,300 tokens, about to go over"—get a distinguishable advantage gap, from which the model can learn to **wrap up early** instead of only feeling the pain once it hits the cap. Length control turns from a binary event (not over / over) into a continuous, learnable task.

## Placing It in the Full "Length Problem" Picture of This Series

The previous few posts in this series all had to do with length; this post covers the final aspect:

- **Gradient dilution from $1/|y|$** (Dr.GRPO, the previous post): the longer a wrong response, the smaller the penalty on each token, so the model tends to make wrong responses overly long. Fix: change the normalization term;
- **token-mean** (DAPO §3.3): instead of averaging within each sample first and then summing over the group, average uniformly over all tokens in the batch, which keeps long responses from taking an outsized weight in the loss. Fix: change the loss aggregation;
- **Overlong** (DAPO §3.4, this post): the truncated samples produced by the generation length cap have undecidable rewards. Fix: mask them (filtering) or change the reward function (soft punishment).

The three address length problems at three different levels: gradient weighting, loss aggregation, and reward judgment. DAPO uses both token-mean and overlong shaping, meaning it fixes both the gradient side and the reward side; Dr.GRPO focuses on normalization. They don't conflict.

One can also see a design orientation in DAPO: **better to use no signal than an unreliable one**. Dynamic sampling filters out prompts with no gradient signal (all-correct / all-wrong problems), overlong filtering masks samples with unreliable rewards (truncated responses), and Dataset Transformation converts all answers to integers to eliminate parser noise (§3.5, DAPO-Math-17K). All four techniques point in the same direction: training collapse is often not because the algorithm isn't good enough, but because noise has crept into the signal.

## My Three Takes

First, **truncated samples are "undecidable" samples, and judging them as wrong is a misjudgment at the reward level**. The premise of RLVR is that rewards are verifiable—you assign a score only if you can judge right or wrong. For a truncated sample, the verifier never got to finish its judgment at all; it is neither a 1 nor a 0. Overlong Filtering acknowledges this undecidability, while Soft Punishment introduces a new decidable dimension (length itself). Both approaches are consistent with the principle that "rewards must be verifiable", whereas the naive punitive reward violates it.

Second, **the fact that entropy is more stable after overlong handling matters more than it appears**. Figure 5(b) shows that the actor's entropy curve is smoother after shaping. This indicates that the chaos near the length cap (high-quality long reasoning that suddenly gets penalized, responses forced into hasty endings) doesn't merely distort the reward; it also adds exploration noise. Read this together with the earlier post on Clip-Higher (entropy collapse): entropy can collapse when wrongly suppressed by the optimizer, and it can also swing wildly when disturbed by reward noise. Entropy monitoring has to be analyzed together with the quality of reward judgments.

Third, **the cache width and the masking ratio are two hyperparameters that need monitoring, and there is currently no public basis for them**. The paper doesn't say how the buffer-ramp length of 4,096/16,384 = 25% was chosen; during training, the truncation rate (the fraction of masked samples) drifts with the model's length behavior—when the masking rate is too high, the effective batch shrinks and the share of hard problems drops; when it is too low, that indicates length control has already broken down or the cap is set too loose. The DAPO paper has no monitoring curves for either number, so reproductions have to fill them in themselves.

## Open Questions

- There are no public experiments on whether the hard-problem weight lost to Overlong Filtering needs to be compensated with a progressive length curriculum in long training runs;
- The rationale and ablations for the choice of cache width (25%) are not disclosed in the paper;
- There is currently no data on whether long but correct exploratory responses get doubly suppressed when Soft Punishment's -1 cap is stacked with Clip-Higher's asymmetric clipping.

## References

- DAPO paper: [arXiv:2503.14476v2](https://arxiv.org/abs/2503.14476) (source of the §3.4 mechanism description, Equation 13, the §4.1 configuration values, and Figure 5)
- Textbook [hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl): Chapter 18 [`docs/chapter18_grpo/deepseek-dapo.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter18_grpo/deepseek-dapo.md) and the appendix DAPO cheatsheet ([`docs/en/appendix_code_cheatsheet/dapo.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/en/appendix_code_cheatsheet/dapo.md))
- Previous posts: [Entropy Collapse: What DAPO's Clip-Higher Fixes](/en/posts/dapo-clip-higher-entropy-collapse/), ["When Wrong, Write Longer": How GRPO's Two Normalization Terms Distort the Optimization Direction](/en/posts/dr-grpo-length-and-difficulty-bias/)
