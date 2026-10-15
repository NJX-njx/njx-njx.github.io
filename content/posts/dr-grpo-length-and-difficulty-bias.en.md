---
title: "“Wrong? Write Longer”: What GRPO Normalizers Skew"
date: 2026-08-29T12:05:00+08:00
draft: false
tags: ["rl", "post-training", "grpo", "dr-grpo", "normalization"]
categories: ["AI"]
summary: "R1-Zero's “length and reward rising together” has been treated as the signature curve of emergent reasoning, but Sea AI Lab's Dr.GRPO paper points out that half of that rise may be the optimizer's own preference: 1/|y| makes the model lean toward writing longer when it answers wrong, and dividing by std lets questions at the difficulty extremes take more of the weight. A mechanistic breakdown of two normalization terms and two biases, plus an implementation-level bias hidden in masked_mean."
---

> This post has a companion [interactive report](/reports/dr-grpo-length-and-difficulty-bias/) (in Chinese), with drill-down worked examples of the biases, a side-by-side comparison of masked_mean implementations, and a decision matrix for choosing a fix.

## An Over-Trusted Curve

The most famous plot from R1-Zero training is the one where response length and training reward rise in lockstep. The community was quick to spin a story around it: the model has learned to think longer, and long CoT keeps taking shape through RL.

Sea AI Lab's Dr.GRPO paper (arXiv:2503.20783) poses a simpler question: on the plateau where reward has stopped rising, why does length keep growing? After pinning down the causes one by one, the paper concludes that this is not the model learning to reason more; rather, **the optimizer keeps pushing the model to generate longer responses**. The driving force comes from two normalization terms in the GRPO objective: they adjust the model's optimization direction in opposite ways for correct and incorrect responses, and their effect on incorrect responses points to one and the same outcome: **when the answer is wrong, the longer the response, the less it gets penalized.**

## Bias 1: 1/|y| — Per-Token Averaging Dilutes the Penalty

The GRPO objective (Equation 3 in the paper) contains an inconspicuous denominator:

$$\mathcal{J}_{GRPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\left(w_{i,t}\hat{A}_i,\ \mathrm{clip}(w_{i,t},1-\varepsilon,1+\varepsilon)\hat{A}_i\right)\right]$$

What $1/|y_i|$ means is this: within a single response, first average the gradient terms over all of its tokens, then add up the $G$ responses. In other words, **the gradient weight each token actually receives is $\hat{A}_i/|y_i|$**. A concrete worked example: sample two responses to the same question, one 100 tokens long and one 1000 tokens long, both with an advantage of absolute value 1 (opposite in sign):

- Correct ($\hat{A}=+1$): each token of the short response is reinforced by $1/100 = 0.01$, while each token of the long response is reinforced by $1/1000 = 0.001$. **Among correct responses, the shorter, the more favored**;
- Incorrect ($\hat{A}=-1$): each token of the short response is penalized by 0.01, while each token of the long response is penalized by only 0.001. **Among incorrect responses, the longer, the less penalized** — a tenfold difference.

This is not an anomaly caused by a math error; it is something the objective writes into every update step: correct answers must be concise; wrong answers may as well be long-winded. Late in training, once reward plateaus, the model's room for improvement on the first half is used up, while the second half keeps pushing the model to generate longer responses — incorrect responses in particular keep getting longer. Fig.5 Plot 2 in the paper's Figure 4 clearly shows this phase: GRPO's length curve keeps rising after reward plateaus.

## Bias 2: Dividing by std — Extreme-Difficulty Questions Take More Weight

The second normalization term sits in the advantage:

$$\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)}{\mathrm{std}(\{R_j\}_{j=1}^G)}$$

Dividing by the within-group reward standard deviation is meant to put advantages on a common scale (a standard RL trick). But the paper points out a key difference: classic RL advantage normalization computes the std over **the whole batch**, with all samples sharing one divisor; GRPO **computes the std separately for each question**, so every question has its own divisor — and different divisors mean different weights.

One more worked example: for question 1, all 8 sampled responses are correct (std≈0.35, $\hat{A}$≈+2.9 on the z-score scale); for question 2, the 8 samples are four correct and four wrong (std≈0.52, $\hat{A}$≈±1.9). The more extreme the question — the all-correct giveaway, the all-wrong dead end — the smaller its std, and the larger its weight after the division. The model spends the most update budget on questions it already gets entirely right (the easy ones) and on questions it cannot solve at all (never answered correctly even once), while questions that are “partly right, partly wrong, right in the zone where learning pays off” get the least weight. This is what the paper calls **difficulty bias**.

A caveat on scope: std normalization is not all downside. Its design intent, amplifying the signal in low-variance cases, is to let the model learn something even from questions it gets entirely right. What the paper criticizes is the unequal weighting caused by **per-question normalization**, not normalization itself.

## Bias 3: Hidden in the Implementation, Not the Formula

The third finding may hit engineers harder. The PPO objective (Equation 2 in the paper) **sums** over the response's tokens, with no step that divides by length. But the paper examined mainstream open-source implementations and found that the losses in frameworks such as trl and verl all contain a `masked_mean` line: it averages using the response's actual token count as the divisor (`loss.sum(dim) / mask.sum(dim)`). In other words, **the formula has no length bias, but the implementation adds one**.

How did this happen? The paper's conjecture traces it to historical practice: in pretraining, all data is packed into fixed-length contexts, and `loss.mean(-1)`, averaging over the context length, is both numerically convenient and stable; it is division by a constant and introduces no bias. Once this idiom was carried over to the RL stage, the divisor quietly changed from “the fixed context length” to “the actual length of each response” — from a constant to a variable, and the effect changed with it. Dr.GRPO's implementation fix changes just one line: replace the divisor in `masked_mean` with a global constant (such as the generation length cap MAX_TOKENS).

The formula's authors and the framework's authors may be running the same set of tests, yet they end up with two different gradients. This kind of “formula–implementation mismatch” is more insidious than paper-level biases: you could read the paper ten times and never catch it; only reading the code reveals it.

## Dr.GRPO's Fix and Its Justification

The fix is simple: remove both normalizations, $1/|y_i|$ and std, so that the advantage only subtracts the group mean:

$$\tilde{A}_i = R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)$$

Appendix A of the paper gives three theoretical justifications: with both terms removed, the objective **reverts to the PPO objective (Equation 2)**, and the advantage is just the baseline-free Monte Carlo return plus an unbiased baseline; **subtracting the mean introduces no bias** — the baseline's expected contribution to the gradient is identically zero, a standard result in policy gradients; and the corrected advantage differs from RLOO (Leave-One-Out) only by a constant factor $G/(G-1)$, which can be folded into the learning rate and does not affect training dynamics.

The controlled experiments are run in the Oat framework on Qwen2.5-1.5B. The results fall into three layers:

- **Training dynamics**: early on, both GRPO and Dr.GRPO show the R1-Zero-style joint rise of reward and length, but after reward plateaus GRPO's length keeps climbing, while Dr.GRPO's length stabilizes (Fig.5 Plot 1&2);
- **Behavior**: on benchmarks, the length of Dr.GRPO's incorrect responses drops significantly — the change also mitigates overthinking (Fig.5 Plot 4);
- **Reproducibility**: after math pretraining, Llama-3.2-3B likewise shows GRPO's length rising together with reward, and Dr.GRPO likewise curbs the continued growth in length (Fig.7). The same recipe also lets the 7B Qwen2.5-Math-7B, using MATH levels 3–5 and training for 27 hours on 8 A100s, reach 43.3% on AIME 2024 — the 7B SOTA at the time, and the whole recipe is extremely simple.

## A Textbook Erratum, and the Division of Labor with the Previous Posts

Section 16.4 of the local textbook hands-on-modern-rl also covers this pair of biases, but its wording deserves a correction: it labels “the advantage gets amplified as std approaches zero” as “length bias”, and attributes length inflation to “dividing by std encourages the model to increase within-group variance”. Checked against the original paper, this account is off on two counts: std causes **difficulty bias** (unequal weights across questions), not length bias; and the direct source of length bias is $1/|y_i|$, which has nothing to do with std and acts on the weight of each token within a response. The two normalizations and the two biases each exist independently — the value of the erratum is that choosing a fix depends on this distinction: removing std does not solve the length problem.

The recent posts in this series happen to tackle this at three different levels:

- **DAPO's token-mean** (§3.3): it changes “average within each sample first, then average over the G samples” to “average all tokens in the batch uniformly by the total token count”, i.e. $\frac{1}{\sum_i|y_i|}\sum_i\sum_t$. All tokens are thus weighted by the total token count, which formally eliminates length bias — but DAPO's advantage keeps the z-score (including std), so it **only solves half the problem**;
- **GSPO**: it uses a single ratio at the sequence level and no longer sums over tokens term by term in the objective, so there is no “divide each token by |y|” averaging step at all — it sidesteps this bias structurally;
- **Dr.GRPO** (this post): it leaves the structure of the objective unchanged and only removes the two variable divisors from the normalization terms.

## My Three Takeaways

First, **an RL training curve must be screened for optimizer-induced false signals before anything else**. When “length keeps rising” is cited by academia as headline evidence that “RL is working” and, at the same time, by engineering teams as an anomaly metric that “calls for intervention”, tracing it back to the root matters all the more. The paper also observes that for Qwen2.5-Math models used with a template, the model's pre-RL ability is first damaged by the template and then rebuilt by RL (§3.3) — another class of factor that needs ruling out: part of the “RL gain” is really just ability that was there all along. Before analyzing an RL curve, rule out, one by one, the biases the optimizer introduces and the artifacts the evaluation setup creates.

Second, **Dr.GRPO and dynamic sampling solve different problems; neither replaces the other**. Removing std normalization corrects weight **distortion** (samples that should not be amplified receive too much weight); DAPO's dynamic sampling (questions with std=0 are simply dropped and resampled) deals with **the absence of gradient signal** (when the responses are all correct/all wrong/all zero, the advantage is still 0 after subtracting the mean, and nothing is learned). Note that after removing std, a zero-signal question is still zero-signal — Dr.GRPO brings no gradient to such questions. One ensures correct relative weights within the samples, the other ensures the samples as a whole carry a learnable signal; a real training system needs both configured at once.

Third, **when checking a normalization, first check “whether the divisor is a variable or a constant”**. The three biases here share one pattern: a normalization that looks like routine practice turns out, on closer inspection, to use a denominator that is not a constant — for $1/|y_i|$ the divisor is the length of each response, for std it is each question's within-group spread, and for `masked_mean` it is the mask sum of the current batch. Going through every mean/divide in the training code and asking “can this divisor change from sample to sample?” is the cheapest taste audit there is.

## Open Questions

- How the reweighting of “extreme-difficulty questions” after Dr.GRPO removes std plays out in very long training runs (tens of thousands of steps): the time window of the paper's 1.5B/7B experiments is relatively short;
- The effective-learning-rate conversion when token-mean (DAPO) and removing the variable divisors (Dr.GRPO) are stacked in the same training stack: there is no public, unified convention;
- A systematic check of “how much of the `loss.mean(-1)` habit from pretraining-era packing still lingers in RL frameworks”: so far there is only the paper's spot check of trl/verl.

## References

- Sea AI Lab / NUS: [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) (the source of this post's formulas, experiments, and bias definitions; v2)
- Local textbook hands-on-modern-rl: Section 16.4 (using the corrected wording) and the appendix DAPO cheatsheet (`docs/en/appendix_code_cheatsheet/dapo.md`, a side-by-side of DAPO's z-score advantage and the token-mean loss granularity)
- RLOO reference: [Back to Basics: Revisiting REINFORCE Style Optimization](https://arxiv.org/abs/2402.14740)
- Previous posts: [Entropy Collapse: What DAPO's Clip-Higher Fixes](/en/posts/dapo-clip-higher-entropy-collapse/), [The Ratio's Wrong Unit: GSPO Goes Sequence-Level](/en/posts/gspo-sequence-level-importance-sampling/), [One Model, Two Probabilities: Cost of Mismatch](/en/posts/tis-train-inference-mismatch/)
