---
title: "The Ratio's Wrong Unit: GSPO Goes Sequence-Level"
date: 2026-08-26T17:40:00+08:00
draft: false
tags: ["rl", "post-training", "grpo", "gspo", "moe"]
summary: "GRPO carries over PPO's token-level importance ratio, which the Qwen team deems a fundamental misuse of importance sampling that leads to irreversible collapse in long-response MoE training. GSPO lifts the ratio, clipping, and optimization all to the sequence level: it clips two orders of magnitude more tokens, yet trains more efficiently. This post unpacks that chain of argument, along with the premises and costs the paper leaves unstated."
---

> This post has a companion [interactive report](/reports/gspo-sequence-level-importance-sampling/) (in Chinese), with drill-down formula breakdowns, a clipping-band comparison, and a dilution-effect calculator.

## The Problem a Patch Exposed

Before mid-2025, training MoE models with GRPO required a dedicated patch.

In the GSPO paper (arXiv:2507.18071), the Qwen team describes the approach they had been using: **Routing Replay**. During sampling, the expert routes activated by the old policy $\pi_{\theta_{old}}$ are cached; when computing the importance ratio, these routes are "replayed" on the new policy $\pi_\theta$, so that the ratio's numerator and denominator run through the same activated network. The patch was needed because of one number they observed: in the 48-layer Qwen3-30B-A3B, after each gradient update, roughly 10% of the experts activated for the same rollout sample under the new policy differ from those under the old policy. Once the routing changes, the per-token probabilities change with it, the token-level importance ratios fluctuate violently, and GRPO simply fails to converge.

The patch works, but its costs are real: extra GPU memory and communication overhead, and forcing the new policy onto the old routes amounts to capping the model's effective capacity. What is more troubling is something else — even with the patch, GRPO on long-response tasks with large models still suffers **irreversible model collapse**. The paper's wording is strong: once collapse occurs, rolling back to a checkpoint, carefully tuning the clipping range, extending the generation length, switching RL queries — none of it can bring the model back.

The token-level importance ratio has been standard since the PPO era, in use for eight years. Why has it suddenly turned pathological in RL for LLMs? GSPO's answer: it did not suddenly fall ill; it was used in the wrong place from the start — this is a misuse of importance sampling, and in the era of small models, short responses, and near-on-policy training, the error was simply masked.

## Why Importance Sampling "Fails"

Let me first write down importance sampling (IS) itself. We want to use samples from a behavior distribution $\pi_{beh}$ to estimate the expectation, under a target distribution $\pi_{tar}$, of a function $f$:

$$\mathbb{E}_{z\sim\pi_{tar}}[f(z)] = \mathbb{E}_{z\sim\pi_{beh}}\left[\frac{\pi_{tar}(z)}{\pi_{beh}(z)} f(z)\right]$$

This identity does not hold by virtue of the ratio itself; it relies on **averaging over many samples**: only when $N \gg 1$ do the fluctuations of the random ratio $w(z)=\pi_{tar}(z)/\pi_{beh}(z)$ average out, so that the estimate converges to the true value. Multiplying a single sample by $w$ provides no correction whatsoever; it merely multiplies $f(z)$ by a high-variance random number.

Now look at what GRPO does. It computes a ratio for every token:

$$w_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}$$

Here $y_{i,t}$ is the $i$-th response's $t$-th token, and $y_{i,<t}$ is the context preceding it. $\pi_{\theta_{old}}(\cdot|x,y_{i,<t})$ is the next-token distribution at step $t$, and **only one sample** is drawn from it: $y_{i,t}$. In other words, every ratio uses a single sample to "correct" a single-step distribution — by the criterion above, this does not constitute an IS correction at all; it merely injects a per-token random weight into the gradient. The longer the sequence, the more random weights accumulate; the larger and sparser the model (MoE), the higher the variance of each individual weight. Clipping then zeroes out the gradients of tokens that fall outside the range, and the noise is amplified into "randomly wiping out the learning signal of a subset of tokens". Chapter 6 of the RLHF Book gives two concrete descriptions of the failure as it plays out: a single token with a large ratio can dominate the entire update; and tokens within the same response are clipped independently of one another, shredding the learning signal.

The GSPO paper distills the crux into a single sentence: **the unit of the optimization objective should match the unit of the reward**. In RLVR, the reward is assigned to the entire sequence (1 point for a correct answer, 0 for a wrong one), whereas GRPO's correction and clipping both happen at the token level. With the units mismatched, the more "conscientious" the correction, the more precisely targeted the damage.

To be honest, I should add one point here: GRPO's formulation is not without theoretical grounding. Treating each token as one action step and decomposing the IS ratio of the whole trajectory into per-step ratios is the standard derivation path in RL. The problem lies at the implementation level, where the per-token ratios are used one by one as independent correction weights and clipped one by one — an approximation that stops being harmless under large vocabularies, long trajectories, and single samples. This is the GSPO paper's critical angle, not an axiom; readers are free to reach their own judgment.

## GSPO: Moving Ratio, Clipping, and Optimization to the Sequence Level

The GSPO objective:

$$\mathcal{J}_{GSPO}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\{y_i\}_{i=1}^G\sim\pi_{\theta_{old}}}\left[\frac{1}{G}\sum_{i=1}^G \min\left(s_i(\theta)\hat{A}_i,\ \mathrm{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_i\right)\right]$$

The difference from GRPO is visible at a glance: the per-token sum $\frac{1}{|y_i|}\sum_t$ outside the min-clip is gone, and each response as a whole occupies just one term in the objective. The advantage is still the group-normalized $\hat{A}_i$ (standardized with the mean and standard deviation of the rewards within the group), shared by the entire response.

The core is the definition of the sequence-level ratio $s_i$:

$$s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}=\exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}\right)$$

This is the geometric mean of the whole-sequence likelihood ratio, equivalent to taking the arithmetic mean of the per-token log-ratios and then exponentiating. The length normalization $1/|y_i|$ cannot be dropped: without it, thousands of ratios get multiplied together, likelihood changes in just a few tokens can make the overall ratio swing violently, and responses of different lengths would need different clipping ranges — there would be no way to set $\varepsilon$ at all. After normalization, the numerical range of $s_i$ is decoupled from sequence length, and a single $\varepsilon$ covers everything. (The hands-on-modern-rl textbook omits this $1/|y_i|$ from its formula when introducing GSPO in Section 16.4; go by Equation 7 in the paper — leave it out and the whole chain of argument breaks.)

The meaning of clipping changes accordingly. In GRPO, clipping zeroes out the gradient of an individual token — the signal is chopped up, but the sample remains; in GSPO, clipping removes an entire response from the gradient estimate — a sample is either used in full or not at all. This aligns, in granularity, with "the reward is given to the whole sequence".

## The Gradient View: Unequal vs. Equal Weighting

Expand the gradients of the two objectives (Equations 8–12 in the paper), and the fundamental difference comes down to one thing: what multiplies each token's $\nabla_\theta\log\pi_\theta$.

In GRPO's gradient, token $t$'s $\nabla\log\pi$ is multiplied by **its own ratio** $w_{i,t}$. After the min-clip, these per-token weights lie in $(0, 1+\varepsilon]$ (when $\hat{A}_i>0$) or $[1-\varepsilon, +\infty)$ (when $\hat{A}_i<0$) — within the same response, the gradient weights that different tokens receive can differ by orders of magnitude, and as they accumulate over training steps, the consequences are hard to predict.

In GSPO's gradient, all tokens of the same response share **the same scalar weight** $s_i$:

$$\nabla_\theta\mathcal{J}_{GSPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G s_i(\theta)\hat{A}_i \cdot \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \nabla_\theta\log\pi_\theta(y_{i,t}|x,y_{i,<t})\right]$$

The tokens' contributions to the update differ only in the magnitude of their own $\nabla\log\pi$; there is no longer any per-token random weight modulating them. The paper describes this as eliminating the destabilizing factor in GRPO.

## Evidence: Three Experimental Facts

The paper's experimental setup is itself worth noting: a cold-start model (obtained by SFT from Qwen3-30B-A3B-Base), with each rollout batch split into 4 mini-batches for gradient updates — a standard off-policy setting, and precisely the setting in which, as the previous post on Clip-Higher put it, "clipping actually starts to do its job" (only then does the ratio deviate from 1). The GRPO baseline's clipping range of 0.2/0.27 was carefully tuned; GSPO uses 3e-4 / 4e-4.

Fact one: **the three-orders-of-magnitude gap in ε is not a typo; it is an inevitable consequence of averaging**. A single token-level ratio can easily shoot up to 1.2; the sequence-level ratio is the exp of the average of the per-token log-ratios, and averaging cancels out positive and negative fluctuations, so $s_i$ naturally stays close to 1, fluctuating only slightly. This can be worked out in detail (what follows is my own estimate; the paper does not report these statistics): suppose that after one mini-batch update, each token's log-ratio drift has, on average, a standard deviation of $\sigma$ (say, 5%), the response length is $L$, and the drifts of different tokens are approximately independent; then the standard deviation of the averaged drift shrinks to $\sigma/\sqrt{L}$ — for $L=2000$, a reduction to about $1/45$. Give such a nearly motionless ratio a clipping band of $\pm 0.2$ and it will never touch the edges; clipping might as well not exist. So GSPO has to tighten the band to a few parts in ten thousand: $0.2 / 3\times10^{-4} \approx 667$ — that is where the three-orders-of-magnitude narrowing comes from. Put another way: a tolerance band of 3e-4 corresponds to allowing an average log drift of only about 0.03% per token. I should also be upfront about how sound this assumption is: in real training, per-token drifts are not independent (a single update raises or lowers the probabilities of similar contexts en masse), so the actual reduction factor falls short of $\sqrt{L}$ — but this only makes the conclusion of Fact two below stronger.

Fact two: **a hundred times more tokens get clipped, yet efficiency is higher**. The band is a thousand times narrower, yet a hundred times more tokens are clipped — this sounds contradictory, but once you multiply two factors together, it isn't (the estimates below are, as above, my own). First factor: relative to their respective fluctuation scales, the band is actually narrower. The fluctuation shrinks by only about 45×, while the band shrinks by 667×; using each ratio's own fluctuation as the yardstick, GSPO's band is about 15× narrower than GRPO's ($\sqrt{L}/667$), so $s_i$ goes out of bounds far more easily than $w_t$ does. Plugging in concrete numbers ($\sigma=5\%$, approximately Gaussian drift): GRPO's 0.2 corresponds to $4\sigma$, so the probability of a single token going out of bounds is a few in ten thousand; GSPO's 3e-4 corresponds to only about $0.27\sigma$ ($0.0003/(0.05/45)$), so each response has a better-than-even chance of being thrown out in its entirety. Second factor: going out of bounds is charged to the whole response. In GRPO, a token that goes out of bounds discards only itself; in GSPO, a single out-of-bounds event removes the entire span of several thousand tokens. Multiply the two factors, and the total number of clipped tokens comes to roughly a hundred times GRPO's — the gap in clipping fractions measured in Figure 2 of the paper is indeed about two orders of magnitude, so the magnitudes line up. The truly counterintuitive part is the second half of the claim: fewer tokens are used for training, yet training efficiency and benchmark performance are better (winning on all three curves: AIME'24 / LiveCodeBench / CodeForces). The paper's interpretation: GRPO's token-level gradient estimates are noisy to begin with, and the portion that happens to survive is no more useful than the portion that gets clipped; GSPO's sequence-level filtering provides a more reliable learning signal — fewer but clean beats more but dirty.

Fact three: **MoE no longer needs Routing Replay**. The sequence likelihood $\pi_\theta(y_i|x)$ is insensitive to routing jitter — when 10% of the experts change, individual token probabilities may shift noticeably, but the likelihood of the whole sequence (a reflection of the model's language-modeling capability) does not change drastically. GSPO cares only about the latter, so simply computing the ratio as usual is enough for it to converge; the patch's memory and communication overhead is saved entirely, and the model is no longer bound to the old routes.

There is also an infrastructure dividend on the side (§5.4 of the paper): in practice, the training engine (Megatron) and the inference engines (SGLang, vLLM) differ in numerical precision, so GRPO has to recompute the old policy's likelihoods with the training engine before it can safely use them; sequence-level likelihoods are far more tolerant of precision discrepancies, so GSPO can optimize directly with the likelihoods returned by the inference engine and skip the recomputation. For partial rollouts, multi-turn RL, and architectures that decouple training from inference, this is a tangible throughput dividend.

## Appendix: How the 3e-4/4e-4 Clipping Range Was Set

At the level of the paper: it isn't stated. The GSPO paper gives only the result (left bound 3e-4, right bound 4e-4), with no selection procedure and no ablation on the clipping range. Why this value rather than 1e-4 or 1e-3 is impossible for outsiders to know; a reasonable guess is that the authors found it by trial and error on their own setup (cold-start Qwen3-30B-A3B, each batch split into 4 mini-batches), but that is only a guess.

At the level of principle (what follows is my inference, not the paper's content): the clipping range's job is to let normal fluctuations through and stop abnormal ones, so it should be set at a small multiple of the typical fluctuation magnitude of $s_i$:

$$\varepsilon \approx k \cdot \frac{\sigma}{\sqrt{L}}$$

Here $\sigma$ is the magnitude of the per-token log-ratio drift after one update, $L$ is the response length, and $k$ is the multiple of the fluctuation you want to let through. Plugging in the illustrative numbers from earlier ($\sigma=5\%$, $L=2000$) gives $\sigma/\sqrt{L} \approx 0.0011$, so the paper's 3e-4 corresponds to $k \approx 0.27$ — letting through only about a quarter of the typical fluctuation, a very strict filter, which squares with the measured "tens of percent of tokens clipped".

This framework also shows that 3e-4 cannot be copied as-is; it is coupled to three things: the degree of off-policyness (the staler the data, the larger $\sigma$, and $\varepsilon$ must be loosened accordingly, or else every sample gets clipped away); response length (the longer $L$, the smaller the fluctuation after averaging, so the same $\varepsilon$ becomes stricter — suitable values for 32k and 2k differ by about 4×); and the desired clipping strength (there is no public data on the optimal clipping fraction, so a reproduction has to run its own sweep). Change the configuration, and this number has to be set again.

## My Three Reservations

First, **the "single-sample" critique applies equally to GSPO itself**. $s_i$ is also a ratio computed from a single sample $y_i$, likewise without any $N \gg 1$ averaging. So GSPO's improvement should not be understood as "restoring the theoretical correctness of importance sampling" — it does no such thing; the paper's IS argument is intuitive, not formal. What it really does is bring three things together: unit alignment (what gets clipped is an entire "bad sample" rather than random tokens), variance normalization ($1/|y|$ squeezes the numerical range onto a common scale), and equal-weight gradients (eliminating the accumulation of unequal per-token weights). All three are engineering arguments — solid, but not theorems.

Second, **the geometric mean is a diluter, and that cuts both ways**. In a 10,000-token CoT, if the probability of one critical token drifts by 50%, its contribution to $s_i$ is only $e^{\log 1.5 / 10000} \approx 1.00004$ — nearly invisible. Against precision noise between the training and inference engines, dilution is a feature (this is exactly where the infrastructure dividend comes from); for detecting "a critical token altered by a policy update", dilution is a loophole. GSPO works properly precisely because it is paired with an extremely narrow clipping band on the order of three parts in ten thousand — making trust-region judgments based on average drift rather than single-point drift. Whether this combination goes blind to certain failure modes on ultra-long CoTs (for example, the policy deviating sharply on only a short stretch of reasoning, which gets let through after averaging) is something the paper does not answer.

Third, **sequence-level clipping gives up on making use of "partially correct" samples**. If a response gets 90% of its reasoning right but miscalculates in the final step, GSPO either uses it whole or throws it out whole. The paper's way out is the GSPO-token variant: using stop-gradient, it combines the sequence-level weight $s_i$ (taking only its value, without backpropagating gradients through it) with the per-token term $\pi_\theta / \mathrm{sg}[\pi_\theta]$; the resulting objective, clipping condition, and theoretical gradient are fully equivalent to GSPO's, but advantages can be customized per token — an opening left for multi-turn RL and fine-grained rewards. At the other end of this line is VAPO: bring the value model back outright to do token-level credit assignment. Sequence-level vs. token-level is not a dispute over right and wrong, but two industrial answers to the old question of "credit-assignment granularity".

## How It Relates to Clip-Higher, and a Detail No One Has Pointed Out

The previous post covered DAPO's Clip-Higher: the pathology is the same — a structural flaw of token-level clipping under off-policy training. DAPO's remedy is to adjust the bounds (raising the upper bound from 0.2 to 0.28 to cut exploratory tokens some slack); GSPO's remedy is to change the unit (the whole response shares a single ratio). The two are not mutually exclusive — Qwen's practice is precisely GSPO stacked with a Dr.GRPO-style bias correction that removes the normalization.

There is one more detail worth pointing out: the GSPO paper's clipping range of 3e-4 / 4e-4 is itself **asymmetric** — the right bound is wider than the left, leaving 33% more room in the reinforcing direction (the ratio breaking upward when $\hat{A}>0$). Isn't that just a sequence-level version of Clip-Higher? The paper does not discuss where this asymmetry comes from, but its direction is consistent with DAPO's logic: let good samples go a little further. Two teams independently prescribed remedies pointing in the same direction for the same pathology — that says more than any single paper's ablation could.

## Open Questions

- The method for choosing $\varepsilon$ has itself not been disclosed: 3e-4/4e-4 comes with results but no ablation, and $\varepsilon \approx k \cdot \sigma/\sqrt{L}$ is only this post's inferred framework; the optimal range of $k$, and the effect of combining it with clip-higher-style asymmetric bounds, all call for public reproduction;
- Whether the dilution effect of the geometric mean systematically lets "abrupt local shifts" slip through on ultra-long CoTs (beyond 32k tokens) needs to be verified with purpose-built experiments;
- The actual effectiveness of GSPO-token paired with turn-level rewards in multi-turn agentic RL is, for now, backed only by Qwen's internal data.

---

> **Update log (2026-08-29)**: Expanded based on feedback from three rounds of reader Q&A — the "Evidence" section now includes the full derivation of the clipping-band arithmetic ($\sigma/\sqrt{L}$ variance reduction, the 667× narrowing, the band being ~15× narrower in relative terms, and the out-of-bounds probability calculation); a new section, "Appendix: How the 3e-4/4e-4 Clipping Range Was Set", was added (the paper does not disclose its selection method; an inferred framework and three couplings); the first open question was rewritten accordingly. All derivations are labeled estimates, not content from the paper.

## References

- GSPO paper: [arXiv:2507.18071v2](https://arxiv.org/abs/2507.18071); the equation numbers and experimental data in this post all come from its v2
- RLHF Book, Chapter 6 (importance sampling background; GSPO/CISPO subsections): [rlhfbook.com](https://rlhfbook.com/); [`book/chapters/06-policy-gradients.md`](https://github.com/natolambert/rlhf-book/blob/main/book/chapters/06-policy-gradients.md)
- hands-on-modern-rl, Section 16.4 (comparison of the GRPO family of improvements; note that its GSPO formula omits the length normalization): [`docs/chapter18_grpo/grpo-family.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter18_grpo/grpo-family.md)
- Previous post: [Entropy Collapse: What DAPO's Clip-Higher Fixes](/en/posts/dapo-clip-higher-entropy-collapse/)
