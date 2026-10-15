---
title: "One Vote or Ten: token-mean vs sample-mean"
date: 2026-08-30T12:05:00+08:00
draft: false
tags: ["rl", "post-training", "grpo", "dapo", "dr-grpo"]
summary: "In the same batch, which contributes more to the gradient: a 500-token response or a 5000-token one? GRPO says they count equally, DAPO says the long one counts ten times as much, and both can justify their answer. This is a dispute over loss aggregation: sample-mean gives each response one vote, token-mean gives each token one vote. The DAPO paper (§3.3) says long-CoT settings must switch to token level, but the switch pushes the length skew from the response level to the question level. This post unpacks the three formulas, the two camps' arguments, and a tradeoff few people talk about."
---

> This post has a companion [interactive report](/reports/token-mean-vs-sample-mean/) (in Chinese), with a weight calculator that switches among the three vote-counting rules and a side-by-side breakdown of the formulas.

## First, Count the Votes

The same training batch contains two responses: one is 500 tokens, the other 5000 tokens. How much does each contribute to this step's gradient?

- **GRPO (sample-mean)**: equally. First average over the tokens within each response, then average over the $G$ responses — every response, long or short, gets one equal vote;
- **DAPO (token-mean)**: the long one contributes ten times as much. All tokens in the batch are flattened together and divided by the total token count — one vote per token, so the 5000-token response gets 5000 votes;
- **Dr.GRPO**: formally also one vote per response, but the divisor changes from "each response's actual length" to a global constant, which changes what a vote means (that was the topic of the previous post).

All three answers are backed by papers, and all three camps are working on **the same spot in the same expression** — the loss's aggregation divisor. This knob is orthogonal to the advantage normalization (the std half) covered in the previous post: the previous post's question was "how is the advantage computed?"; this post's question is "once it is computed, by what rule are the tokens' losses added together?"

## Three Formulas Side by Side

The GRPO objective (DeepSeekMath's Equation 3, with the KL term omitted):

$$\mathcal{J}_{GRPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}\right], \quad \ell_{i,t}=\min\big(w_{i,t}\hat{A}_i,\ \mathrm{clip}(w_{i,t},1-\varepsilon,1+\varepsilon)\hat{A}_i\big)$$

Here $o_i$ is the $i$-th response, $|o_i|$ is its length, $w_{i,t}$ is the probability ratio between the new and old policies for that token, and $\hat{A}_i$ is the advantage shared by the entire response. Note the structure of the sums: $\frac{1}{|o_i|}\sum_t$ first averages each response **on its own** into a single number, and $\frac{1}{G}\sum_i$ then averages those $G$ numbers. The two responses each cast one vote, regardless of length.

DAPO changes just one spot (Equation 12 of the paper, the part highlighted in red):

$$\mathcal{J}_{DAPO} = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\ell_{i,t}\right]$$

The inner $\frac{1}{|o_i|}$ is gone, the two levels of summation are flattened into one, and the divisor becomes the total token count of the whole group. Every token gets exactly the same weight, $1/\sum_i|o_i|$; a response's total weight is proportional to its length.

The third is Dr.GRPO: the same aggregation structure as DAPO (tokens flattened), but the divisor is a global constant (such as the length cap), and the std is removed from the advantage as well — the previous post went into that in detail; this one looks only at the divisor.

## Why DAPO Insists on Switching: Two Places the Penalty Can't Reach

The argument in §3.3 of the DAPO paper is quite concrete and targets the long-CoT setting. Under sample-mean, long samples "contribute disproportionately little to the overall loss" (original wording: disproportionately lower contribution), which leads to two bad outcomes:

First, **reasoning patterns in high-quality long samples don't get learned**. In a correct 5000-token chain of reasoning, each token receives a gradient weight of $1/(G\cdot 5000)$, only one-tenth of what a token in a 500-token short sample gets. What long samples carry is precisely complex reasoning, so under sample-mean, the more important the content, the lower its per-unit weight.

Second, **bad patterns in overlong samples don't get penalized away**. The paper reports an observation: overly long samples often contain low-quality patterns such as gibberish (garbled text) and repeated words. The penalty on these bad tokens is likewise diluted by $1/|o_i|$ — the longer the sample, the more lightly each bad token is penalized. The result is "an unhealthy increase in entropy and response length" (the measured curves in Figures 4a/4b of the paper).

After switching to token-mean, a token carries the same weight whether it sits in a long sample or a short one. DAPO is upfront about the effect: as a result, longer sequences "have more influence on the overall gradient update than shorter ones" (original wording: longer sequences can have more influence). This is not a side effect but a design goal — in the long-CoT era, the gradient should be spent mainly on long responses in the first place.

## A Tradeoff Few People Talk About

But the phrase "long samples get more votes" deserves a pause and a second look: it has two layers of meaning, and DAPO only spells out the first.

The first layer is the response level: within a single response, tokens are no longer diluted by length. This is a direction Dr.GRPO endorses too — the "response-level length bias" that Dr.GRPO criticizes (when correct, shorter responses are favored; when wrong, longer ones are safer) and the "bad tokens don't get penalized" problem that DAPO criticizes are two faces of the same $1/|o_i|$, and the two camps' prescriptions reach the same destination by different roads: equal weight per token.

The second layer is the question level: across the whole batch, **questions with long responses take more of the total weight**. A question on which the model's responses average 6000 tokens and one on which they average 600 tokens differ in batch weight by 10x under token-mean; under sample-mean the two are equal. In other words, token-mean pushes the length skew from the response level to the question level — long questions dominate the gradient. In the long-CoT setting this may be exactly what you want (hard, reasoning-heavy questions tend to have long responses too), but it is a real bias: question weights are no longer uniform; instead, each question is overweighted in proportion to its current length. I have not seen this layer of the tradeoff quantified in any paper (this point is an inference; happy to be proven wrong).

There is also a second second-order effect: DAPO's divisor $\sum_i|o_i|$ is **a random variable that varies from batch to batch**, whereas Dr.GRPO's constant divisor is not. The direct consequence of a random divisor is that the gradient scale fluctuates with the length composition of each batch; the indirect consequence is more interesting — later in training, responses generally get longer, the divisor grows, and each token's effective learning rate decays automatically. In other words, token-mean comes with a built-in implicit learning-rate decay that moves in lockstep with length inflation, a property that neither sample-mean nor a constant divisor has. This is a mechanism-level inference that no paper has measured, but it is a real variable when choosing an aggregation method: what you are choosing is not just a weight allocation but also the dynamics of the gradient scale.

## Both Say "Back to PPO," but Not the Same PPO

An easily overlooked historical fact: PPO's original objective was aggregated per token to begin with (summing or averaging over the tokens of the whole batch, with GAE supplying a per-token advantage). **It is GRPO's sample-mean that is the outlier** — because GRPO's advantage is "one number per response," averaging each response into one number first and then averaging across responses is semantically the tidiest.

Then DAPO and Dr.GRPO both claim to have "restored the PPO objective": DAPO restores **PPO's aggregation scheme** (tokens flattened, but the advantage keeps its z-score, so the std is still there); Dr.GRPO restores **the form of PPO's objective function** (a constant divisor, the advantage reduced to $r-\text{mean}$, the std removed). The two knobs — aggregation granularity and advantage normalization — are independent and can be combined freely: in implementations such as verl, the two really are configured separately (from the implementation-level discussion in the previous post: by default, `masked_mean` in trl/verl adds back the per-response averaging that the formula doesn't have, and auditing whether each divisor "is a variable or a constant" is the cheapest check there is). This also means that "GRPO vs DAPO vs Dr.GRPO" is not three algorithms lined up side by side, but a configuration space of at least 2×2, and the good points known so far are all still being found by trial and error along its edges.

## Which Vote Should Weigh More, and When

Break it down by scenario and the choice isn't mysterious:

- **Short CoT, low length variance**: the two aggregations are approximately equivalent (lengths are similar, so $1/|o_i|$ and $1/\sum|o_i|$ differ by a roughly constant factor), so there's no need to agonize;
- **Long CoT, bad patterns hiding in long samples** (repetition, gibberish, pointless rambling): token-mean is crucial, otherwise bad tokens will never be penalized enough — this is DAPO's core scenario;
- **You want uniform weights across questions** (e.g., question difficulty varies widely and you worry about long questions dominating): sample-mean or a constant divisor is more controllable;
- **You want a stable gradient scale and predictable learning-rate behavior**: a constant divisor (the Dr.GRPO route) is the cleanest; token-mean's random divisor mixes length dynamics into the learning rate.

My take: at its core, the aggregation dispute is about "in what unit the gradient budget is allocated." Under sample-mean it is allocated per response, implicitly assuming every response carries the same amount of information — true in the era of short answers; under token-mean it is allocated per token, implicitly assuming every token carries the same amount of information — closer to the truth in the long-CoT era, but at the cost of letting length decide the weighting across questions. The paper that is really missing is one that makes this budget allocation **explicit**: weights assigned neither per response nor per token, but according to estimated information content or uncertainty. Until then, DAPO's change is the most pragmatic default for long CoT.

## Open Questions

- The effect of token-mean's question-level length skew (cross-question weight ∝ length) on where training ends up has not been publicly quantified;
- Nobody has measured how the implicit learning-rate decay from the random divisor interacts with an explicit LR schedule stacked on top of it;
- For a full 2×2 ablation of the two knobs, aggregation granularity and advantage normalization (sample/token × with std/without std), the public literature only has the points on the diagonal.

## References

- DAPO paper: [arXiv:2503.14476](https://arxiv.org/abs/2503.14476); token-level loss in §3.3 (Equation 12), entropy and length curves in Figure 4
- Dr.GRPO paper: [arXiv:2503.20783](https://arxiv.org/abs/2503.20783); the original statements of the response-level length bias and the question-level difficulty bias
- DeepSeekMath / GRPO: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300); the sample-mean objective (Equation 3)
- Previous post (the one right before this, covering the advantage-normalization half): [Dr.GRPO: Two Normalization Terms, Two Biases](/en/posts/dr-grpo-length-and-difficulty-bias/) (it gave the token-mean section number as §3.2 by mistake; the correct one is §3.3, and that has been corrected in passing)
- The question-level length skew and the random divisor's implicit LR decay are this post's own inferences, each flagged where it appears in the text
