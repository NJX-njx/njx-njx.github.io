---
title: "One Model, Two Probabilities: Cost of Mismatch"
date: 2026-08-27T13:00:00+08:00
draft: false
math: true
tags: ["rl", "post-training", "grpo", "tis", "precision"]
categories: ["AI"]
summary: "Rollouts run on an inference engine and training runs on a training engine, and the same model's numbers don't match across the two: the distribution you actually sample from deviates from the policy you assume, and nominally on-policy training quietly becomes off-policy. This post breaks down where the discrepancy comes from, its two consequences (biased gradients and a deployment gap), the lineage of algorithmic patches, and Sea AI Lab's conclusion tracing the root cause to BF16 precision."
---

> This post has a companion [interactive report](/reports/tis-train-inference-mismatch/) (in Chinese), with a drill-down diagram of the two-engine mechanism, a comparison of the patch lineage, and a bit-field breakdown of the precision formats.

## The Same Hard-to-Explain Collapse

In the second half of 2025, several teams independently reported the same phenomenon: early in GRPO training everything looks normal and reward climbs steadily; then, at some point, it abruptly turns downward until training collapses. There is no gradient spike before the collapse, nothing is wrong with the hyperparameters, and if you rerun with a different batch of training queries, the collapse still comes, just at a different time.

The August 2025 technical report by Defog (Feng Yao, ByteDance Seed), “Your Efficient RL Framework Secretly Brings You Off-Policy RL Training”, offered an explanation that surprised quite a few people: **you think you are doing on-policy training, but the distribution your rollout data actually comes from has already drifted away from the policy you are optimizing**. The problem is not in the algorithm but in the numerics—the probabilities computed by the engine that generates the data don't match those computed by the engine that trains on it.

## Why One Model Computes Two Probabilities

Modern LLM RL frameworks are split into two halves. The rollout side uses an inference engine (vLLM, SGLang), heavily optimized at the kernel level for high throughput; the training side uses a training engine (Megatron, PyTorch FSDP, DeepSpeed), which computes each token's log probability and the gradients. Both sides load the same weights, so in theory they should be the same policy $\pi$.

But because their engineering implementations differ, the engines don't compute exactly the same results. Sea AI Lab's FP16 paper (arXiv:2510.26788) summarizes where the discrepancies come from:

- **Different computation patterns**. Inference generates autoregressively, token by token; training runs a parallel forward pass over the whole sequence under teacher forcing. For the same token string, the different computation processes use different floating-point reduction orders, and errors accumulate differently.
- **Different kernel implementations**. The inference engine's RMSNorm, RoPE, and attention kernels are specially tuned for latency and throughput, and differ numerically, by tiny amounts, from the training-side implementations.
- **Different parallel sharding**. Training and inference use different tensor-parallel and data-parallel sharding, and a different order of cross-GPU reductions changes the floating-point result.
- **MoE top-k gating is sensitive to small changes**. Expert selection is sensitive to tiny fluctuations in the logits, so if the same weights are run once through each of the two engines, the selected experts can differ. For GSPO, covered in the previous post, the 10% expert drift occurs along the temporal dimension of “before vs. after an update”; here we mean the spatial dimension of “same weights, two engines”. Gating sensitivity introduces a discrepancy along each of the two dimensions separately.
- **The precision format's mantissa is too short**. The default BF16 has only 7 mantissa bits, so none of the tiny discrepancies cited above has enough numerical precision headroom, and they end up producing noticeable deviations. This point is the crux of the whole post; more on it later.

So the inference engine actually corresponds to **a numerically different policy** from the training engine's, denoted $\mu$. Data is sampled from $\mu$, but the gradient is computed under the assumption of $\pi$. The boundary between on-policy and off-policy ultimately comes down to the floating-point behavior of kernels.

## Two Consequences: Biased Gradients, Discounted Deployment

Consequence one: **biased gradients**. The theoretical policy gradient requires sampling from the policy being optimized, $\pi$:

$$\nabla_\theta\mathcal{J}(x,\theta)=\mathbb{E}_{y\sim\pi(\cdot|x,\theta)}\left[\nabla_\theta\log\pi(y|x,\theta)\cdot R(x,y)\right]$$

$\pi(y|x,\theta)$ is the likelihood of the response in the training forward pass, $R(x,y)$ is the reward, and the expectation has to be taken under the distribution of $\pi$. But the actual samples come from $\mu$:

$$\nabla_\theta\mathcal{J}_{biased}(x,\theta)=\mathbb{E}_{y\sim\mu(\cdot|x,\theta)}\left[\nabla_\theta\log\pi(y|x,\theta)\cdot R(x,y)\right]\neq \nabla_\theta\mathcal{J}(x,\theta)$$

The distribution underlying the expectation has suddenly changed, and the equality no longer holds. The bias is usually small, but RL is an iterative process that keeps pushing the distribution in one direction, so small differences between $\mu$ and $\pi$ get amplified step by step over training. The paper also observes a phenomenon that can serve as an early-warning signal: in every run that eventually collapsed, the gap between $\pi$ and $\mu$ kept widening before the collapse; in extreme cases, under the same weights, the same token had a probability approaching 1 in one engine and approaching 0 in the other.

Consequence two: **the deployment gap**. Even if the gradient bias in training is fully corrected, the optimization target is still the training engine's probability distribution, while deployment and evaluation run on the inference engine:

$$\arg\max_\theta \mathbb{E}_{y\sim\mu}R \neq \arg\max_\theta \mathbb{E}_{y\sim\pi}R$$

Parameters that are optimal under $\pi$ are not necessarily optimal under $\mu$. This gap is visible in the paper's experiments: the most stable algorithmic patch under BF16 (sequence-level masked correction) reaches a peak training accuracy of 95% and an AIME 2024 score of 34%; under FP16, the numbers are 99% and 39%. On this measure, all the algorithmic patches make zero difference—they only address the bias in the training process and cannot address this deployment gap.

## Algorithmic Patches and Their Limits

At its core, the problem is “sampling from $\mu$ but taking the expectation under $\pi$”, and the textbook fix is an importance-sampling correction: multiply each sample by the train-inference probability ratio $\pi/\mu$. The improvements in the second half of 2025 came mainly in two steps:

**Patch one: token-level truncated correction (Defog)**. For each token, compute the ratio of the training probability to the inference probability, $\rho_t = \pi(y_t|x,y_{<t},\theta')/\mu(y_t|x,y_{<t},\theta')$, truncate it at an upper bound $C$ (3 in the experiments), and multiply it into each term of the GRPO gradient. Note that the denominator $\mu$ is the inference engine's probability, so the training engine has to run one more forward pass over the same sample to obtain the numerator, adding about 25% training overhead (estimated assuming the backward pass costs roughly 2× the forward pass). Result: the collapse is delayed, but training still collapses after peaking at 82% (VeRL) / 88% (Oat).

**Patch two: sequence-level masked correction** (“When Speed Kills Stability”, Liu et al.). The authors point out that the token-level correction yields a biased gradient, and change the correction to compute a single ratio for the entire sequence, $\rho = \pi(y|x,\theta')/\mu(y|x,\theta')$, masking out the gradient of the whole sequence whenever it exceeds $C$. The idea is in the same lineage as GSPO, covered in the previous post (although GSPO's $s_i$ serves only as the quantity being optimized, whereas $\rho$ here is a correction factor). It is the only algorithmic patch that does not collapse under BF16; the price is high variance in the sequence-level ratio and slow convergence, and the deployment gap remains exactly as it was—the peak is only 95%/34%.

What the two patches have in common: both accept the fact that $\mu \neq \pi$ and compensate sample by sample, at the cost of extra compute and added variance, and neither can solve the deployment gap.

## The Root Cause: BF16's 7-Bit Mantissa

Rather than pushing further down the patching path, the Sea AI Lab paper went straight to examining the numerical format.

BF16 and FP16 both use 16 bits but allocate them differently. BF16 uses 8 exponent bits + 7 mantissa bits, giving it a dynamic range on par with FP32: pretraining doesn't have to worry about overflow and doesn't need loss scaling, which is why it became the default. FP16 uses 5 exponent bits + 10 mantissa bits: its dynamic range is much smaller (it relies on loss scaling to prevent gradient underflow), but adjacent representable numbers are 8× as dense as in BF16 ($2^{10}$ vs. $2^7$), so its precision is higher.

The paper's assessment: the requirements change in the RL fine-tuning stage. The dynamic range of weights and activations has already been settled during pretraining, so BF16's FP32-level range no longer serves any practical purpose here, while every kernel discrepancy between training and inference needs FP16's 3 extra mantissa bits to provide enough numerical precision headroom. Once precision is unified to FP16, the two engines' outputs agree numerically on the vast majority of tokens, and $\mu$ and $\pi$ coincide again. Offline analysis shows that under BF16 the sequence-level train-inference mismatch grows exponentially with generation length, while FP16 reduces the mismatch substantially, to about 1/24 of BF16's. In the precision ablations, the “BF16 training + FP32 inference” combination is also fully stable, but inference is 3× slower; using FP16 throughout is the best combination of stability and efficiency.

The most telling result is the next set of experiments: **with FP16, the performance differences between algorithms almost disappear**. GRPO, token-level TIS, GSPO, and sequence-level MIS, which differ markedly under BF16, end up close to one another under FP16, and even the plainest importance-sampling policy gradient, without any correction, achieves the best performance. The reason is not hard to see: once the gap between $\mu$ and $\pi$ is close to zero, training returns to an approximately on-policy regime, and every algorithm meant to “correct the mismatch” loses the thing it was correcting. (One more detail: on VeRL, GSPO's gradients turn into NaN after 1,200 steps, which suggests that algorithm-level improvements may be of no use while the numerical problem remains unsolved.) The engineering routes cited here bear this out too: the SGLang team hand-aligned the kernel implementations of the two compute paths (the November 2025 Unified FP8 work), and Miles/slime aligned training and inference in one go at the system level. Both are effective, but they require deep kernel expertise and a heavy engineering investment, and they are hard to port. The numerical format is the cheapest of all the solutions: it only takes changing a few lines of configuration, while keeping the existing loss scaling mechanism.

## Reading It Alongside the Previous Two Posts

This week's three posts happen to approach the same problem at four levels:

- **DAPO Clip-Higher** (the post before last): assumes training and inference are consistent, and adjusts the clip bounds inside the algorithm to fix entropy collapse;
- **GSPO** (the previous post): moves the optimization objective to sequence-level units, is naturally robust to the discrepancy in $\mu$, and also tolerates MoE routing drift;
- **TIS algorithmic patches** (the first half of this post): accept the fact that $\mu \neq \pi$ and apply importance-sampling compensation sample by sample;
- **FP16 and kernel alignment** (the second half of this post): make $\mu \approx \pi$, eliminating the mismatch directly.

Seen against this arc, GSPO §5.4's observation that “sequence-level likelihoods are highly tolerant of precision differences between training and inference” is no longer just a side benefit—it is a direct algorithm-level response to the train-inference mismatch (TIS) problem.

## My Three Takes

First, **in engineering terms, on-policy is a numerical problem**. The algorithm's definition says “sample from the current policy”; in engineering practice you can only sample from “a numerical approximation of the current policy under some kernel implementation”. Once the two differ by a certain magnitude, the premises of the theoretical analysis no longer hold, and existing alerting mechanisms cannot raise an alarm about it: the training curve looks normal, the loss looks normal, and only the logprob gap between the training side and the inference side is silently diverging. The practical advice is straightforward: make the train-inference logprob gap (or KL) a routine monitoring metric. Its cost is close to zero, and the early-warning value the paper demonstrates is empirically measured.

Second, **BF16's default status deserves to be questioned, but the FP16 conclusion should not be over-generalized**. The evidence covers 1.5B to 30B MoE, two frameworks, LoRA, and two model families, which is solid; the paper itself also states that it does not claim universality: using FP16 on very large models requires managing overflow, and if the industry moves toward FP8 (with only 3–4 mantissa bits), the train-inference mismatch problem will reappear. The current FP8 RL route (LMSYS, NVIDIA) unifies training and inference on FP8 rather than going back to FP16. The precision problem will either be taken seriously again or resurface in another form.

Third, **the deployment gap is the most direct and least studied problem behind all this**. Although the gradient bias can be corrected, and the train-inference mismatch can be eliminated by aligning both the kernels and the precision format on the two sides, the problem that “the trained parameters are not optimal for the deployment engine” has not disappeared; it has only shrunk. When evaluation switches to the inference engine, which policy does the reported score actually belong to? With mixed multi-engine deployment, how should the gap be computed? So far I have not seen a systematic study.

## Open Questions

- Whether the small train-inference discrepancies that remain under FP16 will accumulate back into a problem over ultra-long training runs of thousands of steps;
- The final performance of the unified-FP8 route versus the FP16 route under the same compute budget: there is currently no public head-to-head comparison;
- The deployment gap will still appear, only smaller: when evaluating with an engine different from the training environment, which policy should the score be attributed to?

## References

- Sea AI Lab / NUS: [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788) (the source of this post's experiments, formulas, and data)
- Defog (Feng Yao et al.): “Your Efficient RL Framework Secretly Brings You Off-Policy RL Training”, 2025-08, published on Notion; the original report could not be accessed directly due to technical limitations, so its token-level correction formula is paraphrased from §2.1 of the FP16 paper
- Liu et al.: “When Speed Kills Stability: Demystifying RL Collapse from the Inference-Training Mismatch”, 2025, published on Notion; sequence-level masked correction, paraphrased in the same way
- Engineering routes: [Unified FP8: Moving Beyond Mixed Precision for Stable and Accelerated MoE RL](https://www.lmsys.org/blog/2025-11-25-fp8-rl/) (LMSYS); Miles/slime: All-In-One Solution to Training-Inference Mismatch
- Local textbook hands-on-modern-rl, Appendix B.1: two engineering treatments of the old logprobs produced by the rollout engine (use the results returned by the inference side directly, or recompute them on the training side and check their accuracy)
- Previous posts: [Entropy Collapse: What DAPO's Clip-Higher Fixes](/en/posts/dapo-clip-higher-entropy-collapse/) and [The Ratio's Wrong Unit: GSPO Goes Sequence-Level](/en/posts/gspo-sequence-level-importance-sampling/)
