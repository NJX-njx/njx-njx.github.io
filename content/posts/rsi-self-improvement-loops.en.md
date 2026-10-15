---
title: "“The Model Trains Itself”: RSI and Three Loops"
date: 2026-09-01T12:30:00+08:00
draft: false
tags: ["rl", "post-training", "agentic-rl", "rsi", "self-improvement"]
summary: "The term RSI (recursive self-improvement) became hot again in 2025-2026, but the things it refers to often get lumped together. Starting from a practitioner's dichotomy (change the harness but not the weights vs. change the weights directly), this post breaks “self-improvement” into three levels—the data loop, the reward loop, and the system loop—which differ in their preconditions, how far they have actually paid off, and their failure modes."
---

> This post has a companion [interactive report](/reports/rsi-self-improvement-loops/) (in Chinese), with an interactive diagram of the three nested loops, a three-level exam matrix, and a scissors-gap diagram.

## A Conflated Term

"RSI" (Recursive Self-Improvement) became hot again in 2025–2026, but what people mean by it varies widely: some mean an iterative pipeline that filters rollouts into data and retrains on it, some mean a model rewriting its own training process, and some mean scaling up test-time compute. The term's degree of confusion is directly proportional to the allure of what it promises.

Let me start with a practitioner's framework. In a conversation, Ding Duo, head of post-training at Perplexity, split RSI and continual learning into two routes: **one leaves the model parameters unchanged and optimizes the harness, the loop, and the solving process; the other updates the model weights directly**. His judgment is that the former produces results faster, while the latter may bring deeper changes in capability; and the two routes share three hard problems—controlling cost, avoiding parameter contamination, and proving that what is continually learned is transferable capability. This post will keep coming back to these three problems.

Another common confusion is around the term test-time. Classic TTT (Test-Time Training, Sun et al. 2020) refers to taking gradient updates on the current input at inference time, whereas the "test-time scaling" popularized by s1 refers to lengthening the reasoning chain and adding budget, without touching the weights. When people in an LLM context talk about a "new test-time paradigm", always ask which one they mean—the former actually changes the model; the latter just thinks more. What Ding Duo calls "optimizing the harness/loop" is in fact a third path: leave the gradients alone and change the scaffolding.

## Level One: The Data Loop—Most Mature, and Least Like RSI

The first level is data circulating through the loop: model rollout → filter successful trajectories by verifiable outcomes → mix them into the next round of SFT/RL training → the model gets stronger → roll out again. LoopTool ([arXiv:2511.09148](https://arxiv.org/abs/2511.09148)) has exactly this shape (evaluation uncovers weaknesses → targeted data synthesis → retraining); a more complete industrial example is the WebSailor series (arXiv:2507.02592; the V2 version, 2509.13305, explicitly puts synthetic data + scalable RL in its title): use a strong model to generate browsing trajectories, train with outcome rewards, filter with self-critique, then automatically synthesize harder training tasks. DeepResearcher (2504.03160) reports a phenomenon worth remembering: in this kind of closed-loop training, behaviors such as pre-search planning, information layering, and cross-validation **are not explicitly encouraged by any reward term; they are by-products of RL optimization**.

This level is already industrialized, but strictly speaking it is not "self"-improvement—the choices in the loop are made by a filter designed by engineers, and the model is just a data source. Its ceiling is also clear: the filtering criteria themselves are hackable (inefficient trajectories that "happened to succeed" get kept), and the narrower the data distribution, the faster the loop turns, and the faster the model converges itself toward whatever the current verifier likes.

## Level Two: The Reward Loop—RLSVR Aims to Extend the Self-Verification Dividend to Open-Ended Tasks

The second level is reward circulating through the loop. RLVR's success over the past two years rests on "math and code have deterministic verifiers"; open-ended tasks (writing, research, dialogue) have no verifier and can only rely on preference models or LLM judges, which brings a triple tax: evaluation bias, the judge-capability bottleneck, and inference cost. RLSVR (arXiv:2607.23802) borrows the concept of a "pretext task" from self-supervised learning: **apply a task transformation to the open-ended task to turn it into a self-verifiable form**, so that the reward once again comes from the task itself rather than from an external judge.

The conditions for this idea to hold are worth laying out: it requires that the task admit some fidelity-preserving transformation (getting the transformed task right ≈ doing the original task well), and that the verifier be deterministic in the transformed space. The "task transformation induces self-verifiable rewards" in the paper's title refers to exactly this inducing step. If it holds, open-ended tasks can also tap into the dividends of RLVR's industrialization; if it doesn't—if the transformation loses too much—the self-verifiable reward is optimizing a proxy metric that has come apart from the original task.

What this means for the loops: the reward loop is the most critical of the three levels for RSI. The data loop's filter and the system loop's evaluator both have to answer "what counts as getting better", and self-verifiable reward is currently the only answer that does not depend on a continuous supply from humans.

## Level Three: The System Loop—RSI Proper, Stuck on Oversight

Only the third level is RSI proper: the model (or what it produces) improves the things used to improve it—better assistants, better data, better evaluations—forming a recursion. Leike et al.'s 2018 research agenda for reward modeling (arXiv:1811.07871) already has a section explicitly discussing recursive reward modeling: use AI to help humans evaluate AI outputs, so that evaluation capability grows along with whatever is being evaluated.

The prerequisite for this level is scalable oversight (2211.03540, cited in Chapter 30 of the hands-on-modern-rl textbook as a core direction in alignment): once model capability exceeds human evaluation capability, who verifies that an improvement is real? Every round of the loop makes selections, and once the selection criteria are hackable, **the faster the loop turns, the faster it collapses**—a small-scale version is already visible in the data loop (metric inflation), and the system loop amplifies it to the entire pipeline. This is also why "proving that what is continually learned is transferable capability" is on Ding Duo's list of hard problems: improvement inside the loop is often "getting stronger on this loop's evaluations", which does not necessarily hold outside the loop.

## The Three Levels Share One Exam

Putting the three levels together, each of Ding Duo's three hard problems maps onto concrete mechanisms:

- **Cost**: the data loop's cost is rollout and filtering (every iteration samples in full; we did the rollout = prompt × G math earlier in the series); the reward loop's cost is constructing and maintaining the task transformations and verifiers; the system loop's cost is basically impossible to estimate for now—because no instance has yet run the full pipeline end to end.
- **Parameter contamination**: only level three is involved. Continual gradient updates let new knowledge interfere with old capabilities, and the contamination accumulates with the number of iterations—this is one reason the "don't change the weights" route is currently underrated: optimizing the harness and loop doesn't contaminate parameters, and the gains are immediate; engineering work like DeepSeek V3.2 (self-added verifiers, switching the sampler to mHC) is this route paying off.
- **Transferability**: none of the three levels has a good evaluation protocol. Evaluation inside a loop is contaminated by the loop itself (the training target and the evaluation target share the same source), and cross-loop evaluation is too expensive.

## My Three Takes

First, **the "don't change the weights" route is currently the main way RSI actually pays off, and theorists underrate it**. Scaffold optimization (tools, context engineering, verifier internalization, sampling procedures) doesn't change parameters, doesn't contaminate weights, and can be rolled back, and at the current stage its return per unit of compute is higher than that of changing the weights directly. Expelling it from the ranks of "self-improvement" is a misallocation of resources caused by terminological purism.

Second, **the three levels are bottlenecked in different places**. The data loop is stuck on the hackability of its filter; the reward loop on the fidelity of the task transformation; the system loop on oversight (without a solution to scalable oversight, recursion can only be recursive self-deception). Discuss "RSI progress" without specifying the level, and the conclusions are bound to be vague.

Third, **loop speed is a double-edged sword, and an amplifier of safety problems**. Every paper on self-improvement talks about convergence speed, and few talk about runaway speed at the same time; yet the two share the same structure—the loop compounds "small selection biases". For self-improving systems, what should be monitored is the **scissors gap** between in-loop and out-of-loop metrics, not any single metric.

## Open Questions

- Whether there is a systematic way to construct task transformations for open-ended tasks (rather than hand-tuning them task by task); RLSVR only provides the paradigm;
- Where the capability ceiling of the "don't change the weights" route lies—whether there are capabilities that can only be obtained by changing the weights; there is no clean evidence yet;
- An evaluation protocol for cross-loop transferability: who evaluates, with what data, and how to keep the evaluation from being contaminated by any one of the loops.

## References

- Notes on a conversation with Ding Duo: 《在模型持续进步的时代，创业者应该站在哪里》§20 (the two routes for RSI and continual learning, and the three hard problems)
- RLSVR: [arXiv:2607.23802](https://arxiv.org/abs/2607.23802) (abstract and core claims)
- WebSailor: [arXiv:2507.02592](https://arxiv.org/abs/2507.02592), [V2 2509.13305](https://arxiv.org/abs/2509.13305) (titles and positioning within the series); DeepResearcher [arXiv:2504.03160](https://arxiv.org/abs/2504.03160)
- Recursive reward modeling: Leike et al. [arXiv:1811.07871](https://arxiv.org/abs/1811.07871); Scalable Oversight [arXiv:2211.03540](https://arxiv.org/abs/2211.03540) (hands-on-modern-rl textbook: [`docs/chapter30_alignment_failures/scaling-and-defenses.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter30_alignment_failures/scaling-and-defenses.md))
- Classic Test-Time Training: Sun et al. [arXiv:1909.13231](https://arxiv.org/abs/1909.13231); test-time scaling: s1 [arXiv:2501.19393](https://arxiv.org/abs/2501.19393)
- Previous post: [Who's to Blame When a 7-Turn Interaction Fails: Reward Credit Assignment for Multi-Turn Trajectories](/en/posts/turn-level-reward-credit-assignment/) (previous post: within-episode credit assignment; this post: cross-episode loops)
