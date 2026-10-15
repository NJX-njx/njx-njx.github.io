---
title: "Entropy Collapse: What DAPO's Clip-Higher Fixes"
date: 2026-08-26T14:30:00+08:00
draft: false
tags: ["rl", "post-training", "grpo", "dapo", "rlvr"]
categories: ["AI"]
summary: "When naive GRPO is used to reproduce R1, entropy collapses rapidly and samples within a group converge; DAPO's Clip-Higher merely raises the upper clipping bound from 0.2 to 0.28. Behind that one number lies a structural asymmetry of PPO-Clip on the probability axis. This post unpacks the failure chain, the mechanism, the evidence, and the limits."
---

> This post has a companion [interactive report](/reports/dapo-clip-higher-entropy-collapse/) (in Chinese), with drill-down breakdowns of the formulas and charts of the ablation data.

## A Scene from a Training Run

In early 2025, many teams were doing the same thing: reproducing DeepSeek-R1's training with GRPO. The DAPO paper (arXiv:2503.14476) reports the authors' own starting point—running naive GRPO on the Qwen2.5-32B base model scored only 30 on AIME 2024, while DeepSeek's R1-Zero-Qwen-32B scored 47.

The gap did not open up gradually; it was a very specific way of dying: during training, the actor model's entropy drops rapidly, the responses sampled as a group for the same problem grow more and more alike, and in the end they are nearly identical. The paper calls this **entropy collapse**. Sampling loses its diversity, rewards within the group converge, GRPO's advantages all approach zero, and the gradient signal extinguishes itself. Training is still running, but it can no longer learn anything.

The first remedy DAPO prescribes is called **Clip-Higher**: raise PPO-Clip's upper clipping bound ε from the default 0.2 to 0.28 and leave the lower bound at 0.2. The change is a single number. Why can this one number stop the collapse? The answer lies in a structural property of PPO-Clip that is rarely spelled out.

## First, Pinning Down "Entropy"

This whole post revolves around a single word—"entropy"—so let me pin down its definition first. The entropy here is exactly the entropy of information theory, and the object it applies to is **the probability distribution over the entire vocabulary that the model outputs each time it generates a token**:

$$H = -\sum_{x \in \text{vocab}} p(x) \log p(x)$$

The flatter the distribution (many tokens with similar probabilities), the higher the entropy, and the more the model "can't make up its mind"; the sharper the distribution (one token takes 0.99 and the rest split 0.01), the lower the entropy, and the more "certain" the model is. A numerical example: suppose the vocabulary has only 3 tokens; the distribution [0.9, 0.05, 0.05] has an entropy of about 0.39 nat, while the uniform distribution [1/3, 1/3, 1/3] has about 1.10 nat. Sample 10 times from the former and you get the same token 9 times; samples from the latter spread out evenly—**entropy is the mathematical measure of "sampling diversity"**. The "actor model entropy" that the DAPO paper monitors is the average of these per-step distribution entropies over rollout generation.

Why it is the lifeline of RL training: all of GRPO's training signal comes from the within-group reward variance, and that variance comes from sampling diversity. With high entropy, 16 responses sampled for the same problem all differ, some right and some wrong, and the advantages are nonzero; when entropy collapses, the 16 converge, the rewards are all identical, and the advantages are all zero.

Two caveats, so as not to over-read this: first, **higher entropy is not always better**—excessively high entropy corresponds to runaway modes such as gibberish and repeated words; the paper only says entropy should be kept within an "appropriate range", without saying how to determine that range (one of the open questions at the end of this post). Second, **don't conflate entropy with sampling temperature**: entropy is the intrinsic shape of the distribution, while temperature is how you draw from it; it takes parameters like temperature 1.0 at rollout time to cash entropy out as diversity. Clip-Higher affects the former—it keeps training itself from squeezing the distribution ever sharper.

## First, Working Out the PPO-Clip Math

The PPO/GRPO objective, viewed token by token, is:

$$J(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\,\hat{A}_t\right)\right]$$

The symbols, one by one:

- $r_t(\theta) = \pi_\theta(o_t \mid q, o_{<t}) / \pi_{\theta_{old}}(o_t \mid q, o_{<t})$, the **importance sampling ratio**: the current policy's probability for the $t$-th token divided by the old policy's probability at sampling time. $r=1$ means the policy has not changed.
- $\hat{A}_t$, the **advantage**: how much better this token is than average. In GRPO it is computed by within-group normalization: $\hat{A}_{i,t} = (R_i - \text{mean}(\{R\})) / \text{std}(\{R\})$.
- $\varepsilon$, the **clipping range**: how far the ratio is allowed to deviate from 1; the default in the vast majority of implementations is 0.2—a default that comes from the Atari and robotic-control experiments in the original 2017 PPO paper.

On where the advantage comes from, PPO and GRPO take two different paths; don't mix them up when reading the literature:

- **PPO**: relies on a learned value function (the critic) and estimates the advantage token by token with GAE: $\hat{A}_t = \sum_{l}(\gamma\lambda)^l \delta_{t+l}$, where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. If the critic is accurate, the advantage has low variance; if the critic is off, the whole training run goes off with it.
- **GRPO**: no critic; it samples G responses for the same problem and normalizes with the within-group rewards. In essence it uses the group mean as the baseline—this is REINFORCE's variance-reduction trick, except that the baseline doesn't have to be learned; it is computed directly from the samples in the same group.

The GRPO route has three direct consequences, all of which come up later. First, **identical rewards across a group yield all zeros**: when all G responses are right or all are wrong, the mean equals each response's own reward, every advantage is 0, and the problem contributes no gradient at all. Second, **dividing by std carries an implicit weighting**: groups with a small std (say, only 1 of 16 correct) get their weight amplified by the normalization, and the difficulty bias that Dr.GRPO criticizes comes from this term. Third, **the advantage is computed per whole response and shared by every token in it**: in a wrong response, the first 80% of the reasoning steps may well be correct, yet they receive the same negative advantage as the final step where the calculation went wrong—GRPO by itself has no token-level credit assignment.

How the $\min$ behaves depends on the sign of $\hat{A}$—this is the key to understanding everything:

- **$\hat{A} > 0$ (good token, to be reinforced)**: when $r > 1+\varepsilon$, the clipped branch is selected, the objective becomes a constant, and **the gradient is zero**. In other words, each update step can raise a token's probability to at most $1+\varepsilon$ times its value at sampling time; beyond that, no gradient is given.
- **$\hat{A} < 0$ (bad token, to be suppressed)**: symmetrically, each step can push the probability down at most to $1-\varepsilon$ times its value at sampling time.

There is an easily misread point here that is worth unpacking: **for an advantage of either sign, the clip is one-sided**. Take the case $\hat{A} > 0$ and $r < 1-\varepsilon$: the $\min$ still selects $r \cdot \hat{A}$ (because it is the smaller one). Although the objective's value is small, **the gradient is alive**—$dJ/d\pi > 0$, and the optimizer still pulls this good token's probability back up with the full slope $\hat{A}$. A small value does not mean suppression: **the value is the outcome; the gradient is the action**. What the $\min$ does is **withdraw an incentive**, never **apply a reverse incentive**. So the complete structure is: for good tokens, only the incentive to rise is capped (the plateau on the right); for bad tokens, only the incentive to suppress is capped (the plateau on the left); correction in the opposite direction always goes through with the full gradient.

The core mechanism chart in the interactive report is drawn on exactly this logic. How to read it: the horizontal axis is the ratio $r$, the vertical axis is this token's contribution to the objective, $J(r)$, and $\hat{A}$ is normalized to 1. For $r < 1+\varepsilon$, $J(r)=r$ is a rising straight line (the gradient is alive); once $r > 1+\varepsilon$, it becomes a horizontal plateau (the gradient stops there).

Chapter 6 of the RLHF Book also makes a point of noting that the clip is not always at work. Only when the ratio actually goes out of bounds and the $\min$ happens to select the clipped branch is the gradient cut off. The clip fraction is the metric that monitors exactly this. Keep this in mind; it will come in handy later.

## Symmetric Clipping, Asymmetric Effects

Now for some arithmetic. Let $\varepsilon = 0.2$ and $\hat{A} > 0$, and consider two tokens:

- A token with $\pi_{\theta_{old}} = 0.9$: the upper bound on its probability is $0.9 \times 1.2 = 1.08$. A probability cannot exceed 1, so **it is effectively unconstrained**—in one round it can be pushed from 0.9 to above 0.99.
- A token with $\pi_{\theta_{old}} = 0.01$: its upper bound is $0.01 \times 1.2 = 0.012$. **An absolute increment of 0.002**—it barely moves.

First, let me pin down "one round" precisely: it means one "sample → update" cycle. After sampling, $\pi_{\theta_{old}}$ is frozen as a snapshot, and then multiple gradient updates are run on this batch of data (16 for DAPO, paper §4.1); during this time $\pi_\theta$ gradually drifts away from the snapshot and $r$ starts drifting from 1. Once $r$ crosses $1+\varepsilon$, that token's gradient is cut off, and the remaining updates in this round no longer push it. So the precise meaning of "an upper bound of 0.012" is **the point where the incentive vanishes**, not a hard clamp—with a large step, the probability may overshoot it in a single step; it's just that once past it, the incentive drops to zero: nothing pushes it further, and nothing pulls it back.

And the budget **refreshes every round**: the next round resamples, $\pi_{\theta_{old}}$ is updated, $r$ resets to 1, and the token gets another $\times 1.2$ budget. So a low-probability token grows along a multiplicative path: $0.01 \to 0.012 \to 0.0144 \to \cdots$—exponential growth in ratio terms, yet in absolute terms it climbs very slowly through the first stretch; a high-probability token hits 1.0 and saturates within a round or two. The suppression direction is symmetric but harsher: with $\hat{A} < 0$, each round can cut the probability down to as little as $\times 0.8$, leaving 0.33 after five rounds, so low-probability tokens are quickly beaten down to near 0—**multiplicative decay on the way down is much faster than multiplicative growth on the way up**.

This is exactly where the problem lies: **clipping is measured on the "ratio" scale, but learning takes effect on the "probability" axis.** With the same 20% tolerance on the ratio, a high-probability token gets close to 0.2 of room in absolute probability, while a low-probability token gets 0.002. The ranges of movement of the two kinds of token differ by nearly a hundredfold.

And "exploration" in RL is carried precisely by low-probability tokens—tokens the current policy is not inclined to pick but that may lead to new solutions. DAPO's measurements bear this out: the tokens actually clipped by the upper bound during training have an average probability below 0.2 (Figure 3a of the paper). In other words, **the tokens the clipping mechanism keeps pinning down are precisely the exploration tokens**; exploitation tokens (tokens the model is already very sure about) are almost never constrained.

Long-CoT settings amplify this bias: a reasoning chain runs to thousands of tokens, an exploration token at a critical turning point can move only 0.2% per round, and before it has a chance to grow, the rest of the distribution has already sharpened further.

## The Closed Loop of Collapse

Put this micro-level bias into GRPO's training dynamics and you get a complete failure chain:

1. Exploration tokens' probabilities can't grow, high-probability tokens keep climbing, the distribution sharpens, and **entropy drops**;
2. A sharper distribution means the responses sampled as a group (say, 16) for the same problem become **more and more alike**;
3. GRPO's advantage is computed by within-group normalization; as within-group rewards converge, $\text{std} \to 0$ and all advantages approach zero;
4. **The gradient signal vanishes**; the model loses its direction of improvement and can only keep reinforcing patterns it is already sure of—back to step 1.

In the RLVR setting this loop is especially lethal, because its entire training signal is the within-group reward variance. The reward is a rule-based 0/1 (+1 for a correct answer, -1 for a wrong one), with no reward model to provide a continuous signal. Once diversity dries up, the signal source is cut off. Entropy collapse is not "a monitoring metric that looks bad"; it is training burning its own fuel.

This also explains the logic behind the ordering of DAPO's four techniques: Clip-Higher prevents the collapse, while Dynamic Sampling (filtering out prompts whose samples are all correct or all wrong) stanches the bleeding once the collapse has already produced zero-gradient groups. One treats the root cause, the other the symptoms; the paper puts Clip-Higher in the first section.

## Clip-Higher: Raising Only the Upper Bound Is Deliberate

What Clip-Higher does is split the symmetric clip into two hyperparameters:

$$\text{clip}\left(r_t(\theta),\ 1-\varepsilon_{low},\ 1+\varepsilon_{high}\right),\quad \varepsilon_{low}=0.2,\ \varepsilon_{high}=0.28$$

The upper bound goes from 1.2 to 1.28, so a low-probability exploration token's room to rise in a single round goes from +20% to +28%; compounded by the multiplicative accumulation of the budget across rounds, the effect is considerable.

The question worth asking is: why not touch the lower bound? The paper is blunt about it—**if $\varepsilon_{low}$ were loosened (say, allowing probabilities to be pushed down to 0.6×), the gradient from negative advantages would, by the same ratio logic, push low-probability tokens straight down to near 0**; those tokens would effectively be deleted from the sampling space, and the collapse would only come faster. We did the multiplicative-decay math earlier: the downward direction is already faster than the upward one, so clipping in the suppression direction should, if anything, be tightened, not loosened.

So Clip-Higher is not some vague move like "opening the door to exploration a little wider"; it is a precise piece of asymmetric surgery: **loosen the restraints on the way up, stay on guard on the way down**.

In terms of results, Figure 2 of the paper shows that after Clip-Higher is applied, the entropy curve clearly recovers and AIME accuracy rises in step. The ablation chain (Qwen2.5-32B, AIME 2024 avg@32):

| Configuration | AIME score |
|---|---|
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (full DAPO) | 50 |

To be honest: Clip-Higher on its own contributes only +2 points, not the largest gain in the chain (Overlong Filtering gives +6, and adding Dynamic Sampling at the end gives +8). But its role isn't to add points—what it fixes is the stall point of "whether training can keep scaling". Once entropy collapses, the three techniques that follow have nothing to work with.

## Two Premises Rarely Spelled Out

**Premise one: the clip activates only when training is off-policy.** Chapter 6 of the RLHF Book points out that in RL practice for language models, many algorithms take only one gradient step per batch—in that case the ratio is identically 1, the clip never triggers, and what actually acts as the regularizer is the KL penalty. DAPO, by contrast, takes 16 gradient steps per rollout (mini-batch 512, paper §4.1): the policy is updated multiple times within a batch, and only then does the ratio deviate from 1 and the clip actually start working. **So the value of the Clip-Higher knob is coupled to how off-policy the training is**: if your training is strictly single-step on-policy, this knob simply does not exist. Conversely, asynchronous RL (where rollout and training overlap and samples are inherently off-policy) is the soil in which it really comes into play.

**Premise two: DAPO also drops the KL term.** First, let me be clear about the KL penalty itself. The KL divergence measures how far the current policy $\pi_\theta$ is from the reference policy $\pi_{ref}$ (usually the frozen SFT/base model):

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{x \sim \pi_\theta}\left[\log \pi_\theta(x) - \log \pi_{ref}(x)\right]$$

Classic RLHF puts it into the objective as a penalty term ($-\beta D_{KL}$), for two reasons: first, **to prevent reward hacking**—the reward model is a leaky proxy, and a policy that strays far enough can find outputs the RM has never seen and blindly scores high; the KL tethers the search to the region of the distribution the RM has seen; second, to preserve the initial model's language abilities and keep formatting from breaking down. The engineering cost is that a frozen reference model has to stay resident in GPU memory, and every token needs a forward pass through both models.

DAPO removes it entirely ($\beta=0$, so the reference model doesn't need to be loaded), with the reasoning given in §2.3 of the paper: the KL penalty was originally meant to "align, but don't stray too far from the initial model", whereas the goal of long-CoT training is **precisely to depart substantially from the initial model**—the base model cannot write long chains of thought or reflect on itself to begin with; these behaviors depend entirely on RL reshaping the distribution, and staying tethered to the anchor amounts to not letting the ship sail. There is also an unstated enabling condition: RLVR rewards are rule-based judgments, so **there is no RM to hack**, and the anti-hacking half of the KL's motivation simply disappears.

Putting the two premises together reveals an implicit gear shift in DAPO: in the classic setup, drift is held in check by the KL—an **absolute anchor** that forbids straying too far from the initial model; once DAPO drops the KL, only the clip remains—a **relative anchor** that forbids each step from straying too far from the previous one but allows the policy to keep moving farther away. The trust-region role has passed from one anchor to the other. Note also that dropping the KL and raising the upper bound point in the same direction—both loosen the restraints on exploration: the KL penalty pulls the distribution toward the reference model token by token, and only with it gone does the policy dare to venture into new regions. The paper does not say outright that the two were designed to work together—this is my inference; but judged by the goal of "preventing collapse", they are indeed fighting in the same trench. Of course, removing the anchor is not free: when training spins out of control, nothing pulls the model back, which is why DAPO relies so heavily on its monitoring dashboard (entropy, length, mean probability), and why it needs substitute stabilizers such as the clip and dynamic sampling.

## Cross-Checking the Implementations

Beyond the mechanism on paper, I checked three implementations, and the details largely line up:

- **rlhf-book**'s `DAPOLoss` in `code/policy_gradients/loss.py`: decoupled clip (`clip_eps_lo=0.2 / clip_eps_hi=0.28`, see `configs/dapo.yaml`), token-level loss normalization (the loss is summed and then divided by the total number of valid tokens across the whole batch, rather than GRPO's original per-sequence mean followed by a mean over sequences), and no KL. The teaching implementation is consistent with the paper.
- **verl's production implementation** (`verl/trainer/ppo/core_algos.py`): `clip_ratio_low` / `clip_ratio_high` are independent config options that fall back to the symmetric `clip_ratio` when unset—that is, DAPO's asymmetric clipping is already a first-class citizen, but the default behavior is still classic PPO. The file also contains dual-clip's `clip_ratio_c=3.0`, which guards against ratio blow-up on negative-advantage samples—a problem in a different direction from the one Clip-Higher solves.
- **Industry corroboration**: MiniMax's CISPO likewise allows asymmetric clipping bounds; VAPO adopts clip-higher directly. This insight is no longer unique to DAPO; it has become an engineering consensus in long-CoT RL.

Let me also mention a fact-checking case: the Chinese-language course hands-on-modern-rl, in its explanation of Clip-Higher, says "a lower bound of 0.8 means a low-probability action can be pushed down to 0.008 at most, almost completely suppressed", conflating the suppressive effect of the lower bound with the room to rise under the upper bound, which can easily lead readers to think it is the lower bound that was changed. The paper's actual mechanism is exactly the opposite: what changes is the upper bound; the lower bound stays at 0.2 precisely because loosening it further would crush low-probability tokens. The course states that it was generated with AI assistance and has not yet been fully reviewed; when reading secondhand material, always check back against the original—this very check turned up one such error.

## Addendum: RLVR and GRPO Are Not on the Same Layer

When reading this post, it is easy to say RLVR and GRPO in one breath, but they are actually two orthogonal axes:

| | Question it answers | Which layer | Other options on the same layer |
|---|---|---|---|
| **RLVR** | Where does the reward come from? | Reward source / training paradigm | Human preferences (RLHF's starting point), learned reward models, verifiable rule-based rewards |
| **GRPO** | Given the reward, how do you update the model? | Policy optimization algorithm | REINFORCE, PPO, DAPO, GSPO |

The two can be combined freely: RLVR + PPO works perfectly well (VAPO is exactly the value-based route running on verifiable rewards); RM + GRPO holds up too (when GRPO came out of DeepSeekMath it was not tied to rule-based rewards, and within-group normalization works for any scalar reward). RLVR + GRPO became the mainstream recipe because the two bring out the best in each other: RLVR's rule-based rewards are cheap, objective, and robust to reward hacking, but they are a sparse 0/1 signal; GRPO's within-group normalization happens to suit 0/1 well—sample a group for the same problem and the ratio of right to wrong naturally forms a baseline, with no need to train a separate critic to estimate "how many points this line of reasoning is worth". Conversely, PPO's critic is hard to train accurately on sparse 0/1 rewards over sequences thousands of tokens long.

Placed back into this post's failure chain, the division of labor between the two is clear as well: RLVR determines **how fragile the signal source is** (0/1 rewards come in only three patterns—all correct, all wrong, or a mix—and the first two give zero gradient outright), while GRPO determines **what the signal depends on** (within-group reward variance, i.e., sampling diversity). All four of DAPO's techniques are patches at GRPO's algorithm layer and leave RLVR's reward source untouched; improvements in the other direction—building more verifiable tasks for RLVR, refining rewards down to the process level (PRM)—are surgery at the reward-paradigm layer, a path orthogonal to the algorithm-layer patches.

## My Assessment and Open Questions

**Assessment**: At its core, Clip-Higher turns the "pace of exploration" from an implicit default left over from history (ε=0.2, from the continuous-control settings of 2017) into an explicit knob geared to long-CoT reasoning. The 0.28 is not theoretically derived; it is an empirical value, but the diagnostic framework behind it—the asymmetry of ratio clipping on the probability axis—has general explanatory power. From now on, whenever you see "training was going fine, then entropy suddenly collapsed", first checking the probability distribution of the tokens held down by the clip should become standard practice.

**Open questions**, in order of how much I care about them:

1. **Entropy has no target range.** The paper's empirical conclusion is that "a slow rise in entropy is beneficial", but it also acknowledges that excessively high entropy corresponds to gibberish and repetition. So should ε_high be adjusted adaptively along with the entropy curve rather than fixed at 0.28? I haven't seen anyone study this systematically.
2. **Order effects in the ablation.** Table 1 is a cumulative chain: Clip-Higher is tested after Overlong Filtering, so its +2 points were measured on a base that already includes earlier changes. What its clean, standalone contribution is remains unknown.
3. **Compatibility with sequence-level clipping.** GSPO lifts the importance sampling ratio to the sequence level (to fix the numerical instability of token-level ratios in MoE models); does the logic of "raising the token-level upper bound" still hold then? A later post in this series will be dedicated to unpacking GSPO.
4. **Robustness under asynchronous training.** In asynchronous RL the ratio distribution drifts further and is more skewed; whether a fixed 0.28 is appropriate across different degrees of off-policyness is not covered by the paper.

## References

- DAPO paper: [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) (primary source for the mechanism, numerical examples, ablation data, and training details)
- RLHF Book Chapter 6 and its companion implementation: `code/policy_gradients/loss.py`, `configs/dapo.yaml` (local repo, including the [book-zh Chinese parallel edition](https://github.com/natolambert/rlhf-book))
- verl production implementation: [`verl/trainer/ppo/core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py) (clip_ratio_low/high, dual-clip)
- Bilibili video: [Analysis of the GRPO loss/objective and its biases (DAPO, Dr. GRPO), BV1LgXbY5EFD](https://www.bilibili.com/video/BV1LgXbY5EFD) (note: this episode mainly covers dynamic sampling and length bias, and Dr. GRPO's standard-deviation issue; it spends very little time on Clip-Higher)
- hands-on-modern-rl `docs/chapter18_grpo/deepseek-dapo.md` (a Chinese-language explainer; the wording in its Clip-Higher section is muddled, as flagged in the main text)
- Notion "Common Language" notes page (the three rounds of industrial reworking of GRPO: DAPO / Dr.GRPO / GSPO)

---

> Update log: 2026-08-26, first version published; the same day, additions based on reader Q&A—the definition and caveats of "entropy", the two sources of the advantage in PPO and GRPO and their consequences, how to read the one-sided clipping of the min (value vs. gradient), the precise meaning of the "single-round budget" and its multiplicative accumulation, the layering of RLVR and GRPO, and a detailed look at the KL penalty and the absolute-anchor/relative-anchor gear shift.
