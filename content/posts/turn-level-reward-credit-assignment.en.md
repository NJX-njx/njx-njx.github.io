---
title: "A 7-Turn Failure: Assigning Blame in GRPO/GSPO"
date: 2026-08-31T12:55:00+08:00
draft: false
tags: ["rl", "post-training", "agentic-rl", "grpo", "credit-assignment"]
summary: "Multi-turn agentic RL has one reward per trajectory. How to split it across turns: the formalism, the credit-assignment spectrum, and GSPO-token's opening."
---

In single-turn RL the whole sequence shares one advantage, and nobody has ever lost sleep over it; in multi-turn agentic RL, the same trajectory can get the turn-1 search right, send the turn-2 query off course, and end with a wrong answer—so how to split a single trajectory reward becomes a real design decision. This post unpacks the formalism (action mask, trajectory probability), the credit-assignment spectrum (ORM/PRM/turn discounting/three middle-ground routes), and the opening GSPO-token leaves for multi-turn training.

> This post comes with a companion [interactive report](/reports/turn-level-reward-credit-assignment/) (in Chinese), featuring a toggleable ORM/PRM credit-assignment comparison, backward discounting with a γ slider, and a breakdown of the GSPO-token formula.

## A Problem That Doesn't Exist in Single-Turn Training

Consider a five-turn trajectory: turn 1's search hits the key evidence; turn 2 drifts off course while rewriting the query and brings back irrelevant content; turn 3 corrects course and gets halfway back; turn 4 makes a calculation error; turn 5 gives a wrong answer. The environment returns just one number: failure, reward = 0.

Single-turn RL doesn't have this problem—the entire response shares one advantage, and "all right or all wrong" is taken for granted. In the multi-turn setting this crude logic breaks down immediately: turn 1 clearly did the right thing, so why should it be penalized along with turn 4? On top of that there is a further engineering layer: the trajectory mixes two kinds of tokens, those the model generated itself (thinking, tool-call arguments, the final answer) and those returned by the environment (search results, code output), and the latter should never enter the gradient at all. So "how does turn-level reward get into GRPO/GSPO" is really three questions rolled into one: **which tokens count as actions, how much advantage each turn gets, and which tokens that advantage gets multiplied onto**. This post takes them apart in that order.

## First, Pin Down the Object: Trajectory Probability and the Action Mask

In the formalization of multi-turn trajectories (the textbook adopts the POMDP formulation from the AppWorld paper, arXiv:2504.11536), the full state is $z_t = [s_0, c, x_{1:t}]$: $s_0$ is the hidden initial environment state (database snapshot, REPL state), $c$ is the task context, and $x_{1:t}$ is the entire token history so far—both what the model generated and what the environment returned. The model sees only the text part, which is why it is called partially observable.

The chain-rule factorization of the trajectory probability has one key detail:

$$\rho_\theta(x \mid s_0, c) = \mathbb{I}(s_0, x) \prod_{t \in a(x)} p_\theta(x_t \mid c, x_{1:t-1})$$

The product runs **only over $a(x)$, the set of positions of model-generated tokens**; the observation tokens returned by the environment are determined by the environment dynamics and are not part of the product. Translated into the loss, this is the action mask: a 0/1 vector the same length as the trajectory, with model tokens marked 1 and the prompt and tool returns marked 0. SearchR1's implementation makes this very explicit: the contents of `<think>` and `<search>` take part in training, while the `<information>` segments returned by the retriever are masked out entirely. Letting environment tokens into the gradient as well would amount to forcing the model to "learn to predict which web pages the search engine returns", contaminating the policy gradient.

This sets the skeleton of this post's answer: turn-level reward enters training **not by weighting environment tokens, but by constructing a separate advantage for each turn's model tokens**.

## Three Layers of Signal: How Reward Travels from Trajectory to Weights

The textbook splits the signal flow into three layers, and this layering is the map for understanding the whole spectrum of approaches:

1. **Trajectory reward** $R(\tau_i)$: the raw scalar given by the environment—"did this episode succeed or not";
2. **Step advantage** $A_{i,t}$: breaks the final outcome down into per-turn scores, answering "should turn $t$ be rewarded"—this is the main battleground of credit assignment;
3. **Token gradient**: $A_{i,t}$ is multiplied onto the log-probs of that turn's action tokens, $\Delta\theta = A_{i,t} \sum_{k \in a_{i,t}} \nabla \log \pi_\theta(y_k \mid \cdot)$.

What "putting turn-level reward into GRPO" actually comes down to is layer 2: GRPO's within-group comparison naturally works at the trajectory level (ranking multiple trajectories for the same problem against each other), so to get down to the turn level you have to define $A_{i,t}$ yourself and feed it in.

## Why the Naive Approach Fails: A Controlled Experiment

The textbook's Mini Agent Loop experiment gives the most convincing set of numbers. The task: look up Earth's radius → compute the equatorial circumference → verify → answer. Construct a "bad" trajectory: **only step 2 is wrong** (π taken as 3), steps 1 and 3 are both correct, and the final answer is wrong.

- **ORM (final outcome only)**: on failure, every step's credit is 0—because $0 \times \gamma = 0$, after the zero signal propagates backward every step gets "no signal"; if you switch to "-1 on failure", the correct search in step 1 receives a penalty of -0.857. Either the innocent get blamed, or everyone simply goes silent.
- **PRM (each step scored independently)**: correct steps get positive scores and wrong steps negative ones; the discriminability is 19 times that of ORM, and at the same number of steps the task success rate is about 30 percentage points higher. The cost is that every step has to be evaluated—which in real settings means annotation cost or training yet another scoring model.

PRM is too expensive and ORM too coarse; the middle-ground routes of 2025–2026 fall into three categories by "where the signal comes from", and all of them are essentially looking for a compromise between the two.

## The Credit-Assignment Spectrum: Three Routes Between ORM and PRM

**Route 1: compare actions from the same state (state-anchored).** GiGPO's idea is that if multiple rollouts within a group converge on some state $s$, you compare "the subsequent returns of the different actions taken from the same situation", and the within-group relative score is the step advantage. Its refined version, HGPO, points out that the same page is not the same context (depending on whether a constraint was missed earlier, the same "add to cart" can mean something completely different), so it stratifies the comparison groups by $k$-step history and fuses the strata with weights. This route needs no extra annotation at all, but it depends on states being able to collide—web tasks are naturally friendly to it, while in open-ended dialogue collisions almost never happen.

**Route 2: redistribute the final reward back to each step (progress redistribution).** SPA-RL trains a progress estimator under the constraint that the per-step contributions sum back to the final reward ($\hat R = \sum_t \hat c_t \approx R$): the contributions along a successful trajectory sum to nearly 1 and along a failed one to nearly 0, which forces the model to assign positive contribution to the steps that actually push the task forward. IGPO is cheaper: it directly uses "the increase in the model's log probability of the ground-truth answer" as the turn-level reward—if after turn $t$ the model is more confident in the correct answer (log-prob rises from -4.0 to -1.5), it records $r^{IG} = +2.5$; if the model was led astray, it records a negative value. What it yields is a ready-made turn-level scalar that can go straight into discounted accumulation, with no Monte Carlo value estimation needed; the cost is a dependence on high-quality reference answers, and questions with multiple valid answers get unfairly penalized.

**Route 3: change the optimization granularity (the step-aligned paradigm).** StepPO and Turn-PPO take the most radical view: existing algorithms inherit RLHF's token-centric paradigm, but an agent's decisions are naturally turn-grained, so there is a granularity mismatch between token-level optimization and turn-level decisions. Turn-PPO uses PPO instead of GRPO for multi-turn training, on the grounds that trajectory-level within-group advantages have too much variance, and switches to turn-level advantage estimation—each turn has its own advantage, shared among the tokens within that turn. This route is orthogonal to all the methods above: the others change where $A_{i,t}$ comes from; this one changes the basic unit of optimization.

Whichever route you take, multi-turn RL has one more problem, along a purely temporal dimension: **the earlier the mistake, the greater the responsibility** (if step 1 goes wrong, every later step unfolds on top of the error). Both the textbook and engineering implementations handle this with backward discounting: working back from the final outcome, the earlier the step, the larger the discount—for a 7-turn interaction in which only the last turn has reward 1.0 and $\gamma = 0.9$, the returns assigned to the turns are $[0.531, 0.590, 0.656, 0.729, 0.810, 0.900, 1.000]$.

## Where the GRPO/GSPO Family Sits on This Map

This post settles the loose end left by the [GSPO post](/en/posts/gspo-sequence-level-importance-sampling/). GSPO's principle that "the unit of optimization should match the unit of reward" runs into real tension in multi-turn settings: GSPO's objective treats the entire response as a single action, which suits outcome-level rewards; multi-turn RL wants turn-level credit assignment, which amounts to demanding a granularity finer than the sequence. The GSPO-token variant that §4.3 of the GSPO paper designs for exactly this has a very telling formula:

$$s_{i,t}(\theta) = \mathrm{sg}[s_i(\theta)] \cdot \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\mathrm{sg}[\pi_\theta(y_{i,t}|x,y_{i,<t})]}$$

$\mathrm{sg}[\cdot]$ is stop-gradient (it takes only the value and passes no gradient back). The beauty of this construction: numerically, $s_{i,t}$ is identically equal to the sequence-level ratio $s_i$, so the objective, the clipping condition, and the theoretical gradient are fully equivalent to GSPO's; yet the gradient expression now contains a "per-token customizable advantage slot". Set all tokens of the same turn to the same $A_{i,t}$ and you get exactly turn-level credit assignment—this is a formal opening the algorithm layer leaves for multi-turn training, not an engineering hack.

The GRPO family takes the other, symmetric route: it leaves the mechanism of comparing multiple trajectories within a group untouched and outsources the construction of $A_{i,t}$—whether you hand it trajectory-level comparisons (the default) or turn-level scores (ORM + discounting / PRM / IGPO / SALT), it multiplies them in all the same. So **in implementation terms, "turn-level reward in GRPO" is not a change to the algorithm but a change to the advantage constructor + making sure the action mask is clean**. The combination of the two routes at the paper level (GSPO-token × turn-level $A_{i,t}$) currently has no public ablation—a genuine gap.

## My Three Takes

First, **the number of turns sets the tier; the algorithm can't.** The textbook's selection guide tiers by turn count: for 3–5 turns, plain ORM/GRPO is enough (episodes are short, so sparsity isn't fatal); for 5–15 turns, bring in milestone rewards or annotation-free methods like SALT/GiGPO; beyond 15 turns, PRM or progress rewards plus tree search are a must. This tiering matters more than any single technique—when reward density can't sustain learning, switching to a stronger RL algorithm yields zero gain. The five-turn trajectory from the top of this post actually sits right on the boundary between the first and second tiers.

Second, **the link between credit-assignment quality and entropy is more direct than you might think.** Under ORM every step of a failed trajectory gets penalized, and the easiest countermeasure the model learns is to reduce behavioral diversity (take fewer actions, answer earlier)—in the multi-turn setting, entropy collapse is not a by-product but a direct consequence of wrong credit assignment. Conversely, fine-grained signals like PRM or IGPO let the model "keep the right steps and fix only the wrong ones", which is what preserves exploration. In multi-turn training, watching the agent's turn-count distribution and tool-call diversity exposes credit-assignment problems earlier than watching the reward curve.

Third, **four reward types stand between "it runs" and "the credit is assigned correctly".** The textbook divides agentic rewards into four categories: Outcome (is the final result correct), Format (can the action be parsed and executed), Cost (number of turns, API spend), and Process (does the step advance the task). When getting started, people usually implement only the first two, to make sure training runs, and Process is exactly what the credit-assignment layer needs to make dense. But many engineering setups also fold the Format penalty into the outcome signal and assign credit for both together—a turn with a format error and a turn with a reasoning error bear the same blame in the trajectory, and almost no public report teases this kind of conflation apart.

## Open Questions

- There is no public ablation of how GSPO-token combined with turn-level $A_{i,t}$ performs on multi-turn agentic tasks—whether Qwen has validated it internally is impossible to know;
- The $\gamma$ of turn-level discounting has not been systematically swept in multi-turn LLM training (the textbook example uses 0.9, which comes from classical RL convention rather than an ablation);
- The mix and isolation of the four reward types (outcome/format/cost/process) at the credit-assignment layer lack public controlled experiments—most reports give only final metrics, not the credit-assignment structure.

## References

- Textbook [hands-on-modern-rl](https://github.com/walkinglabs/hands-on-modern-rl): 22.2 Multi-turn RL formalization (AppWorld POMDP, action mask, the four reward types) and 22.3 Trajectory credit assignment (three signal layers, the ORM/PRM experiment, three middle-ground routes, turn-level discounting, selection tiers), in [`docs/chapter22_agentic/formulation.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter22_agentic/formulation.md) and [`docs/chapter22_agentic/credit-assignment.md`](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/docs/chapter22_agentic/credit-assignment.md)
- GSPO paper: [arXiv:2507.18071](https://arxiv.org/abs/2507.18071) §4.3 (the GSPO-token formula and the equivalence argument)
- AppWorld: [AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://arxiv.org/abs/2407.18901); original ORM/PRM framework: [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman et al., 2023)
- Middle-ground routes: SALT [2510.20022], GiGPO [2505.10978], HGPO [2602.22817], SPA-RL [2505.20732], IGPO [2510.14967], Turn-PPO [2512.17008], StepPO (the ID given in the textbook points to a different paper; the original could not be verified) (arXiv IDs checked against arXiv)
- Previous posts: [The Ratio's Wrong Unit: GSPO Goes Sequence-Level](/en/posts/gspo-sequence-level-importance-sampling/), [Entropy Collapse: What DAPO's Clip-Higher Fixes](/en/posts/dapo-clip-higher-entropy-collapse/)
