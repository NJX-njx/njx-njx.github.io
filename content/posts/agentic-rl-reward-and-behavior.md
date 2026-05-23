---
title: "Agentic RL: Reward, Behavior, and The Long Shadow Of Feedback"
date: 2026-05-22T15:40:00+08:00
draft: false
tags: ["agentic-rl", "reinforcement-learning", "agent", "reward", "eval"]
categories: ["AI"]
summary: "A field note on agentic RL: reward design, behavior shaping, credit assignment, online and offline evaluation, and the feedback loops behind agents."
---

## Before Reward

Before reward, there is behavior.

I think this is easy to forget. We name the reward, tune the policy, plot the curve, and only later ask what kind of creature we have trained into the interface. For agentic systems, the curve is not enough. The behavior has texture: hesitation, overconfidence, tool obsession, premature final answers, stubborn loops, graceful recovery.

Agentic RL should begin with a behavioral vocabulary.

What should the agent notice? When should it ask? When should it search? When should it stop? What does a good recovery sound like? What is forbidden even if it helps the score?

Reward is only useful after the desired behavior has a shape.

## The Credit Problem

Agents stretch credit assignment across time.

A final answer may fail because the model misunderstood the user, chose the wrong tool, retrieved the wrong note, trusted stale evidence, summarized too early, or never noticed an exception. If the reward only lands at the end, it becomes a distant bell. The agent hears that something was wrong, but not where the wrongness entered.

This is why agentic RL needs intermediate signals:

- plan quality,
- tool selection,
- argument correctness,
- evidence relevance,
- state tracking,
- recovery behavior,
- final answer quality.

The signals do not need to be perfect. They need to be useful enough to guide repair.

## Reward Is A Product Decision

Reward design is not only a modeling decision. It is a product decision.

If you reward speed too hard, the agent may stop asking clarifying questions. If you reward task completion without user trust, it may become forceful. If you reward tool use, it may turn every thought into an API call. If you reward short answers, it may hide uncertainty.

The reward function is a compressed philosophy of the product.

For a research assistant, citation faithfulness may matter more than speed. For a coding agent, preserving user changes may matter more than finishing fast. For a travel planner, constraints and taste may matter more than itinerary density. The correct reward depends on what kind of harm the product must avoid.

## Offline And Online

Offline training is a cold room. Online behavior is weather.

Offline data can teach style, structure, and many useful habits. It can also preserve the assumptions of the dataset. Online feedback reveals the friction of real use: ambiguous intent, partial context, broken tools, impatient users, and tasks that do not fit the benchmark.

I like to separate three loops:

- **Imitation loop:** learn from good traces.
- **Critique loop:** learn from labeled failures and model judges.
- **Interaction loop:** learn from real user outcomes with guardrails.

The third loop is the most tempting and the most dangerous. It needs conservative rollout, logging, privacy boundaries, and a way to stop learning from noisy reward.

An agent that learns from the world must also be protected from the world.

## Evals For Agentic RL

Agentic RL without evals is just hope with gradients.

The eval suite should measure both outcomes and manners. Did the task finish? Did it finish for the right reason? Did the agent keep track of state? Did it avoid unnecessary tools? Did it ask when ambiguity was real? Did it recover when a tool failed?

I want eval cases that include:

- long-horizon tasks,
- tool failures,
- conflicting constraints,
- stale retrieval,
- partial success,
- user correction,
- and cases where the right answer is to stop.

The last case is essential. A capable agent should know when not to continue.

## Behavior Shaping

The beautiful part of agentic RL is also the frightening part: small signals can shape large habits.

A reward for "helpfulness" can become verbosity. A reward for "autonomy" can become impatience. A reward for "using evidence" can become citation theater. The policy learns the shadow of the metric, not the intention behind it.

So the work is slow. Watch traces. Cluster failures. Write sharper evals. Add counterexamples. Penalize the habit, not only the final mistake. Reward restraint when restraint is the product value.

There is no pure reward. There is only a better conversation between metric and behavior.

## The Long Shadow

Every feedback signal casts a shadow.

In agentic RL, the shadow reaches into how the system plans, asks, searches, acts, and stops. That is why I want reward design to stay close to product design and evaluation design. The three should not live in separate rooms.

The agent is a loop. RL changes the loop. Evals listen to the loop.

If those three are aligned, the system may become not only stronger, but more legible.
