---
title: "Agent Design: Loops, Tools, and the Shape of Memory"
date: 2026-05-22T15:20:00+08:00
draft: false
tags: ["agent", "llm", "tool-use", "memory", "product"]
categories: ["AI"]
summary: "A field note on designing agents as observable loops, with tools, memory, failure recovery, and product boundaries."
---

## A Loop You Can Inspect

An agent is not a magic noun. It is a loop.

Something observes. Something decides. Something acts. Something reads the trace of that action and changes its next step. The whole thing becomes useful only when the loop is visible enough to debug.

I like to design agents from the trace outward. Before choosing a model, I ask what a good run should leave behind: the user intent, the plan, the chosen tools, the failed calls, the recovered branch, the final answer, and the small reasons in between. If the trace is poor, the agent may look clever for one demo and become unknowable in production.

The first design question is therefore not "how autonomous should it be?" It is "where can we look when it is wrong?"

## Tools Have Contracts

Tool use is where language meets the hard floor.

A tool should not be a decoration attached to a prompt. It needs a contract: input schema, output schema, timeout, error vocabulary, permission boundary, and a small set of examples that make failure ordinary. The model should not have to guess whether a failed search means "no result", "bad query", "network error", or "try again later".

Good tool design reduces poetry at the interface. The tool may serve an imaginative task, but its surface should be almost boring. Names are literal. Arguments are narrow. Errors are typed. Return values are small enough to fit back into the loop.

When a tool is vague, the model improvises. Improvisation is lovely in writing and dangerous in systems.

## Memory Is A Working Surface

Memory is often discussed as if it were a drawer. I prefer to treat it as a working surface.

The agent should not remember everything. It should keep only what changes future behavior: user preferences, unfinished commitments, durable facts, task state, and decisions that would be expensive or rude to ask again. Everything else can remain in the run trace.

There are at least three kinds of memory:

- **Session memory:** what matters inside this conversation.
- **Project memory:** what matters across related tasks.
- **World memory:** what should be retrieved, cited, and refreshed because it may become stale.

Mixing these layers makes agents feel haunted. They recall the wrong thing at the wrong time. They confuse a temporary instruction with a durable preference. They carry yesterday's debugging assumption into today's production question.

Good memory design is mostly forgetting with discipline.

## The Boundary Of Autonomy

Autonomy should be granted like access to a database: narrowly, intentionally, and with logs.

For product agents, I tend to separate actions into four bands:

- **Think:** reason, summarize, compare, draft.
- **Prepare:** create a plan, stage a change, preview a command.
- **Act:** call tools, edit files, submit forms, trigger jobs.
- **Commit:** publish, pay, delete, notify, merge.

The higher the band, the more the agent needs either explicit user confirmation or a reversible path. A good agent does not merely do more. It knows which actions deserve friction.

The user should feel the system moving with them, not around them.

## Failure As A Design Material

Agent design becomes real when the first run breaks.

The browser times out. The search result is stale. The tool returns a malformed payload. The model cites a source that was never provided. The user changes their mind halfway through. The environment has a file with the same name and different intent.

These are not edge cases. They are the material.

For every agent workflow, I want a small failure table:

- What can fail?
- How can we detect it?
- What should the agent try once?
- When should it stop?
- What should it tell the user?

The difference between a demo and a system is often the sentence after failure. A demo says "something went wrong." A system says what happened, what it preserved, what it will not touch, and what can be tried next.

## Product Shape

The best agent is not always the most general one.

Sometimes the right shape is a quiet assistant that drafts and waits. Sometimes it is a batch runner that disappears into logs. Sometimes it is a narrow workflow with one excellent button. The product decides how much intelligence should be visible.

I do not want agents that perform intelligence theatrically. I want agents that make the next human action lighter.

Design the loop. Name the tools. Keep the memory clean. Draw the boundary. Make failure legible.

Then the agent has a chance to become something better than a prompt with ambition.
