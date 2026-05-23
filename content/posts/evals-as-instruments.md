---
title: "Evals As Instruments: Measuring What The Demo Hides"
date: 2026-05-22T15:30:00+08:00
draft: false
tags: ["eval", "benchmark", "llm", "agent", "product"]
categories: ["AI"]
summary: "A note on evaluation as an instrument: failure cases, metrics, benchmark design, product loops, and the discipline of measuring agents."
---

## The Instrument

An eval is an instrument. It does not tell the whole truth. It tells one truth repeatedly enough that we can notice a change.

This is why I distrust both empty demos and overloaded benchmarks. A demo is often a lantern pointed at the best path. A benchmark can become a museum of numbers with no living user nearby. The useful eval sits between them: close enough to the product to hurt, clean enough to compare.

When I build an eval, I start with the failure I want to hear more clearly.

Does the model lose long-context facts? Does the agent call tools too early? Does it refuse when it should ask a clarifying question? Does retrieval bring back pretty but irrelevant context? Does a metric improve while user trust gets worse?

The eval should make one of these failures audible.

## Units Of Failure

A good eval has small units.

For language systems, a single "accuracy" score is usually too thick. I want to split the behavior into pieces that can be inspected:

- **Intent capture:** did the system understand what the user was trying to do?
- **Retrieval quality:** did the right evidence enter the context?
- **Reasoning path:** did the answer use the evidence in a sane order?
- **Tool behavior:** did the agent call the right tool with the right arguments?
- **Final usefulness:** did the result reduce the user's next step?
- **Recovery:** did the system notice and repair failure?

The unit of failure should point to the unit of repair. If a score drops and nobody knows where to look, the eval is only weather.

## OOD Cases Are The Real Room

Out-of-distribution cases are not a footnote. They are where the room becomes visible.

For agents, I like OOD cases that bend the workflow rather than merely change the topic:

- The user asks for two conflicting goals.
- The document is long and the key line is near the end.
- A tool returns partial data.
- A source disagrees with another source.
- The user gives a casual phrase that hides a hard constraint.
- The task requires saying "not enough information."

These cases do not exist to embarrass the model. They exist to map the boundary. A system without a known boundary is not more powerful; it is less honest.

## The Dataset Is A Weather Station

A benchmark dataset should not be a pile of prompts. It should be a weather station.

Each case should carry metadata: task type, domain, difficulty, expected evidence, known trap, required tool, answer form, and the behavior being tested. Without metadata, failures become anecdotes. With metadata, patterns appear.

If long-context cases fail together, you know where to dig. If tool-call cases pass offline but fail in product, you look at latency, schema, and user timing. If retrieval-heavy cases degrade after adding new content, you inspect the index instead of blaming the model.

The dataset should make debugging less mystical.

## Product Evals

A product eval is not finished when it returns a score.

For real products, I care about three layers:

- **Offline evals:** controlled, repeatable, cheap enough to run often.
- **Shadow evals:** run against real traffic without affecting users.
- **User-loop evals:** measure whether the system changed behavior in the world.

Each layer has a different truth. Offline evals are good for regression. Shadow evals reveal distribution. User loops reveal whether anyone cares.

The danger is optimizing the first layer until it becomes beautiful and irrelevant.

## Taste In Measurement

There is taste in measurement.

A good eval is small enough to run, sharp enough to fail, stable enough to compare, and honest enough to admit what it does not measure. It should create fewer arguments, not more. It should make the next experiment obvious.

I like eval reports that contain:

- the headline metric,
- the top regressions,
- representative failures,
- suspected causes,
- next actions,
- and the cases that should be added because the current eval was blind.

The last part matters. Every eval should improve after being surprised.

## The Quiet Standard

The goal is not to make models look good. The goal is to make systems less self-deceptive.

An eval is a quiet standard kept beside the workbench. Each run leaves a mark. Some marks are small. Some marks change the whole design.

When the metric moves, ask what moved with it. When the metric does not move, ask what it cannot hear.

That is where the real engineering begins.
