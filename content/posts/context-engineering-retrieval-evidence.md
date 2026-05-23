---
title: "Context Engineering: Retrieval, Memory, and The Shape Of Evidence"
date: 2026-05-22T15:50:00+08:00
draft: false
tags: ["retrieval", "rag", "context-engineering", "llm", "memory", "agent"]
categories: ["AI"]
summary: "A note on RAG and context engineering: retrieval quality, evidence shape, memory boundaries, and why context is a product surface."
---

## The Room Before The Answer

Before the model answers, there is a room.

The room has documents on the table, old notes in the corner, tool results still warm from the network, and a user sentence that may be smaller than the thing it wants. Context engineering is the work of arranging that room so the next token has fewer ways to lie.

I do not think of RAG as "attach search to a model." That phrase is too thin. Retrieval is a decision about what the system is allowed to see, what it must ignore, and what kind of evidence should become visible at the moment of reasoning.

Bad context makes the model sound fluent inside a fog.

## Retrieval Is Not Recall Alone

Recall matters. But recall alone is a wide net thrown into dark water.

A useful retrieval pipeline needs several questions:

- Did the query preserve the user's real intent?
- Did we retrieve the right unit: page, heading, paragraph, table, trace, or code block?
- Did the result include enough surrounding structure to be understood?
- Did ranking prefer evidence over decorative similarity?
- Did the final context contain contradictions that need to be named?

For blog search, this is why I prefer heading-level chunks over whole posts. A whole article is too large to be a source. A chunk has a doorway. It can say: this answer came from here, under this section, with this nearby meaning.

## The Shape Of Evidence

Evidence has shape.

A product metric is not shaped like a personal preference. A tool error is not shaped like a paragraph. A Notion note is not shaped like a benchmark row. If the retrieval layer flattens all of them into anonymous text, the model has to guess what kind of object it is holding.

Good context should carry metadata:

- title,
- section,
- date,
- tags,
- anchor,
- source URL,
- and a small summary of why the chunk exists.

This metadata is not clerical. It is part of reasoning. It tells the model whether a sentence is a memory, a measurement, a project note, or a public claim.

## Memory And Retrieval

Memory and retrieval are cousins, not twins.

Memory should hold durable preferences and working state. Retrieval should bring in refreshable evidence. If an agent treats both as the same thing, it becomes confused in a very human-looking way: it remembers stale facts, cites private assumptions, and carries temporary instructions too far.

I like the split to be explicit:

- **Memory:** what changes future behavior.
- **Retrieval:** what grounds the present answer.
- **Trace:** what explains how this run happened.

The trace is especially important. Without it, retrieval quality becomes a mood. With it, you can see the query, the candidates, the rejected chunks, the final sources, and the answer that grew from them.

## RAG As A Product Surface

RAG is often treated as backend plumbing. In real products, it becomes a user-facing surface.

Users notice when search feels literal but not thoughtful. They notice when an answer cites a page but misses the decisive line. They notice when the system remembers something it should have forgotten. They may not name the retrieval layer, but they feel its weather.

So I want retrieval to be designed with product taste:

- show the source,
- jump to the section,
- expose a snippet,
- prefer precise chunks,
- and let failure be honest.

If the system cannot find enough evidence, it should say so. There is grace in a clean boundary.

## The Quiet Work

Context engineering is quiet work. It has fewer fireworks than model choice and more small knives: chunking, normalization, anchors, deduping, reranking, freshness, permissions, and tests that ask whether the right piece of text reached the model.

But when it works, the answer feels less like a performance and more like a window opening.

The model still speaks. The context teaches it where to look.
