---
title: "Product Evals: Travel Planner, Long Context, and The Weight Of Taste"
date: 2026-05-22T16:00:00+08:00
draft: false
tags: ["travel-planner", "eval", "product", "agent", "long-context"]
categories: ["AI"]
summary: "A product note on evaluating an AI travel planner: itinerary quality, OOD scenes, long-context consistency, recommendation taste, and user loops."
---

## A Map Is Not A Trip

A travel planner can produce a map and still fail the trip.

This is the first lesson. The model may know attractions, distances, opening hours, and the polite grammar of an itinerary. But a trip is made of smaller forces: tired feet, budget, weather, a parent's pace, a museum closed on Monday, the user's private idea of what a good afternoon feels like.

The product problem is not only generation. It is judgment.

## The Eval Behind The Itinerary

For an AI travel product, I would not trust a single "itinerary quality" score. It is too smooth. The useful eval needs to break the trip into inspectable pieces:

- **Constraint following:** dates, budget, cities, pace, forbidden places.
- **Recommendation relevance:** places match user taste, not only popularity.
- **Route sanity:** distance, order, time windows, transportation friction.
- **Long-context consistency:** preferences survive across the whole plan.
- **OOD scenes:** rainy days, elderly travelers, children, sudden closures, niche interests.
- **Conversion usefulness:** the plan should make the next user action lighter.

The score should tell us where the journey cracked.

## Long Context Has A Memory Of Its Own

Travel planning is a long-context task wearing a friendly face.

The user gives constraints at the beginning, modifies one detail in the middle, asks for alternatives near the end, and still expects the system to remember that dinner should stay near the hotel. A weak system forgets politely. It does not crash. It simply drifts.

That drift is dangerous because it looks acceptable in a screenshot.

This is why long-context evals need consistency checks. Did the plan preserve the original budget? Did it repeat an attraction? Did it schedule two faraway places back to back? Did it obey the user's "quiet morning" preference after adding a new museum?

The best failures are visible enough to repair.

## OOD Is Where Taste Appears

Out-of-distribution scenes are where the product starts to show its character.

A normal tourist route is easy to imitate. A useful planner has to handle stranger requests:

- "I like bookstores and quiet alleys, not landmarks."
- "My parents cannot walk much."
- "Plan around a concert at night."
- "It may rain, give me a soft fallback."
- "I want the city to feel local without becoming inconvenient."

These cases test taste as much as intelligence. They force the system to trade off density, distance, novelty, and comfort. They also reveal whether the agent is optimizing for an impressive page or a livable day.

## Product Loop

The product loop matters more than the first answer.

Users revise travel plans conversationally. They remove a place, ask for cheaper options, swap a neighborhood, or say something wonderfully vague like "make it less tiring." The agent has to update the plan without losing the structure that still works.

A strong travel planner should keep a working state:

- current itinerary,
- hard constraints,
- soft preferences,
- rejected options,
- unresolved questions,
- and the reason each recommendation survived.

This is where agent design meets product design. The system is not just writing. It is carrying a small world forward.

## What I Would Measure

If I were building the eval suite again, I would keep three views side by side:

- **Case-level metrics:** constraint pass rate, route validity, preference match, repetition rate.
- **Trace-level metrics:** retrieval quality, tool calls, revision behavior, memory consistency.
- **User-loop metrics:** satisfaction, edit distance after first answer, conversion, retention.

The offline eval protects the system from regression. The trace eval explains why. The user loop decides whether the whole machine deserves to exist.

## The Weight Of Taste

Travel is a gentle domain until it is not.

A good itinerary is not the densest one. It has rhythm. It knows when not to add another place. It leaves a little air between the train station and dinner. It understands that recommendation is a form of taste under constraint.

That is the work I find interesting: turning a beautiful but fragile idea into something that can survive real users, real edges, and real days.

The map matters. The trip matters more.
