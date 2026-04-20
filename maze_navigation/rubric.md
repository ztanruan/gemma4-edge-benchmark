# Maze Navigation Rubric

Use this rubric for later human or LLM review of the maze-navigation experiment.

## Success Condition

The run succeeds only if the simulator reaches the exit state.

## Required Output Contract

Each model call must produce valid JSON with exactly one key:

```json
{"action":"move_forward"}
```

Allowed actions:

- `move_forward`
- `turn_left`
- `turn_right`
- `ascend`
- `descend`
- `hover`
- `stop`

Any other visible output is an invalid-format response.

## Deterministic Ground Truth

The benchmark computes the optimal shortest path with graph search over:

- `x`
- `y`
- `z`
- `heading`

Reference data is written to `maze_reference.json`.

## What to Review

At run level:

- did the run escape
- total call count
- optimal action count
- call-count gap versus optimal
- invalid-format count
- invalid-transition count
- non-optimal-action count
- repeated-state count

At step level:

- raw model response
- parsed action
- whether the action was allowed
- whether the action changed state
- whether the action was on an optimal path from that state

## Common Failure Modes

- invalid JSON
- action outside the allowed enum
- legal token but illegal movement from the current state
- early `stop`
- repeated loops
- unnecessary `hover`
- valid but non-optimal detours
