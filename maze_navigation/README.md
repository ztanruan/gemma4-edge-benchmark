# Maze Navigation Experiment

This is a separate deterministic navigation experiment for Gemma 4 on `vLLM`.

It is designed to test offline edge reasoning on a simulated drone-style maze task without internet or external tools.

## Goal

The model receives the full maze, current state, heading, and the allowed action set. It must return structured JSON with exactly one `action` field:

- `move_forward`
- `turn_left`
- `turn_right`
- `ascend`
- `descend`
- `hover`
- `stop`

Expected output shape:

```json
{"action":"move_forward"}
```

Any other output is counted as an incorrect format.

The simulator then:

- validates the action
- applies it if legal
- records the new state, prompt, response, reasoning, raw SSE, and metrics deltas
- stops immediately when the exit is reached

## Deterministic Evaluation

Each level has a true optimal action sequence computed by graph search over:

- position
- altitude layer
- heading

The experiment records:

- whether the exit was reached
- how many model calls were needed
- how many valid actions were returned
- how many state-changing actions were executed
- optimal action count
- call-count gap versus optimal
- state-change gap versus optimal
- invalid output-format count
- invalid movement count
- hover count
- early `stop` count
- non-optimal action count
- repeated-state count
- latency, TTFT, token counts, and optional `tegrastats`
- per-step prompts, messages, request payloads, response text, reasoning text, raw SSE, and Prometheus metric deltas

The reference rubric is algorithmic:

- success means reaching the exit
- valid format means the visible output is valid JSON with exactly one allowed action in the `action` field
- optimality is measured against the BFS shortest path over `(x, y, z, heading)`
- `hover` and invalid outputs are allowed to happen, but they count against efficiency

The run writes:

- `records.jsonl`: one run-level record per `level x thinking x repeat`
- `step_records.jsonl`: one step-level record per model call
- `maze_reference.json`: level renderings, optimal paths, and evaluation rubric
- `rubric.md`: human/LLM review guidance for the deterministic metrics and step traces
- prompt / response / metrics artifacts under `prompts/`, `responses/`, and `metrics/`

## Prompting Path

The maze experiment uses the same Gemma 4 / vLLM chat path as the rest of the benchmark:

- `system` + `user` chat messages
- few-shot chat examples for forward, turn, and ascend decisions
- `chat_template_kwargs.enable_thinking` for `thinking=false` / `thinking=true`
- `response_format.json_schema` to constrain the output to `{"action": "<enum>"}`  

The model is still judged strictly by the resulting action and the shortest-path ground truth.

## Levels

- `easy`: single-layer corridor escape
- `medium`: one required ascent
- `hard`: ascend, navigate across the upper layer, then descend to escape

## Validate Levels

Run this first to verify the shortest-path ground truth:

```bash
python3 scripts/validate_maze_navigation.py
```

## Smoke Check

Run this before the full maze experiment to confirm the level runner, records, artifacts, and summary inputs are all working:

```bash
python3 scripts/run_maze_navigation_smoke.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking both
```

This runs one smoke pass across `easy`, `medium`, and `hard`, then writes:

- `maze_smoke_report.json`
- `records.jsonl`
- `step_records.jsonl`
- `maze_reference.json`
- full prompt / response / metrics artifacts for each step

## Run

```bash
python3 scripts/run_maze_navigation.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking both \
  --with-tegrastats
```

Run one difficulty only:

```bash
python3 scripts/run_maze_navigation.py \
  --backend-config configs/backends/vllm.yaml \
  --difficulty hard \
  --thinking both
```

Dry-run the full experiment without hitting the model:

```bash
python3 scripts/run_maze_navigation.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking both \
  --dry-run
```

Run only one thinking mode:

```bash
python3 scripts/run_maze_navigation.py \
  --backend-config configs/backends/vllm.yaml \
  --thinking true
```

Common tuning flags:

- `--repeat-count 3`
- `--max-calls 80`
- `--max-tokens 1024`
- `--temperature 0.0`
- `--with-tegrastats`

## Summarize

```bash
python3 scripts/summarize_maze_navigation.py \
  --run-dir outputs/maze_navigation_runs/<run_id>
```

This produces:

- `summary.csv`: one row per maze run
- `condition_summary.csv`: aggregated comparison by level and thinking mode
- `steps.csv`: one row per model call

## Interpreting Results

- `call_count` tells you how many model calls it took to escape
- `optimal_action_count` is the true shortest-path length
- `call_count_gap_vs_optimal` shows how far the model was from the minimum possible call count
- `invalid_format_count` shows how often the model failed the strict JSON action contract
- `invalid_transition_count` shows how often it chose legal tokens that were illegal moves from the current state
- `nonoptimal_action_count` shows how often it chose an allowed but non-shortest-path action
