from __future__ import annotations

import json
import statistics
import time
from collections import deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .runner import (
    _backend_record_fields,
    _chat_template_kwargs,
    _completion_status,
    _derive_server_stats,
    _write_json,
    _write_text,
    collect_run_metadata,
    ensure_run_dirs,
    load_yaml,
    metrics_delta,
    request_chat_completion,
    scrape_metrics,
    tokenize_chat_messages_detailed,
    update_run_metadata,
    verify_model,
    write_record,
)
from .tegrastats import (
    TegraStatsSession,
    summarize_tegrastats_log,
    write_tegrastats_timeseries,
)

ALLOWED_ACTIONS = (
    "move_forward",
    "turn_left",
    "turn_right",
    "ascend",
    "descend",
    "hover",
    "stop",
)

SEARCH_ACTIONS = ("move_forward", "turn_left", "turn_right", "ascend", "descend")
HEADINGS = ("north", "east", "south", "west")
TURN_LEFT = {"north": "west", "west": "south", "south": "east", "east": "north"}
TURN_RIGHT = {"north": "east", "east": "south", "south": "west", "west": "north"}
FORWARD_DELTA = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
OPEN_SYMBOLS = {".", "S", "E", "U", "D", "B"}
ASCEND_SYMBOLS = {"U", "B"}
DESCEND_SYMBOLS = {"D", "B"}
MAZE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": list(ALLOWED_ACTIONS),
        }
    },
    "required": ["action"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class MazeState:
    x: int
    y: int
    z: int
    heading: str


@dataclass(frozen=True)
class MazeLevel:
    id: str
    difficulty: str
    title: str
    description: str
    start_heading: str
    layers: tuple[tuple[str, ...], ...]
    start_position: tuple[int, int, int]
    exit_position: tuple[int, int, int]
    required_optimal_actions: tuple[str, ...] = ()
    recommended_max_calls: int | None = None

    @property
    def width(self) -> int:
        return len(self.layers[0][0])

    @property
    def height(self) -> int:
        return len(self.layers[0])

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def start_state(self) -> MazeState:
        x, y, z = self.start_position
        return MazeState(x=x, y=y, z=z, heading=self.start_heading)

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= z < self.depth and 0 <= y < self.height and 0 <= x < self.width

    def cell(self, x: int, y: int, z: int) -> str:
        return self.layers[z][y][x]

    def is_open(self, x: int, y: int, z: int) -> bool:
        return self.in_bounds(x, y, z) and self.cell(x, y, z) in OPEN_SYMBOLS

    def is_goal(self, state: MazeState) -> bool:
        return (state.x, state.y, state.z) == self.exit_position

    def render(self) -> str:
        lines: list[str] = []
        sx, sy, sz = self.start_position
        ex, ey, ez = self.exit_position
        for z, layer in enumerate(self.layers):
            lines.append(f"Layer z={z}")
            for y, row in enumerate(layer):
                rendered_row = list(row)
                for x, char in enumerate(rendered_row):
                    if (x, y, z) == (sx, sy, sz):
                        rendered_row[x] = "S"
                    elif (x, y, z) == (ex, ey, ez):
                        rendered_row[x] = "E"
                    elif char not in OPEN_SYMBOLS:
                        rendered_row[x] = "#"
                lines.append("".join(rendered_row))
            lines.append("")
        lines.append(
            "Legend: # wall, . open, S start, E exit, U ascend-only shaft, D descend-only shaft, B bidirectional shaft"
        )
        return "\n".join(lines).strip()


def load_maze_levels(path: Path) -> list[MazeLevel]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    levels: list[MazeLevel] = []
    for item in payload.get("levels", []):
        levels.append(_level_from_payload(item))
    return levels


def _level_from_payload(item: dict[str, Any]) -> MazeLevel:
    raw_layers = item["layers"]
    normalized_layers: list[tuple[str, ...]] = []
    start_position: tuple[int, int, int] | None = None
    exit_position: tuple[int, int, int] | None = None
    width: int | None = None
    height: int | None = None
    for z, layer in enumerate(raw_layers):
        rows = tuple(layer["rows"])
        if width is None:
            width = len(rows[0])
            height = len(rows)
        if any(len(row) != width for row in rows):
            raise ValueError(f"Level {item['id']} has inconsistent row widths on layer {z}.")
        if len(rows) != height:
            raise ValueError(f"Level {item['id']} has inconsistent row counts across layers.")
        mutable_rows = []
        for y, row in enumerate(rows):
            chars = list(row)
            for x, char in enumerate(chars):
                if char not in OPEN_SYMBOLS and char != "#":
                    raise ValueError(
                        f"Level {item['id']} has unsupported cell marker {char!r} at {(x, y, z)}."
                    )
                if char == "S":
                    if start_position is not None:
                        raise ValueError(f"Level {item['id']} has multiple start markers.")
                    start_position = (x, y, z)
                if char == "E":
                    if exit_position is not None:
                        raise ValueError(f"Level {item['id']} has multiple exit markers.")
                    exit_position = (x, y, z)
            mutable_rows.append("".join(chars))
        normalized_layers.append(tuple(mutable_rows))
    if start_position is None or exit_position is None:
        raise ValueError(f"Level {item['id']} must include exactly one S and one E marker.")
    start_heading = item.get("start_heading", "east")
    if start_heading not in HEADINGS:
        raise ValueError(f"Level {item['id']} has unsupported start heading {start_heading!r}.")
    required_optimal_actions = tuple(item.get("required_optimal_actions") or [])
    for action in required_optimal_actions:
        if action not in SEARCH_ACTIONS:
            raise ValueError(f"Level {item['id']} requires unsupported optimal action {action!r}.")
    recommended_max_calls = item.get("recommended_max_calls")
    if recommended_max_calls is not None and (
        not isinstance(recommended_max_calls, int) or recommended_max_calls < 1
    ):
        raise ValueError(
            f"Level {item['id']} has invalid recommended_max_calls={recommended_max_calls!r}."
        )
    return MazeLevel(
        id=item["id"],
        difficulty=item["difficulty"],
        title=item["title"],
        description=item["description"],
        start_heading=start_heading,
        layers=tuple(normalized_layers),
        start_position=start_position,
        exit_position=exit_position,
        required_optimal_actions=required_optimal_actions,
        recommended_max_calls=recommended_max_calls,
    )


def parse_action(content: str) -> dict[str, Any]:
    normalized = (content or "").strip()
    if not normalized:
        return {
            "action": None,
            "format_valid": False,
            "message": "Output must be valid JSON with exactly one allowed action in the action field.",
            "raw": normalized,
            "parsed_json": None,
        }
    try:
        parsed = json.loads(normalized)
    except (TypeError, ValueError) as exc:
        return {
            "action": None,
            "format_valid": False,
            "message": f"Output must be valid JSON with exactly one allowed action in the action field. JSON parse failed: {exc}",
            "raw": normalized,
            "parsed_json": None,
        }
    if not isinstance(parsed, dict):
        return {
            "action": None,
            "format_valid": False,
            "message": "Output JSON must be an object with exactly one key: action.",
            "raw": normalized,
            "parsed_json": parsed,
        }
    if set(parsed.keys()) != {"action"}:
        return {
            "action": None,
            "format_valid": False,
            "message": f"Output JSON must contain only the action key. Got keys: {sorted(parsed.keys())}",
            "raw": normalized,
            "parsed_json": parsed,
        }
    action = parsed.get("action")
    if action in ALLOWED_ACTIONS:
        return {
            "action": action,
            "format_valid": True,
            "message": None,
            "raw": normalized,
            "parsed_json": parsed,
        }
    return {
        "action": None,
        "format_valid": False,
        "message": f"Output JSON action must be one of the allowed actions. Got: {action!r}",
        "raw": normalized,
        "parsed_json": parsed,
    }


def apply_action(level: MazeLevel, state: MazeState, action: str) -> dict[str, Any]:
    if action not in ALLOWED_ACTIONS:
        return {
            "format_valid": False,
            "action_valid": False,
            "transition_valid": False,
            "state_after": state,
            "state_changed": False,
            "terminal": False,
            "reason": "invalid_action_token",
            "message": f"Action {action!r} is not in the allowed action set.",
        }
    if action == "turn_left":
        return {
            "format_valid": True,
            "action_valid": True,
            "transition_valid": True,
            "state_after": MazeState(state.x, state.y, state.z, TURN_LEFT[state.heading]),
            "state_changed": True,
            "terminal": False,
            "reason": "ok",
            "message": "Turned left.",
        }
    if action == "turn_right":
        return {
            "format_valid": True,
            "action_valid": True,
            "transition_valid": True,
            "state_after": MazeState(state.x, state.y, state.z, TURN_RIGHT[state.heading]),
            "state_changed": True,
            "terminal": False,
            "reason": "ok",
            "message": "Turned right.",
        }
    if action == "move_forward":
        dx, dy = FORWARD_DELTA[state.heading]
        nx, ny = state.x + dx, state.y + dy
        if not level.is_open(nx, ny, state.z):
            return {
                "format_valid": True,
                "action_valid": True,
                "transition_valid": False,
                "state_after": state,
                "state_changed": False,
                "terminal": False,
                "reason": "blocked_forward",
                "message": "Forward cell is blocked or outside the maze.",
            }
        return {
            "format_valid": True,
            "action_valid": True,
            "transition_valid": True,
            "state_after": MazeState(nx, ny, state.z, state.heading),
            "state_changed": True,
            "terminal": False,
            "reason": "ok",
            "message": "Moved forward.",
        }
    if action == "ascend":
        if level.cell(state.x, state.y, state.z) not in ASCEND_SYMBOLS:
            return {
                "format_valid": True,
                "action_valid": True,
                "transition_valid": False,
                "state_after": state,
                "state_changed": False,
                "terminal": False,
                "reason": "no_ascend_shaft",
                "message": "Ascend is only allowed from a U or B shaft cell.",
            }
        nz = state.z + 1
        if not level.is_open(state.x, state.y, nz):
            return {
                "format_valid": True,
                "action_valid": True,
                "transition_valid": False,
                "state_after": state,
                "state_changed": False,
                "terminal": False,
                "reason": "blocked_ascend",
                "message": "Cannot ascend because the target cell above is blocked or out of bounds.",
            }
        return {
            "format_valid": True,
            "action_valid": True,
            "transition_valid": True,
            "state_after": MazeState(state.x, state.y, nz, state.heading),
            "state_changed": True,
            "terminal": False,
            "reason": "ok",
            "message": "Ascended one layer.",
        }
    if action == "descend":
        if level.cell(state.x, state.y, state.z) not in DESCEND_SYMBOLS:
            return {
                "format_valid": True,
                "action_valid": True,
                "transition_valid": False,
                "state_after": state,
                "state_changed": False,
                "terminal": False,
                "reason": "no_descend_shaft",
                "message": "Descend is only allowed from a D or B shaft cell.",
            }
        nz = state.z - 1
        if not level.is_open(state.x, state.y, nz):
            return {
                "format_valid": True,
                "action_valid": True,
                "transition_valid": False,
                "state_after": state,
                "state_changed": False,
                "terminal": False,
                "reason": "blocked_descend",
                "message": "Cannot descend because the target cell below is blocked or out of bounds.",
            }
        return {
            "format_valid": True,
            "action_valid": True,
            "transition_valid": True,
            "state_after": MazeState(state.x, state.y, nz, state.heading),
            "state_changed": True,
            "terminal": False,
            "reason": "ok",
            "message": "Descended one layer.",
        }
    if action == "hover":
        return {
            "format_valid": True,
            "action_valid": True,
            "transition_valid": True,
            "state_after": state,
            "state_changed": False,
            "terminal": False,
            "reason": "hover",
            "message": "Hovered in place.",
        }
    return {
        "format_valid": True,
        "action_valid": True,
        "transition_valid": True,
        "state_after": state,
        "state_changed": False,
        "terminal": True,
        "reason": "stop",
        "message": "Mission stopped by model output.",
    }


class MazeSolver:
    def __init__(self, level: MazeLevel):
        self.level = level

    # Caching on the method pins the solver instance, which is fine here:
    # solvers are created per level and discarded with their cache.
    @cache  # noqa: B019
    def shortest_path(self, state: MazeState) -> tuple[str, ...] | None:
        if self.level.is_goal(state):
            return ()
        queue: deque[tuple[MazeState, tuple[str, ...]]] = deque([(state, ())])
        visited = {state}
        while queue:
            current, path = queue.popleft()
            for action in SEARCH_ACTIONS:
                outcome = apply_action(self.level, current, action)
                if not outcome["transition_valid"]:
                    continue
                next_state = outcome["state_after"]
                if next_state in visited:
                    continue
                next_path = path + (action,)
                if self.level.is_goal(next_state):
                    return next_path
                visited.add(next_state)
                queue.append((next_state, next_path))
        return None

    def optimal_action_count(self, state: MazeState) -> int | None:
        path = self.shortest_path(state)
        return None if path is None else len(path)

    def optimal_actions(self, state: MazeState) -> list[str]:
        path = self.shortest_path(state)
        if path is None:
            return []
        if len(path) == 0:
            return ["stop"]
        best = len(path)
        actions: list[str] = []
        for action in SEARCH_ACTIONS:
            outcome = apply_action(self.level, state, action)
            if not outcome["transition_valid"]:
                continue
            next_path = self.shortest_path(outcome["state_after"])
            if next_path is None:
                continue
            if 1 + len(next_path) == best:
                actions.append(action)
        return actions


def build_system_prompt() -> str:
    return (
        "You are controlling a simulated drone in an offline maze-navigation benchmark on Jetson Thor. "
        "Your objective is to reach the exit using the fewest total actions possible. "
        "Reason over the current state, heading, and vertical-shaft rules, then produce only the next action. "
        'Output must be valid JSON matching this structure: {"action": "<allowed_action>"}. '
        "Do not add explanations, markdown, or extra keys. "
        "Allowed actions are: move_forward, turn_left, turn_right, ascend, descend, hover, stop."
    )


def build_user_prompt(
    level: MazeLevel,
    state: MazeState,
    *,
    step_index: int,
    max_calls: int,
    previous_feedback: str | None,
    recent_trace: list[str] | None = None,
) -> str:
    x, y, z = state.x, state.y, state.z
    lines = [
        f"Level ID: {level.id}",
        f"Difficulty: {level.difficulty}",
        f"Title: {level.title}",
        f"Step: {step_index} of at most {max_calls}",
        "",
        "Maze:",
        level.render(),
        "",
        f"Current state: x={x}, y={y}, z={z}, heading={state.heading}",
        f"Exit position: x={level.exit_position[0]}, y={level.exit_position[1]}, z={level.exit_position[2]}",
        "",
        "Action semantics:",
        "- move_forward: move one cell in the current heading if the cell is open",
        "- turn_left / turn_right: rotate 90 degrees in place",
        "- ascend: move to z+1 at the same x,y only from U or B cells",
        "- descend: move to z-1 at the same x,y only from D or B cells",
        "- hover: stay in place",
        "- stop: terminate the mission",
    ]
    if recent_trace:
        lines.extend(["", "Recent action trace:"])
        lines.extend(recent_trace)
    if previous_feedback:
        lines.extend(["", "Previous step feedback:", previous_feedback])
    lines.extend(
        [
            "",
            "Goal: choose the single next action that keeps the route to the exit as short as possible.",
            "Return valid JSON with exactly one key named action.",
            'Example valid output: {"action":"move_forward"}',
        ]
    )
    return "\n".join(lines).strip()


def build_few_shot_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": (
                "Mini example 1\n"
                "Current state: x=1, y=1, z=0, heading=east\n"
                "Exit position: x=3, y=1, z=0\n"
                "Local map:\n"
                "#####\n"
                "#S.E#\n"
                "#####\n"
                "Return the next optimal action as JSON."
            ),
        },
        {"role": "assistant", "content": '{"action":"move_forward"}'},
        {
            "role": "user",
            "content": (
                "Mini example 2\n"
                "Current state: x=1, y=1, z=0, heading=east\n"
                "Exit position: x=1, y=2, z=0\n"
                "Local map:\n"
                "#####\n"
                "#S###\n"
                "#E..#\n"
                "#####\n"
                "move_forward would hit a wall. Return the next optimal action as JSON."
            ),
        },
        {"role": "assistant", "content": '{"action":"turn_right"}'},
        {
            "role": "user",
            "content": (
                "Mini example 3\n"
                "Current state: x=2, y=2, z=0, heading=north\n"
                "Exit position: x=2, y=2, z=1\n"
                "Current cell marker: U\n"
                "Return the next optimal action as JSON."
            ),
        },
        {"role": "assistant", "content": '{"action":"ascend"}'},
    ]


def build_messages_for_step(
    level: MazeLevel,
    state: MazeState,
    *,
    step_index: int,
    max_calls: int,
    previous_feedback: str | None,
    recent_trace: list[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        *build_few_shot_messages(),
        {
            "role": "user",
            "content": build_user_prompt(
                level,
                state,
                step_index=step_index,
                max_calls=max_calls,
                previous_feedback=previous_feedback,
                recent_trace=recent_trace,
            ),
        },
    ]


def maze_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "maze_next_action",
            "schema": MAZE_RESPONSE_SCHEMA,
        },
    }


def build_level_reference(level: MazeLevel) -> dict[str, Any]:
    solver = MazeSolver(level)
    optimal_path = solver.shortest_path(level.start_state)
    return {
        "level_id": level.id,
        "difficulty": level.difficulty,
        "title": level.title,
        "description": level.description,
        "allowed_actions": list(ALLOWED_ACTIONS),
        "start_state": _state_dict(level.start_state),
        "exit_position": {
            "x": level.exit_position[0],
            "y": level.exit_position[1],
            "z": level.exit_position[2],
        },
        "required_optimal_actions": list(level.required_optimal_actions),
        "recommended_max_calls": level.recommended_max_calls,
        "maze_render": level.render(),
        "optimal_action_count": None if optimal_path is None else len(optimal_path),
        "optimal_path": list(optimal_path or ()),
    }


def validate_maze_levels(levels: list[MazeLevel]) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for level in levels:
        solver = MazeSolver(level)
        optimal_path = solver.shortest_path(level.start_state)
        if optimal_path is None:
            raise ValueError(f"Level {level.id} has no path from start to exit.")
        missing_required = [
            action for action in level.required_optimal_actions if action not in optimal_path
        ]
        if missing_required:
            raise ValueError(
                f"Level {level.id} optimal path is missing required actions: {', '.join(missing_required)}. "
                f"Actual optimal path: {list(optimal_path)}"
            )
        validations.append(
            {
                "level_id": level.id,
                "difficulty": level.difficulty,
                "title": level.title,
                "optimal_action_count": len(optimal_path),
                "optimal_path": list(optimal_path),
                "required_optimal_actions": list(level.required_optimal_actions),
                "missing_required_actions": [],
                "recommended_max_calls": level.recommended_max_calls,
            }
        )
    return validations


def run_maze_navigation_suite(
    *,
    project_root: Path,
    backend_config_path: Path,
    output_root: Path,
    levels_path: Path,
    difficulties: set[str] | None = None,
    level_ids: set[str] | None = None,
    thinking_modes: list[bool] | None = None,
    repeat_count: int = 1,
    max_calls_override: int | None = None,
    generation: dict[str, Any] | None = None,
    dry_run: bool = False,
    with_tegrastats: bool = False,
    seed: int = 20260417,
) -> Path:
    backend = load_yaml(backend_config_path)
    levels = load_maze_levels(levels_path)
    selected_levels = [
        level
        for level in levels
        if (not difficulties or level.difficulty in difficulties)
        and (not level_ids or level.id in level_ids)
    ]
    if not selected_levels:
        raise ValueError("No maze levels matched the requested filters.")
    validations = validate_maze_levels(selected_levels)
    thinking_modes = thinking_modes or [False, True]
    generation = generation or {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": 1024,
    }

    run_paths = ensure_run_dirs(output_root, dry_run=dry_run)
    step_records_path = run_paths.root / "step_records.jsonl"
    backend_fields = _backend_record_fields(backend, backend_config_path)
    model_verification = (
        {
            "requested_model": backend.get("model"),
            "advertised_models": [],
            "requested_model_found": None,
            "dry_run": True,
        }
        if dry_run
        else verify_model(backend["base_url"], backend.get("api_key"), backend["model"])
    )
    run_options = {
        "experiment": "maze_navigation",
        "levels_path": str(levels_path),
        "selected_level_ids": [level.id for level in selected_levels],
        "selected_difficulties": sorted(difficulties) if difficulties else None,
        "thinking_modes": ["true" if mode else "false" for mode in thinking_modes],
        "repeat_count": repeat_count,
        "max_calls_override": max_calls_override,
        "dry_run": dry_run,
        "with_tegrastats": with_tegrastats,
        "generation": generation,
        "seed": seed,
    }
    collect_run_metadata(run_paths, backend, model_verification, run_options)

    reference_payload = {
        "experiment": "maze_navigation",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evaluation_rubric": {
            "success_condition": "The run succeeds when the simulator state reaches the exit position.",
            "format_rule": "A model output is valid only if it is valid JSON with exactly one key named action and that action is in the allowed action enum.",
            "allowed_actions": list(ALLOWED_ACTIONS),
            "optimality_rule": "Optimality is measured against the BFS shortest path over x, y, z, and heading.",
            "structured_output_schema": MAZE_RESPONSE_SCHEMA,
            "notes": [
                "hover is allowed but non-progress unless it happens to be optimal from the current state, which is not expected here.",
                "stop before reaching the exit is counted as an early terminal failure.",
                "Invalid-format output does not change the simulator state.",
            ],
        },
        "levels": {level.id: build_level_reference(level) for level in selected_levels},
        "validations": validations,
    }
    reference_path = run_paths.root / "maze_reference.json"
    _write_json(reference_path, reference_payload)
    update_run_metadata(
        run_paths,
        lambda metadata: metadata.update(
            {
                "maze_reference_path": str(reference_path),
                "step_records_path": str(step_records_path),
            }
        ),
    )

    run_count = 0
    error_count = 0
    for level_index, level in enumerate(selected_levels):
        for thinking_index, thinking_enabled in enumerate(thinking_modes):
            for repeat_index in range(repeat_count):
                run_seed = seed + level_index * 10000 + thinking_index * 1000 + repeat_index * 100
                record = _run_single_experiment(
                    project_root=project_root,
                    run_paths=run_paths,
                    step_records_path=step_records_path,
                    backend=backend,
                    backend_config_path=backend_config_path,
                    backend_fields=backend_fields,
                    level=level,
                    generation=generation,
                    thinking_enabled=thinking_enabled,
                    repeat_index=repeat_index,
                    max_calls_override=max_calls_override,
                    dry_run=dry_run,
                    with_tegrastats=with_tegrastats,
                    seed=run_seed,
                )
                write_record(run_paths.records_path, record)
                run_count += 1
                if record.get("status") == "error":
                    error_count += 1

    update_run_metadata(
        run_paths,
        lambda metadata: metadata.update(
            {
                "maze_run_count": run_count,
                "maze_error_count": error_count,
            }
        ),
    )
    return run_paths.root


def _run_single_experiment(
    *,
    project_root: Path,
    run_paths: Any,
    step_records_path: Path,
    backend: dict[str, Any],
    backend_config_path: Path,
    backend_fields: dict[str, Any],
    level: MazeLevel,
    generation: dict[str, Any],
    thinking_enabled: bool,
    repeat_index: int,
    max_calls_override: int | None,
    dry_run: bool,
    with_tegrastats: bool,
    seed: int,
) -> dict[str, Any]:
    solver = MazeSolver(level)
    optimal_path = solver.shortest_path(level.start_state)
    if optimal_path is None:
        raise ValueError(f"Level {level.id} has no valid reference path.")
    optimal_action_count = len(optimal_path)
    max_calls = _effective_max_calls(level, optimal_action_count, max_calls_override)
    experiment_id = (
        f"{level.id}__thinking_{'true' if thinking_enabled else 'false'}__repeat_{repeat_index:02d}"
    )

    tegrastats_path = run_paths.telemetry_dir / f"{experiment_id}.log"
    tegrastats_session = TegraStatsSession(tegrastats_path)
    telemetry_enabled = with_tegrastats and tegrastats_session.enabled and not dry_run
    if telemetry_enabled:
        tegrastats_session.start()

    run_metrics_before = (
        None if dry_run else scrape_metrics(backend["base_url"], backend.get("api_key"))
    )
    current_state = level.start_state
    previous_feedback: str | None = None
    recent_trace: list[str] = []
    visited_counts: dict[MazeState, int] = {current_state: 1}
    revisited_state_count = 0
    steps_compact: list[dict[str, Any]] = []
    total_latency_values: list[float] = []
    ttft_values: list[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    invalid_format_count = 0
    invalid_transition_count = 0
    hover_count = 0
    stop_count = 0
    stop_before_goal_count = 0
    unexpected_tool_call_count = 0
    nonoptimal_action_count = 0
    optimal_action_match_count = 0
    state_change_count = 0
    valid_action_count = 0
    truncated_step_count = 0
    parser_failure_count = 0
    call_count = 0
    closest_remaining_action_count = optimal_action_count
    termination_reason = "max_calls_exhausted"
    success_reached_exit = False
    error_type = None
    error_message = None

    try:
        for step_index in range(1, max_calls + 1):
            prompt_text = build_user_prompt(
                level,
                current_state,
                step_index=step_index,
                max_calls=max_calls,
                previous_feedback=previous_feedback,
                recent_trace=recent_trace[-5:],
            )
            messages = build_messages_for_step(
                level,
                current_state,
                step_index=step_index,
                max_calls=max_calls,
                previous_feedback=previous_feedback,
                recent_trace=recent_trace[-5:],
            )
            prompt_token_detail = None
            prompt_token_estimate = None
            if not dry_run:
                prompt_token_detail = tokenize_chat_messages_detailed(
                    base_url=backend["base_url"],
                    api_key=backend.get("api_key"),
                    model=backend["model"],
                    messages=messages,
                    add_special_tokens=True,
                    add_generation_prompt=True,
                    continue_final_message=False,
                    chat_template_kwargs=_chat_template_kwargs(thinking_enabled),
                )
                if prompt_token_detail.get("ok") and prompt_token_detail.get("count") is not None:
                    prompt_token_estimate = int(prompt_token_detail["count"])

            prompt_token_debug = _tokenize_debug_summary(prompt_token_detail)
            prompt_dir, response_dir, metrics_dir = _prepare_step_dirs(
                run_paths, experiment_id, step_index
            )
            _write_text(prompt_dir / "user_prompt.txt", prompt_text)
            _write_json(prompt_dir / "messages.json", messages)
            if prompt_token_detail is not None:
                _write_json(prompt_dir / "prompt_token_debug.json", prompt_token_detail)

            metrics_before_text = (
                None if dry_run else scrape_metrics(backend["base_url"], backend.get("api_key"))
            )
            step_seed = seed + step_index - 1
            call_count += 1
            try:
                completion = (
                    _dry_run_completion(
                        action=(solver.optimal_actions(current_state) or ["stop"])[0],
                        prompt_token_estimate=prompt_token_estimate,
                        messages=messages,
                        generation=generation,
                        thinking_enabled=thinking_enabled,
                        seed=step_seed,
                    )
                    if dry_run
                    else request_chat_completion(
                        base_url=backend["base_url"],
                        api_key=backend.get("api_key"),
                        model=backend["model"],
                        messages=messages,
                        generation=generation,
                        thinking_enabled=thinking_enabled,
                        prompt_token_estimate=prompt_token_estimate,
                        seed=step_seed,
                        response_format=maze_response_format(),
                    )
                )
            except Exception as exc:
                metrics_after_text = (
                    None if dry_run else scrape_metrics(backend["base_url"], backend.get("api_key"))
                )
                metric_values = metrics_delta(metrics_before_text, metrics_after_text)
                _write_metrics_artifacts(
                    metrics_dir, metrics_before_text, metrics_after_text, metric_values
                )
                error_type = type(exc).__name__
                error_message = str(exc)
                step_record = {
                    **backend_fields,
                    "record_type": "maze_step",
                    "experiment_id": experiment_id,
                    "level_id": level.id,
                    "difficulty": level.difficulty,
                    "title": level.title,
                    "repeat_index": repeat_index,
                    "thinking_enabled": thinking_enabled,
                    "step_index": step_index,
                    "seed": step_seed,
                    "dry_run": dry_run,
                    "generation": generation,
                    "status": "error",
                    "error_type": error_type,
                    "error": error_message,
                    "state_before": _state_dict(current_state),
                    "state_after": _state_dict(current_state),
                    "prompt_token_estimate": prompt_token_estimate,
                    "prompt_token_debug": prompt_token_debug,
                    "prompt_token_debug_path": str(prompt_dir / "prompt_token_debug.json")
                    if prompt_token_detail is not None
                    else None,
                    "server_metrics_delta": metric_values,
                    "artifact_paths": _artifact_paths(
                        prompt_dir,
                        response_dir,
                        metrics_dir,
                        prompt_token_detail is not None,
                    ),
                }
                write_record(step_records_path, step_record)
                steps_compact.append(
                    {
                        "step_index": step_index,
                        "status": "error",
                        "error_type": error_type,
                        "error": error_message,
                        "state_before": _state_dict(current_state),
                        "state_after": _state_dict(current_state),
                        "artifact_paths": step_record["artifact_paths"],
                    }
                )
                termination_reason = "request_error"
                break

            metrics_after_text = (
                None if dry_run else scrape_metrics(backend["base_url"], backend.get("api_key"))
            )
            metric_values = metrics_delta(metrics_before_text, metrics_after_text)
            _write_completion_artifacts(
                response_dir=response_dir,
                metrics_dir=metrics_dir,
                completion=completion,
                metrics_before_text=metrics_before_text,
                metrics_after_text=metrics_after_text,
                metric_values=metric_values,
            )

            parse_result = parse_action(completion["content"])
            if parse_result["format_valid"]:
                valid_action_count += 1
                outcome = apply_action(level, current_state, parse_result["action"])
            else:
                invalid_format_count += 1
                outcome = {
                    "format_valid": False,
                    "action_valid": False,
                    "transition_valid": False,
                    "state_after": current_state,
                    "state_changed": False,
                    "terminal": False,
                    "reason": "invalid_format",
                    "message": parse_result["message"],
                }

            if not outcome["transition_valid"] and parse_result["format_valid"]:
                invalid_transition_count += 1
            if parse_result["action"] == "hover":
                hover_count += 1
            if parse_result["action"] == "stop":
                stop_count += 1
            if parse_result["action"] == "stop" and not level.is_goal(current_state):
                stop_before_goal_count += 1
            if completion.get("tool_calls"):
                unexpected_tool_call_count += len(completion["tool_calls"])
            if completion.get("reasoning_only_truncated"):
                truncated_step_count += 1
            if completion.get("empty_stream_bug"):
                parser_failure_count += 1

            optimal_path_from_state = list(solver.shortest_path(current_state) or ())
            optimal_actions_from_state = solver.optimal_actions(current_state)
            optimal_remaining_before = len(optimal_path_from_state)
            chosen_action_is_optimal = (
                parse_result["action"] in optimal_actions_from_state
                if parse_result["action"]
                else False
            )
            if chosen_action_is_optimal:
                optimal_action_match_count += 1
            elif parse_result["action"] is not None:
                nonoptimal_action_count += 1

            state_after = outcome["state_after"]
            goal_reached_after_step = level.is_goal(state_after)
            if outcome["state_changed"]:
                state_change_count += 1
            optimal_remaining_after = solver.optimal_action_count(state_after)
            if optimal_remaining_after is not None:
                closest_remaining_action_count = min(
                    closest_remaining_action_count, optimal_remaining_after
                )
            if goal_reached_after_step:
                success_reached_exit = True
                termination_reason = "escaped"
            elif outcome["reason"] == "stop":
                termination_reason = "stopped_before_goal"

            visited_counts[state_after] = visited_counts.get(state_after, 0) + 1
            if visited_counts[state_after] > 1:
                revisited_state_count += 1

            server_stats = _derive_server_stats(completion, metric_values)
            latency_ms = completion.get("latency_ms")
            if isinstance(latency_ms, (int, float)):
                total_latency_values.append(float(latency_ms))
            effective_ttft_ms = completion.get("ttft_ms")
            if effective_ttft_ms is None:
                effective_ttft_ms = server_stats.get("server_ttft_ms")
            if isinstance(effective_ttft_ms, (int, float)):
                ttft_values.append(float(effective_ttft_ms))
            usage = completion.get("usage") or {}
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens_total += int(usage["prompt_tokens"])
            if isinstance(usage.get("completion_tokens"), int):
                completion_tokens_total += int(usage["completion_tokens"])

            step_record = {
                **backend_fields,
                "record_type": "maze_step",
                "experiment_id": experiment_id,
                "level_id": level.id,
                "difficulty": level.difficulty,
                "title": level.title,
                "repeat_index": repeat_index,
                "thinking_enabled": thinking_enabled,
                "step_index": step_index,
                "seed": step_seed,
                "dry_run": dry_run,
                "generation": generation,
                "status": _completion_status(completion),
                "state_before": _state_dict(current_state),
                "state_after": _state_dict(state_after),
                "raw_response": completion["content"],
                "raw_reasoning": completion["reasoning"],
                "parsed_action": parse_result["action"],
                "response_raw_normalized": parse_result["raw"],
                "parsed_json": parse_result.get("parsed_json"),
                "format_valid": parse_result["format_valid"],
                "action_valid": outcome["action_valid"],
                "transition_valid": outcome["transition_valid"],
                "state_changed": outcome["state_changed"],
                "goal_reached_after_step": goal_reached_after_step,
                "terminal": outcome["terminal"],
                "termination_reason_if_terminal": termination_reason
                if outcome["terminal"] or goal_reached_after_step
                else None,
                "feedback": outcome["message"],
                "outcome_reason": outcome["reason"],
                "prompt_token_estimate": prompt_token_estimate,
                "prompt_token_debug": prompt_token_debug,
                "prompt_token_debug_path": str(prompt_dir / "prompt_token_debug.json")
                if prompt_token_detail is not None
                else None,
                "usage": usage,
                "latency_ms": completion["latency_ms"],
                "ttft_ms": completion["ttft_ms"],
                "ttft_ms_effective": effective_ttft_ms,
                "ttft_source": "client_stream"
                if completion["ttft_ms"] is not None
                else ("server_metrics" if effective_ttft_ms is not None else None),
                "finish_reason": completion["finish_reason"],
                "content_chars": len(completion["content"]),
                "reasoning_chars": len(completion["reasoning"]),
                "tool_call_count": len(completion["tool_calls"]),
                "raw_event_count": len(completion["raw_events"]),
                "data_event_count": completion["data_event_count"],
                "content_event_count": completion["content_event_count"],
                "reasoning_event_count": completion["reasoning_event_count"],
                "tool_call_event_count": completion["tool_call_event_count"],
                "reasoning_only_truncated": completion["reasoning_only_truncated"],
                "empty_stream_bug": completion["empty_stream_bug"],
                "end_to_end_tokens_per_second": completion["end_to_end_tokens_per_second"],
                "server_stats": server_stats,
                "server_metrics_delta": metric_values,
                "optimal_action_count_from_state": optimal_remaining_before,
                "optimal_actions_from_state": optimal_actions_from_state,
                "optimal_path_from_state": optimal_path_from_state,
                "optimal_action_count_after_step": optimal_remaining_after,
                "chosen_action_is_optimal": chosen_action_is_optimal,
                "artifact_paths": _artifact_paths(
                    prompt_dir,
                    response_dir,
                    metrics_dir,
                    prompt_token_detail is not None,
                ),
            }
            write_record(step_records_path, step_record)

            steps_compact.append(
                {
                    "step_index": step_index,
                    "status": step_record["status"],
                    "state_before": step_record["state_before"],
                    "state_after": step_record["state_after"],
                    "parsed_action": step_record["parsed_action"],
                    "response_raw_normalized": step_record["response_raw_normalized"],
                    "parsed_json": step_record["parsed_json"],
                    "format_valid": step_record["format_valid"],
                    "transition_valid": step_record["transition_valid"],
                    "state_changed": step_record["state_changed"],
                    "goal_reached_after_step": step_record["goal_reached_after_step"],
                    "outcome_reason": step_record["outcome_reason"],
                    "optimal_actions_from_state": step_record["optimal_actions_from_state"],
                    "chosen_action_is_optimal": step_record["chosen_action_is_optimal"],
                    "latency_ms": step_record["latency_ms"],
                    "ttft_ms_effective": step_record["ttft_ms_effective"],
                    "finish_reason": step_record["finish_reason"],
                    "artifact_paths": step_record["artifact_paths"],
                }
            )

            previous_feedback = (
                f"{outcome['message']} "
                f"Current state is now x={state_after.x}, y={state_after.y}, z={state_after.z}, heading={state_after.heading}."
            )
            recent_trace.append(
                f"- Step {step_index}: response={parse_result['raw']!r}, action={parse_result['action']!r}, "
                f"reason={outcome['reason']}, state_after=(x={state_after.x}, y={state_after.y}, z={state_after.z}, heading={state_after.heading})"
            )
            current_state = state_after
            if success_reached_exit or termination_reason == "stopped_before_goal":
                break
    finally:
        if telemetry_enabled:
            tegrastats_session.stop()

    run_metrics_after = (
        None if dry_run else scrape_metrics(backend["base_url"], backend.get("api_key"))
    )
    run_metrics_delta = metrics_delta(run_metrics_before, run_metrics_after)
    telemetry_summary = (
        summarize_tegrastats_log(tegrastats_path) if tegrastats_path.exists() else None
    )
    telemetry_timeseries_path = (
        write_tegrastats_timeseries(tegrastats_path, tegrastats_session.interval_ms)
        if tegrastats_path.exists()
        else None
    )

    total_latency_ms = round(sum(total_latency_values), 3) if total_latency_values else None
    avg_call_latency_ms = (
        round(statistics.mean(total_latency_values), 3) if total_latency_values else None
    )
    avg_ttft_ms = round(statistics.mean(ttft_values), 3) if ttft_values else None
    unique_state_count = len(visited_counts)
    status = "error" if termination_reason == "request_error" else "completed"
    call_count_gap = call_count - optimal_action_count if success_reached_exit else None
    state_change_gap = state_change_count - optimal_action_count if success_reached_exit else None

    return {
        **backend_fields,
        "record_type": "maze_run",
        "experiment_id": experiment_id,
        "level_id": level.id,
        "difficulty": level.difficulty,
        "title": level.title,
        "description": level.description,
        "repeat_index": repeat_index,
        "thinking_enabled": thinking_enabled,
        "seed": seed,
        "dry_run": dry_run,
        "generation": generation,
        "status": status,
        "error_type": error_type,
        "error": error_message,
        "success_reached_exit": success_reached_exit,
        "termination_reason": termination_reason,
        "start_state": _state_dict(level.start_state),
        "final_state": _state_dict(current_state),
        "exit_position": {
            "x": level.exit_position[0],
            "y": level.exit_position[1],
            "z": level.exit_position[2],
        },
        "call_count": call_count,
        "valid_action_count": valid_action_count,
        "state_change_count": state_change_count,
        "optimal_action_count": optimal_action_count,
        "optimal_path": list(optimal_path),
        "required_optimal_actions": list(level.required_optimal_actions),
        "closest_remaining_action_count": closest_remaining_action_count,
        "progress_ratio": round(
            (optimal_action_count - closest_remaining_action_count) / optimal_action_count,
            6,
        )
        if optimal_action_count
        else None,
        "call_count_gap_vs_optimal": call_count_gap,
        "state_change_gap_vs_optimal": state_change_gap,
        "call_efficiency_ratio": round(optimal_action_count / call_count, 6)
        if call_count and success_reached_exit
        else None,
        "state_change_efficiency_ratio": round(optimal_action_count / state_change_count, 6)
        if state_change_count and success_reached_exit
        else None,
        "invalid_format_count": invalid_format_count,
        "invalid_transition_count": invalid_transition_count,
        "hover_count": hover_count,
        "stop_count": stop_count,
        "stop_before_goal_count": stop_before_goal_count,
        "unexpected_tool_call_count": unexpected_tool_call_count,
        "parser_failure_count": parser_failure_count,
        "truncated_step_count": truncated_step_count,
        "nonoptimal_action_count": nonoptimal_action_count,
        "optimal_action_match_count": optimal_action_match_count,
        "revisited_state_count": revisited_state_count,
        "unique_state_count": unique_state_count,
        "max_calls": max_calls,
        "goal_reached_call_index": call_count if success_reached_exit else None,
        "total_latency_ms": total_latency_ms,
        "avg_call_latency_ms": avg_call_latency_ms,
        "avg_ttft_ms": avg_ttft_ms,
        "total_prompt_tokens": prompt_tokens_total,
        "total_completion_tokens": completion_tokens_total,
        "telemetry_summary": telemetry_summary,
        "telemetry_path": str(tegrastats_path) if tegrastats_path.exists() else None,
        "telemetry_timeseries_path": telemetry_timeseries_path,
        "run_metrics_delta": run_metrics_delta,
        "reference_path": str(run_paths.root / "maze_reference.json"),
        "step_records_path": str(step_records_path),
        "steps": steps_compact,
    }


def _effective_max_calls(
    level: MazeLevel, optimal_action_count: int, max_calls_override: int | None
) -> int:
    if max_calls_override is not None:
        return max_calls_override
    if level.recommended_max_calls is not None:
        return level.recommended_max_calls
    return max(20, optimal_action_count * 2 + 10)


def _state_dict(state: MazeState) -> dict[str, Any]:
    return {"x": state.x, "y": state.y, "z": state.z, "heading": state.heading}


def _tokenize_debug_summary(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    if not detail:
        return None
    raw = detail.get("raw") or {}
    summary = {
        "request_type": detail.get("request_type"),
        "status_code": detail.get("status_code"),
        "ok": detail.get("ok"),
        "count": detail.get("count"),
        "error_type": detail.get("error_type"),
        "error": detail.get("error"),
    }
    if isinstance(raw, dict) and raw.get("max_model_len") is not None:
        summary["max_model_len"] = raw.get("max_model_len")
    return summary


def _prepare_step_dirs(
    run_paths: Any, experiment_id: str, step_index: int
) -> tuple[Path, Path, Path]:
    step_label = f"step_{step_index:04d}"
    prompt_dir = run_paths.prompts_dir / experiment_id / step_label
    response_dir = run_paths.responses_dir / experiment_id / step_label
    metrics_dir = run_paths.metrics_dir / experiment_id / step_label
    prompt_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return prompt_dir, response_dir, metrics_dir


def _write_completion_artifacts(
    *,
    response_dir: Path,
    metrics_dir: Path,
    completion: dict[str, Any],
    metrics_before_text: str | None,
    metrics_after_text: str | None,
    metric_values: dict[str, Any] | None,
) -> None:
    _write_text(response_dir / "response.txt", completion["content"])
    _write_text(response_dir / "reasoning.txt", completion["reasoning"])
    _write_json(response_dir / "tool_calls.json", completion["tool_calls"])
    _write_text(response_dir / "response_events.sse", "\n".join(completion["raw_events"]))
    _write_text(
        response_dir / "response_events_timeline.jsonl",
        "\n".join(json.dumps(item) for item in completion["raw_event_records"]),
    )
    _write_json(response_dir / "request_payload.json", completion["request_payload"])
    _write_json(
        response_dir / "completion_snapshot.json",
        {
            "content": completion["content"],
            "reasoning": completion["reasoning"],
            "tool_calls": completion["tool_calls"],
            "ttft_ms": completion["ttft_ms"],
            "latency_ms": completion["latency_ms"],
            "usage": completion["usage"],
            "finish_reason": completion["finish_reason"],
            "end_to_end_tokens_per_second": completion["end_to_end_tokens_per_second"],
            "reasoning_only_truncated": completion["reasoning_only_truncated"],
            "empty_stream_bug": completion["empty_stream_bug"],
        },
    )
    if completion.get("completion_token_fallback_debug") is not None:
        _write_json(
            response_dir / "completion_token_fallback_debug.json",
            completion["completion_token_fallback_debug"],
        )
    _write_metrics_artifacts(metrics_dir, metrics_before_text, metrics_after_text, metric_values)


def _write_metrics_artifacts(
    metrics_dir: Path,
    metrics_before_text: str | None,
    metrics_after_text: str | None,
    metric_values: dict[str, Any] | None,
) -> None:
    if metrics_before_text is not None:
        _write_text(metrics_dir / "metrics_before.prom", metrics_before_text)
    if metrics_after_text is not None:
        _write_text(metrics_dir / "metrics_after.prom", metrics_after_text)
    if metric_values is not None:
        _write_json(metrics_dir / "metrics_delta.json", metric_values)


def _artifact_paths(
    prompt_dir: Path,
    response_dir: Path,
    metrics_dir: Path,
    has_prompt_token_debug: bool,
) -> dict[str, str | None]:
    completion_token_fallback_path = response_dir / "completion_token_fallback_debug.json"
    return {
        "prompt_path": str(prompt_dir / "user_prompt.txt"),
        "messages_path": str(prompt_dir / "messages.json"),
        "prompt_token_debug_path": str(prompt_dir / "prompt_token_debug.json")
        if has_prompt_token_debug
        else None,
        "response_path": str(response_dir / "response.txt"),
        "reasoning_path": str(response_dir / "reasoning.txt"),
        "tool_calls_path": str(response_dir / "tool_calls.json"),
        "request_payload_path": str(response_dir / "request_payload.json"),
        "completion_snapshot_path": str(response_dir / "completion_snapshot.json"),
        "raw_events_path": str(response_dir / "response_events.sse"),
        "raw_event_timeline_path": str(response_dir / "response_events_timeline.jsonl"),
        "completion_token_fallback_debug_path": str(completion_token_fallback_path)
        if completion_token_fallback_path.exists()
        else None,
        "metrics_before_path": str(metrics_dir / "metrics_before.prom")
        if (metrics_dir / "metrics_before.prom").exists()
        else None,
        "metrics_after_path": str(metrics_dir / "metrics_after.prom")
        if (metrics_dir / "metrics_after.prom").exists()
        else None,
        "metrics_delta_path": str(metrics_dir / "metrics_delta.json")
        if (metrics_dir / "metrics_delta.json").exists()
        else None,
    }


def _dry_run_completion(
    *,
    action: str,
    prompt_token_estimate: int | None,
    messages: list[dict[str, Any]],
    generation: dict[str, Any],
    thinking_enabled: bool,
    seed: int,
) -> dict[str, Any]:
    usage = {
        "prompt_tokens": prompt_token_estimate,
        "completion_tokens": 1,
        "prompt_tokens_source": "dry_run",
        "completion_tokens_source": "dry_run",
    }
    content = json.dumps({"action": action}, separators=(",", ":"))
    return {
        "content": content,
        "reasoning": "",
        "tool_calls": [],
        "ttft_ms": 0.0,
        "latency_ms": 0.0,
        "usage": usage,
        "finish_reason": "stop",
        "raw_events": [],
        "raw_event_records": [],
        "end_to_end_tokens_per_second": None,
        "server_decode_tokens_per_second": None,
        "last_event": None,
        "completion_token_fallback_debug": None,
        "data_event_count": 0,
        "content_event_count": 1,
        "non_empty_text_event_count": 1,
        "reasoning_event_count": 0,
        "tool_call_event_count": 0,
        "output_event_offsets_ms": [0.0],
        "reasoning_only_truncated": None,
        "empty_stream_bug": None,
        "request_payload": {
            "dry_run": True,
            "model": "dry_run",
            "messages": messages,
            "max_tokens": generation["max_tokens"],
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "top_k": generation["top_k"],
            "chat_template_kwargs": {"enable_thinking": thinking_enabled},
            "seed": seed,
            "response_format": maze_response_format(),
        },
    }
