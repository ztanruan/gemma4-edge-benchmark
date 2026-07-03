from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma_vllm_benchmark.maze_navigation import (
    ALLOWED_ACTIONS,
    MazeSolver,
    MazeState,
    apply_action,
    load_maze_levels,
    parse_action,
    validate_maze_levels,
)


@pytest.fixture(scope="module")
def levels(project_root: Path):
    return load_maze_levels(project_root / "maze_navigation" / "levels.yaml")


def test_levels_load_and_have_unique_ids(levels):
    assert levels, "levels.yaml must define at least one level"
    ids = [level.id for level in levels]
    assert len(ids) == len(set(ids))


def test_all_levels_are_solvable(levels):
    for validation in validate_maze_levels(levels):
        assert validation["optimal_action_count"] is not None, (
            f"level {validation['level_id']} has no path from start to exit"
        )


def test_required_optimal_actions_match_solver(levels):
    for level in levels:
        if not level.required_optimal_actions:
            continue
        solver = MazeSolver(level)
        path = solver.shortest_path(level.start_state)
        assert path is not None
        for action in level.required_optimal_actions:
            assert action in path, (
                f"level {level.id} requires {action!r} on the optimal path, "
                f"but the solver found {path}"
            )


def test_parse_action_accepts_every_allowed_action():
    for action in ALLOWED_ACTIONS:
        result = parse_action(json.dumps({"action": action}))
        assert result["format_valid"], result["message"]
        assert result["action"] == action


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        json.dumps(["move_forward"]),
        json.dumps({"action": "fly"}),
        json.dumps({"action": "move_forward", "extra": 1}),
        json.dumps({"move": "move_forward"}),
    ],
)
def test_parse_action_rejects_malformed_output(content):
    result = parse_action(content)
    assert not result["format_valid"]
    assert result["action"] is None
    assert result["message"]


def test_turning_changes_heading_only(levels):
    level = levels[0]
    state = level.start_state
    outcome = apply_action(level, state, "turn_left")
    after = outcome["state_after"]
    assert outcome["transition_valid"]
    assert (after.x, after.y, after.z) == (state.x, state.y, state.z)
    assert after.heading != state.heading


def test_full_turn_returns_to_start_heading(levels):
    level = levels[0]
    state = level.start_state
    for _ in range(4):
        state = apply_action(level, state, "turn_right")["state_after"]
    assert state == level.start_state


def test_invalid_action_token_is_rejected(levels):
    level = levels[0]
    outcome = apply_action(level, level.start_state, "teleport")
    assert not outcome["action_valid"]
    assert outcome["state_after"] == level.start_state


def test_solver_optimal_actions_reach_exit(levels):
    for level in levels:
        solver = MazeSolver(level)
        state: MazeState = level.start_state
        path = solver.shortest_path(state)
        assert path is not None
        for action in path:
            outcome = apply_action(level, state, action)
            assert outcome["transition_valid"], (
                f"level {level.id}: solver action {action!r} was rejected"
            )
            state = outcome["state_after"]
        assert level.is_goal(state)
