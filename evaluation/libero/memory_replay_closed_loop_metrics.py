from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np


CLOSED_LOOP_CONDITIONS = (
    "clean_on",
    "clean_twin",
    "replay_on",
    "clean_off",
    "replay_off",
)


def exact_two_sided_binomial_pvalue(left_count: int, right_count: int) -> float:
    """Return the exact two-sided p-value for discordant paired outcomes."""
    if left_count < 0 or right_count < 0:
        raise ValueError("Paired outcome counts must be non-negative")
    discordant = left_count + right_count
    if discordant == 0:
        return 1.0
    tail_count = min(left_count, right_count)
    tail_probability = sum(comb(discordant, k) for k in range(tail_count + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail_probability)


def _index_episodes(episodes: list[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for episode in episodes:
        key = (episode["pair_id"], episode["condition"])
        if key in indexed:
            raise ValueError(f"Duplicate closed-loop episode: {key}")
        indexed[key] = episode
    return indexed


def _index_control_steps(steps: list[dict]) -> dict[tuple[str, str, int], dict]:
    indexed = {}
    for step in steps:
        if step.get("phase") != "control":
            continue
        key = (step["pair_id"], step["condition"], int(step["model_call_index"]))
        if key in indexed:
            raise ValueError(f"Duplicate closed-loop control step: {key}")
        indexed[key] = step
    return indexed


def _action_max_abs(left: dict, right: dict) -> float:
    left_action = np.asarray(left["normalized_action"], dtype=np.float64)
    right_action = np.asarray(right["normalized_action"], dtype=np.float64)
    if left_action.shape != right_action.shape or left_action.shape[-1] != 7:
        raise ValueError(
            f"Expected matching normalized actions ending in 7 dimensions, got "
            f"{left_action.shape} and {right_action.shape}"
        )
    return float(np.max(np.abs(left_action - right_action)))


def _condition_steps(
    indexed_steps: dict[tuple[str, str, int], dict],
    pair_id: str,
    condition: str,
) -> dict[int, dict]:
    return {
        model_call_index: step
        for (step_pair_id, step_condition, model_call_index), step in indexed_steps.items()
        if step_pair_id == pair_id and step_condition == condition
    }


def _compare_identity(
    left_steps: dict[int, dict],
    right_steps: dict[int, dict],
) -> tuple[bool, bool, float]:
    lengths_identical = set(left_steps) == set(right_steps)
    common_calls = sorted(set(left_steps).intersection(right_steps))
    if not common_calls:
        return False, False, float("inf")
    frames_identical = all(
        left_steps[index]["input_frame_sha256"] == right_steps[index]["input_frame_sha256"]
        for index in common_calls
    )
    max_abs = max(_action_max_abs(left_steps[index], right_steps[index]) for index in common_calls)
    return lengths_identical, frames_identical, max_abs


def _prefix_consistent(
    indexed_steps: dict[tuple[str, str, int], dict],
    pair_id: str,
    intervention_call_index: int,
    tolerance: float,
) -> bool:
    reference = _condition_steps(indexed_steps, pair_id, "clean_on")
    for condition in CLOSED_LOOP_CONDITIONS:
        candidate = _condition_steps(indexed_steps, pair_id, condition)
        for model_call_index in range(intervention_call_index):
            if model_call_index not in reference or model_call_index not in candidate:
                return False
            if (
                reference[model_call_index]["input_frame_sha256"]
                != candidate[model_call_index]["input_frame_sha256"]
            ):
                return False
            if _action_max_abs(reference[model_call_index], candidate[model_call_index]) > tolerance:
                return False
    return True


def _first_divergence_delay(
    left_steps: dict[int, dict],
    right_steps: dict[int, dict],
    intervention_call_index: int,
    tolerance: float,
) -> int | None:
    common_calls = sorted(set(left_steps).intersection(right_steps))
    for model_call_index in common_calls:
        if model_call_index < intervention_call_index:
            continue
        if _action_max_abs(left_steps[model_call_index], right_steps[model_call_index]) > tolerance:
            return model_call_index - intervention_call_index + 1
    if set(left_steps) != set(right_steps):
        first_unmatched = min(set(left_steps).symmetric_difference(right_steps))
        if first_unmatched >= intervention_call_index:
            return first_unmatched - intervention_call_index + 1
    return None


def build_pairwise_outcome_rows(
    episodes: list[dict],
    steps: list[dict],
    *,
    identity_tolerance: float = 1e-4,
) -> list[dict]:
    indexed_episodes = _index_episodes(episodes)
    indexed_steps = _index_control_steps(steps)
    pair_ids = sorted({episode["pair_id"] for episode in episodes})
    rows = []

    for pair_id in pair_ids:
        missing = [
            condition
            for condition in CLOSED_LOOP_CONDITIONS
            if (pair_id, condition) not in indexed_episodes
        ]
        if missing:
            raise ValueError(f"Missing closed-loop conditions for {pair_id}: {missing}")

        pair_episodes = {
            condition: indexed_episodes[(pair_id, condition)]
            for condition in CLOSED_LOOP_CONDITIONS
        }
        intervention_indices = {
            int(episode["intervention_call_index"]) for episode in pair_episodes.values()
        }
        if len(intervention_indices) != 1:
            raise ValueError(f"Inconsistent intervention indices for {pair_id}")
        intervention_call_index = intervention_indices.pop()

        for field in ("diffusion_seed", "initial_state_id"):
            values = {int(episode[field]) for episode in pair_episodes.values()}
            if len(values) != 1:
                raise ValueError(f"Inconsistent {field} values for {pair_id}")

        condition_steps = {
            condition: _condition_steps(indexed_steps, pair_id, condition)
            for condition in CLOSED_LOOP_CONDITIONS
        }
        for condition, condition_episode in pair_episodes.items():
            expected_calls = set(range(int(condition_episode["model_calls"])))
            actual_calls = set(condition_steps[condition])
            if actual_calls != expected_calls:
                raise ValueError(f"Incomplete control-step records for {pair_id}/{condition}")

        clean_steps = condition_steps["clean_on"]
        twin_steps = condition_steps["clean_twin"]
        replay_steps = condition_steps["replay_on"]
        clean_off_steps = condition_steps["clean_off"]
        replay_off_steps = condition_steps["replay_off"]

        twin_lengths_identical, twin_frames_identical, twin_max_abs = _compare_identity(
            clean_steps, twin_steps
        )
        prefix_consistent = _prefix_consistent(
            indexed_steps,
            pair_id,
            intervention_call_index,
            identity_tolerance,
        )
        replay_divergence_delay = _first_divergence_delay(
            clean_steps,
            replay_steps,
            intervention_call_index,
            identity_tolerance,
        )
        replay_off_divergence_delay = _first_divergence_delay(
            clean_off_steps,
            replay_off_steps,
            intervention_call_index,
            identity_tolerance,
        )

        clean_success = bool(pair_episodes["clean_on"]["success"])
        twin_success = bool(pair_episodes["clean_twin"]["success"])
        replay_success = bool(pair_episodes["replay_on"]["success"])
        clean_off_success = bool(pair_episodes["clean_off"]["success"])
        replay_off_success = bool(pair_episodes["replay_off"]["success"])
        post_intervention_replay_calls = max(
            0,
            int(pair_episodes["replay_on"]["model_calls"]) - intervention_call_index,
        )
        harmful_flip_on = clean_success and not replay_success
        beneficial_flip_on = not clean_success and replay_success
        harmful_flip_off = clean_off_success and not replay_off_success
        beneficial_flip_off = not clean_off_success and replay_off_success

        rows.append(
            {
                "pair_id": pair_id,
                "diffusion_seed": int(pair_episodes["clean_on"]["diffusion_seed"]),
                "initial_state_id": int(pair_episodes["clean_on"]["initial_state_id"]),
                "intervention_call_index": intervention_call_index,
                "clean_on_success": clean_success,
                "clean_twin_success": twin_success,
                "replay_on_success": replay_success,
                "clean_off_success": clean_off_success,
                "replay_off_success": replay_off_success,
                "harmful_flip_on": harmful_flip_on,
                "beneficial_flip_on": beneficial_flip_on,
                "harmful_flip_off": harmful_flip_off,
                "beneficial_flip_off": beneficial_flip_off,
                "clean_twin_lengths_identical": twin_lengths_identical,
                "clean_twin_frames_identical": twin_frames_identical,
                "clean_twin_max_abs": twin_max_abs,
                "prefix_consistent": prefix_consistent,
                "identity_consistent": (
                    twin_lengths_identical
                    and twin_frames_identical
                    and twin_max_abs <= identity_tolerance
                    and clean_success == twin_success
                    and prefix_consistent
                ),
                "first_replay_divergence_delay": replay_divergence_delay,
                "first_replay_off_divergence_delay": replay_off_divergence_delay,
                "post_intervention_replay_calls": post_intervention_replay_calls,
                "delayed_harmful_flip": bool(
                    harmful_flip_on
                    and post_intervention_replay_calls >= 2
                ),
            }
        )

    return rows


def summarize_conditions(episodes: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["condition"]].append(episode)

    missing = set(CLOSED_LOOP_CONDITIONS).difference(grouped)
    if missing:
        raise ValueError(f"Missing closed-loop conditions: {sorted(missing)}")

    rows = []
    for condition in CLOSED_LOOP_CONDITIONS:
        condition_episodes = grouped[condition]
        successes = sum(bool(episode["success"]) for episode in condition_episodes)
        rows.append(
            {
                "condition": condition,
                "n": len(condition_episodes),
                "successes": successes,
                "success_rate": successes / len(condition_episodes),
                "mean_model_calls": float(
                    np.mean([episode["model_calls"] for episode in condition_episodes])
                ),
                "mean_low_level_steps": float(
                    np.mean([episode["low_level_steps"] for episode in condition_episodes])
                ),
            }
        )
    return rows


def evaluate_closed_loop_go_no_go(
    rows: list[dict],
    *,
    identity_tolerance: float = 1e-4,
    minimum_clean_success_rate: float = 0.8,
    minimum_harmful_flips: int = 1,
    minimum_retrieval_reduction: float = 0.7,
) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty closed-loop result")

    clean_successes = sum(row["clean_on_success"] for row in rows)
    replay_successes = sum(row["replay_on_success"] for row in rows)
    clean_off_successes = sum(row["clean_off_success"] for row in rows)
    replay_off_successes = sum(row["replay_off_success"] for row in rows)
    harmful_flips_on = sum(row["harmful_flip_on"] for row in rows)
    beneficial_flips_on = sum(row["beneficial_flip_on"] for row in rows)
    harmful_flips_off = sum(row["harmful_flip_off"] for row in rows)
    beneficial_flips_off = sum(row["beneficial_flip_off"] for row in rows)
    delayed_harmful_flips = sum(row["delayed_harmful_flip"] for row in rows)
    identity_max_abs = max(float(row["clean_twin_max_abs"]) for row in rows)
    harmful_rate_on = harmful_flips_on / clean_successes if clean_successes else None
    harmful_rate_off = harmful_flips_off / clean_off_successes if clean_off_successes else None
    retrieval_reduction = (
        1.0 - harmful_rate_off / harmful_rate_on
        if harmful_rate_on and harmful_rate_off is not None
        else None
    )
    clean_success_rate = clean_successes / len(rows)

    checks = {
        "identity_consistent": (
            identity_max_abs <= identity_tolerance
            and all(row["identity_consistent"] for row in rows)
        ),
        "clean_baseline_reliable": clean_success_rate >= minimum_clean_success_rate,
        "harmful_outcome_observed": harmful_flips_on >= minimum_harmful_flips,
        "delayed_harmful_outcome_observed": delayed_harmful_flips >= minimum_harmful_flips,
        "retrieval_mediated": (
            harmful_flips_on >= minimum_harmful_flips
            and retrieval_reduction is not None
            and retrieval_reduction >= minimum_retrieval_reduction
        ),
    }
    return {
        "decision": "GO_CLOSED_LOOP_SCREEN" if all(checks.values()) else "NO_GO_REVIEW",
        "checks": checks,
        "n_pairs": len(rows),
        "clean_on_successes": clean_successes,
        "replay_on_successes": replay_successes,
        "clean_off_successes": clean_off_successes,
        "replay_off_successes": replay_off_successes,
        "clean_on_success_rate": clean_success_rate,
        "replay_on_success_rate": replay_successes / len(rows),
        "clean_off_success_rate": clean_off_successes / len(rows),
        "replay_off_success_rate": replay_off_successes / len(rows),
        "harmful_flips_on": harmful_flips_on,
        "beneficial_flips_on": beneficial_flips_on,
        "harmful_flips_off": harmful_flips_off,
        "beneficial_flips_off": beneficial_flips_off,
        "delayed_harmful_flips": delayed_harmful_flips,
        "identity_max_abs": identity_max_abs,
        "harmful_flip_rate_on_clean_successes": harmful_rate_on,
        "harmful_flip_rate_off_clean_successes": harmful_rate_off,
        "retrieval_harm_reduction": retrieval_reduction,
        "paired_exact_pvalue": exact_two_sided_binomial_pvalue(
            harmful_flips_on, beneficial_flips_on
        ),
        "thresholds": {
            "identity_tolerance": identity_tolerance,
            "minimum_clean_success_rate": minimum_clean_success_rate,
            "minimum_harmful_flips": minimum_harmful_flips,
            "minimum_retrieval_reduction": minimum_retrieval_reduction,
        },
    }
