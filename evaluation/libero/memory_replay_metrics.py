from collections import defaultdict

import numpy as np


COMPARISONS = {
    "identity": ("clean_twin", "clean_on"),
    "replay_on": ("replay_on", "clean_on"),
    "replay_off": ("replay_off", "clean_off"),
    "rng_noise": ("clean_rng", "clean_on"),
}


def action_metrics(left, right) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim == 1:
        left_array = left_array[None, :]
    if right_array.ndim == 1:
        right_array = right_array[None, :]
    if left_array.shape != right_array.shape or left_array.shape[-1] != 7:
        raise ValueError(f"Expected matching [T, 7] actions, got {left_array.shape} and {right_array.shape}")

    difference = left_array - right_array
    return {
        "action_rms": float(np.sqrt(np.mean(np.square(difference)))),
        "translation_rms": float(np.sqrt(np.mean(np.square(difference[:, :3])))),
        "rotation_rms": float(np.sqrt(np.mean(np.square(difference[:, 3:6])))),
        "gripper_flip_rate": float(np.mean(left_array[:, 6] != right_array[:, 6])),
        "max_abs": float(np.max(np.abs(difference))),
    }


def build_pairwise_rows(records: list[dict]) -> list[dict]:
    by_probe = {}
    for record in records:
        if record.get("phase") != "probe":
            continue
        key = (record["pair_id"], int(record["delay"]), record["condition"])
        if key in by_probe:
            raise ValueError(f"Duplicate probe record: {key}")
        by_probe[key] = record

    pair_delays = sorted({(pair_id, delay) for pair_id, delay, _ in by_probe})
    rows = []
    for pair_id, delay in pair_delays:
        for comparison, (left_condition, right_condition) in COMPARISONS.items():
            left_key = (pair_id, delay, left_condition)
            right_key = (pair_id, delay, right_condition)
            if left_key not in by_probe or right_key not in by_probe:
                raise ValueError(
                    f"Missing conditions for {pair_id} at delay {delay}: {left_condition}, {right_condition}"
                )
            metrics = action_metrics(
                by_probe[left_key]["normalized_action"],
                by_probe[right_key]["normalized_action"],
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "delay": delay,
                    "comparison": comparison,
                    **metrics,
                }
            )
    return rows


def bootstrap_mean_interval(values, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sample")
    if values.size == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def aggregate_rows(rows: list[dict], bootstrap_samples: int = 2000) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["comparison"], row["delay"])].append(row["action_rms"])

    aggregates = []
    for (comparison, delay), values in sorted(grouped.items()):
        low, high = bootstrap_mean_interval(values, samples=bootstrap_samples)
        aggregates.append(
            {
                "comparison": comparison,
                "delay": delay,
                "n": len(values),
                "mean_action_rms": float(np.mean(values)),
                "median_action_rms": float(np.median(values)),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return aggregates


def evaluate_go_no_go(
    rows: list[dict],
    *,
    identity_tolerance: float = 1e-4,
    minimum_signal_ratio: float = 2.0,
    minimum_retrieval_reduction: float = 0.7,
) -> dict:
    values_by_comparison = defaultdict(list)
    for row in rows:
        values_by_comparison[row["comparison"]].append(row["action_rms"])

    missing = set(COMPARISONS).difference(values_by_comparison)
    if missing:
        raise ValueError(f"Missing comparisons: {sorted(missing)}")

    identity_max_abs = max(row["max_abs"] for row in rows if row["comparison"] == "identity")
    noise_floor = float(np.median(values_by_comparison["rng_noise"]))
    replay_on = float(np.median(values_by_comparison["replay_on"]))
    replay_off = float(np.median(values_by_comparison["replay_off"]))
    signal_ratio = replay_on / max(noise_floor, 1e-8)
    retrieval_reduction = 1.0 - replay_off / max(replay_on, 1e-8)

    delayed_values = defaultdict(list)
    for row in rows:
        if row["comparison"] == "replay_on" and row["delay"] >= 2:
            delayed_values[row["delay"]].append(row["action_rms"])
    persistent_delayed_effect = any(
        np.median(values) >= minimum_signal_ratio * max(noise_floor, 1e-8)
        for values in delayed_values.values()
    )

    checks = {
        "identity_consistent": identity_max_abs <= identity_tolerance,
        "signal_above_rng_floor": signal_ratio >= minimum_signal_ratio,
        "persistent_after_delay_2": bool(persistent_delayed_effect),
        "retrieval_mediated": retrieval_reduction >= minimum_retrieval_reduction,
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO_GO_REVIEW",
        "checks": checks,
        "identity_max_abs": identity_max_abs,
        "rng_noise_median_action_rms": noise_floor,
        "replay_on_median_action_rms": replay_on,
        "replay_off_median_action_rms": replay_off,
        "signal_to_rng_ratio": signal_ratio,
        "retrieval_reduction": retrieval_reduction,
        "thresholds": {
            "identity_tolerance": identity_tolerance,
            "minimum_signal_ratio": minimum_signal_ratio,
            "minimum_retrieval_reduction": minimum_retrieval_reduction,
        },
    }
