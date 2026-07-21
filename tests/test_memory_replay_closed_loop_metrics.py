import sys
import unittest
from pathlib import Path


LIBERO_EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "libero"
sys.path.insert(0, str(LIBERO_EVAL_DIR))

from memory_replay_closed_loop_metrics import (  # noqa: E402
    CLOSED_LOOP_CONDITIONS,
    build_pairwise_outcome_rows,
    evaluate_closed_loop_go_no_go,
    exact_two_sided_binomial_pvalue,
    summarize_conditions,
)


def make_episode(pair_id: str, condition: str, success: bool) -> dict:
    return {
        "pair_id": pair_id,
        "condition": condition,
        "success": success,
        "diffusion_seed": 7 if pair_id == "pair-1" else 17,
        "initial_state_id": 0,
        "intervention_call_index": 2,
        "model_calls": 4,
        "low_level_steps": 42,
    }


def make_step(pair_id: str, condition: str, call_index: int, action_value: float) -> dict:
    return {
        "pair_id": pair_id,
        "condition": condition,
        "phase": "control",
        "model_call_index": call_index,
        "input_frame_sha256": f"{pair_id}-frame-{call_index}",
        "normalized_action": [[action_value] * 7],
    }


def make_synthetic_results() -> tuple[list[dict], list[dict]]:
    outcomes = {
        "pair-1": {
            "clean_on": True,
            "clean_twin": True,
            "replay_on": False,
            "clean_off": True,
            "replay_off": True,
        },
        "pair-2": {condition: True for condition in CLOSED_LOOP_CONDITIONS},
    }
    episodes = []
    steps = []
    for pair_id, pair_outcomes in outcomes.items():
        for condition, success in pair_outcomes.items():
            episodes.append(make_episode(pair_id, condition, success))
            for call_index in range(4):
                value = float(call_index)
                if pair_id == "pair-1" and condition == "replay_on" and call_index == 2:
                    value += 0.25
                steps.append(make_step(pair_id, condition, call_index, value))
    return episodes, steps


class ClosedLoopMetricsTest(unittest.TestCase):
    def test_exact_two_sided_binomial_pvalue(self):
        self.assertEqual(exact_two_sided_binomial_pvalue(0, 0), 1.0)
        self.assertEqual(exact_two_sided_binomial_pvalue(4, 0), 0.125)
        self.assertEqual(exact_two_sided_binomial_pvalue(0, 4), 0.125)
        with self.assertRaises(ValueError):
            exact_two_sided_binomial_pvalue(-1, 2)

    def test_pairwise_rows_and_screening_decision(self):
        episodes, steps = make_synthetic_results()
        rows = build_pairwise_outcome_rows(episodes, steps)

        first = next(row for row in rows if row["pair_id"] == "pair-1")
        self.assertTrue(first["harmful_flip_on"])
        self.assertFalse(first["harmful_flip_off"])
        self.assertEqual(first["first_replay_divergence_delay"], 1)
        self.assertTrue(first["delayed_harmful_flip"])
        self.assertTrue(first["identity_consistent"])

        decision = evaluate_closed_loop_go_no_go(rows)
        self.assertEqual(decision["decision"], "GO_CLOSED_LOOP_SCREEN")
        self.assertEqual(decision["harmful_flips_on"], 1)
        self.assertEqual(decision["harmful_flips_off"], 0)
        self.assertEqual(decision["retrieval_harm_reduction"], 1.0)

    def test_retrieval_mediation_requires_clean_off_successes(self):
        episodes, steps = make_synthetic_results()
        for episode in episodes:
            if episode["condition"] in {"clean_off", "replay_off"}:
                episode["success"] = False
        rows = build_pairwise_outcome_rows(episodes, steps)
        decision = evaluate_closed_loop_go_no_go(rows)
        self.assertIsNone(decision["retrieval_harm_reduction"])
        self.assertFalse(decision["checks"]["retrieval_mediated"])

    def test_condition_summary(self):
        episodes, _ = make_synthetic_results()
        summary = summarize_conditions(episodes)
        replay_on = next(row for row in summary if row["condition"] == "replay_on")
        self.assertEqual(replay_on["n"], 2)
        self.assertEqual(replay_on["successes"], 1)
        self.assertEqual(replay_on["success_rate"], 0.5)

    def test_missing_condition_is_rejected(self):
        episodes, steps = make_synthetic_results()
        episodes = [
            episode
            for episode in episodes
            if not (episode["pair_id"] == "pair-1" and episode["condition"] == "replay_off")
        ]
        with self.assertRaisesRegex(ValueError, "Missing closed-loop conditions"):
            build_pairwise_outcome_rows(episodes, steps)

    def test_incomplete_control_steps_are_rejected(self):
        episodes, steps = make_synthetic_results()
        steps = [
            step
            for step in steps
            if not (
                step["pair_id"] == "pair-1"
                and step["condition"] == "replay_on"
                and step["model_call_index"] == 3
            )
        ]
        with self.assertRaisesRegex(ValueError, "Incomplete control-step records"):
            build_pairwise_outcome_rows(episodes, steps)

    def test_clean_twin_frame_mismatch_fails_identity(self):
        episodes, steps = make_synthetic_results()
        twin_step = next(
            step
            for step in steps
            if step["pair_id"] == "pair-1"
            and step["condition"] == "clean_twin"
            and step["model_call_index"] == 3
        )
        twin_step["input_frame_sha256"] = "different-frame"
        rows = build_pairwise_outcome_rows(episodes, steps)
        first = next(row for row in rows if row["pair_id"] == "pair-1")
        self.assertFalse(first["identity_consistent"])


if __name__ == "__main__":
    unittest.main()
