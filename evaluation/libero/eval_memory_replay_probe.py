from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import draccus
import numpy as np
from PIL import Image
import tqdm

os.environ.setdefault("MUJOCO_GL", "osmesa")

from libero.libero import benchmark

from libero_utils import get_libero_env, get_libero_image, quat2axisangle
from vla_policy import LLaVAClient

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


@dataclass
class ProbeConfig:
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_initial_states: int = 5
    diffusion_seeds: str = "7,17,27"
    intervention_call_index: int = 2
    replay_source_call_index: int = -1
    probe_horizon: int = 5
    num_steps_wait: int = 10
    resolution: int = 256
    port: int = 23456
    request_timeout: int = 600
    local_log_dir: str = "./run_logs/memory_replay_probe"
    run_id_note: str = "replay-minimal"


def parse_diffusion_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("diffusion_seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("At least one diffusion seed is required")
    return seeds


def get_max_steps(task_suite_name: str) -> int:
    max_steps_by_suite = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in max_steps_by_suite:
        raise ValueError(f"Unsupported task suite: {task_suite_name}")
    return max_steps_by_suite[task_suite_name]


def observation_state(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )


def action_for_environment(action) -> np.ndarray:
    actions = np.asarray(action, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action shape [T, 7], got {actions.shape}")

    actions = actions.copy()
    gripper = actions[:, 6]
    actions[np.isclose(gripper, 1.0), 6] = -1.0
    actions[np.isclose(gripper, 0.0), 6] = 1.0
    return actions


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def validate_diagnostic_response(payload: dict) -> None:
    required = {"action", "normalized_action", "memory"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Model response is missing diagnostic fields: {sorted(missing)}")
    normalized_action = np.asarray(payload["normalized_action"])
    if normalized_action.ndim not in (1, 2) or normalized_action.shape[-1] != 7:
        raise ValueError(f"Invalid normalized action shape: {normalized_action.shape}")


def reset_environment(env, initial_state, num_steps_wait: int):
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs


def collect_clean_trajectory(
    env,
    initial_state,
    policy: LLaVAClient,
    task_description: str,
    cfg: ProbeConfig,
    diffusion_seed: int,
    snapshot_name: str,
) -> tuple[list[np.ndarray], bool]:
    obs = reset_environment(env, initial_state, cfg.num_steps_wait)
    policy.set_experiment_seed(diffusion_seed)

    frames = []
    done = False
    low_level_step = cfg.num_steps_wait
    inference_call = 0
    max_steps = get_max_steps(cfg.task_suite_name) + cfg.num_steps_wait

    while low_level_step < max_steps:
        frame = get_libero_image(obs, cfg.resolution)
        frames.append(frame.copy())

        if inference_call == cfg.intervention_call_index:
            policy.save_experiment_state(snapshot_name)

        payload = policy.process_frame_with_diagnostics(
            text=task_description,
            episode_first_frame="True" if inference_call == 0 else "False",
            base_cam=frame,
            states=observation_state(obs),
        )
        validate_diagnostic_response(payload)

        for action in action_for_environment(payload["action"]):
            obs, _, done, _ = env.step(action)
            low_level_step += 1
            if done or low_level_step >= max_steps:
                break

        inference_call += 1
        if done:
            break

    return frames, bool(done)


def resolve_replay_source_index(configured_index: int, frame_count: int, intervention_call_index: int) -> int:
    source_index = configured_index if configured_index >= 0 else frame_count + configured_index
    if source_index < 0 or source_index >= frame_count:
        raise ValueError(
            f"replay_source_call_index={configured_index} is invalid for a {frame_count}-frame trajectory"
        )
    if source_index <= intervention_call_index:
        raise ValueError("The replay frame must occur after intervention_call_index in the clean trajectory")
    return source_index


def make_record(
    *,
    pair_id: str,
    cfg: ProbeConfig,
    diffusion_seed: int,
    condition: str,
    phase: str,
    delay: int,
    input_kind: str,
    input_frame: np.ndarray,
    source_index: int,
    payload: dict,
) -> dict:
    return {
        "pair_id": pair_id,
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "diffusion_seed": diffusion_seed,
        "intervention_call_index": cfg.intervention_call_index,
        "replay_source_call_index": source_index,
        "condition": condition,
        "phase": phase,
        "delay": delay,
        "input_kind": input_kind,
        "input_frame_sha256": frame_hash(input_frame),
        "action": payload["action"],
        "normalized_action": payload["normalized_action"],
        "memory": payload["memory"],
    }


def run_condition(
    *,
    policy: LLaVAClient,
    snapshot_name: str,
    task_description: str,
    cfg: ProbeConfig,
    pair_id: str,
    diffusion_seed: int,
    condition: str,
    write_frame: np.ndarray,
    input_kind: str,
    probe_frames: list[np.ndarray],
    source_index: int,
    retrieval_during_probe: bool,
    rng_seed_override: int | None = None,
) -> list[dict]:
    policy.restore_experiment_state(snapshot_name)
    policy.set_memory_retrieval(True)
    if rng_seed_override is not None:
        policy.set_experiment_seed(rng_seed_override)

    write_payload = policy.process_frame_with_diagnostics(
        text=task_description,
        episode_first_frame="False",
        base_cam=write_frame,
    )
    validate_diagnostic_response(write_payload)
    records = [
        make_record(
            pair_id=pair_id,
            cfg=cfg,
            diffusion_seed=diffusion_seed,
            condition=condition,
            phase="write",
            delay=0,
            input_kind=input_kind,
            input_frame=write_frame,
            source_index=source_index,
            payload=write_payload,
        )
    ]

    policy.set_memory_retrieval(retrieval_during_probe)
    for delay, frame in enumerate(probe_frames, start=1):
        payload = policy.process_frame_with_diagnostics(
            text=task_description,
            episode_first_frame="False",
            base_cam=frame,
        )
        validate_diagnostic_response(payload)
        records.append(
            make_record(
                pair_id=pair_id,
                cfg=cfg,
                diffusion_seed=diffusion_seed,
                condition=condition,
                phase="probe",
                delay=delay,
                input_kind="clean_future",
                input_frame=frame,
                source_index=source_index,
                payload=payload,
            )
        )

    return records


def save_pair_frames(output_dir: Path, pair_id: str, clean_frame: np.ndarray, replay_frame: np.ndarray) -> None:
    frame_dir = output_dir / "frames" / pair_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(clean_frame).save(frame_dir / "clean_write.png")
    Image.fromarray(replay_frame).save(frame_dir / "replay_write.png")


@draccus.wrap()
def eval_memory_replay_probe(cfg: ProbeConfig) -> None:
    if cfg.num_initial_states < 1:
        raise ValueError("num_initial_states must be positive")
    if cfg.intervention_call_index < 1:
        raise ValueError("intervention_call_index must be at least 1 so the snapshot contains clean history")
    if cfg.probe_horizon < 2:
        raise ValueError("probe_horizon must be at least 2 to test a delayed effect")

    diffusion_seeds = parse_diffusion_seeds(cfg.diffusion_seeds)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.task_suite_name}-task{cfg.task_id}-{cfg.run_id_note}-{timestamp}"
    output_dir = Path(cfg.local_log_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "records.jsonl"
    errors_path = output_dir / "errors.jsonl"
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    policy = LLaVAClient(
        base_url=f"http://localhost:{cfg.port}",
        request_timeout=cfg.request_timeout,
    )
    status = policy.get_experiment_status()
    print(f"Experiment API ready: {status}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    if cfg.task_id < 0 or cfg.task_id >= task_suite.n_tasks:
        raise ValueError(f"task_id must be between 0 and {task_suite.n_tasks - 1}")

    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    if cfg.num_initial_states > len(initial_states):
        raise ValueError(
            f"Requested {cfg.num_initial_states} initial states, but only {len(initial_states)} are available"
        )

    env, task_description = get_libero_env(task, resolution=cfg.resolution)
    completed_pairs = 0
    skipped_pairs = 0

    try:
        total_pairs = cfg.num_initial_states * len(diffusion_seeds)
        progress = tqdm.tqdm(total=total_pairs, desc="paired memory probes")
        for initial_state_id in range(cfg.num_initial_states):
            for diffusion_seed in diffusion_seeds:
                pair_id = f"init{initial_state_id:03d}-seed{diffusion_seed}"
                snapshot_name = f"memory-probe-{pair_id}"
                snapshot_saved = False
                try:
                    frames, clean_success = collect_clean_trajectory(
                        env=env,
                        initial_state=initial_states[initial_state_id],
                        policy=policy,
                        task_description=task_description,
                        cfg=cfg,
                        diffusion_seed=diffusion_seed,
                        snapshot_name=snapshot_name,
                    )
                    snapshot_saved = len(frames) > cfg.intervention_call_index

                    required_frames = cfg.intervention_call_index + cfg.probe_horizon + 1
                    if not clean_success:
                        raise RuntimeError("Clean source rollout did not complete the task")
                    if len(frames) < required_frames:
                        raise RuntimeError(
                            f"Clean rollout has {len(frames)} inference frames; at least {required_frames} are required"
                        )

                    source_index = resolve_replay_source_index(
                        cfg.replay_source_call_index,
                        len(frames),
                        cfg.intervention_call_index,
                    )
                    clean_write_frame = frames[cfg.intervention_call_index]
                    replay_write_frame = frames[source_index]
                    probe_frames = frames[
                        cfg.intervention_call_index + 1:cfg.intervention_call_index + cfg.probe_horizon + 1
                    ]
                    save_pair_frames(output_dir, pair_id, clean_write_frame, replay_write_frame)

                    conditions = (
                        ("clean_on", clean_write_frame, "clean_current", True, None),
                        ("clean_twin", clean_write_frame, "clean_current", True, None),
                        ("replay_on", replay_write_frame, "late_stage_replay", True, None),
                        ("clean_off", clean_write_frame, "clean_current", False, None),
                        ("replay_off", replay_write_frame, "late_stage_replay", False, None),
                        ("clean_rng", clean_write_frame, "clean_current", True, diffusion_seed + 1_000_003),
                    )

                    with records_path.open("a", encoding="utf-8") as records_file:
                        for condition, write_frame, input_kind, retrieval_enabled, seed_override in conditions:
                            records = run_condition(
                                policy=policy,
                                snapshot_name=snapshot_name,
                                task_description=task_description,
                                cfg=cfg,
                                pair_id=pair_id,
                                diffusion_seed=diffusion_seed,
                                condition=condition,
                                write_frame=write_frame,
                                input_kind=input_kind,
                                probe_frames=probe_frames,
                                source_index=source_index,
                                retrieval_during_probe=retrieval_enabled,
                                rng_seed_override=seed_override,
                            )
                            for record in records:
                                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            records_file.flush()

                    completed_pairs += 1
                except (RuntimeError, ValueError) as exc:
                    skipped_pairs += 1
                    error = {"pair_id": pair_id, "error": str(exc)}
                    with errors_path.open("a", encoding="utf-8") as errors_file:
                        errors_file.write(json.dumps(error, ensure_ascii=False) + "\n")
                    print(f"Skipping {pair_id}: {exc}")
                finally:
                    if snapshot_saved:
                        policy.delete_experiment_state(snapshot_name)
                    policy.set_memory_retrieval(True)
                    progress.update(1)
        progress.close()
    finally:
        env.close()

    run_summary = {
        "output_dir": str(output_dir),
        "task_description": task_description,
        "completed_pairs": completed_pairs,
        "skipped_pairs": skipped_pairs,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))
    if completed_pairs == 0:
        raise RuntimeError("No valid paired probes were completed; inspect errors.jsonl")


if __name__ == "__main__":
    eval_memory_replay_probe()
