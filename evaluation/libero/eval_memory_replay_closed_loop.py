import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import draccus
import numpy as np
import tqdm

os.environ.setdefault("MUJOCO_GL", "osmesa")

from libero.libero import benchmark

from eval_memory_replay_probe import (
    action_for_environment,
    frame_hash,
    get_max_steps,
    observation_state,
    parse_diffusion_seeds,
    reset_environment,
    resolve_replay_source_index,
    save_pair_frames,
    validate_diagnostic_response,
)
from libero_utils import get_libero_env, get_libero_image, save_rollout_video
from memory_replay_closed_loop_metrics import CLOSED_LOOP_CONDITIONS
from vla_policy import LLaVAClient

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    replay_write: bool
    retrieval_after_write: bool


CONDITION_SPECS = (
    ConditionSpec("clean_on", replay_write=False, retrieval_after_write=True),
    ConditionSpec("clean_twin", replay_write=False, retrieval_after_write=True),
    ConditionSpec("replay_on", replay_write=True, retrieval_after_write=True),
    ConditionSpec("clean_off", replay_write=False, retrieval_after_write=False),
    ConditionSpec("replay_off", replay_write=True, retrieval_after_write=False),
)


@dataclass
class ClosedLoopConfig:
    task_suite_name: str = "libero_10"
    task_id: int = 0
    num_initial_states: int = 1
    diffusion_seeds: str = "7"
    intervention_call_index: int = 2
    replay_source_call_index: int = -1
    num_steps_wait: int = 10
    resolution: int = 256
    port: int = 23456
    request_timeout: int = 600
    local_log_dir: str = "./run_logs/memory_replay_closed_loop"
    run_id_note: str = "long-debug"
    save_videos: bool = True


def execute_action_chunk(env, action, obs, low_level_step: int, max_steps: int):
    done = False
    for environment_action in action_for_environment(action):
        obs, _, done, _ = env.step(environment_action)
        low_level_step += 1
        if done or low_level_step >= max_steps:
            break
    return obs, bool(done), low_level_step


def reset_paired_environment(env, initial_state, num_steps_wait: int):
    env.seed(0)
    return reset_environment(env, initial_state, num_steps_wait)


def collect_clean_source(
    *,
    env,
    initial_state,
    policy: LLaVAClient,
    task_description: str,
    cfg: ClosedLoopConfig,
    diffusion_seed: int,
) -> dict:
    obs = reset_paired_environment(env, initial_state, cfg.num_steps_wait)
    policy.set_memory_retrieval(True)
    policy.set_experiment_seed(diffusion_seed)

    frames = []
    video_frames = []
    done = False
    model_call_index = 0
    low_level_step = cfg.num_steps_wait
    max_steps = get_max_steps(cfg.task_suite_name) + cfg.num_steps_wait

    while low_level_step < max_steps:
        frame = get_libero_image(obs, cfg.resolution)
        frames.append(frame.copy())
        video_frames.append(frame.copy())
        payload = policy.process_frame_with_diagnostics(
            text=task_description,
            episode_first_frame="True" if model_call_index == 0 else "False",
            base_cam=frame,
            states=observation_state(obs),
        )
        validate_diagnostic_response(payload)
        obs, done, low_level_step = execute_action_chunk(
            env,
            payload["action"],
            obs,
            low_level_step,
            max_steps,
        )
        model_call_index += 1
        if done:
            break

    return {
        "success": done,
        "frames": frames,
        "video_frames": video_frames,
        "model_calls": model_call_index,
        "low_level_steps": low_level_step,
    }


def make_step_record(
    *,
    pair_id: str,
    condition: str,
    phase: str,
    cfg: ClosedLoopConfig,
    diffusion_seed: int,
    initial_state_id: int,
    model_call_index: int,
    low_level_step: int,
    input_kind: str,
    input_frame: np.ndarray,
    source_index: int,
    payload: dict,
) -> dict:
    if phase == "write":
        delay = 0
    elif model_call_index >= cfg.intervention_call_index:
        delay = model_call_index - cfg.intervention_call_index + 1
    else:
        delay = None
    return {
        "pair_id": pair_id,
        "condition": condition,
        "phase": phase,
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "diffusion_seed": diffusion_seed,
        "initial_state_id": initial_state_id,
        "intervention_call_index": cfg.intervention_call_index,
        "replay_source_call_index": source_index,
        "model_call_index": model_call_index,
        "low_level_step": low_level_step,
        "delay": delay,
        "input_kind": input_kind,
        "input_frame_sha256": frame_hash(input_frame),
        "action": payload["action"],
        "normalized_action": payload["normalized_action"],
        "memory": payload["memory"],
    }


def run_closed_loop_condition(
    *,
    env,
    initial_state,
    policy: LLaVAClient,
    task_description: str,
    cfg: ClosedLoopConfig,
    pair_id: str,
    initial_state_id: int,
    diffusion_seed: int,
    condition: ConditionSpec,
    replay_frame: np.ndarray,
    source_index: int,
) -> tuple[dict, list[dict], list[np.ndarray]]:
    obs = reset_paired_environment(env, initial_state, cfg.num_steps_wait)
    policy.set_memory_retrieval(True)
    policy.set_experiment_seed(diffusion_seed)

    records = []
    video_frames = []
    done = False
    model_call_index = 0
    low_level_step = cfg.num_steps_wait
    max_steps = get_max_steps(cfg.task_suite_name) + cfg.num_steps_wait

    while low_level_step < max_steps:
        frame = get_libero_image(obs, cfg.resolution)
        video_frames.append(frame.copy())

        if model_call_index == cfg.intervention_call_index:
            write_frame = replay_frame if condition.replay_write else frame
            write_payload = policy.process_frame_with_diagnostics(
                text=task_description,
                episode_first_frame="False",
                base_cam=write_frame,
            )
            validate_diagnostic_response(write_payload)
            records.append(
                make_step_record(
                    pair_id=pair_id,
                    condition=condition.name,
                    phase="write",
                    cfg=cfg,
                    diffusion_seed=diffusion_seed,
                    initial_state_id=initial_state_id,
                    model_call_index=model_call_index,
                    low_level_step=low_level_step,
                    input_kind="late_stage_replay" if condition.replay_write else "clean_current",
                    input_frame=write_frame,
                    source_index=source_index,
                    payload=write_payload,
                )
            )
            policy.set_memory_retrieval(condition.retrieval_after_write)

        payload = policy.process_frame_with_diagnostics(
            text=task_description,
            episode_first_frame="True" if model_call_index == 0 else "False",
            base_cam=frame,
            states=observation_state(obs),
        )
        validate_diagnostic_response(payload)
        records.append(
            make_step_record(
                pair_id=pair_id,
                condition=condition.name,
                phase="control",
                cfg=cfg,
                diffusion_seed=diffusion_seed,
                initial_state_id=initial_state_id,
                model_call_index=model_call_index,
                low_level_step=low_level_step,
                input_kind="environment_current",
                input_frame=frame,
                source_index=source_index,
                payload=payload,
            )
        )

        obs, done, low_level_step = execute_action_chunk(
            env,
            payload["action"],
            obs,
            low_level_step,
            max_steps,
        )
        model_call_index += 1
        if done:
            break

    episode = {
        "pair_id": pair_id,
        "condition": condition.name,
        "task_suite_name": cfg.task_suite_name,
        "task_id": cfg.task_id,
        "task_description": task_description,
        "diffusion_seed": diffusion_seed,
        "initial_state_id": initial_state_id,
        "intervention_call_index": cfg.intervention_call_index,
        "replay_source_call_index": source_index,
        "retrieval_after_write": condition.retrieval_after_write,
        "success": done,
        "model_calls": model_call_index,
        "low_level_steps": low_level_step,
    }
    return episode, records, video_frames


def append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()


@draccus.wrap()
def eval_memory_replay_closed_loop(cfg: ClosedLoopConfig) -> None:
    if cfg.num_initial_states < 1:
        raise ValueError("num_initial_states must be positive")
    if cfg.intervention_call_index < 1:
        raise ValueError("intervention_call_index must be at least 1")
    if tuple(spec.name for spec in CONDITION_SPECS) != CLOSED_LOOP_CONDITIONS:
        raise RuntimeError("Closed-loop conditions do not match the metric definitions")

    diffusion_seeds = parse_diffusion_seeds(cfg.diffusion_seeds)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.task_suite_name}-task{cfg.task_id}-{cfg.run_id_note}-{timestamp}"
    output_dir = Path(cfg.local_log_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes_path = output_dir / "episodes.jsonl"
    steps_path = output_dir / "steps.jsonl"
    sources_path = output_dir / "sources.jsonl"
    errors_path = output_dir / "errors.jsonl"
    (output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    policy = LLaVAClient(
        base_url=f"http://localhost:{cfg.port}",
        request_timeout=cfg.request_timeout,
    )
    print(f"Experiment API ready: {policy.get_experiment_status()}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    if cfg.task_id < 0 or cfg.task_id >= task_suite.n_tasks:
        raise ValueError(f"task_id must be between 0 and {task_suite.n_tasks - 1}")

    task = task_suite.get_task(cfg.task_id)
    initial_states = task_suite.get_task_init_states(cfg.task_id)
    if cfg.num_initial_states > len(initial_states):
        raise ValueError(
            f"Requested {cfg.num_initial_states} initial states, but only "
            f"{len(initial_states)} are available"
        )

    env, task_description = get_libero_env(task, resolution=cfg.resolution)
    completed_pairs = 0
    skipped_pairs = 0

    try:
        total_pairs = cfg.num_initial_states * len(diffusion_seeds)
        progress = tqdm.tqdm(total=total_pairs, desc="closed-loop pairs")
        for initial_state_id in range(cfg.num_initial_states):
            for diffusion_seed in diffusion_seeds:
                pair_id = f"init{initial_state_id:03d}-seed{diffusion_seed}"
                try:
                    source = collect_clean_source(
                        env=env,
                        initial_state=initial_states[initial_state_id],
                        policy=policy,
                        task_description=task_description,
                        cfg=cfg,
                        diffusion_seed=diffusion_seed,
                    )
                    if not source["success"]:
                        raise RuntimeError("Clean source rollout did not complete the task")

                    source_index = resolve_replay_source_index(
                        cfg.replay_source_call_index,
                        len(source["frames"]),
                        cfg.intervention_call_index,
                    )
                    clean_write_frame = source["frames"][cfg.intervention_call_index]
                    replay_frame = source["frames"][source_index]
                    if frame_hash(clean_write_frame) == frame_hash(replay_frame):
                        raise RuntimeError("Replay source frame is identical to the clean write frame")
                    save_pair_frames(output_dir, pair_id, clean_write_frame, replay_frame)

                    source_record = {
                        "pair_id": pair_id,
                        "task_suite_name": cfg.task_suite_name,
                        "task_id": cfg.task_id,
                        "task_description": task_description,
                        "diffusion_seed": diffusion_seed,
                        "initial_state_id": initial_state_id,
                        "success": source["success"],
                        "model_calls": source["model_calls"],
                        "low_level_steps": source["low_level_steps"],
                        "replay_source_call_index": source_index,
                        "clean_write_frame_sha256": frame_hash(clean_write_frame),
                        "replay_frame_sha256": frame_hash(replay_frame),
                    }
                    if cfg.save_videos:
                        source_record["video_path"] = save_rollout_video(
                            source["video_frames"],
                            f"{pair_id}-source",
                            success=source["success"],
                            task_description=task_description,
                            rollout_dir=str(output_dir / "videos"),
                        )

                    pair_episodes = []
                    pair_steps = []
                    for condition in CONDITION_SPECS:
                        episode, records, video_frames = run_closed_loop_condition(
                            env=env,
                            initial_state=initial_states[initial_state_id],
                            policy=policy,
                            task_description=task_description,
                            cfg=cfg,
                            pair_id=pair_id,
                            initial_state_id=initial_state_id,
                            diffusion_seed=diffusion_seed,
                            condition=condition,
                            replay_frame=replay_frame,
                            source_index=source_index,
                        )
                        if cfg.save_videos:
                            episode["video_path"] = save_rollout_video(
                                video_frames,
                                f"{pair_id}-{condition.name}",
                                success=episode["success"],
                                task_description=task_description,
                                rollout_dir=str(output_dir / "videos"),
                            )
                        pair_episodes.append(episode)
                        pair_steps.extend(records)

                    append_jsonl(sources_path, [source_record])
                    append_jsonl(steps_path, pair_steps)
                    append_jsonl(episodes_path, pair_episodes)
                    completed_pairs += 1
                except (RuntimeError, ValueError) as exc:
                    skipped_pairs += 1
                    append_jsonl(
                        errors_path,
                        [{"pair_id": pair_id, "error_type": type(exc).__name__, "error": str(exc)}],
                    )
                    print(f"Skipping {pair_id}: {type(exc).__name__}: {exc}")
                finally:
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
        "episodes_per_pair": len(CONDITION_SPECS),
        "source_rollouts_per_pair": 1,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))
    if completed_pairs == 0:
        raise RuntimeError("No valid closed-loop pairs were completed; inspect errors.jsonl")


if __name__ == "__main__":
    eval_memory_replay_closed_loop()
