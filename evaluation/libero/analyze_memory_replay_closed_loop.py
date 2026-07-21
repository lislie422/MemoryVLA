import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_replay_closed_loop_metrics import (
    build_pairwise_outcome_rows,
    evaluate_closed_loop_go_no_go,
    summarize_conditions,
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_success_rates(path: Path, condition_rows: list[dict]) -> None:
    labels = {
        "clean_on": "Clean / retrieval on",
        "clean_twin": "Clean deterministic twin",
        "replay_on": "Replay / retrieval on",
        "clean_off": "Clean / retrieval off",
        "replay_off": "Replay / retrieval off",
    }
    colors = ["#0072B2", "#777777", "#D55E00", "#56B4E9", "#E69F00"]
    x_positions = range(len(condition_rows))
    success_rates = [row["success_rate"] for row in condition_rows]

    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(x_positions, success_rates, color=colors)
    axis.set_xticks(
        list(x_positions),
        [labels[row["condition"]] for row in condition_rows],
        rotation=18,
        ha="right",
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Task success rate")
    axis.set_title("Closed-loop outcome after controlled memory replay")
    axis.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, condition_rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.02, bar.get_height() + 0.025),
            f'{row["successes"]}/{row["n"]}',
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze paired closed-loop MemoryVLA replay outcomes."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--identity_tolerance", type=float, default=1e-4)
    parser.add_argument("--minimum_clean_success_rate", type=float, default=0.8)
    parser.add_argument("--minimum_harmful_flips", type=int, default=1)
    parser.add_argument("--minimum_retrieval_reduction", type=float, default=0.7)
    args = parser.parse_args()

    episodes = load_jsonl(args.run_dir / "episodes.jsonl")
    steps = load_jsonl(args.run_dir / "steps.jsonl")
    pairwise_rows = build_pairwise_outcome_rows(
        episodes,
        steps,
        identity_tolerance=args.identity_tolerance,
    )
    condition_rows = summarize_conditions(episodes)
    decision = evaluate_closed_loop_go_no_go(
        pairwise_rows,
        identity_tolerance=args.identity_tolerance,
        minimum_clean_success_rate=args.minimum_clean_success_rate,
        minimum_harmful_flips=args.minimum_harmful_flips,
        minimum_retrieval_reduction=args.minimum_retrieval_reduction,
    )

    write_csv(args.run_dir / "pairwise_outcomes.csv", pairwise_rows)
    write_csv(args.run_dir / "condition_summary.csv", condition_rows)
    (args.run_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_success_rates(args.run_dir / "closed_loop_success_rates.png", condition_rows)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
