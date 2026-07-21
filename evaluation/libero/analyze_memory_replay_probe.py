import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from memory_replay_metrics import aggregate_rows, build_pairwise_rows, evaluate_go_no_go


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
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
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_delayed_deviation(path: Path, aggregates: list[dict]) -> None:
    labels = {
        "replay_on": "Replay / retrieval on",
        "replay_off": "Replay / retrieval off",
        "rng_noise": "Clean RNG reference",
        "identity": "Deterministic twin",
    }
    colors = {
        "replay_on": "#D55E00",
        "replay_off": "#0072B2",
        "rng_noise": "#009E73",
        "identity": "#777777",
    }

    fig, axis = plt.subplots(figsize=(8, 5))
    comparisons = sorted({row["comparison"] for row in aggregates})
    for comparison in comparisons:
        rows = sorted(
            (row for row in aggregates if row["comparison"] == comparison),
            key=lambda row: row["delay"],
        )
        delays = [row["delay"] for row in rows]
        means = [row["mean_action_rms"] for row in rows]
        lows = [row["ci95_low"] for row in rows]
        highs = [row["ci95_high"] for row in rows]
        axis.plot(
            delays,
            means,
            marker="o",
            label=labels[comparison],
            color=colors[comparison],
        )
        axis.fill_between(delays, lows, highs, alpha=0.15, color=colors[comparison])

    axis.set_xlabel("Delay after replay write (model calls)")
    axis.set_ylabel("Normalized action RMS deviation")
    axis.set_title("Delayed action deviation under controlled memory replay")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze controlled MemoryVLA replay-probe records.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing records.jsonl")
    parser.add_argument("--identity_tolerance", type=float, default=1e-4)
    parser.add_argument("--minimum_signal_ratio", type=float, default=2.0)
    parser.add_argument("--minimum_retrieval_reduction", type=float, default=0.7)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    args = parser.parse_args()

    records_path = args.run_dir / "records.jsonl"
    records = load_records(records_path)
    pairwise_rows = build_pairwise_rows(records)
    aggregates = aggregate_rows(pairwise_rows, bootstrap_samples=args.bootstrap_samples)
    decision = evaluate_go_no_go(
        pairwise_rows,
        identity_tolerance=args.identity_tolerance,
        minimum_signal_ratio=args.minimum_signal_ratio,
        minimum_retrieval_reduction=args.minimum_retrieval_reduction,
    )

    write_csv(args.run_dir / "pairwise_metrics.csv", pairwise_rows)
    write_csv(args.run_dir / "aggregate_metrics.csv", aggregates)
    (args.run_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_delayed_deviation(args.run_dir / "delayed_action_deviation.png", aggregates)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
