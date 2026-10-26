import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
experiment_id = sys.argv[1]
selection_path = root / "evidence/task_selection.yaml"
main_task = None
if selection_path.exists():
    for line in selection_path.read_text().splitlines():
        if line.startswith("main_task_name: "):
            main_task = line.split(": ", 1)[1]
            break
task_name = sys.argv[2] if len(sys.argv) > 2 else main_task
experiment_dir = root / "evidence" / experiment_id
metrics_path = experiment_dir / "metrics.jsonl"
records = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []

if records:
    x = [record["iteration"] + 1 for record in records]
    score_key = next((key for key in records[0] if key.startswith("rl_train_succeess_rate/")), None)
    scores = [record.get(score_key, record.get("mean_scores")) for record in records]
    accepted = [record.get("accepted_episodes", 0) for record in records]
    discarded = [record.get("discarded_episodes", 0) for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, scores, marker="o")
    axes[0].set(title="Training rollout success", xlabel="Iteration", ylabel="Success rate", ylim=(0, 1.05))
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, accepted, marker="o", label="Accepted episodes")
    axes[1].plot(x, discarded, marker="s", label="Discarded episodes")
    axes[1].set(title="Sampling volume", xlabel="Iteration", ylabel="Episodes")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    output = experiment_dir / "training_curve.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)

if not task_name:
    raise SystemExit(0)

rows = []
with (root / "evidence/results.csv").open() as handle:
    for row in csv.DictReader(handle):
        if row["task_name"] == task_name and row["experiment_id"].startswith(("E1-", "E2-", "E3-")) and "smoke" not in row["experiment_id"]:
            rows.append(row)
if not rows:
    raise SystemExit(0)

preferred_order = [
    "E1-main-sft",
    "E1-main-ript",
    "E2-main-rloo-par8-seed00-eval",
    "E2-main-rloo-seed00-eval",
    "E3-main-bript-seed00-eval",
]
order = {name: index for index, name in enumerate(preferred_order)}
rows.sort(key=lambda row: (order.get(row["experiment_id"], 100), row["experiment_id"]))
labels = [row["experiment_id"] for row in rows]
rates = [float(row["success_rate"]) for row in rows]
errors = []
for row in rows:
    n = int(row["rollout_count"])
    p = float(row["success_rate"])
    z = 1.96
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    errors.append((center - half, center + half))

fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.6), 4.5))
x = list(range(len(rows)))
ax.bar(x, rates, yerr=[[rate - low for rate, (low, high) in zip(rates, errors)], [high - rate for rate, (low, high) in zip(rates, errors)]], capsize=5)
ax.set_xticks(x, labels, rotation=25, ha="right")
ax.set(title="Success rate on main task", ylabel="Success rate", ylim=(0, 1.05))
ax.grid(axis="y", alpha=0.3)
for index, rate in enumerate(rates):
    ax.text(index, rate + 0.02, f"{rate:.1%}", ha="center", va="bottom")
fig.tight_layout()
output = root / "evidence" / f"success_comparison_{experiment_id}.png"
fig.savefig(output, dpi=160)
plt.close(fig)
print(output)
