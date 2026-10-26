import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

log_path = Path(__file__).resolve().parent.parent / "evidence" / sys.argv[1]
metrics_path = log_path / "metrics.jsonl"
summary_path = log_path / "training_summary.json"
text = (log_path / "logs" / "train.log").read_text(errors="replace")
rollout_times = [float(value) for value in re.findall(r"finished generate rollout episodes in ([0-9.]+) seconds", text)]
metric_lines = []
for line in text.splitlines():
    if line.startswith("rank 0 metrics "):
        metric_lines.append(line.split("rank 0 metrics ", 1)[1].strip())

records = []
for index, raw in enumerate(metric_lines):
    values = ast.literal_eval(raw)
    values["iteration"] = index
    if index < len(rollout_times):
        values["rollout_seconds"] = rollout_times[index]
    records.append(values)

with metrics_path.open("w") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

termination_counts = Counter(record.get("termination_reason", "unknown") for record in records)
summary = {
    "iterations": len(records),
    "skipped_no_advantage_iterations": sum(record.get("skipped_no_advantage", 0) for record in records),
    "rollout_budget_exhausted_iterations": sum(record.get("rollout_budget_exhausted", 0) for record in records),
    "samples_checked": sum(record.get("samples_checked", 0) for record in records),
    "accepted_groups": sum(record.get("accepted_groups", 0) for record in records),
    "accepted_episodes": sum(record.get("accepted_episodes", 0) for record in records),
    "discarded_uniform_groups": sum(record.get("discarded_uniform_groups", 0) for record in records),
    "discarded_all_zero_groups": sum(record.get("discarded_all_zero_groups", 0) for record in records),
    "discarded_all_one_groups": sum(record.get("discarded_all_one_groups", 0) for record in records),
    "discarded_episodes": sum(record.get("discarded_episodes", 0) for record in records),
    "termination_reasons": dict(termination_counts),
    "rollout_seconds": sum(rollout_times),
}
for key in ("pg_loss_stats", "pg_ratio_stats", "mean_scores", "mean_advantage", "grad_norm"):
    values = [record[key] for record in records if key in record]
    if values:
        summary[f"mean_{key}"] = sum(values) / len(values)
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
