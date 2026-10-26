import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

log_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
experiment_id = sys.argv[3]
task_name = sys.argv[4]
expected_count = int(sys.argv[5])
exit_code = int(sys.argv[6])
text = log_path.read_text(errors="replace")
matches = []
for line in text.splitlines():
    marker = "[RANK 0] Rollout results:"
    if marker in line:
        matches.append(line.split(marker, 1)[1].strip())
if not matches:
    raise RuntimeError(f"no rollout results found in {log_path}")
result = ast.literal_eval(matches[-1])
rollout = result["rollout"]
actual_count = int(rollout["rollout_count"])
summary = {
    "experiment_id": experiment_id,
    "task_name": task_name,
    "expected_rollout_count": expected_count,
    "actual_rollout_count": actual_count,
    "success_count": round(float(rollout["overall_success_rate"]) * actual_count),
    "success_rate": float(rollout["overall_success_rate"]),
    "average_reward": float(rollout["overall_average_reward"]),
    "environments_solved": int(rollout["environments_solved"]),
    "per_task_success_rate": result.get("rollout_success_rate", {}),
    "exit_code": exit_code,
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
if actual_count != expected_count:
    raise RuntimeError(f"rollout count mismatch: {actual_count} != {expected_count}")
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
