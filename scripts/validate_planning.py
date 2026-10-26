import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
payload = json.loads(path.read_text())
summary = payload["summary"]
assert summary["policy_inference_calls"] > 0, "No candidate generation calls recorded"
assert summary["all_actions_finite"], "Non-finite candidate actions"
assert summary["mean_unique_candidate_fraction"] >= 0.5, "Candidates are highly duplicated"
assert summary["mean_pairwise_l1"] > 0, "Candidates have zero action diversity"
print(json.dumps(summary, indent=2))
