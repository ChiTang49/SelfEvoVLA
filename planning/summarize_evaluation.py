import argparse
import json
import math
from pathlib import Path

import numpy as np


def summarize(episodes):
    count = len(episodes)
    successes = sum(bool(episode["success"]) for episode in episodes)
    rate = successes / count
    z = 1.96
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": rate,
        "wilson_95": [center - half, center + half],
        "mean_episode_steps": float(np.mean([episode["episode_steps"] for episode in episodes])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    directory = args.experiment_dir
    episodes = json.loads((directory / "episode_results.json").read_text())["episodes"]
    assert len(episodes) > 0
    result = {
        "experiment_id": directory.name,
        "overall": summarize(episodes),
        "state_strata": {
            "value_train_states_0_29": summarize([e for e in episodes if e["init_state_index"] <= 29]),
            "value_validation_states_30_39": summarize([e for e in episodes if 30 <= e["init_state_index"] <= 39]),
            "value_heldout_states_40_49": summarize([e for e in episodes if e["init_state_index"] >= 40]),
        },
    }
    planning_path = directory / "planning_metrics.json"
    if planning_path.exists():
        planning = json.loads(planning_path.read_text())
        result["planning"] = planning["summary"]
        scores = [record["candidate_scores"] for record in planning["records"]]
        result["planning"]["mean_within_context_score_std"] = float(np.mean(np.std(scores, axis=1)))
    output = directory / "planning_eval_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
