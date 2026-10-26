import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "E3, K=1": ("E10-e3-k1", "E10-e3-k1-eval-seed01"),
    "Feedback, K=1": ("E10-feedback-k1", "E10-feedback-k1-eval-seed01"),
    "E3 + Planning, K=4": ("E10-e3-planning-k4", "E10-e3-planning-k4-eval-seed01"),
    "Feedback + Planning, K=4": ("E10-feedback-planning-k4", "E10-feedback-planning-k4-eval-seed01"),
}


def main():
    root = Path(__file__).resolve().parent.parent
    results = {}
    for method, experiments in METHODS.items():
        seed_results = []
        for seed, experiment in enumerate(experiments):
            if seed == 0 and experiment.startswith("E10-e3-"):
                source = experiment.replace("E10-e3-", "E9-e3-")
            else:
                source = experiment
            path = root / "evidence" / source / "planning_eval_summary.json"
            payload = json.loads(path.read_text())
            seed_results.append({
                "evaluation_seed": seed,
                "experiment_id": source,
                "overall": payload["overall"],
                "heldout": payload["state_strata"]["value_heldout_states_40_49"],
            })
        results[method] = {
            "seeds": seed_results,
            "mean_overall_success_rate": float(np.mean([r["overall"]["success_rate"] for r in seed_results])),
            "mean_heldout_success_rate": float(np.mean([r["heldout"]["success_rate"] for r in seed_results])),
        }
    training = json.loads((root / "evidence" / "E10-feedback-e3-k4-v2" / "checkpoints" / "feedback_training_summary.json").read_text())
    summary = {
        "scope": "One policy training seed and two evaluation seeds. Evaluation-seed replication does not establish cross-training-seed stability.",
        "feedback_training": training,
        "methods": results,
    }
    (root / "evidence" / "feedback_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(METHODS))
    width = 0.26
    colors = ("#94a3b8", "#2563eb")
    for axis, key, title in zip(axes, ("overall", "heldout"), ("All 50 states", "10 value-heldout states")):
        for seed in range(2):
            values = [results[method]["seeds"][seed][key]["success_rate"] for method in METHODS]
            axis.bar(x + (seed - 0.5) * width, values, width, label=f"Evaluation seed {seed}", color=colors[seed])
        means = [results[method][f"mean_{key}_success_rate"] for method in METHODS]
        axis.scatter(x, means, marker="D", color="#111827", s=45, label="Two-seed mean", zorder=3)
        for index, mean in enumerate(means):
            axis.text(index, mean + 0.025, f"{mean:.0%}", ha="center", fontsize=10)
        axis.set_xticks(x, ["E3\nK=1", "Feedback\nK=1", "E3+Plan\nK=4", "Feedback+Plan\nK=4"])
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Success rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("One verified feedback round: two evaluation seeds")
    fig.tight_layout()
    fig.savefig(root / "assets" / "feedback_results.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"summary": "evidence/feedback_summary.json", "methods": results}, indent=2))


if __name__ == "__main__":
    main()
