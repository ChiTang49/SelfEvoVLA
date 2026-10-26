import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = [
    ("sft", 1, "E9-sft-k1"),
    ("sft", 4, "E9-sft-planning-k4"),
    ("sft", 8, "E9-sft-planning-k8"),
    ("e3", 1, "E9-e3-k1"),
    ("e3", 4, "E9-e3-planning-k4"),
    ("e3", 8, "E9-e3-planning-k8"),
]


def main():
    root = Path(__file__).resolve().parent.parent
    experiments = {}
    for policy, candidates, experiment in EXPERIMENTS:
        path = root / "evidence" / experiment / "planning_eval_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        experiments[experiment] = json.loads(path.read_text())
        experiments[experiment]["policy"] = policy
        experiments[experiment]["num_candidates"] = candidates

    value_dir = root / "evidence" / "E9-value-final-v2"
    summary = {
        "scope": "Main LIBERO-Long task; one policy training seed. Planning evaluations have 50 states overall and 10 value-heldout states; they are independent of the earlier E1-E3 formal runs.",
        "value_metrics": json.loads((value_dir / "value_metrics.json").read_text()),
        "value_data": json.loads((value_dir / "value_data_summary.json").read_text()),
        "experiments": experiments,
    }
    (root / "evidence" / "planning_core_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    width = 0.35
    x = np.arange(3)
    for policy_index, (policy, label, color) in enumerate((("sft", "SFT", "#64748b"), ("e3", "Bounded RIPT", "#2563eb"))):
        records = [experiments[experiment] for p, _, experiment in EXPERIMENTS if p == policy]
        overall = [record["overall"]["success_rate"] for record in records]
        heldout = [record["state_strata"]["value_heldout_states_40_49"]["success_rate"] for record in records]
        offset = (policy_index - 0.5) * width
        axes[0].bar(x + offset, overall, width, label=label, color=color)
        axes[1].bar(x + offset, heldout, width, label=label, color=color)
        latency = [1000 * record["planning"]["mean_inference_seconds"] for record in records]
        axes[2].plot([1, 4, 8], latency, marker="o", label=label, color=color)
    for axis, title in zip(axes[:2], ("All 50 evaluation states", "10 value-heldout states")):
        axis.set_xticks(x, ["K=1", "K=4", "K=8"])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Success rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[2].set_xticks([1, 4, 8])
    axes[2].set_title("Candidate generation + scoring")
    axes[2].set_ylabel("Mean inference time (ms)")
    axes[2].grid(alpha=0.25)
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(assets / "planning_results.png", dpi=170)
    plt.close(fig)

    history = json.loads((value_dir / "value_training_history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [record["epoch"] for record in history]
    axes[0].plot(epochs, [record["train_loss"] for record in history], marker="o")
    axes[0].set(title="Outcome-value training", xlabel="Epoch", ylabel="Weighted BCE loss", yscale="log")
    axes[1].plot(epochs, [record["auroc"] for record in history], label="Validation AUROC")
    axes[1].plot(epochs, [record["auprc"] for record in history], label="Validation AUPRC")
    axes[1].set(title="State-disjoint validation", xlabel="Epoch", ylim=(0, 1.05))
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(assets / "value_training_curve.png", dpi=170)
    plt.close(fig)

    planning = json.loads((root / "evidence" / "E9-sft-planning-k4" / "planning_metrics.json").read_text())
    records = planning["records"][:24]
    fig, ax = plt.subplots(figsize=(11, 4))
    scores = np.asarray([record["candidate_scores"] for record in records])
    for candidate in range(scores.shape[1]):
        ax.plot(scores[:, candidate], marker="o", label=f"Candidate {candidate}")
    chosen = [record["chosen_candidate"] for record in records]
    ax.scatter(np.arange(len(records)), scores[np.arange(len(records)), chosen], color="black", s=75, facecolors="none", label="Selected")
    ax.set(title="Outcome-value scores on the first 24 replanning calls", xlabel="Replanning call", ylabel="Score logit")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(assets / "candidate_scores.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"summary": "evidence/planning_core_summary.json", "experiments": len(experiments)}))


if __name__ == "__main__":
    main()
