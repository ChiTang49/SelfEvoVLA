import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = {
    "Main": {
        "value": "E9-value-final-v2",
        "SFT": "E9-sft-k1",
        "RIPT": "E9-e3-k1",
        "SFT+Plan": "E9-sft-planning-k4",
        "RIPT+Plan": "E9-e3-planning-k4",
    },
    "Task 5": {
        "value": "E11-task5-value",
        "SFT": "E11-task5-sft",
        "RIPT": "E11-task5-bript-eval",
        "SFT+Plan": "E11-task5-sft-planning-k4",
        "RIPT+Plan": "E11-task5-bript-planning-k4",
    },
    "Task 6": {
        "value": "E11-task6-value",
        "SFT": "E11-task6-sft",
        "RIPT": "E11-task6-bript-eval",
        "SFT+Plan": "E11-task6-sft-planning-k4",
        "RIPT+Plan": "E11-task6-bript-planning-k4",
    },
}
METHODS = ("SFT", "RIPT", "SFT+Plan", "RIPT+Plan")


def main():
    root = Path(__file__).resolve().parent.parent
    summary = {"scope": "Three LIBERO-Long tasks, one policy training seed per task, evaluation seed 0. Main-task feedback is reported separately with two evaluation seeds.", "tasks": {}}
    for task, entries in TASKS.items():
        task_result = {"value_metrics": json.loads((root / "evidence" / entries["value"] / "value_metrics.json").read_text()), "methods": {}}
        for method in METHODS:
            payload = json.loads((root / "evidence" / entries[method] / "planning_eval_summary.json").read_text())
            task_result["methods"][method] = {
                "experiment_id": entries[method],
                "overall": payload["overall"],
                "heldout": payload["state_strata"]["value_heldout_states_40_49"],
            }
        task_result["macro_experiment_id"] = entries["value"]
        summary["tasks"][task] = task_result
    summary["macro_average"] = {
        method: {
            "overall_success_rate": float(np.mean([summary["tasks"][task]["methods"][method]["overall"]["success_rate"] for task in TASKS])),
            "heldout_success_rate": float(np.mean([summary["tasks"][task]["methods"][method]["heldout"]["success_rate"] for task in TASKS])),
        }
        for method in METHODS
    }
    (root / "evidence" / "multitask_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(TASKS))
    width = 0.2
    colors = ("#94a3b8", "#2563eb", "#f59e0b", "#7c3aed")
    for axis, key, title in zip(axes[:2], ("overall", "heldout"), ("All 50 states", "10 value-heldout states")):
        for index, method in enumerate(METHODS):
            values = [summary["tasks"][task]["methods"][method][key]["success_rate"] for task in TASKS]
            axis.bar(x + (index - 1.5) * width, values, width, label=method, color=colors[index])
        axis.set_xticks(x, list(TASKS))
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Success rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)
    for index, task in enumerate(TASKS):
        metrics = summary["tasks"][task]["value_metrics"]["test"]
        axis_x = index * 2
        axes[2].bar(axis_x - 0.35, metrics["auroc"], 0.7, label="AUROC", color="#2563eb")
        axes[2].bar(axis_x + 0.35, metrics["auprc"], 0.7, label="AUPRC", color="#f59e0b")
        axes[2].text(axis_x - 0.35, metrics["auroc"] + 0.02, f"{metrics['auroc']:.3f}", ha="center", fontsize=9)
        axes[2].text(axis_x + 0.35, metrics["auprc"] + 0.02, f"{metrics['auprc']:.3f}", ha="center", fontsize=9)
    axes[2].set_xticks([0, 2, 4], list(TASKS))
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Outcome-value heldout metrics")
    axes[2].grid(axis="y", alpha=0.25)
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles[:2], labels[:2])
    fig.tight_layout()
    fig.savefig(root / "assets" / "multitask_results.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"summary": "evidence/multitask_summary.json", "macro_average": summary["macro_average"]}, indent=2))


if __name__ == "__main__":
    main()
