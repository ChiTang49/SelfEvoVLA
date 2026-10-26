import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from planning.outcome_value import OutcomeValueModel


def ranking_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return {"auroc": None, "auprc": None, "positive_count": positives, "negative_count": negatives}
    order = np.argsort(scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    _, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    sorted_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    for index, count in enumerate(counts):
        sorted_ranks[inverse == index] = start + (count + 1) / 2
        start += count
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = sorted_ranks
    auroc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    descending_order = np.argsort(-scores, kind="stable")
    descending_labels = labels[descending_order]
    descending_scores = scores[descending_order]
    cumulative_positives = np.cumsum(descending_labels)
    group_ends = np.flatnonzero(np.r_[descending_scores[1:] != descending_scores[:-1], True])
    auprc = 0.0
    previous_recall = 0.0
    for end in group_ends:
        true_positives = cumulative_positives[end]
        recall = true_positives / positives
        precision = true_positives / (end + 1)
        auprc += (recall - previous_recall) * precision
        previous_recall = recall
    auprc = float(auprc)
    probabilities = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (probabilities >= lower) & (probabilities < lower + 0.1)
        if mask.any():
            ece += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return {
        "auroc": float(auroc),
        "auprc": auprc,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece_10bin": float(ece),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_fraction": float(labels.mean()),
    }


def load_dataset(data_dirs):
    contexts, actions, labels, states, sources, steps = [], [], [], [], [], []
    source_files = []
    for data_dir in data_dirs:
        for path in sorted(Path(data_dir).glob("*.npz")):
            with np.load(path) as data:
                context = data["context_tokens"]
                action = data["action_indices"]
                count = context.shape[0]
                assert context.shape[:1] == action.shape[:1]
                assert np.isfinite(context).all()
                state = int(data["init_state_index"])
                label = int(data["success"])
                contexts.append(context)
                actions.append(action)
                labels.extend([label] * count)
                states.extend([state] * count)
                sources.extend([str(path)] * count)
                steps.extend(range(count))
                source_files.append({"path": str(path), "checkpoint_sha256": str(data["checkpoint_sha256"])})
    if not contexts:
        raise ValueError("No rollout .npz files found")
    return (
        np.concatenate(contexts).astype(np.float32),
        np.concatenate(actions).astype(np.int64),
        np.asarray(labels, dtype=np.float32),
        np.asarray(states, dtype=np.int64),
        np.asarray(sources),
        np.asarray(steps, dtype=np.int64),
        source_files,
    )


def split_summary(states, labels):
    unique = sorted(set(int(state) for state in states))
    digest = hashlib.sha256(np.asarray(unique, dtype=np.int64).tobytes()).hexdigest()
    return {
        "sample_count": len(states),
        "state_count": len(unique),
        "state_indices": unique,
        "state_set_sha256": digest,
        "positive_samples": int(labels.sum()),
        "negative_samples": int(len(labels) - labels.sum()),
    }


def predict(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for context, action, label in loader:
            score = model(context.to(device), action.to(device))
            labels.append(label.numpy())
            scores.append(score.detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-max-state", type=int, default=29)
    parser.add_argument("--val-max-state", type=int, default=39)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    contexts, actions, labels, states, sources, steps, source_files = load_dataset(args.data_dirs)
    train_mask = states <= args.train_max_state
    val_mask = (states > args.train_max_state) & (states <= args.val_max_state)
    test_mask = states > args.val_max_state
    assert train_mask.any() and val_mask.any() and test_mask.any()
    assert not set(states[train_mask]) & set(states[val_mask])
    assert not set(states[train_mask]) & set(states[test_mask])
    assert not set(states[val_mask]) & set(states[test_mask])

    splits = {}
    for name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        split_labels = labels[mask]
        split = split_summary(states[mask], split_labels)
        splits[name] = split

    datasets = {}
    loaders = {}
    for name, mask, shuffle in (("train", train_mask, True), ("val", val_mask, False), ("test", test_mask, False)):
        datasets[name] = TensorDataset(
            torch.from_numpy(contexts[mask]), torch.from_numpy(actions[mask]), torch.from_numpy(labels[mask])
        )
        loaders[name] = DataLoader(
            datasets[name], batch_size=args.batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(args.seed), pin_memory=device.startswith("cuda"),
        )

    model = OutcomeValueModel(context_dim=contexts.shape[-1]).to(device)
    train_labels = labels[train_mask]
    positive = train_labels.sum()
    negative = len(train_labels) - positive
    pos_weight = torch.tensor(float(negative / max(positive, 1)), device=device).clamp(0.1, 10.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for context, action, label in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            score = model(context.to(device), action.to(device))
            loss = criterion(score, label.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_labels, val_scores = predict(model, loaders["val"], device)
        val_metrics = ranking_metrics(val_labels, val_scores)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(record)
        score = val_metrics["auroc"] if val_metrics["auroc"] is not None else -record["train_loss"]
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    metrics = {}
    for name in ("train", "val", "test"):
        split_labels, split_scores = predict(model, loaders[name], device)
        metrics[name] = ranking_metrics(split_labels, split_scores)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    payload.update({
        "splits": splits,
        "metrics": metrics,
        "training_config": vars(args),
        "source_files": source_files,
        "pos_weight": float(pos_weight),
        "label_semantics": "Terminal simulator success for the action chunk and observation context; learned outcome-value proxy, not exact Bellman Q.",
    })
    torch.save(payload, output)
    output.with_name("value_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    output.with_name("value_data_summary.json").write_text(json.dumps(splits, indent=2) + "\n")
    output.with_name("value_training_history.json").write_text(json.dumps(history, indent=2) + "\n")

    reloaded = OutcomeValueModel.load(output, device)
    test_labels, test_scores = predict(reloaded, loaders["test"], device)
    assert np.allclose(test_scores, split_scores, atol=1e-6)
    print(json.dumps({"checkpoint": str(output), "metrics": metrics, "reload_verified": True}, indent=2))


if __name__ == "__main__":
    main()
