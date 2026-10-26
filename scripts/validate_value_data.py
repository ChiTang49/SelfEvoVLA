import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--expected-per-directory", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()
    reports = []
    for directory in args.directories:
        paths = sorted(Path(directory).glob("*.npz"))
        assert len(paths) == args.expected_per_directory, (directory, len(paths))
        states = []
        successes = []
        chunks = 0
        checkpoint_hashes = set()
        for path in paths:
            with np.load(path) as data:
                state = int(data["init_state_index"])
                states.append(state)
                successes.append(int(data["success"]))
                context = data["context_tokens"]
                actions = data["action_indices"]
                assert context.ndim == 3 and context.shape[1] == 2, (path, context.shape)
                assert actions.ndim == 2 and actions.shape[1] == 8, (path, actions.shape)
                assert context.shape[0] == actions.shape[0] > 0
                assert np.isfinite(context).all()
                assert actions.min() >= 0 and actions.max() < 1024
                chunks += context.shape[0]
                checkpoint_hashes.add(str(data["checkpoint_sha256"]))
        assert sorted(states) == list(range(args.expected_per_directory))
        assert len(checkpoint_hashes) == 1
        state_hash = hashlib.sha256(np.asarray(states, dtype=np.int64).tobytes()).hexdigest()
        reports.append({
            "directory": directory,
            "episodes": len(paths),
            "action_chunks": chunks,
            "successes": sum(successes),
            "state_set_sha256": state_hash,
            "checkpoint_sha256": next(iter(checkpoint_hashes)),
        })
    payload = {
        "directories": reports,
        "split_protocol": {"train": "0-29", "validation": "30-39", "heldout": "40-49"},
        "validation_passed": True,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
