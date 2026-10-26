import os
import json
import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from ript.model_loader import load_quest_model


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="../sources/ript-vla/config", version_base=None)
def main(cfg):
    data_dir = Path(os.environ["RIPT_FEEDBACK_DATA_DIR"])
    output = Path(os.environ["RIPT_FEEDBACK_OUTPUT"])
    epochs = int(os.environ.get("RIPT_FEEDBACK_EPOCHS", "2"))
    learning_rate = float(os.environ.get("RIPT_FEEDBACK_LR", "1e-5"))
    max_state = int(os.environ.get("RIPT_FEEDBACK_MAX_STATE", "29"))
    seed = int(os.environ.get("RIPT_FEEDBACK_SEED", "0"))
    torch.manual_seed(seed)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=10800))
    device = "cuda:0"
    torch.cuda.set_device(0)
    model, _, _, _ = load_quest_model(
        cfg=cfg,
        local_tasks=cfg.task.task_names_to_use,
        device=device,
        device_id=0,
        world_size=1,
        use_ddp=True,
    )

    contexts, actions, source_files = [], [], []
    states = []
    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path) as data:
            state = int(data["init_state_index"])
            if not bool(data["success"]) or state > max_state:
                continue
            contexts.append(data["context_tokens"])
            actions.append(data["action_indices"])
            states.append(state)
            source_files.append(str(path))
    if not contexts:
        raise ValueError("No verified successful trajectories in the feedback training states")
    context_array = np.concatenate(contexts).astype(np.float32)
    action_array = np.concatenate(actions).astype(np.int64)
    dataset = TensorDataset(torch.from_numpy(context_array), torch.from_numpy(action_array))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))

    policy = model.module
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.policy_prior.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(policy.policy_prior.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        policy.train()
        losses = []
        for context, indices in loader:
            context = context.to(device)
            indices = indices.to(device)
            start = torch.full((indices.shape[0], 1), policy.start_token, device=device, dtype=torch.long)
            inputs = torch.cat([start, indices[:, :-1]], dim=1)
            logits = policy.policy_prior(inputs, context)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), indices.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.policy_prior.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "batches": len(losses)}
        history.append(record)
        print(json.dumps(record), flush=True)

    policy.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "initial_checkpoint": cfg.checkpoint_path,
        "data_dir": str(data_dir),
        "source_files": source_files,
        "feedback_states": sorted(set(states)),
        "feedback_episodes": len(source_files),
        "feedback_action_chunks": len(dataset),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "max_state": max_state,
        "history": history,
        "verified_simulator_success_only": True,
    }
    torch.save({"model": policy.state_dict(), "feedback_metadata": metadata}, output)
    output.with_name("feedback_training_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(output), "feedback_episodes": len(source_files), "feedback_action_chunks": len(dataset)}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
