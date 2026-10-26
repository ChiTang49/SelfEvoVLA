import time

import numpy as np
import torch

import ript.utils.obs_utils as ObsUtils
import ript.utils.tensor_utils as TensorUtils
from ript.utils.utils import map_tensor_to_device

from planning.candidate_sampler import sample_action_candidates


class CandidatePlanningPolicy:
    def __init__(self, base_policy, num_candidates, value_model=None, device="cuda:0"):
        self.base_policy = base_policy
        self.num_candidates = int(num_candidates)
        self.value_model = value_model
        self.device = device
        self.records = []

    def reset(self):
        self.base_policy.reset()

    def get_action(self, obs, task_id, task_emb=None):
        policy = self.base_policy
        assert hasattr(policy, "action_queue") and policy.action_queue is not None
        policy.eval()
        chosen_context = None
        chosen_indices = None
        if len(policy.action_queue) == 0:
            started = time.perf_counter()
            for key, value in obs.items():
                if key in policy.image_encoders:
                    value = ObsUtils.process_frame(value, channel_dim=3)
                elif key in policy.lowdim_encoders:
                    value = TensorUtils.to_float(value)
                obs[key] = torch.tensor(value)
            batch = {"obs": obs}
            if task_emb is not None:
                batch["task_emb"] = task_emb
            else:
                batch["task_id"] = torch.tensor([task_id], dtype=torch.long)
            batch = map_tensor_to_device(batch, policy.device)
            with torch.no_grad():
                actions, context, action_indices, diagnostics = sample_action_candidates(
                    policy, batch, self.num_candidates
                )
                if actions.shape[0] != 1:
                    raise ValueError("Planning evaluation currently requires num_parallel_envs=1")
                if self.value_model is None:
                    scores = torch.zeros((1, self.num_candidates), device=actions.device)
                else:
                    scores = self.value_model(context, action_indices)
                chosen = int(scores[0].argmax().item())
                policy.action_queue.extend(actions[0, chosen, :policy.action_horizon].detach().cpu().numpy())
                chosen_context = context[:1].detach().cpu().numpy()
                chosen_indices = action_indices[:1, chosen].detach().cpu().numpy()
                record = {
                    **diagnostics,
                    "candidate_scores": scores[0].detach().cpu().tolist(),
                    "chosen_candidate": chosen,
                    "inference_seconds": time.perf_counter() - started,
                    "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                }
                self.records.append(record)
        action = policy.action_queue.popleft()
        return action, chosen_context, chosen_indices

    def summary(self):
        if not self.records:
            return {"policy_inference_calls": 0}
        return {
            "policy_inference_calls": len(self.records),
            "num_candidates": self.num_candidates,
            "uses_value_model": self.value_model is not None,
            "mean_unique_candidate_fraction": float(np.mean([r["unique_candidate_fraction"] for r in self.records])),
            "mean_pairwise_l1": float(np.mean([r["mean_pairwise_l1"] for r in self.records])),
            "all_actions_finite": all(r["finite_actions"] for r in self.records),
            "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in self.records])),
            "max_memory_allocated_bytes": max(r["max_memory_allocated_bytes"] or 0 for r in self.records),
        }
