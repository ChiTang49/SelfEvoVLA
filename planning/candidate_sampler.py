import torch


@torch.no_grad()
def sample_action_candidates(policy, data, num_candidates):
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    data = policy.preprocess_input(data, train_mode=False)
    context = policy.get_context(data)
    batch_size = context.shape[0]
    expanded_context = context.repeat_interleave(num_candidates, dim=0)
    action_indices = policy.policy_prior.get_indices_top_k(expanded_context, policy.codebook_size)
    actions = policy.autoencoder.decode_actions(action_indices)
    actions = actions.reshape(batch_size, num_candidates, *actions.shape[1:])
    action_indices = action_indices.reshape(batch_size, num_candidates, -1)
    if batch_size == 1:
        unique = torch.unique(action_indices[0], dim=0).shape[0]
        unique_fraction = unique / num_candidates
        if num_candidates > 1:
            pairwise = (actions[0, :, None] - actions[0, None, :]).abs().mean()
        else:
            pairwise = actions.new_zeros(())
    else:
        unique_fraction = actions.new_tensor(float("nan"))
        pairwise = actions.new_tensor(float("nan"))
    diagnostics = {
        "unique_candidate_fraction": float(unique_fraction),
        "mean_pairwise_l1": float(pairwise),
        "finite_actions": bool(torch.isfinite(actions).all()),
    }
    return actions, context, action_indices, diagnostics
