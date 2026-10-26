import torch
from torch import nn


class OutcomeValueModel(nn.Module):
    def __init__(self, context_dim, vocab_size=1024, action_tokens=8, action_embedding_dim=32, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.config = {
            "context_dim": int(context_dim),
            "vocab_size": int(vocab_size),
            "action_tokens": int(action_tokens),
            "action_embedding_dim": int(action_embedding_dim),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
        }
        self.action_tokens = int(action_tokens)
        self.action_embedding = nn.Embedding(vocab_size, action_embedding_dim)
        feature_dim = 2 * context_dim + action_tokens * action_embedding_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context, action_indices):
        single_candidate = action_indices.ndim == 2
        if single_candidate:
            action_indices = action_indices.unsqueeze(1)
        if action_indices.ndim != 3 or action_indices.shape[-1] != self.action_tokens:
            raise ValueError("action_indices has an incompatible shape")
        if context.ndim != 3 or context.shape[0] != action_indices.shape[0]:
            raise ValueError("context must have shape [batch, sequence, dim]")
        context = context.float()
        context_feature = torch.cat([context.mean(dim=1), context[:, -1]], dim=-1).unsqueeze(1)
        context_feature = context_feature.expand(-1, action_indices.shape[1], -1)
        action_feature = self.action_embedding(action_indices).flatten(2)
        feature = torch.cat([context_feature, action_feature], dim=-1)
        scores = self.head(feature).squeeze(-1)
        return scores[:, 0] if single_candidate else scores

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
