from __future__ import annotations

import torch
from torch import nn


class RewardModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1]
        reward = self.value_head(last_hidden)
        return reward.squeeze(-1)


def pairwise_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
