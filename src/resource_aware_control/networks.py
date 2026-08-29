from pathlib import Path

import torch
from torch import nn

from resource_aware_control.actions import action_count
from resource_aware_control.environment import OBSERVATION_DIM


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_count()),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "hidden_size": self.hidden_size},
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PolicyNetwork":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        policy = cls(hidden_size=int(payload["hidden_size"]))
        policy.load_state_dict(payload["state_dict"])
        policy.eval()
        return policy


class ValueNetwork(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)
