import numpy as np
import torch

from resource_aware_control.actions import action_count
from resource_aware_control.environment import OBSERVATION_DIM
from resource_aware_control.networks import PolicyNetwork, ValueNetwork


def test_network_output_shapes() -> None:
    observation = torch.zeros(OBSERVATION_DIM)
    assert PolicyNetwork(hidden_size=8)(observation).shape == (action_count(),)
    assert ValueNetwork(hidden_size=8)(observation).shape == ()


def test_policy_save_and_load(tmp_path) -> None:
    torch.manual_seed(2)
    policy = PolicyNetwork(hidden_size=8)
    path = tmp_path / "policy.pt"
    policy.save(path)
    restored = PolicyNetwork.load(path)
    observation = torch.as_tensor(np.ones(OBSERVATION_DIM), dtype=torch.float32)
    assert torch.equal(policy(observation), restored(observation))
