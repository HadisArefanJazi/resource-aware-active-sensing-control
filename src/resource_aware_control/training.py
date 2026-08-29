import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from resource_aware_control.config import (
    CostLimits,
    EnvironmentConfig,
    Scenario,
    TrainingConfig,
    evaluation_scenarios,
)
from resource_aware_control.environment import ActiveSensingControlEnv
from resource_aware_control.networks import PolicyNetwork, ValueNetwork


@dataclass(frozen=True)
class TrainingSummary:
    constrained: bool
    episodes: int
    mean_reward_last_100: float
    mean_information_last_100: float
    violation_rate_last_100: float
    information_multiplier: float
    violation_multiplier: float

    def to_dict(self) -> dict[str, bool | float | int]:
        return {
            "constrained": self.constrained,
            "episodes": self.episodes,
            "mean_reward_last_100": round(self.mean_reward_last_100, 4),
            "mean_information_last_100": round(self.mean_information_last_100, 4),
            "violation_rate_last_100": round(self.violation_rate_last_100, 4),
            "information_multiplier": round(self.information_multiplier, 4),
            "violation_multiplier": round(self.violation_multiplier, 4),
        }


def train_policy(
    training: TrainingConfig,
    environment_config: EnvironmentConfig | None = None,
    limits: CostLimits | None = None,
    constrained: bool = True,
    scenarios: list[Scenario] | None = None,
) -> tuple[PolicyNetwork, TrainingSummary]:
    """Train a primal-dual actor-critic policy with two average constraints."""

    environment_config = environment_config or EnvironmentConfig()
    limits = limits or CostLimits()
    scenarios = scenarios or evaluation_scenarios()
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)

    policy = PolicyNetwork(training.hidden_size)
    value = ValueNetwork(training.hidden_size)
    network_parameters = list(policy.parameters()) + list(value.parameters())
    optimizer = torch.optim.Adam(network_parameters, lr=training.learning_rate)
    multipliers = torch.zeros(2, dtype=torch.float32)
    reward_history: list[float] = []
    information_history: list[float] = []
    violation_history: list[float] = []

    for episode in range(training.episodes):
        scenario = random.choice(scenarios)
        environment = ActiveSensingControlEnv(
            scenario,
            environment_config,
            limits,
            seed=training.seed * 10_000 + episode,
        )
        observation = environment.reset()
        log_probabilities: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        information_costs: list[float] = []
        violations: list[float] = []
        done = False

        while not done:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            distribution = Categorical(logits=policy(observation_tensor))
            action = distribution.sample()
            result = environment.step(int(action.item()))
            log_probabilities.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            values.append(value(observation_tensor))
            rewards.append(result.reward)
            information_costs.append(result.information_cost)
            violations.append(result.violation)
            observation = result.observation
            done = result.done

        # Apply the current constraint penalties to each step reward.
        information_multiplier = float(multipliers[0])
        violation_multiplier = float(multipliers[1])
        lagrangian_rewards = []
        for index, reward in enumerate(rewards):
            information_gap = information_costs[index] - limits.information_per_step
            violation_gap = violations[index] - limits.violation_per_step
            adjusted_reward = (
                reward
                - information_multiplier * information_gap
                - violation_multiplier * violation_gap
            )
            lagrangian_rewards.append(adjusted_reward)

        # Compute discounted returns from the end of the episode to the start.
        reversed_returns = []
        running_return = 0.0
        for reward in reversed(lagrangian_rewards):
            running_return = reward + training.gamma * running_return
            reversed_returns.append(running_return)
        returns = list(reversed(reversed_returns))

        return_tensor = torch.tensor(returns, dtype=torch.float32)
        value_tensor = torch.stack(values)
        advantage = return_tensor - value_tensor.detach()
        policy_loss = -(torch.stack(log_probabilities) * advantage).mean()
        value_loss = 0.5 * torch.square(value_tensor - return_tensor).mean()
        entropy_bonus = torch.stack(entropies).mean()
        loss = policy_loss + value_loss - training.entropy_coefficient * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network_parameters, max_norm=1.0)
        optimizer.step()

        average_information = float(np.mean(information_costs))
        average_violation = float(np.mean(violations))
        if constrained:
            constraint_gap = torch.tensor(
                [
                    average_information - limits.information_per_step,
                    average_violation - limits.violation_per_step,
                ]
            )
            multipliers = torch.clamp(
                multipliers + training.dual_learning_rate * constraint_gap,
                min=0.0,
                max=25.0,
            )

        reward_history.append(float(sum(rewards)))
        information_history.append(float(sum(information_costs)))
        violation_history.append(average_violation)

    window = min(100, training.episodes)
    summary = TrainingSummary(
        constrained=constrained,
        episodes=training.episodes,
        mean_reward_last_100=float(np.mean(reward_history[-window:])),
        mean_information_last_100=float(np.mean(information_history[-window:])),
        violation_rate_last_100=float(np.mean(violation_history[-window:])),
        information_multiplier=float(multipliers[0]),
        violation_multiplier=float(multipliers[1]),
    )
    return policy, summary
