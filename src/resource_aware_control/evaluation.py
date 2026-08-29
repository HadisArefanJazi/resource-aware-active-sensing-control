import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from resource_aware_control.actions import InformationMode, JointAction
from resource_aware_control.baselines import Controller
from resource_aware_control.config import CostLimits, EnvironmentConfig, Scenario
from resource_aware_control.environment import ActiveSensingControlEnv


@dataclass(frozen=True)
class EpisodeMetrics:
    cumulative_reward: float
    state_squared_error: float
    control_squared: float
    information_cost: float
    state_violations: float
    sensed_steps: int
    communicated_steps: int


def run_episode(
    controller: Controller,
    scenario: Scenario,
    seed: int,
    environment_config: EnvironmentConfig | None = None,
    limits: CostLimits | None = None,
) -> EpisodeMetrics:
    environment_config = environment_config or EnvironmentConfig()
    limits = limits or CostLimits()
    environment = ActiveSensingControlEnv(scenario, environment_config, limits, seed=seed)
    observation = environment.reset()
    reward = 0.0
    state_squared_error = 0.0
    control_squared = 0.0
    information_cost = 0.0
    state_violations = 0.0
    sensed_steps = 0
    communicated_steps = 0
    done = False
    while not done:
        action_index = controller.select_action(observation, environment)
        action = JointAction.decode(action_index)
        result = environment.step(action_index)
        reward += result.reward
        state_squared_error += result.true_state**2
        control_squared += result.control**2
        information_cost += result.information_cost
        state_violations += result.violation
        sensed_steps += int(action.information != InformationMode.NONE)
        communicated_steps += int(action.information == InformationMode.COMMUNICATE)
        observation = result.observation
        done = result.done
    return EpisodeMetrics(
        reward,
        state_squared_error,
        control_squared,
        information_cost,
        state_violations,
        sensed_steps,
        communicated_steps,
    )


def evaluate_controller(
    controller: Controller,
    scenario: Scenario,
    episodes: int = 100,
    seed: int = 100,
    environment_config: EnvironmentConfig | None = None,
    limits: CostLimits | None = None,
) -> dict[str, Any]:
    environment_config = environment_config or EnvironmentConfig()
    limits = limits or CostLimits()
    results = []
    for episode in range(episodes):
        result = run_episode(
            controller,
            scenario,
            seed + episode,
            environment_config,
            limits,
        )
        results.append(result)

    steps = episodes * environment_config.horizon
    reward_values = []
    information_values = []
    total_state_squared_error = 0.0
    total_control_squared = 0.0
    total_state_violations = 0.0
    total_sensed_steps = 0
    total_communicated_steps = 0
    budget_violation_count = 0
    budget = limits.information_per_step * environment_config.horizon

    for result in results:
        reward_values.append(result.cumulative_reward)
        information_values.append(result.information_cost)
        total_state_squared_error += result.state_squared_error
        total_control_squared += result.control_squared
        total_state_violations += result.state_violations
        total_sensed_steps += result.sensed_steps
        total_communicated_steps += result.communicated_steps
        if result.information_cost > budget:
            budget_violation_count += 1

    return {
        "episodes": episodes,
        "mean_cumulative_reward": round(float(np.mean(reward_values)), 4),
        "reward_std": round(float(np.std(reward_values)), 4),
        "control_rmse": round(math.sqrt(total_state_squared_error / steps), 4),
        "mean_control_effort": round(total_control_squared / steps, 4),
        "mean_information_cost": round(float(np.mean(information_values)), 4),
        "information_cost_per_step": round(sum(information_values) / steps, 4),
        "state_violation_rate": round(total_state_violations / steps, 4),
        "information_budget_violation_frequency": round(budget_violation_count / episodes, 4),
        "sensing_rate": round(total_sensed_steps / steps, 4),
        "communication_rate": round(total_communicated_steps / steps, 4),
    }
