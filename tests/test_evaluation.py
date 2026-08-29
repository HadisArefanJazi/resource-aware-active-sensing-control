import torch

from resource_aware_control.baselines import (
    ClassicalController,
    FixedSensingController,
    RLController,
)
from resource_aware_control.config import EnvironmentConfig, evaluation_scenarios
from resource_aware_control.environment import ActiveSensingControlEnv
from resource_aware_control.evaluation import evaluate_controller, run_episode
from resource_aware_control.networks import PolicyNetwork


def test_fixed_sensing_respects_expected_schedule() -> None:
    result = run_episode(
        FixedSensingController(interval=4),
        evaluation_scenarios()[0],
        seed=1,
        environment_config=EnvironmentConfig(horizon=8),
    )
    assert result.sensed_steps == 2
    assert result.information_cost == 2


def test_evaluation_reports_control_and_constraint_metrics() -> None:
    metrics = evaluate_controller(
        ClassicalController(),
        evaluation_scenarios()[0],
        episodes=2,
        seed=1,
        environment_config=EnvironmentConfig(horizon=8),
    )
    assert metrics["episodes"] == 2
    assert metrics["control_rmse"] >= 0
    assert metrics["mean_information_cost"] == 8
    assert 0 <= metrics["state_violation_rate"] <= 1
    assert metrics["information_budget_violation_frequency"] == 1


def test_rl_controller_can_use_greedy_evaluation() -> None:
    policy = PolicyNetwork(hidden_size=8)
    for parameter in policy.parameters():
        torch.nn.init.zeros_(parameter)
    environment = ActiveSensingControlEnv(evaluation_scenarios()[0], seed=1)
    observation = environment.reset()
    assert RLController(policy, stochastic=False).select_action(observation, environment) == 0
