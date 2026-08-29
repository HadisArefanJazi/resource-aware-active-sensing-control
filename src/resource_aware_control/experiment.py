import json
from pathlib import Path
from typing import Any

import numpy as np

from resource_aware_control.baselines import (
    ClassicalController,
    FixedSensingController,
    FullInformationController,
    RLController,
)
from resource_aware_control.config import (
    CostLimits,
    EnvironmentConfig,
    TrainingConfig,
    evaluation_scenarios,
)
from resource_aware_control.evaluation import evaluate_controller
from resource_aware_control.training import train_policy


def run_experiment(
    training: TrainingConfig,
    evaluation_episodes: int = 100,
    limits: CostLimits | None = None,
    environment_config: EnvironmentConfig | None = None,
    artifact_directory: str | Path | None = None,
) -> dict[str, Any]:
    limits = limits or CostLimits()
    environment_config = environment_config or EnvironmentConfig()
    scenarios = evaluation_scenarios()
    constrained_policy, constrained_summary = train_policy(
        training,
        environment_config,
        limits,
        constrained=True,
        scenarios=scenarios,
    )
    unconstrained_policy, unconstrained_summary = train_policy(
        training,
        environment_config,
        limits,
        constrained=False,
        scenarios=scenarios,
    )
    if artifact_directory is not None:
        artifact_path = Path(artifact_directory)
        constrained_policy.save(artifact_path / "constrained_policy.pt")
        unconstrained_policy.save(artifact_path / "unconstrained_policy.pt")

    controllers = {
        "resource_aware_rl": RLController(constrained_policy),
        "full_information": FullInformationController(),
        "fixed_sensing": FixedSensingController(interval=4),
        "unconstrained_rl": RLController(unconstrained_policy),
        "classical_control": ClassicalController(),
    }

    scenario_metrics = {}
    for policy_name, controller in controllers.items():
        policy_metrics = {}
        for scenario in scenarios:
            metrics = evaluate_controller(
                controller,
                scenario,
                episodes=evaluation_episodes,
                seed=training.seed * 1_000,
                environment_config=environment_config,
                limits=limits,
            )
            policy_metrics[scenario.name] = metrics
        scenario_metrics[policy_name] = policy_metrics

    overall = {}
    for policy_name, policy_metrics in scenario_metrics.items():
        overall[policy_name] = _aggregate(policy_metrics)

    scenario_records = []
    for scenario in scenarios:
        scenario_records.append(scenario.to_dict())

    return {
        "experiment": "resource_aware_active_sensing_control",
        "training": training.to_dict(),
        "environment": {
            "horizon": environment_config.horizon,
            "dynamics": environment_config.dynamics,
            "control_gain": environment_config.control_gain,
            "state_limit": environment_config.state_limit,
        },
        "limits": limits.to_dict(),
        "scenarios": scenario_records,
        "training_summary": {
            "resource_aware_rl": constrained_summary.to_dict(),
            "unconstrained_rl": unconstrained_summary.to_dict(),
        },
        "overall": overall,
        "by_scenario": scenario_metrics,
    }


def _aggregate(scenario_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    metrics = list(scenario_metrics.values())
    reward_values = []
    rmse_values = []
    information_values = []
    violation_values = []
    budget_violation_values = []
    sensing_values = []
    communication_values = []
    for metric in metrics:
        reward_values.append(metric["mean_cumulative_reward"])
        rmse_values.append(metric["control_rmse"])
        information_values.append(metric["mean_information_cost"])
        violation_values.append(metric["state_violation_rate"])
        budget_violation_values.append(metric["information_budget_violation_frequency"])
        sensing_values.append(metric["sensing_rate"])
        communication_values.append(metric["communication_rate"])

    nominal_rmse = scenario_metrics["nominal"]["control_rmse"]
    worst_rmse = max(rmse_values)
    return {
        "mean_cumulative_reward": round(float(np.mean(reward_values)), 4),
        "mean_control_rmse": round(float(np.mean(rmse_values)), 4),
        "mean_information_cost": round(float(np.mean(information_values)), 4),
        "mean_state_violation_rate": round(float(np.mean(violation_values)), 4),
        "information_budget_violation_frequency": round(
            float(np.mean(budget_violation_values)),
            4,
        ),
        "mean_sensing_rate": round(float(np.mean(sensing_values)), 4),
        "mean_communication_rate": round(float(np.mean(communication_values)), 4),
        "robustness_score": round(nominal_rmse / max(worst_rmse, 1e-6), 4),
    }


def write_results(results: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def format_results(results: dict[str, Any]) -> str:
    headers = ("policy", "reward", "rmse", "info", "violations", "budget", "robust")
    rows = []
    for name, metric in results["overall"].items():
        rows.append(
            (
                name,
                f"{metric['mean_cumulative_reward']:.2f}",
                f"{metric['mean_control_rmse']:.3f}",
                f"{metric['mean_information_cost']:.2f}",
                f"{metric['mean_state_violation_rate']:.3f}",
                f"{metric['information_budget_violation_frequency']:.2f}",
                f"{metric['robustness_score']:.2f}",
            )
        )

    widths = []
    for index, header in enumerate(headers):
        width = len(header)
        for row in rows:
            width = max(width, len(row[index]))
        widths.append(width)

    lines = []
    header_cells = []
    for index, header in enumerate(headers):
        header_cells.append(header.ljust(widths[index]))
    lines.append("  ".join(header_cells))

    for row in rows:
        cells = []
        for index, value in enumerate(row):
            cells.append(value.ljust(widths[index]))
        lines.append("  ".join(cells))

    return "\n".join(lines)
