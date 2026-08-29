from resource_aware_control.config import EnvironmentConfig, TrainingConfig
from resource_aware_control.experiment import run_experiment


def test_experiment_includes_all_policies_and_scenarios() -> None:
    results = run_experiment(
        TrainingConfig(episodes=3, hidden_size=8, seed=3),
        evaluation_episodes=1,
        environment_config=EnvironmentConfig(horizon=5),
    )
    assert set(results["overall"]) == {
        "resource_aware_rl",
        "full_information",
        "fixed_sensing",
        "unconstrained_rl",
        "classical_control",
    }
    assert len(results["scenarios"]) == 6
    assert set(results["by_scenario"]["resource_aware_rl"]) == {
        scenario["name"] for scenario in results["scenarios"]
    }
