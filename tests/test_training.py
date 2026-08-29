from resource_aware_control.config import EnvironmentConfig, TrainingConfig
from resource_aware_control.training import train_policy


def test_constrained_training_smoke() -> None:
    _, summary = train_policy(
        TrainingConfig(episodes=5, hidden_size=8, seed=2),
        environment_config=EnvironmentConfig(horizon=8),
        constrained=True,
    )
    assert summary.episodes == 5
    assert summary.constrained
    assert summary.information_multiplier >= 0
    assert summary.violation_multiplier >= 0


def test_unconstrained_training_keeps_dual_variables_zero() -> None:
    _, summary = train_policy(
        TrainingConfig(episodes=3, hidden_size=8, seed=2),
        environment_config=EnvironmentConfig(horizon=5),
        constrained=False,
    )
    assert summary.information_multiplier == 0
    assert summary.violation_multiplier == 0
