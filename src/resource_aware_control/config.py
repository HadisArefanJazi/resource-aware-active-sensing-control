from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostLimits:
    information_per_step: float = 0.80
    violation_per_step: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentConfig:
    horizon: int = 40
    dynamics: float = 1.05
    control_gain: float = 0.75
    max_control: float = 1.0
    state_limit: float = 4.0
    control_effort_weight: float = 0.05


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 1_500
    learning_rate: float = 2e-3
    dual_learning_rate: float = 5e-2
    gamma: float = 0.98
    entropy_coefficient: float = 0.03
    hidden_size: int = 64
    seed: int = 7

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class Scenario:
    name: str
    observation_noise: float
    disturbance_std: float
    sensor_dropout: float = 0.0
    communication_dropout: float = 0.0
    dynamics_scale: float = 1.0

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def evaluation_scenarios() -> list[Scenario]:
    return [
        Scenario("nominal", observation_noise=0.25, disturbance_std=0.08),
        Scenario("noisy", observation_noise=0.80, disturbance_std=0.08),
        Scenario("disturbed", observation_noise=0.25, disturbance_std=0.30),
        Scenario(
            "sensor_dropout",
            observation_noise=0.35,
            disturbance_std=0.12,
            sensor_dropout=0.35,
        ),
        Scenario(
            "communication_limited",
            observation_noise=0.35,
            disturbance_std=0.12,
            communication_dropout=0.55,
        ),
        Scenario(
            "dynamics_shift",
            observation_noise=0.40,
            disturbance_std=0.18,
            sensor_dropout=0.15,
            dynamics_scale=1.08,
        ),
    ]
