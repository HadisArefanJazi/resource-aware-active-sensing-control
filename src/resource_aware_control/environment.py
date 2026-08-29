import math
from dataclasses import dataclass

import numpy as np

from resource_aware_control.actions import InformationMode, JointAction
from resource_aware_control.config import CostLimits, EnvironmentConfig, Scenario

OBSERVATION_DIM = 6
INFORMATION_COSTS = {
    InformationMode.NONE: 0.0,
    InformationMode.SENSE: 1.0,
    InformationMode.COMMUNICATE: 3.0,
}


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    information_cost: float
    violation: float
    done: bool
    true_state: float
    control: float
    measurement_received: bool


class ActiveSensingControlEnv:
    """Partially observed unstable linear system with costly sensing channels."""

    def __init__(
        self,
        scenario: Scenario,
        config: EnvironmentConfig | None = None,
        limits: CostLimits | None = None,
        seed: int = 0,
    ) -> None:
        self.scenario = scenario
        self.config = config or EnvironmentConfig()
        self.limits = limits or CostLimits()
        self.rng = np.random.default_rng(seed)
        self.true_state = 0.0
        self.estimate = 0.0
        self.uncertainty = 1.0
        self.measurement_age = 0
        self.cumulative_information = 0.0
        self.last_control = 0.0
        self.time_step = 0

    def reset(self, initial_state: float | None = None) -> np.ndarray:
        if initial_state is None:
            self.true_state = float(self.rng.normal(0.0, 1.25))
        else:
            self.true_state = float(initial_state)

        self.estimate = 0.0
        self.uncertainty = 1.0
        self.measurement_age = self.config.horizon
        self.cumulative_information = 0.0
        self.last_control = 0.0
        self.time_step = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        budget = max(self.limits.information_per_step * self.config.horizon, 1e-6)
        normalized_estimate = np.clip(
            self.estimate / self.config.state_limit,
            -2.0,
            2.0,
        )
        normalized_uncertainty = np.clip(self.uncertainty / 2.0, 0.0, 2.0)
        normalized_age = min(self.measurement_age / self.config.horizon, 1.0)
        normalized_information = min(self.cumulative_information / budget, 2.0)
        episode_progress = self.time_step / self.config.horizon
        normalized_control = self.last_control / self.config.max_control

        return np.asarray(
            [
                normalized_estimate,
                normalized_uncertainty,
                normalized_age,
                normalized_information,
                episode_progress,
                normalized_control,
            ],
            dtype=np.float32,
        )

    def step(self, action_index: int) -> StepResult:
        if self.time_step >= self.config.horizon:
            raise RuntimeError("Episode is complete; call reset() before stepping again.")
        action = JointAction.decode(action_index)
        control = action.control * self.config.max_control

        # Update the hidden physical state.
        dynamics = self.config.dynamics * self.scenario.dynamics_scale
        disturbance = float(self.rng.normal(0.0, self.scenario.disturbance_std))
        next_state = dynamics * self.true_state + self.config.control_gain * control + disturbance
        self.true_state = float(np.clip(next_state, -10.0, 10.0))

        # Predict the belief before using a new measurement.
        predicted_estimate = dynamics * self.estimate + self.config.control_gain * control
        predicted_variance = dynamics**2 * self.uncertainty**2 + self.scenario.disturbance_std**2
        information_cost = INFORMATION_COSTS[action.information]
        measurement_received = self._measurement_received(action.information)

        # Correct the belief when sensing or communication succeeds.
        if measurement_received:
            noise_std = self.scenario.observation_noise
            if action.information == InformationMode.COMMUNICATE:
                noise_std *= 0.25
            measurement = self.true_state + float(self.rng.normal(0.0, noise_std))
            measurement_variance = max(noise_std**2, 1e-6)
            kalman_gain = predicted_variance / (predicted_variance + measurement_variance)
            self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
            self.uncertainty = math.sqrt(max((1 - kalman_gain) * predicted_variance, 1e-6))
            self.measurement_age = 0
        else:
            self.estimate = predicted_estimate
            self.uncertainty = math.sqrt(max(predicted_variance, 1e-6))
            self.measurement_age += 1

        self.time_step += 1
        self.cumulative_information += information_cost
        self.last_control = control
        violation = float(abs(self.true_state) > self.config.state_limit)
        reward = -(self.true_state**2 + self.config.control_effort_weight * control**2)
        return StepResult(
            observation=self.observation(),
            reward=float(reward),
            information_cost=information_cost,
            violation=violation,
            done=self.time_step >= self.config.horizon,
            true_state=self.true_state,
            control=control,
            measurement_received=measurement_received,
        )

    def _measurement_received(self, mode: InformationMode) -> bool:
        if mode == InformationMode.NONE:
            return False

        if mode == InformationMode.SENSE:
            dropout = self.scenario.sensor_dropout
        else:
            dropout = self.scenario.communication_dropout
        return bool(self.rng.random() >= dropout)
