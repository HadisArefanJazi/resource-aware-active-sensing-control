import numpy as np
import pytest

from resource_aware_control.actions import InformationMode, JointAction
from resource_aware_control.config import EnvironmentConfig, Scenario
from resource_aware_control.environment import OBSERVATION_DIM, ActiveSensingControlEnv


def scenario(**overrides) -> Scenario:
    values = {
        "name": "test",
        "observation_noise": 0.2,
        "disturbance_std": 0.0,
        "sensor_dropout": 0.0,
        "communication_dropout": 0.0,
    }
    values.update(overrides)
    return Scenario(**values)


def test_reset_and_step_return_partial_observations() -> None:
    environment = ActiveSensingControlEnv(scenario(), seed=3)
    observation = environment.reset(initial_state=1.0)
    result = environment.step(JointAction(-1.0, InformationMode.SENSE).encode())
    assert observation.shape == (OBSERVATION_DIM,)
    assert result.observation.shape == (OBSERVATION_DIM,)
    assert result.information_cost == 1.0
    assert result.measurement_received
    assert np.isfinite(result.reward)


@pytest.mark.parametrize(
    ("mode", "cost"),
    [
        (InformationMode.NONE, 0.0),
        (InformationMode.SENSE, 1.0),
        (InformationMode.COMMUNICATE, 3.0),
    ],
)
def test_information_modes_have_explicit_costs(mode: InformationMode, cost: float) -> None:
    environment = ActiveSensingControlEnv(scenario(), seed=1)
    environment.reset(initial_state=0.5)
    result = environment.step(JointAction(0.0, mode).encode())
    assert result.information_cost == cost


def test_sensor_dropout_prevents_measurement() -> None:
    environment = ActiveSensingControlEnv(scenario(sensor_dropout=1.0), seed=1)
    environment.reset(initial_state=0.5)
    result = environment.step(JointAction(0.0, InformationMode.SENSE).encode())
    assert not result.measurement_received
    assert result.information_cost == 1.0


def test_episode_terminates_at_horizon() -> None:
    environment = ActiveSensingControlEnv(scenario(), EnvironmentConfig(horizon=2), seed=1)
    environment.reset(initial_state=0.5)
    action = JointAction(0.0, InformationMode.NONE).encode()
    assert not environment.step(action).done
    assert environment.step(action).done
    with pytest.raises(RuntimeError):
        environment.step(action)
