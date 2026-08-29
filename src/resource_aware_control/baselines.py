import numpy as np
import torch

from resource_aware_control.actions import InformationMode, JointAction
from resource_aware_control.environment import ActiveSensingControlEnv
from resource_aware_control.networks import PolicyNetwork


def stabilizing_control(state_estimate: float, deadband: float = 0.15) -> float:
    if state_estimate > deadband:
        return -1.0
    if state_estimate < -deadband:
        return 1.0
    return 0.0


class Controller:
    """Common interface used by every controller."""

    def select_action(
        self,
        observation: np.ndarray,
        environment: ActiveSensingControlEnv,
    ) -> int:
        raise NotImplementedError


class RLController(Controller):
    def __init__(self, policy: PolicyNetwork, stochastic: bool = True) -> None:
        self.policy = policy
        self.stochastic = stochastic

    def select_action(
        self,
        observation: np.ndarray,
        environment: ActiveSensingControlEnv,
    ) -> int:
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(observation, dtype=torch.float32))
            if not self.stochastic:
                return int(torch.argmax(logits).item())

            probabilities = torch.softmax(logits, dim=-1).numpy()
            action = environment.rng.choice(len(probabilities), p=probabilities)
            return int(action)


class FullInformationController(Controller):
    """Oracle upper bound that communicates the true state every step."""

    def select_action(
        self,
        observation: np.ndarray,
        environment: ActiveSensingControlEnv,
    ) -> int:
        del observation
        control = stabilizing_control(environment.true_state)
        return JointAction(control, InformationMode.COMMUNICATE).encode()


class FixedSensingController(Controller):
    def __init__(self, interval: int = 4) -> None:
        self.interval = interval

    def select_action(
        self,
        observation: np.ndarray,
        environment: ActiveSensingControlEnv,
    ) -> int:
        control = stabilizing_control(environment.estimate)
        mode = InformationMode.NONE
        if environment.time_step % self.interval == 0:
            mode = InformationMode.SENSE
        return JointAction(control, mode).encode()


class ClassicalController(Controller):
    """Certainty-equivalent control with fixed local sensing."""

    def select_action(
        self,
        observation: np.ndarray,
        environment: ActiveSensingControlEnv,
    ) -> int:
        del observation
        control = stabilizing_control(environment.estimate)
        return JointAction(control, InformationMode.SENSE).encode()
