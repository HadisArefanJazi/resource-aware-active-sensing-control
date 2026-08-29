from dataclasses import dataclass
from enum import IntEnum

CONTROL_LEVELS = (-1.0, 0.0, 1.0)


class InformationMode(IntEnum):
    NONE = 0
    SENSE = 1
    COMMUNICATE = 2


@dataclass(frozen=True)
class JointAction:
    control: float
    information: InformationMode

    @classmethod
    def decode(cls, index: int) -> "JointAction":
        total_actions = action_count()
        if index < 0 or index >= total_actions:
            raise ValueError(f"Action index must be in [0, {action_count() - 1}].")

        information_mode_count = len(InformationMode)
        control_index = index // information_mode_count
        information_index = index % information_mode_count
        control = CONTROL_LEVELS[control_index]
        information = InformationMode(information_index)
        return cls(control, information)

    def encode(self) -> int:
        control_index = CONTROL_LEVELS.index(self.control)
        information_mode_count = len(InformationMode)
        return control_index * information_mode_count + int(self.information)


def action_count() -> int:
    return len(CONTROL_LEVELS) * len(InformationMode)
