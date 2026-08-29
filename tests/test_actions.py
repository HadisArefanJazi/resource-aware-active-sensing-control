import pytest

from resource_aware_control.actions import InformationMode, JointAction, action_count


def test_joint_actions_round_trip() -> None:
    assert action_count() == 9
    for index in range(action_count()):
        assert JointAction.decode(index).encode() == index


def test_invalid_action_is_rejected() -> None:
    with pytest.raises(ValueError):
        JointAction.decode(action_count())


def test_information_modes_are_ordered() -> None:
    assert int(InformationMode.NONE) < int(InformationMode.SENSE)
    assert int(InformationMode.SENSE) < int(InformationMode.COMMUNICATE)
