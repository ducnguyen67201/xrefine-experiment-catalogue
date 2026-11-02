"""No-GT count-matched whole-pair allocation control for the frozen panel."""
from pathlib import Path

from joint_utility_core import UtilityAdapter


METADATA = {
    "name": "shuffled_gate",
    "requires": ["stock"],
    "provenance_files": ["joint_utility_core.py", "../../research/joint-utility-20261020/training/checkpoint.json",
                         "../../research/joint-utility-20261020/training/shuffled.json"],
}


def build(config):
    if config:
        raise ValueError("The shuffled control has no configurable search parameters")
    directory = Path(__file__).resolve().parents[2] / "research/joint-utility-20261020/training"
    return UtilityAdapter(directory / "checkpoint.json", "shuffled_gate", directory / "shuffled.json")
