"""Research OOF raw/joint utility action, using only pre-refinement features."""
from pathlib import Path

from joint_utility_core import UtilityAdapter


METADATA = {
    "name": "utility_gate",
    "requires": ["stock", "descriptors", "intrinsics"],
    "provenance_files": ["joint_utility_core.py", "../../research/joint-utility-20261020/training/checkpoint.json"],
}


def build(config):
    if config:
        raise ValueError("The utility experiment has no configurable search parameters")
    checkpoint = Path(__file__).resolve().parents[2] / "research/joint-utility-20261020/training/checkpoint.json"
    return UtilityAdapter(checkpoint, "utility_gate")
