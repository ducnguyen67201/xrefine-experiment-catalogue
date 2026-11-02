"""Mean-descriptor-confidence control with the same fixed OOF utility target."""
from pathlib import Path

from joint_utility_core import UtilityAdapter


METADATA = {
    "name": "confidence_gate",
    "requires": ["stock", "descriptors"],
    "provenance_files": ["joint_utility_core.py", "../../research/joint-utility-20261020/training/checkpoint.json"],
}


def build(config):
    if config:
        raise ValueError("The confidence control has no configurable search parameters")
    checkpoint = Path(__file__).resolve().parents[2] / "research/joint-utility-20261020/training/checkpoint.json"
    return UtilityAdapter(checkpoint, "confidence_gate")
