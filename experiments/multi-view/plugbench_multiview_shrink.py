"""Own-pair pixel-centered half-offset control, with matched support."""
from plugbench_multiview_huber import PreparedAdapter

METADATA = {"name": "multiview_shrink", "requires": ["stock", "descriptors"],
    "provenance_files": ["plugbench_multiview_huber.py", "multiview_consensus_core.py", "plugbench_multiview_transport.py",
    "../../research/multiview-consensus-20261020/prepared/manifest.json",
    "../../research/multiview-consensus-20261020/prepared/packed/bank-0000-0500.zip",
    "../../research/multiview-consensus-20261020/prepared/packed/bank-0500-1000.zip",
    "../../research/multiview-consensus-20261020/prepared/packed/bank-1000-1500.zip"]}


def build(config):
    if config:
        raise ValueError("The frozen consensus experiment has no search parameters")
    return PreparedAdapter("shrink")
