"""Example candidate: keep the raw source and use the stock-refined target."""
from plugbench_contract import Prediction

METADATA = {
    "name": "fixed_source",
    "requires": ["stock"],
    "provenance_files": [],
}


class FixedSource:
    def refine(self, pair):
        return Prediction(pair.raw[0], pair.stock[1])


def build(config):
    if config:
        raise ValueError("The fixed-source ablation has no tunable parameters.")
    return FixedSource()
