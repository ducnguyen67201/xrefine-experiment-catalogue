# Refinement benefit and endpoint actions

I tested whether we should apply a refinement, rather than assuming every refined endpoint helps.

**Whole-pair utility.** Twelve raw-match-set features and opposite-scene out-of-fold ridge predict capped pose-loss benefit. The decision uses both raw or both refined endpoints for the entire pair. Confidence and shuffled gates are explicit controls. Full 1500-pair × ten-draw AUC5: stock48.4091 → utility48.2133. Only 29 pairs selected raw; unchanged-input solver variability limits causal attribution.

**Fixed-source readout.** Retaining raw source and refined target scored43.0018 vs joint stock48.4865 in its full comparison.

**RM050 per-match action risk.** Own-match frozen features and displacements predict QQ/PP/PQ/QP actions using ridge regression. Training120pairs and held30pairs came from previously exposed ScanNet FIT rooms. Weak epipolar risk was3.115828 vs always-refine3.096052 (lower better); frozen gates failed before any pose pilot. AUC5 and severe errors were not measured.

**Learning.** Both whole-pair benefit and this specific per-match weak-target contract have been tried. Epipolar compatibility is not pose utility. These results do not reject every possible benefit target or representation.

## Handpicked files

| File | Purpose |
|---|---|
| [whole-pair/joint_utility_core.py](whole-pair/joint_utility_core.py) | Utility core or fixed control |
| [whole-pair/plugbench_utility_gate.py](whole-pair/plugbench_utility_gate.py) | Utility core or fixed control |
| [whole-pair/plugbench_confidence_gate.py](whole-pair/plugbench_confidence_gate.py) | Utility core or fixed control |
| [whole-pair/plugbench_shuffled_gate.py](whole-pair/plugbench_shuffled_gate.py) | Utility core or fixed control |
| [whole-pair/test_joint_utility.py](whole-pair/test_joint_utility.py) | Checks |
| [whole-pair/train.py](whole-pair/train.py) | Whole-pair fitting workflow |
| [whole-pair/protocol.md](whole-pair/protocol.md) | Frozen protocol |
| [fixed-source/plugbench_fixed_source.py](fixed-source/plugbench_fixed_source.py) | Fixed-source endpoint plugin |
| [per-match-rm050/math050.py](per-match-rm050/math050.py) | Action features, ridge and weak-risk arithmetic |
| [per-match-rm050/test_math050.py](per-match-rm050/test_math050.py) | Checks |
| [per-match-rm050/run.py](per-match-rm050/run.py) | Original fit/predict/screen driver; historical environment required |
| [per-match-rm050/PROTOCOL.md](per-match-rm050/PROTOCOL.md) | Frozen protocol |

Exact original locations, source bindings and hashes are in [PROVENANCE.json](../../PROVENANCE.json). [Back to catalogue](../../README.md).
