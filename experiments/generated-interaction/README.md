# Generated cross-match interaction

## Where this changes the pipeline

![Step diagram showing the retained inputs and the changed block](../../docs/diagrams/proxy-experiments.svg)

[Full visual guide and scope](../../docs/diagrams/README.md). Each lane is a separate experiment.

I tested whether extra image evidence between matches resolves a deliberately ambiguous local task.

**Construction.** RM029 jointly moves patches with cross-match relations. RM032 uses bridge pixels, finite displacement-state factors and four min-sum belief-propagation sweeps. Controls include local-only, independent bridge marginals, relationship-shuffled BP and simple expected-factor mean messages.

**Outcome: generated-task negative versus strongest controls.** RM029 joint 139/256 did not beat mean 140/256 or shuffle 142/256. RM032 BP succeeded on 179/256 paired units (69.92%), below mean 206/256 (80.47%); local-only was 0/256 and shuffled BP 30/256. These are generated paired success rates, not natural-image pose AUC. Mean messages already exchange cross-match information.

**Learning.** The construction contains useful interaction information, but the selected joint/BP mechanisms did not beat simpler same-data alternatives. It does not isolate four sweeps as the cause or establish pose improvement.

The historical filename026 belongs to the later RM029 S1G operator. Earlier H-route qualification and generated S1G evidence remain separate. `.snapshot` sources were renamed to `.py` without changing their bytes.

## Handpicked files

| File | Purpose |
|---|---|
| [moving-patch-rm029/moving_patch_relations_026.py](moving-patch-rm029/moving_patch_relations_026.py) | Evaluated relation operator |
| [moving-patch-rm029/test_moving_patch_relations_026.py](moving-patch-rm029/test_moving_patch_relations_026.py) | Independent fixture suite |
| [moving-patch-rm029/S1-PROSPECTIVE-029.md](moving-patch-rm029/S1-PROSPECTIVE-029.md) | Prospective protocol; generated S1G distinct from unqualified H route |
| [bridge-bp-rm032/bridge_context.py](bridge-bp-rm032/bridge_context.py) | Bridge factors, BP and controls |
| [bridge-bp-rm032/test_bridge_context.py](bridge-bp-rm032/test_bridge_context.py) | Scientific fixture suite |
| [bridge-bp-rm032/protocol.md](bridge-bp-rm032/protocol.md) | Frozen generated-task protocol |

Exact original locations, source bindings and hashes are in [PROVENANCE.json](../../PROVENANCE.json). [Back to catalogue](../../README.md).
